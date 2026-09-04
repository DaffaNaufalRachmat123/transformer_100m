"""
Index Builder — per DATA_PIPELINE_PLAN A.14.

Builds global index mapping: global_idx → (shard_id, offset_in_samples)
Supports efficient random access and deterministic distributed assignment.

Index format:
  - index.bin: array of (shard_id: uint32, offset: uint32) pairs
  - index.json: metadata (version, total_sequences, shard_paths, config_hash)
"""

import hashlib
import json
import os
import struct
from datetime import datetime
from typing import Dict, List, Optional, Tuple


INDEX_ENTRY_FORMAT = "<II"
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FORMAT)


class IndexBuilder:
    """Build global index for binary shard dataset."""

    def __init__(
        self,
        output_dir: str = "data/corpus_final",
        version_prefix: str = "pretrain_v1",
    ):
        self.output_dir = output_dir
        self.version_prefix = version_prefix
        self._config_hash = ""

    def build(
        self,
        shard_paths_by_language: Dict[str, List[str]],
        shard_writer,
        config: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """Build global index from sharded files.

        Returns (index_bin_path, index_json_path).
        """
        all_shards = []
        for lang, paths in shard_paths_by_language.items():
            for path in paths:
                header = shard_writer.read_shard_header(path)
                all_shards.append(header)

        # Sort shards for deterministic ordering
        all_shards.sort(key=lambda s: (s["language"], s["shard_idx"]))

        # Build index entries
        index_path = os.path.join(self.output_dir, "index.bin")
        global_idx = 0
        total_sequences = 0

        with open(index_path, "wb") as f:
            for shard in all_shards:
                n_samples = shard["n_samples"]
                for offset in range(n_samples):
                    entry = struct.pack(INDEX_ENTRY_FORMAT, shard["shard_idx"], offset)
                    f.write(entry)
                global_idx += n_samples
                total_sequences += n_samples

        # Write metadata
        metadata = {
            "version": f"{self.version_prefix}_{datetime.now().strftime('%Y%m%d')}",
            "total_sequences": total_sequences,
            "total_tokens": total_sequences * all_shards[0]["seq_len"] if all_shards else 0,
            "seq_len": all_shards[0]["seq_len"] if all_shards else 0,
            "num_shards": len(all_shards),
            "shard_paths": {s["path"]: s for s in all_shards},
            "languages": sorted(set(s["language"] for s in all_shards)),
            "config_hash": self._compute_config_hash(config),
            "created_at": datetime.now().isoformat(),
        }

        meta_path = os.path.join(self.output_dir, "index.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[index] Built index: {total_sequences:,} sequences across {len(all_shards)} shards")
        return index_path, meta_path

    def distributed_assignment(
        self,
        index_meta_path: str,
        world_size: int,
        rank: int,
        seed: int = 42,
    ) -> List[int]:
        """Compute deterministic shard assignment for distributed training.

        Each rank gets shard_id % world_size == rank.
        Returns list of global_indices assigned to this rank.
        """
        with open(index_meta_path, "r") as f:
            meta = json.load(f)

        total = meta["total_sequences"]
        shard_paths = meta["shard_paths"]

        assigned = []
        for global_idx in range(total):
            with open(os.path.join(self.output_dir, "index.bin"), "rb") as f:
                f.seek(global_idx * INDEX_ENTRY_SIZE)
                shard_id, offset = struct.unpack(INDEX_ENTRY_FORMAT, f.read(INDEX_ENTRY_SIZE))

            if shard_id % world_size == rank:
                assigned.append(global_idx)

        return assigned

    def lookup(self, global_idx: int, index_bin_path: str = None) -> Tuple[int, int]:
        """Look up (shard_id, offset) for a global index."""
        path = index_bin_path or os.path.join(self.output_dir, "index.bin")
        with open(path, "rb") as f:
            f.seek(global_idx * INDEX_ENTRY_SIZE)
            return struct.unpack(INDEX_ENTRY_FORMAT, f.read(INDEX_ENTRY_SIZE))

    def _compute_config_hash(self, config: Optional[Dict]) -> str:
        """Compute a hash of the pipeline config for reproducibility."""
        if config is None:
            return "no_config"
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
