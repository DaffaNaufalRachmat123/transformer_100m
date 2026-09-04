"""
Shard Writer — per DATA_PIPELINE_PLAN A.14.

Writes packed token sequences to memory-mapped binary files:
  - Shard size: 256MB–1GB per shard file
  - Format: raw uint32 arrays (numpy .bin compatible)
  - Each shard: header (magic, version, seq_len, n_samples, language) + token data
  - Supports mmap for zero-copy reading by C++ dataloader
"""

import json
import os
import struct
from typing import Dict, List, Optional, Tuple


SHARD_MAGIC = b"TS650M\x00\x00"
SHARD_VERSION = 2
SHARD_HEADER_FORMAT = "<8sIIII32s"
SHARD_HEADER_SIZE = struct.calcsize(SHARD_HEADER_FORMAT)


class ShardWriter:
    """Write token sequences to binary shard files."""

    def __init__(
        self,
        output_dir: str = "data/corpus_final",
        shard_size_mb: int = 512,
        seq_len: int = 256,
        mmap_enabled: bool = True,
    ):
        self.output_dir = output_dir
        self.shard_size_bytes = shard_size_mb * 1024 * 1024
        self.seq_len = seq_len
        self.mmap_enabled = mmap_enabled

        self.stats = {
            "total_shards": 0,
            "total_sequences": 0,
            "total_tokens": 0,
            "total_bytes": 0,
            "shards_by_language": {},
        }

    def write_shards(
        self,
        sequences: List[List[int]],
        language: str = "en",
        metadata: Optional[Dict] = None,
    ) -> List[str]:
        """Write token sequences to shard files for a language.

        Returns list of shard file paths.
        """
        lang_dir = os.path.join(self.output_dir, language)
        os.makedirs(lang_dir, exist_ok=True)

        seq_bytes = self.seq_len * 4  # 4 bytes per uint32
        seqs_per_shard = max(1, self.shard_size_bytes // (seq_bytes + 16))

        shard_paths = []
        shard_idx = 0
        n_seqs = len(sequences)

        for start in range(0, n_seqs, seqs_per_shard):
            chunk = sequences[start:start + seqs_per_shard]
            path = os.path.join(lang_dir, f"shard_{shard_idx:06d}.bin")

            self._write_shard_file(path, chunk, language, shard_idx, metadata)
            shard_paths.append(path)

            self.stats["total_shards"] += 1
            self.stats["total_sequences"] += len(chunk)
            self.stats["total_tokens"] += len(chunk) * self.seq_len
            self.stats["total_bytes"] += os.path.getsize(path)
            self.stats["shards_by_language"][language] = self.stats["shards_by_language"].get(language, 0) + 1

            shard_idx += 1

        print(f"[shard_writer] {language}: wrote {shard_idx} shards, {n_seqs:,} sequences")
        return shard_paths

    def write_shards_from_documents(
        self,
        documents: List[Dict],
        language: str = "en",
        token_ids_field: str = "token_ids",
    ) -> List[str]:
        """Extract token_ids from documents and write shards."""
        sequences = []
        for doc in documents:
            ids = doc.get(token_ids_field)
            if ids and len(ids) == self.seq_len:
                sequences.append(ids)

        return self.write_shards(sequences, language, {"source": "pipeline"})

    def write_all_languages(
        self,
        documents_by_language: Dict[str, List[Dict]],
        token_ids_field: str = "token_ids",
    ) -> Dict[str, List[str]]:
        """Write shards for all languages. Returns {lang: [paths]}."""
        all_paths = {}
        for lang, docs in documents_by_language.items():
            paths = self.write_shards_from_documents(docs, lang, token_ids_field)
            all_paths[lang] = paths
        return all_paths

    def _write_shard_file(self, path: str, sequences: List[List[int]],
                          language: str, shard_idx: int,
                          metadata: Optional[Dict] = None):
        """Write a single shard binary file with header."""
        n_samples = len(sequences)
        lang_encoded = language.encode("utf-8")[:32].ljust(32, b"\x00")

        header = struct.pack(
            SHARD_HEADER_FORMAT,
            SHARD_MAGIC,
            SHARD_VERSION,
            self.seq_len,
            n_samples,
            shard_idx,
            lang_encoded,
        )

        with open(path, "wb") as f:
            f.write(header)

            for seq in sequences:
                if len(seq) != self.seq_len:
                    padded = list(seq) + [0] * (self.seq_len - len(seq))
                    f.write(struct.pack(f"<{self.seq_len}I", *padded[:self.seq_len]))
                else:
                    f.write(struct.pack(f"<{self.seq_len}I", *seq))

            # Write metadata footer
            if metadata:
                meta_bytes = json.dumps(metadata).encode("utf-8")
                f.write(struct.pack("<I", len(meta_bytes)))
                f.write(meta_bytes)
            else:
                f.write(struct.pack("<I", 0))

    def read_shard_header(self, path: str) -> Dict:
        """Read header from a shard file."""
        with open(path, "rb") as f:
            data = f.read(SHARD_HEADER_SIZE)
            magic, version, seq_len, n_samples, shard_idx, lang_bytes = struct.unpack(
                SHARD_HEADER_FORMAT, data
            )
            language = lang_bytes.rstrip(b"\x00").decode("utf-8")

        return {
            "path": path,
            "magic": magic,
            "version": version,
            "seq_len": seq_len,
            "n_samples": n_samples,
            "shard_idx": shard_idx,
            "language": language,
            "size_bytes": os.path.getsize(path),
            "tokens": n_samples * seq_len,
        }

    def read_shard_tokens(self, path: str, sample_idx: int) -> List[int]:
        """Read a single sample from a shard file."""
        header = self.read_shard_header(path)
        offset = SHARD_HEADER_SIZE + sample_idx * header["seq_len"] * 4
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(header["seq_len"] * 4)
            return list(struct.unpack(f"<{header['seq_len']}I", raw))

    def get_stats(self) -> Dict:
        return dict(self.stats)

    def print_summary(self):
        s = self.stats
        print(f"\n{'='*60}\n BINARY SHARD WRITER SUMMARY\n{'='*60}")
        print(f"  Shard size          : {self.shard_size_bytes // (1024*1024)} MB")
        print(f"  Seq length          : {self.seq_len}")
        print(f"  Total shards        : {s['total_shards']:,}")
        print(f"  Total sequences     : {s['total_sequences']:,}")
        print(f"  Total tokens        : {s['total_tokens']:,}")
        print(f"  Total written       : {s['total_bytes'] // (1024*1024):,} MB")
        print("-" * 60)
        for lang, count in s.get("shards_by_language", {}).items():
            print(f"  {lang:<15s}: {count:>6,} shards")
        print("=" * 60)
