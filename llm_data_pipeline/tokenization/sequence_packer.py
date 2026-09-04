"""
Sequence Packer — per DATA_PIPELINE_PLAN A.13.

Packs variable-length token sequences into fixed-length training blocks:
  - Algorithm: First-fit-decreasing bin packing (greedy)
  - Document boundary markers: Insert BOS/EOS between documents
  - Attention mask: track document boundaries per position
  - Target packing efficiency: > 99.5% token utilization

Cross-document attention is NOT allowed (document_id per position).
"""

import struct
from typing import Dict, List, Optional, Tuple


class SequencePacker:
    """Pack token sequences into fixed-length training blocks."""

    def __init__(
        self,
        seq_len: int = 256,
        bos_id: int = 0,
        eos_id: int = 0,
        pad_id: int = 15,
        packing_algorithm: str = "first_fit_decreasing",
        document_boundary_marker: bool = True,
        generate_attention_mask: bool = True,
        efficiency_target: float = 0.995,
    ):
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.algorithm = packing_algorithm
        self.doc_boundary_marker = document_boundary_marker
        self.generate_attn_mask = generate_attention_mask
        self.efficiency_target = efficiency_target

        self.stats = {
            "total_documents_packed": 0,
            "total_tokens": 0,
            "tokens_used": 0,
            "tokens_wasted": 0,
            "sequences_produced": 0,
            "documents_truncated": 0,
        }

    def pack(self, documents: List[Dict], token_ids_field: str = "token_ids",
             return_attention_mask: bool = True) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        """Pack variable-length documents into fixed-length sequences.

        Returns (packed_sequences, attention_masks) or (packed_sequences, None).
        Each sequence is length seq_len.
        Attention mask: [seq_len] with document_id per position.
        """
        token_lists = []
        for doc in documents:
            ids = doc.get(token_ids_field, doc.get("text", ""))
            if isinstance(ids, str):
                continue
            if ids and len(ids) > 0:
                token_lists.append(list(ids))
                self.stats["total_documents_packed"] += 1
                self.stats["total_tokens"] += len(ids)

        if not token_lists:
            return [], None if return_attention_mask else None

        # Sort by length (decreasing) for first-fit-decreasing
        if self.algorithm == "first_fit_decreasing":
            token_lists.sort(key=len, reverse=True)

        sequences = []
        attention_masks = [] if return_attention_mask else None
        current_seq = []
        current_mask = [] if return_attention_mask else None
        current_doc_id = 0

        for doc_ids in token_lists:
            doc_len = len(doc_ids)

            # Check if document fits in current sequence
            space_left = self.seq_len - len(current_seq)

            if doc_len <= space_left:
                # Fits — add to current sequence
                if self.doc_boundary_marker and current_seq:
                    current_seq.append(self.eos_id)
                    if current_mask is not None:
                        current_mask.append(-1)  # boundary marker

                current_seq.extend(doc_ids)
                if current_mask is not None:
                    current_mask.extend([current_doc_id] * doc_len)
                current_doc_id += 1
            else:
                # Doesn't fit — finalize current sequence and start new
                if current_seq:
                    self._finalize_sequence(current_seq, current_mask, sequences, attention_masks)

                # New sequence
                current_seq = []
                current_mask = [] if return_attention_mask else None

                if doc_len <= self.seq_len:
                    current_seq = list(doc_ids)
                    if current_mask is not None:
                        current_mask = [current_doc_id] * doc_len
                    current_doc_id += 1
                else:
                    # Document too long — truncate
                    current_seq = list(doc_ids[:self.seq_len])
                    if current_mask is not None:
                        current_mask = [current_doc_id] * self.seq_len
                    current_doc_id += 1
                    self.stats["documents_truncated"] += 1

        # Finalize last sequence
        if current_seq:
            self._finalize_sequence(current_seq, current_mask, sequences, attention_masks)

        return sequences, attention_masks

    def _finalize_sequence(self, seq: List[int], mask: Optional[List[int]],
                           sequences: List[List[int]], masks: Optional[List[List[int]]]):
        """Pad and finalize a sequence."""
        pad_len = self.seq_len - len(seq)
        seq.extend([self.pad_id] * pad_len)
        sequences.append(seq)
        self.stats["sequences_produced"] += 1
        self.stats["tokens_used"] += len(seq) - pad_len
        self.stats["tokens_wasted"] += pad_len

        if masks is not None and mask is not None:
            mask.extend([-2] * pad_len)
            masks.append(mask)

    def pack_to_binary(self, documents: List[Dict],
                       token_ids_field: str = "token_ids") -> Tuple[bytes, Optional[bytes]]:
        """Pack documents into binary token arrays.

        Returns (token_bytes, attention_mask_bytes) as binary blobs.
        Uses uint32 for token IDs and int32 for attention masks.
        """
        sequences, masks = self.pack(documents, token_ids_field, return_attention_mask=True)

        tokens_binary = b"".join(
            struct.pack(f">{len(seq)}I", *seq) for seq in sequences
        )

        mask_binary = None
        if masks:
            mask_binary = b"".join(
                struct.pack(f">{len(m)}i", *m) for m in masks
            )

        return tokens_binary, mask_binary

    def get_efficiency(self) -> float:
        """Return current packing efficiency (0-1)."""
        total = self.stats["tokens_used"] + self.stats["tokens_wasted"]
        if total == 0:
            return 0.0
        return self.stats["tokens_used"] / total

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        s["packing_efficiency"] = self.get_efficiency()
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n SEQUENCE PACKING SUMMARY\n{'='*60}")
        print(f"  Seq length          : {self.seq_len}")
        print(f"  Algorithm           : {self.algorithm}")
        print(f"  Documents packed    : {s['total_documents_packed']:,}")
        print(f"  Total tokens        : {s['total_tokens']:,}")
        print(f"  Sequences produced  : {s['sequences_produced']:,}")
        print(f"  Tokens used         : {s['tokens_used']:,}")
        print(f"  Tokens wasted       : {s['tokens_wasted']:,}")
        print(f"  Packing efficiency  : {s['packing_efficiency']:.4f} ({100*s['packing_efficiency']:.2f}%)")
        print(f"  Target              : > {self.efficiency_target:.4f}")
        print("=" * 60)
