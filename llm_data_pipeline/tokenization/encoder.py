"""
Tokenization Encoder — per DATA_PIPELINE_PLAN A.13.

BPE encodes all documents using the trained tokenizer (vocab_6/).
Inserts BOS at document start and EOS at document end.
Handles batch encoding with length tracking.
"""

import os
import re
from typing import Dict, List, Optional, Tuple


class TokenizationEncoder:
    """Encode text into BPE token IDs using the project's vocab_6 tokenizer."""

    def __init__(
        self,
        vocab_path: str = "vocab_6/vocab.txt",
        merges_path: str = "vocab_6/merges.txt",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|pad|>",
        bos_id: int = 0,
        eos_id: int = 0,
        pad_id: int = 15,
        max_seq_len: int = 2048,
    ):
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab_size = 0

        self.stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "truncated": 0,
            "empty": 0,
        }

    def load_vocab(self):
        """Load BPE vocabulary and merges from vocab_6/."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocab not found: {self.vocab_path}")
        if not os.path.exists(self.merges_path):
            raise FileNotFoundError(f"Merges not found: {self.merges_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx
                self.id_to_token[idx] = token

        self.vocab_size = len(self.vocab)

        with open(self.merges_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = idx

        print(f"[tokenization] Loaded vocab: {self.vocab_size:,} tokens, {len(self.merges):,} merges")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True,
               max_len: Optional[int] = None) -> List[int]:
        """BPE encode a single text string.

        Returns token IDs. Truncates to max_len if specified.
        """
        if not text or not text.strip():
            self.stats["empty"] += 1
            ids = []
            if add_bos:
                ids.append(self.bos_id)
            if add_eos:
                ids.append(self.eos_id)
            return ids

        encoded = self._bpe_encode(text)

        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(encoded)
        if add_eos:
            ids.append(self.eos_id)

        max_l = max_len or self.max_seq_len
        if len(ids) > max_l:
            ids = ids[:max_l - 1] + [self.eos_id]
            self.stats["truncated"] += 1

        self.stats["total_documents"] += 1
        self.stats["total_tokens"] += len(ids)

        return ids

    def encode_documents(self, documents: List[Dict], text_field: str = "text",
                         add_bos: bool = True, add_eos: bool = True) -> List[Dict]:
        """Encode a batch of documents, adding token_ids field."""
        results = []
        for doc in documents:
            text = doc.get(text_field, "")
            token_ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            doc_out = dict(doc)
            doc_out["token_ids"] = token_ids
            doc_out["token_count"] = len(token_ids)
            results.append(doc_out)
        return results

    def encode_to_binary(self, documents: List[Dict], text_field: str = "text") -> List[bytes]:
        """Encode documents to binary token arrays."""
        results = []
        for doc in documents:
            text = doc.get(text_field, "")
            ids = self.encode(text)
            import struct
            binary = struct.pack(f">{len(ids)}I", *ids)
            results.append(binary)
        return results

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = {self.bos_id, self.eos_id, self.pad_id}

        tokens = []
        for tid in ids:
            if skip_special and tid in special_ids:
                continue
            token = self.id_to_token.get(tid, "")
            if token:
                tokens.append(token)

        # Simple byte-level decoding (GPT-2 style)
        result = "".join(tokens)
        result = bytearray([self.vocab.get(t, 0) & 0xFF for t in tokens if t in self.vocab]).decode("utf-8", errors="replace")
        return result

    def _bpe_encode(self, text: str) -> List[int]:
        """Apply BPE merges to encode text into token IDs."""
        words = re.findall(r"\S+\s*", text)
        if not words:
            words = [text]

        ids = []
        for word in words:
            chars = list(word)
            while len(chars) > 0:
                # Find the best merge pair
                best_pair = None
                best_rank = float("inf")

                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    rank = self.merges.get(pair, float("inf"))
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = i

                if best_pair is None:
                    # No more merges — convert remaining chars to IDs
                    for ch in chars:
                        tid = self.vocab.get(ch)
                        if tid is not None:
                            ids.append(tid)
                        else:
                            # Byte fallback
                            for b in ch.encode("utf-8"):
                                byte_token = f"<0x{b:02X}>"
                                ids.append(self.vocab.get(byte_token, self.pad_id))
                    break

                # Apply the best merge
                i = best_pair
                merged = chars[i] + chars[i + 1]
                chars = chars[:i] + [merged] + chars[i + 2:]

        return ids

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        total_docs = max(s["total_documents"], 1)
        s["avg_tokens_per_doc"] = s["total_tokens"] / total_docs
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n TOKENIZATION SUMMARY\n{'='*60}")
        print(f"  Vocab size          : {self.vocab_size:,}")
        print(f"  Documents encoded   : {s['total_documents']:,}")
        print(f"  Total tokens        : {s['total_tokens']:,}")
        print(f"  Avg tokens/doc      : {s.get('avg_tokens_per_doc', 0):.1f}")
        print(f"  Truncated           : {s['truncated']:,}")
        print(f"  Empty documents     : {s['empty']:,}")
        print("=" * 60)
