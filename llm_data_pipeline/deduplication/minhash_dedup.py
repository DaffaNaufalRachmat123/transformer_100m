"""
MinHash + LSH Near-Duplicate Detection (Level 2) — per DATA_PIPELINE_PLAN A.8.

Configuration:
  - 128 hash functions, 20 bands × 6.4 rows
  - Jaccard similarity threshold: >= 0.8
  - Character-level 5-grams as features
  - Expected reduction: 10-20%

Uses datasketch library for MinHash LSH implementation.
"""

import re
from typing import Dict, List, Set, Tuple


class MinHashDeduplicator:
    """Near-duplicate detection via MinHash + LSH."""

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5,
        num_bands: int = 20,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.num_bands = num_bands

        self.stats = {
            "total_documents": 0,
            "unique": 0,
            "near_duplicates": 0,
            "clusters_found": 0,
        }

    def deduplicate(self, documents: List[Dict], text_field: str = "text") -> List[Dict]:
        try:
            from datasketch import MinHash, MinHashLSH
            import mmh3
        except ImportError:
            raise ImportError("Install datasketch + mmh3: pip install datasketch mmh3")

        lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            params=(self.num_bands, self.num_perm // self.num_bands),
        )

        doc_minhashes: Dict[int, MinHash] = {}
        doc_indices: List[int] = []
        seen_doc_ids: Set[str] = set()

        for idx, doc in enumerate(documents):
            self.stats["total_documents"] += 1
            text = doc.get(text_field, "")
            m = self._compute_minhash(text)

            doc_minhashes[idx] = m
            doc_indices.append(idx)

            # Check against existing docs
            candidates = lsh.query(m)
            is_dup = False

            for cand_idx in candidates:
                cand_m = doc_minhashes.get(cand_idx)
                if cand_m and m.jaccard(cand_m) >= self.threshold:
                    is_dup = True
                    self.stats["near_duplicates"] += 1
                    break

            if not is_dup:
                lsh.insert(idx, m)

        # Collect unique documents
        unique = []
        seen = set()
        for idx, doc in enumerate(documents):
            self.stats["total_documents"] += 1
            text = doc.get(text_field, "")
            m = doc_minhashes[idx]

            is_dup = False
            for existing_idx in seen:
                existing_m = doc_minhashes.get(existing_idx)
                if existing_m and m.jaccard(existing_m) >= self.threshold:
                    is_dup = True
                    break

            if not is_dup:
                seen.add(idx)
                unique.append(doc)
                self.stats["unique"] += 1
            else:
                self.stats["near_duplicates"] += 1
                self.stats["total_documents"] -= 1  # counted once in loop above

        return unique

    def _compute_minhash(self, text: str):
        """Compute MinHash from character n-grams of text."""
        from datasketch import MinHash
        import mmh3

        m = MinHash(num_perm=self.num_perm)
        ngrams = self._extract_ngrams(text, self.ngram_size)
        for ng in ngrams:
            m.update(ng.encode("utf-8"))
        return m

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract character-level n-grams, lowercased and whitespace-normalized."""
        text = re.sub(r"\s+", " ", text.lower().strip())
        if len(text) < n:
            return [text]
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def deduplicate_streaming(self, documents, text_field: str = "text", max_docs: int = 5000000):
        """Memory-bounded streaming dedup. Resets LSH index when doc count exceeds max_docs."""
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            raise ImportError("Install datasketch + mmh3: pip install datasketch mmh3")

        lsh = MinHashLSH(
            threshold=self.threshold, num_perm=self.num_perm,
            params=(self.num_bands, self.num_perm // self.num_bands),
        )
        doc_minhashes: Dict[int, MinHash] = {}
        batch_count = 0

        for doc in documents:
            self.stats["total_documents"] += 1
            batch_count += 1

            # Reset LSH when memory bound
            if batch_count >= max_docs:
                lsh = MinHashLSH(
                    threshold=self.threshold, num_perm=self.num_perm,
                    params=(self.num_bands, self.num_perm // self.num_bands),
                )
                doc_minhashes.clear()
                batch_count = 0

            text = doc.get(text_field, "")
            m = self._compute_minhash(text)
            doc_id = self.stats["total_documents"]
            doc_minhashes[doc_id] = m

            candidates = lsh.query(m)
            is_dup = False
            for cand_id in candidates:
                cand_m = doc_minhashes.get(cand_id)
                if cand_m and m.jaccard(cand_m) >= self.threshold:
                    is_dup = True
                    self.stats["near_duplicates"] += 1
                    break

            if not is_dup:
                lsh.insert(doc_id, m)
                self.stats["unique"] += 1
                yield doc

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        total = max(s["total_documents"], 1)
        s["reduction_pct"] = 100.0 * s["near_duplicates"] / total
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n MINHASH DEDUP — Level 2\n{'='*60}")
        print(f"  Num perm / bands    : {self.num_perm} / {self.num_bands}")
        print(f"  Jaccard threshold   : {self.threshold}")
        print(f"  N-gram size         : {self.ngram_size}")
        print(f"  Documents processed : {s['total_documents']:,}")
        print(f"  Unique             : {s['unique']:,}")
        print(f"  Near-duplicates    : {s['near_duplicates']:,}")
        print(f"  Reduction          : {s['reduction_pct']:.1f}%")
        print("=" * 60)
