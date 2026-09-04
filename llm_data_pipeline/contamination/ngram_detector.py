"""
N-Gram Contamination Detector — per GPT-4 contamination methodology.

Extracts all 13-grams from benchmark test sets and flags any training
document that shares ≥1 matching 13-gram with a benchmark.

Approach:
  1. Build a Bloom filter or in-memory set of all benchmark 13-grams.
  2. Stream training documents, extract their 13-grams.
  3. Flag documents with any overlap.
  4. Optionally also check via embedding cosine similarity.
"""

import re
import mmh3
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from ..common import text_sha256


class NgramContaminationDetector:
    """Detect n-gram overlap between training data and benchmark test sets."""

    def __init__(
        self,
        ngram_size: int = 13,
        use_bloom_filter: bool = False,
        bloom_size: int = 10_000_000,
        bloom_hashes: int = 5,
    ):
        self.ngram_size = ngram_size
        self.use_bloom = use_bloom_filter
        self.bloom_size = bloom_size
        self.bloom_hashes = bloom_hashes

        # Storage for benchmark n-grams
        self._ngram_set: Set[int] = set()
        self._bloom: List[int] = []
        self._tokenizer_pattern = re.compile(r"\S+")

        self.stats = {
            "benchmark_ngrams_inserted": 0,
            "documents_scanned": 0,
            "documents_contaminated": 0,
            "total_overlaps_found": 0,
            "contaminated_by_benchmark": defaultdict(int),
        }

    def build_index(self, benchmark_texts: List[str], benchmark_name: str = "all"):
        """Build the n-gram index from all benchmark reference texts."""
        ngrams_added = 0
        for text in benchmark_texts:
            tokens = self._tokenize(text)
            for i in range(len(tokens) - self.ngram_size + 1):
                ngram = " ".join(tokens[i:i + self.ngram_size])
                ngrams_added += 1
                if self.use_bloom:
                    self._bloom_add(ngram)
                else:
                    self._ngram_set.add(hash(ngram))
        self.stats["benchmark_ngrams_inserted"] += ngrams_added

    def check_document(self, doc_id: str, text: str) -> Tuple[bool, int]:
        """Return (is_contaminated, overlap_count) for a single document."""
        tokens = self._tokenize(text)
        if len(tokens) < self.ngram_size:
            return False, 0

        overlaps = 0
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = " ".join(tokens[i:i + self.ngram_size])
            h = hash(ngram)
            if self.use_bloom:
                if self._bloom_check(ngram):
                    overlaps += 1
            else:
                if h in self._ngram_set:
                    overlaps += 1

        self.stats["documents_scanned"] += 1
        if overlaps > 0:
            self.stats["documents_contaminated"] += 1
            self.stats["total_overlaps_found"] += overlaps
            return True, overlaps
        return False, 0

    def check_corpus(
        self, documents: List[Dict], text_field: str = "text"
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split corpus into (clean_docs, contaminated_docs)."""
        clean = []
        contaminated = []
        for doc in documents:
            doc_id = doc.get("doc_id", text_sha256(doc.get(text_field, "")))
            text = doc.get(text_field, "")
            is_contaminated, overlap_count = self.check_document(doc_id, text)
            if is_contaminated:
                doc["contamination_overlaps"] = overlap_count
                contaminated.append(doc)
            else:
                clean.append(doc)
        return clean, contaminated

    def _bloom_add(self, item: str):
        """Add item to Bloom filter."""
        if not self._bloom:
            self._bloom = [0] * self.bloom_size
        for seed in range(self.bloom_hashes):
            idx = mmh3.hash(item, seed) % self.bloom_size
            self._bloom[idx] = 1

    def _bloom_check(self, item: str) -> bool:
        """Check if item *might* be in Bloom filter."""
        if not self._bloom:
            return False
        for seed in range(self.bloom_hashes):
            idx = mmh3.hash(item, seed) % self.bloom_size
            if self._bloom[idx] == 0:
                return False
        return True

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization after lowercasing."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return self._tokenizer_pattern.findall(text)

    def get_stats(self) -> Dict:
        """Return contamination statistics."""
        stats = dict(self.stats)
        stats["contaminated_by_benchmark"] = dict(stats["contaminated_by_benchmark"])
        if stats["documents_scanned"] > 0:
            stats["contamination_pct"] = 100.0 * stats["documents_contaminated"] / stats["documents_scanned"]
        return stats

    def print_summary(self):
        """Print contamination detection summary."""
        s = self.get_stats()
        print("\n" + "=" * 60)
        print(" N-GRAM CONTAMINATION DETECTION RESULTS ")
        print("=" * 60)
        print(f"  N-gram size          : {self.ngram_size}")
        print(f"  Benchmark n-grams    : {s['benchmark_ngrams_inserted']:,}")
        print(f"  Documents scanned    : {s['documents_scanned']:,}")
        print(f"  Contaminated docs    : {s['documents_contaminated']:,}")
        print(f"  Contamination rate   : {s.get('contamination_pct', 0):.4f}%")
        print(f"  Total n-gram overlaps: {s['total_overlaps_found']:,}")
        print("=" * 60)
