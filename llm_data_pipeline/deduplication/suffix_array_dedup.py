"""
Suffix Array Substring Dedup (Level 3) — per DATA_PIPELINE_PLAN A.8.

Per Llama 3 approach:
  - Build suffix array over corpus to find shared substrings
  - Remove documents containing >= 50 token exact substrings shared with other docs
  - Catches: template text, boilerplate, legal disclaimers, mirrored content
  - Expected reduction: 5-10%

Implementation uses a simplified approach:
  1. Extract all n-grams of length >= min_shared_tokens from each document
  2. Build inverted index of n-gram → set of document IDs
  3. Flag documents where >50% of their n-grams appear in other documents
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class SuffixArrayDeduplicator:
    """Remove documents sharing long substrings with other documents."""

    def __init__(
        self,
        min_shared_tokens: int = 50,
        max_docs_indexed: int = 1_000_000,
        ngram_skip: int = 10,
    ):
        self.min_shared_tokens = min_shared_tokens
        self.max_docs_indexed = max_docs_indexed
        self.ngram_skip = ngram_skip

        self.stats = {
            "total_documents": 0,
            "unique": 0,
            "duplicates": 0,
            "total_substrings_checked": 0,
            "shared_substrings_found": 0,
        }

    def deduplicate(self, documents: List[Dict], text_field: str = "text") -> List[Dict]:
        ngram_index: Dict[str, Set[int]] = defaultdict(set)
        doc_ngrams: Dict[int, List[str]] = {}
        is_duplicate: Dict[int, bool] = {}

        # Phase 1: Index substrings from all documents
        for idx, doc in enumerate(documents):
            if idx >= self.max_docs_indexed:
                break

            self.stats["total_documents"] += 1
            text = doc.get(text_field, "")
            tokens = self._tokenize(text)

            if len(tokens) < self.min_shared_tokens:
                doc_ngrams[idx] = []
                is_duplicate[idx] = False
                continue

            ngrams = self._extract_substrings(tokens)
            doc_ngrams[idx] = ngrams

            for ng in ngrams:
                ngram_index[ng].add(idx)
                self.stats["total_substrings_checked"] += 1

        # Phase 2: Identify duplicate documents
        for idx, ngrams in doc_ngrams.items():
            if not ngrams:
                is_duplicate[idx] = False
                continue

            shared_count = 0
            for ng in ngrams:
                shared_docs = ngram_index[ng]
                if len(shared_docs) > 1:
                    shared_count += 1
                    self.stats["shared_substrings_found"] += 1

            shared_ratio = shared_count / max(1, len(ngrams))
            is_duplicate[idx] = shared_ratio > 0.5

        # Phase 3: Collect unique documents
        unique = []
        for idx, doc in enumerate(documents):
            if idx in is_duplicate and is_duplicate[idx]:
                self.stats["duplicates"] += 1
            else:
                unique.append(doc)
                self.stats["unique"] += 1

        return unique

    def deduplicate_streaming(self, documents, text_field: str = "text", window_size: int = 500000):
        """Process in windows to limit memory usage."""
        buffer = []
        all_unique = []

        for doc in documents:
            buffer.append(doc)
            if len(buffer) >= window_size:
                unique_chunk = self.deduplicate(buffer, text_field)
                all_unique.extend(unique_chunk)
                buffer = []

        if buffer:
            unique_chunk = self.deduplicate(buffer, text_field)
            all_unique.extend(unique_chunk)

        return all_unique

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        text = re.sub(r"\s+", " ", text.lower().strip())
        return text.split()

    def _extract_substrings(self, tokens: List[str]) -> List[str]:
        """Extract token-level substrings of length min_shared_tokens."""
        ngrams = []
        step = max(1, self.ngram_skip)
        n = self.min_shared_tokens

        for i in range(0, len(tokens) - n + 1, step):
            ng = " ".join(tokens[i:i + n])
            ngrams.append(ng)

        return ngrams

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        total = max(s["total_documents"], 1)
        s["reduction_pct"] = 100.0 * s["duplicates"] / total
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n SUFFIX ARRAY DEDUP — Level 3\n{'='*60}")
        print(f"  Min shared tokens   : {self.min_shared_tokens}")
        print(f"  Documents processed : {s['total_documents']:,}")
        print(f"  Unique             : {s['unique']:,}")
        print(f"  Duplicates removed : {s['duplicates']:,}")
        print(f"  Reduction          : {s['reduction_pct']:.1f}%")
        print("=" * 60)
