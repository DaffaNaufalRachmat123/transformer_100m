"""4-Level deduplication per DATA_PIPELINE_PLAN A.8.

Run order (each level catches things the previous missed):
  1. exact_dedup       — SHA-256 on normalized text (~5-10% reduction)
  2. minhash_dedup     — MinHash + LSH, Jaccard >= 0.8 (~10-20% reduction)
  3. suffix_array_dedup — >=50 token substrings across docs (~5-10% reduction)
  4. paragraph_dedup   — Paragraph hash, >50% shared → remove (~5-10% reduction)

Total expected reduction: 25-40% of raw corpus.
"""

from .exact_dedup import ExactDeduplicator
from .minhash_dedup import MinHashDeduplicator
from .suffix_array_dedup import SuffixArrayDeduplicator
from .paragraph_dedup import ParagraphDeduplicator

__all__ = [
    "ExactDeduplicator", "MinHashDeduplicator",
    "SuffixArrayDeduplicator", "ParagraphDeduplicator",
]
