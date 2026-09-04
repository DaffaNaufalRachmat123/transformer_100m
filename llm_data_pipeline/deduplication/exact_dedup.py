"""
Exact Deduplication (Level 1) — per DATA_PIPELINE_PLAN A.8.

SHA-256 on normalized text. Keep first occurrence, remove all others.
Expected reduction: 5-10%.
"""

from typing import Dict, List, Set

from ..common import text_sha256


class ExactDeduplicator:
    """Remove exact duplicate documents by SHA-256 hash."""

    def __init__(self):
        self.stats = {
            "total_documents": 0,
            "unique": 0,
            "duplicates": 0,
            "hash_collisions": 0,
        }

    def deduplicate(self, documents: List[Dict], text_field: str = "text") -> List[Dict]:
        seen: Set[str] = set()
        unique = []

        for doc in documents:
            text = doc.get(text_field, "")
            self.stats["total_documents"] += 1

            digest = text_sha256(text)

            if digest in seen:
                self.stats["duplicates"] += 1
                continue

            seen.add(digest)
            doc_out = dict(doc)
            doc_out["sha256"] = digest[:16]
            unique.append(doc_out)
            self.stats["unique"] += 1

        return unique

    def deduplicate_streaming(self, documents, text_field: str = "text"):
        """Generator version for streaming through large datasets."""
        seen: Set[str] = set()
        for doc in documents:
            text = doc.get(text_field, "")
            self.stats["total_documents"] += 1
            digest = text_sha256(text)
            if digest in seen:
                self.stats["duplicates"] += 1
                continue
            seen.add(digest)
            self.stats["unique"] += 1
            doc["sha256"] = digest[:16]
            yield doc

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        total = max(s["total_documents"], 1)
        s["reduction_pct"] = 100.0 * s["duplicates"] / total
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n EXACT DEDUP (SHA-256) — Level 1\n{'='*60}")
        print(f"  Documents processed: {s['total_documents']:,}")
        print(f"  Unique             : {s['unique']:,}")
        print(f"  Duplicates removed : {s['duplicates']:,}")
        print(f"  Reduction          : {s['reduction_pct']:.1f}%")
        print("=" * 60)
