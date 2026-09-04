"""
Paragraph-Level Dedup (Level 4) — per DATA_PIPELINE_PLAN A.8.

Hash individual paragraphs (SHA-256).
If >50% of a document's paragraphs appear in other documents → remove.
Catches: articles that remix paragraphs from multiple sources.
"""

import re
from typing import Dict, List, Set, Tuple

from ..common import text_sha256


class ParagraphDeduplicator:
    """Remove documents whose paragraphs are mostly duplicates of other documents."""

    def __init__(self, max_paragraph_share: float = 0.5):
        self.max_paragraph_share = max_paragraph_share

        self.stats = {
            "total_documents": 0,
            "unique": 0,
            "duplicates": 0,
            "total_paragraphs": 0,
            "unique_paragraphs": 0,
            "shared_paragraphs_found": 0,
        }

    def deduplicate(self, documents: List[Dict], text_field: str = "text") -> List[Dict]:
        para_index: Dict[str, Set[int]] = {}
        doc_paragraphs: Dict[int, List[str]] = {}

        # Phase 1: Index all paragraphs
        for idx, doc in enumerate(documents):
            self.stats["total_documents"] += 1
            text = doc.get(text_field, "")
            paragraphs = self._split_paragraphs(text)
            doc_paragraphs[idx] = paragraphs
            self.stats["total_paragraphs"] += len(paragraphs)

            for para in paragraphs:
                para_hash = text_sha256(para)
                self.stats["unique_paragraphs"] += 1
                if para_hash not in para_index:
                    para_index[para_hash] = set()
                para_index[para_hash].add(idx)

        # Phase 2: Flag documents with high paragraph overlap
        unique = []
        for idx, doc in enumerate(documents):
            paragraphs = doc_paragraphs.get(idx, [])
            if len(paragraphs) < 2:
                unique.append(doc)
                self.stats["unique"] += 1
                continue

            shared_count = 0
            for para in paragraphs:
                para_hash = text_sha256(para)
                owners = para_index.get(para_hash, set())
                if len(owners) > 1:
                    shared_count += 1
                    self.stats["shared_paragraphs_found"] += 1

            share_ratio = shared_count / len(paragraphs)

            if share_ratio > self.max_paragraph_share:
                self.stats["duplicates"] += 1
            else:
                unique.append(doc)
                self.stats["unique"] += 1

        return unique

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs by double newlines or single newlines."""
        paragraphs = re.split(r"\n{2,}", text.strip())
        result = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # skip very short paragraphs
                result.append(para)
        return result

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        total_docs = max(s["total_documents"], 1)
        total_paras = max(s["total_paragraphs"], 1)
        s["reduction_pct"] = 100.0 * s["duplicates"] / total_docs
        s["avg_paragraphs_per_doc"] = s["total_paragraphs"] / total_docs
        s["paragraph_uniqueness"] = 100.0 * s["unique_paragraphs"] / total_paras
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n PARAGRAPH DEDUP — Level 4\n{'='*60}")
        print(f"  Max paragraph share : {self.max_paragraph_share}")
        print(f"  Documents processed : {s['total_documents']:,}")
        print(f"  Unique             : {s['unique']:,}")
        print(f"  Duplicates removed : {s['duplicates']:,}")
        print(f"  Reduction          : {s['reduction_pct']:.1f}%")
        print(f"  Avg paras/doc      : {s['avg_paragraphs_per_doc']:.1f}")
        print(f"  Para uniqueness    : {s['paragraph_uniqueness']:.1f}%")
        print("=" * 60)
