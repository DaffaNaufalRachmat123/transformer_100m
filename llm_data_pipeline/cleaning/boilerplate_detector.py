"""
Boilerplate Detector — per DATA_PIPELINE_PLAN A.5.

Detects and filters documents with high boilerplate-to-content ratio.

Metrics:
  - main_content_ratio: ratio of main content to total text
  - Reject if main_content_ratio < threshold (default 0.3)

Uses Trafilatura's content/comments/metadata separation when available,
otherwise heuristic-based estimation.
"""

import re
from typing import Dict, List, Tuple


# Boilerplate indicator phrases
BOILERPLATE_INDICATORS = [
    "all rights reserved", "terms of service", "privacy policy",
    "cookie policy", "sign up for our newsletter", "subscribe now",
    "click here to", "read more at", "this article originally appeared",
    "originally published", "republished with permission",
    "the views expressed", "disclaimer:", "affiliate link",
    "sponsored content", "advertisement", "promoted by",
    "share this article", "follow us on", "like us on facebook",
    "comments are closed", "leave a comment", "log in to",
    "create an account", "forgot password", "reset password",
]


class BoilerplateDetector:
    """Detect and filter boilerplate-heavy documents."""

    def __init__(
        self,
        main_content_ratio_threshold: float = 0.3,
        prefer_trafilatura_content: bool = True,
    ):
        self.content_ratio_threshold = main_content_ratio_threshold
        self.prefer_trafilatura = prefer_trafilatura_content

        self.stats = {
            "total_documents": 0,
            "passed": 0,
            "rejected_boilerplate": 0,
            "avg_content_ratio": 0.0,
        }

    def detect(self, text: str) -> Tuple[float, bool]:
        """Compute main content ratio and boilerplate decision.

        Returns (content_ratio, is_acceptable).
        """
        if not text or not text.strip():
            self.stats["total_documents"] += 1
            return 0.0, False

        self.stats["total_documents"] += 1

        lines = text.split("\n")
        if not lines:
            return 0.0, False

        content_lines = 0
        boilerplate_lines = 0

        for line in lines:
            stripped = line.strip().lower()
            if not stripped:
                continue

            if self._is_boilerplate_line(stripped):
                boilerplate_lines += 1
            else:
                content_lines += 1

        total = content_lines + boilerplate_lines
        if total == 0:
            return 0.0, False

        content_ratio = content_lines / total
        is_acceptable = content_ratio >= self.content_ratio_threshold

        if is_acceptable:
            self.stats["passed"] += 1
        else:
            self.stats["rejected_boilerplate"] += 1

        # Running average
        n = self.stats["total_documents"]
        old_avg = self.stats["avg_content_ratio"]
        self.stats["avg_content_ratio"] = old_avg + (content_ratio - old_avg) / n

        return content_ratio, is_acceptable

    def filter_documents(
        self, documents: List[Dict], text_field: str = "text"
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split documents into (content_docs, boilerplate_docs)."""
        kept = []
        rejected = []

        for doc in documents:
            text = doc.get(text_field, "")
            ratio, acceptable = self.detect(text)
            doc_out = dict(doc)
            doc_out["content_ratio"] = ratio
            if acceptable:
                kept.append(doc_out)
            else:
                rejected.append(doc_out)

        return kept, rejected

    def _is_boilerplate_line(self, line: str) -> bool:
        """Check if a line looks like boilerplate."""
        # Very short lines with links
        if len(line) < 15 and ("http" in line or "www." in line):
            return True

        # Lines that are purely navigation/links
        if line.count("|") >= 3 and len(line) < 200:
            return True

        # Lines with known boilerplate indicators
        for indicator in BOILERPLATE_INDICATORS:
            if indicator in line:
                return True

        # Lines that are mostly non-alphabetic (symbols, numbers)
        alpha_ratio = sum(1 for c in line if c.isalpha()) / max(1, len(line))
        if alpha_ratio < 0.3 and len(line) > 10:
            return True

        return False

    def get_stats(self) -> Dict:
        """Return boilerplate detection statistics."""
        return dict(self.stats)

    def print_summary(self):
        """Print boilerplate detection summary."""
        s = self.stats
        print("\n" + "=" * 60)
        print(" BOILERPLATE DETECTION SUMMARY ")
        print("=" * 60)
        print(f"  Documents processed   : {s['total_documents']:,}")
        print(f"  Passed (≥ {self.content_ratio_threshold})   : {s['passed']:,}")
        print(f"  Rejected (boilerplate): {s['rejected_boilerplate']:,}")
        print(f"  Avg content ratio    : {s['avg_content_ratio']:.3f}")
        print("=" * 60)
