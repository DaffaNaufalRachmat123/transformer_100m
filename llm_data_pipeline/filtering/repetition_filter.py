"""
Repetition Filter — per DATA_PIPELINE_PLAN A.10 (C4 heuristics).

Checks:
  - Line duplicate ratio: reject if > 0.3
  - Paragraph duplicate ratio: reject if > 0.2
  - N-gram repetition: reject if any 10-gram appears > 10x
  - Character repetition: reject if any char > 0.2 of total
  - Bullet/list ratio: reject if > 0.5 (filters listicle spam)
"""

import re
from collections import Counter
from typing import Dict, List, Tuple


class RepetitionFilter:
    """Filter documents with excessive repetition."""

    def __init__(
        self,
        line_dup_max: float = 0.3,
        paragraph_dup_max: float = 0.2,
        ngram_max_repeat: int = 10,
        ngram_size: int = 10,
        char_max_ratio: float = 0.2,
        bullet_list_max: float = 0.5,
        all_caps_max: float = 0.1,
    ):
        self.line_dup_max = line_dup_max
        self.paragraph_dup_max = paragraph_dup_max
        self.ngram_max_repeat = ngram_max_repeat
        self.ngram_size = ngram_size
        self.char_max_ratio = char_max_ratio
        self.bullet_list_max = bullet_list_max
        self.all_caps_max = all_caps_max

        self.stats = {
            "total": 0, "passed": 0,
            "high_line_dup": 0, "high_paragraph_dup": 0,
            "high_ngram_repeat": 0, "high_char_ratio": 0,
            "high_bullet_ratio": 0, "high_all_caps": 0,
        }

    def check(self, text: str) -> Tuple[bool, Dict]:
        self.stats["total"] += 1
        failures = []

        if not text or not text.strip():
            return False, {"passed": False, "reasons": ["empty"]}

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        n_lines = len(lines)

        # 1. Line duplication
        if n_lines > 0:
            line_counts = Counter(lines)
            max_line_count = max(line_counts.values())
            line_dup_ratio = max_line_count / n_lines
            if line_dup_ratio > self.line_dup_max:
                failures.append(f"line_dup_ratio={line_dup_ratio:.3f}")
                self.stats["high_line_dup"] += 1

        # 2. Paragraph duplication
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        n_paras = len(paragraphs)
        if n_paras > 0:
            para_counts = Counter(paragraphs)
            max_para_count = max(para_counts.values())
            para_dup_ratio = max_para_count / n_paras
            if para_dup_ratio > self.paragraph_dup_max:
                failures.append(f"paragraph_dup_ratio={para_dup_ratio:.3f}")
                self.stats["high_paragraph_dup"] += 1

        # 3. N-gram repetition
        words = text.lower().split()
        if len(words) >= self.ngram_size:
            ngram_counts = Counter()
            for i in range(len(words) - self.ngram_size + 1):
                ng = " ".join(words[i:i + self.ngram_size])
                ngram_counts[ng] += 1
            max_ngram_count = max(ngram_counts.values()) if ngram_counts else 0
            if max_ngram_count > self.ngram_max_repeat:
                failures.append(f"max_ngram_repeat={max_ngram_count}")
                self.stats["high_ngram_repeat"] += 1

        # 4. Character repetition
        if len(text) > 0:
            char_counts = Counter(text)
            max_char_count = max(char_counts.values())
            char_ratio = max_char_count / len(text)
            if char_ratio > self.char_max_ratio:
                failures.append(f"char_ratio={char_ratio:.3f}")
                self.stats["high_char_ratio"] += 1

        # 5. Bullet/list ratio
        bullet_lines = sum(1 for l in lines if re.match(r"^\s*[-*+•▪▸►]\s", l) or re.match(r"^\s*\d+[.)]\s", l))
        bullet_ratio = bullet_lines / max(1, n_lines)
        if bullet_ratio > self.bullet_list_max:
            failures.append(f"bullet_ratio={bullet_ratio:.3f}")
            self.stats["high_bullet_ratio"] += 1

        # 6. ALL-CAPS ratio
        if n_lines > 0:
            all_caps_lines = sum(1 for l in lines if len(l) > 5 and l == l.upper() and re.search(r"[A-Z]{3,}", l))
            all_caps_ratio = all_caps_lines / n_lines
            if all_caps_ratio > self.all_caps_max:
                failures.append(f"all_caps_ratio={all_caps_ratio:.3f}")
                self.stats["high_all_caps"] += 1

        if failures:
            return False, {"passed": False, "reasons": failures}
        self.stats["passed"] += 1
        return True, {"passed": True, "reasons": []}

    def filter_documents(self, documents: List[Dict], text_field: str = "text") -> Tuple[List[Dict], List[Dict]]:
        passed = []
        rejected = []
        for doc in documents:
            text = doc.get(text_field, "")
            ok, info = self.check(text)
            doc_out = dict(doc)
            doc_out["repetition_filter"] = info
            if ok:
                passed.append(doc_out)
            else:
                rejected.append(doc_out)
        return passed, rejected

    def get_stats(self) -> Dict:
        return dict(self.stats)

    def print_summary(self):
        s = self.stats
        total = max(s["total"], 1)
        rejected = total - s["passed"]
        print(f"\n{'='*60}\n REPETITION FILTER SUMMARY\n{'='*60}")
        print(f"  Documents scanned   : {s['total']:,}")
        print(f"  Passed              : {s['passed']:,} ({100*s['passed']/total:.1f}%)")
        print(f"  Rejected            : {rejected:,}")
        print(f"    Line dup          : {s['high_line_dup']:,}")
        print(f"    Paragraph dup     : {s['high_paragraph_dup']:,}")
        print(f"    N-gram repeat     : {s['high_ngram_repeat']:,}")
        print(f"    Char repeat       : {s['high_char_ratio']:,}")
        print(f"    Bullet/list       : {s['high_bullet_ratio']:,}")
        print(f"    ALL-CAPS          : {s['high_all_caps']:,}")
        print("=" * 60)
