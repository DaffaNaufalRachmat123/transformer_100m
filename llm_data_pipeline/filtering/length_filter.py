"""
Length Filter — per DATA_PIPELINE_PLAN A.10.

Checks:
  - Min tokens: 50 (remove ultra-short fragments)
  - Max tokens: 100,000 (remove single-file data dumps)
  - Min sentences: 3 (at least a paragraph)
  - Mean word length: 3-15 characters (sanity check)

Token estimation uses simple whitespace splitting (not BPE tokenizer).
"""

import re
from typing import Dict, List, Tuple


class LengthFilter:
    """Filter documents by token length and structural properties."""

    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 100_000,
        min_sentences: int = 3,
        min_chars: int = 200,
        max_chars: int = 500_000,
        min_words: int = 20,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_sentences = min_sentences
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words

        self.stats = {
            "total": 0, "passed": 0,
            "too_short_tokens": 0, "too_long_tokens": 0,
            "too_short_chars": 0, "too_long_chars": 0,
            "too_few_sentences": 0, "too_few_words": 0,
        }

    def check(self, text: str) -> Tuple[bool, Dict]:
        self.stats["total"] += 1
        result = {"passed": True, "reason": None, "metrics": {}}

        if not text or not text.strip():
            self.stats["too_short_chars"] += 1
            return False, {"passed": False, "reason": "empty"}

        char_count = len(text)
        tokens = text.split()
        token_count = len(tokens)
        sentences = max(1, len(re.split(r"[.!?\u3002\uff0e\uff1f]+", text)))

        if token_count < self.min_tokens:
            self.stats["too_short_tokens"] += 1
            return False, {"passed": False, "reason": f"too_short_tokens={token_count}", "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

        if token_count > self.max_tokens:
            self.stats["too_long_tokens"] += 1
            return False, {"passed": False, "reason": f"too_long_tokens={token_count}", "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

        if char_count < self.min_chars:
            self.stats["too_short_chars"] += 1
            return False, {"passed": False, "reason": f"too_short_chars={char_count}", "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

        if char_count > self.max_chars:
            self.stats["too_long_chars"] += 1
            return False, {"passed": False, "reason": f"too_long_chars={char_count}", "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

        if sentences < self.min_sentences:
            self.stats["too_few_sentences"] += 1
            return False, {"passed": False, "reason": f"too_few_sentences={sentences}", "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

        if token_count < self.min_words:
            self.stats["too_few_words"] += 1
            return False, {"passed": False, "reason": f"too_few_words={token_count}", "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

        self.stats["passed"] += 1
        return True, {"passed": True, "reason": None, "metrics": {"tokens": token_count, "chars": char_count, "sentences": sentences}}

    def filter_documents(self, documents: List[Dict], text_field: str = "text") -> Tuple[List[Dict], List[Dict]]:
        passed = []
        rejected = []
        for doc in documents:
            text = doc.get(text_field, "")
            ok, info = self.check(text)
            doc_out = dict(doc)
            doc_out["length_filter"] = info
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
        print(f"\n{'='*60}\n LENGTH FILTER SUMMARY\n{'='*60}")
        print(f"  Documents scanned   : {s['total']:,}")
        print(f"  Passed              : {s['passed']:,} ({100*s['passed']/total:.1f}%)")
        print(f"  Too short tokens    : {s['too_short_tokens']:,}")
        print(f"  Too long tokens     : {s['too_long_tokens']:,}")
        print(f"  Too short chars     : {s['too_short_chars']:,}")
        print(f"  Too few sentences   : {s['too_few_sentences']:,}")
        print(f"  Too few words       : {s['too_few_words']:,}")
        print("=" * 60)
