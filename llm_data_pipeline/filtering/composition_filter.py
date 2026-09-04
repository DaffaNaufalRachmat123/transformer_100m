"""
Composition Filter — per DATA_PIPELINE_PLAN A.10.

Checks:
  - Alphabetic ratio: > 0.6 (for Latin-script languages)
  - Numeric ratio: < 0.2 (not a data table dump)
  - Symbol ratio: < 0.1 (not code or formula-heavy)
  - Non-printable character ratio: < 0.01
  - Mean word length: 3-15 characters
  - Stop word presence: must contain common function words
"""

import re
from typing import Dict, List, Tuple, Set

STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "and", "but", "or",
    "if", "while", "because", "about", "it", "its", "that", "this", "these",
    "those", "he", "she", "they", "we", "you", "i", "me", "not", "no",
    "so", "than", "then", "also", "just", "now", "only", "very",
}

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "adalah",
    "ini", "itu", "atau", "tersebut", "juga", "sudah", "saya", "kamu", "dia",
    "mereka", "kami", "kita", "akan", "bisa", "dapat", "tidak", "ada",
    "merupakan", "dalam", "sebagai", "karena", "oleh", "saat", "setelah",
    "sebelum", "antara", "telah", "masih", "hanya", "semua", "lain",
}

STOPWORDS_JA = {
    "の", "に", "を", "は", "が", "と", "で", "も", "へ", "や",
    "か", "な", "だ", "です", "ます", "から", "まで", "より",
    "こと", "もの", "これ", "それ", "あれ", "どれ",
    "この", "その", "あの", "どの", "ここ", "そこ",
    "いる", "ある", "する", "なる", "できる",
}

STOPWORDS_KO = {
    "이", "가", "을", "를", "에", "에서", "로", "으로", "와", "과",
    "의", "은", "는", "도", "만", "까지", "부터", "이나", "나",
    "것", "수", "있다", "없다", "하다", "되다", "그", "이", "저",
    "한", "두", "세", "네", "많은", "있는", "하는",
}

LANG_STOPWORDS = {
    "en": STOPWORDS_EN,
    "id": STOPWORDS_ID,
    "ja": STOPWORDS_JA,
    "ko": STOPWORDS_KO,
}

NON_PRINTABLE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class CompositionFilter:
    """Filter documents by character and word composition properties."""

    def __init__(
        self,
        alpha_min_ratio: float = 0.6,
        numeric_max_ratio: float = 0.2,
        symbol_max_ratio: float = 0.1,
        non_printable_max: float = 0.01,
        mean_word_length_min: int = 3,
        mean_word_length_max: int = 15,
        require_stopwords: bool = True,
    ):
        self.alpha_min_ratio = alpha_min_ratio
        self.numeric_max_ratio = numeric_max_ratio
        self.symbol_max_ratio = symbol_max_ratio
        self.non_printable_max = non_printable_max
        self.mean_wl_min = mean_word_length_min
        self.mean_wl_max = mean_word_length_max
        self.require_stopwords = require_stopwords

        self.stats = {
            "total": 0, "passed": 0,
            "low_alpha": 0, "high_numeric": 0, "high_symbol": 0,
            "high_non_printable": 0, "bad_word_length": 0, "no_stopwords": 0,
        }

    def check(self, text: str, language: str = "en") -> Tuple[bool, Dict]:
        self.stats["total"] += 1
        failures = []

        if not text or not text.strip():
            return False, {"passed": False, "reasons": ["empty"]}

        total_chars = len(text)
        words = text.split()
        n_words = len(words)

        # 1. Character composition
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        non_printable = len(NON_PRINTABLE_PATTERN.findall(text))

        alpha_ratio = alpha_count / max(1, total_chars)
        numeric_ratio = digit_count / max(1, total_chars)
        symbol_ratio = punct_count / max(1, total_chars)
        np_ratio = non_printable / max(1, total_chars)

        if alpha_ratio < self.alpha_min_ratio:
            failures.append(f"alpha_ratio={alpha_ratio:.3f}")
            self.stats["low_alpha"] += 1

        if numeric_ratio > self.numeric_max_ratio:
            failures.append(f"numeric_ratio={numeric_ratio:.3f}")
            self.stats["high_numeric"] += 1

        if symbol_ratio > self.symbol_max_ratio:
            failures.append(f"symbol_ratio={symbol_ratio:.3f}")
            self.stats["high_symbol"] += 1

        if np_ratio > self.non_printable_max:
            failures.append(f"non_printable_ratio={np_ratio:.4f}")
            self.stats["high_non_printable"] += 1

        # 2. Mean word length
        if n_words > 0:
            mean_wl = sum(len(w) for w in words) / n_words
            if mean_wl < self.mean_wl_min or mean_wl > self.mean_wl_max:
                failures.append(f"mean_word_length={mean_wl:.1f}")
                self.stats["bad_word_length"] += 1

        # 3. Stop word presence
        if self.require_stopwords:
            stopwords = LANG_STOPWORDS.get(language, STOPWORDS_EN)
            text_lower = text.lower()
            has_stopword = any(sw in text_lower.split() for sw in stopwords)
            if not has_stopword:
                failures.append("no_stopwords")
                self.stats["no_stopwords"] += 1

        if failures:
            return False, {"passed": False, "reasons": failures}
        self.stats["passed"] += 1
        return True, {"passed": True, "reasons": []}

    def filter_documents(self, documents: List[Dict], text_field: str = "text",
                         language_field: str = "language") -> Tuple[List[Dict], List[Dict]]:
        passed = []
        rejected = []
        for doc in documents:
            text = doc.get(text_field, "")
            lang = doc.get(language_field, "en")
            ok, info = self.check(text, lang)
            doc_out = dict(doc)
            doc_out["composition_filter"] = info
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
        print(f"\n{'='*60}\n COMPOSITION FILTER SUMMARY\n{'='*60}")
        print(f"  Documents scanned   : {s['total']:,}")
        print(f"  Passed              : {s['passed']:,} ({100*s['passed']/total:.1f}%)")
        print(f"  Low alpha ratio     : {s['low_alpha']:,}")
        print(f"  High numeric ratio  : {s['high_numeric']:,}")
        print(f"  High symbol ratio   : {s['high_symbol']:,}")
        print(f"  Non-printable       : {s['high_non_printable']:,}")
        print(f"  Bad word length     : {s['bad_word_length']:,}")
        print(f"  No stopwords        : {s['no_stopwords']:,}")
        print("=" * 60)
