"""
Language Classifier — FastText + CLD3 dual classifier for multilingual pipeline.

Detects language per document and per paragraph.
Routes based on script (CJK → JA/KO, Arabic → AR, Cyrillic → RU).

Target languages: EN, ID, JA, KO (others discarded unless parallel text).
"""

import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


LANG_CODE_MAP = {
    "__label__en": "en",
    "__label__id": "id",
    "__label__ja": "ja",
    "__label__ko": "ko",
    "__label__zh": "zh",
    "__label__ar": "ar",
    "__label__ru": "ru",
    "__label__fr": "fr",
    "__label__de": "de",
    "__label__es": "es",
    "__label__pt": "pt",
    "__label__it": "it",
    "__label__nl": "nl",
    "__label__pl": "pl",
    "__label__tr": "tr",
    "__label__vi": "vi",
    "__label__th": "th",
    "__label__ms": "ms",
    "__label__tl": "tl",
    "__label__hi": "hi",
    "__label__bn": "bn",
}

TARGET_LANGUAGES = {"en", "id", "ja", "ko"}

# Unicode script ranges
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x30A0, 0x30FF),    # Katakana
    (0x3040, 0x309F),    # Hiragana
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
]


class LanguageClassifier:
    """Identify language of text using FastText + CLD3."""

    def __init__(
        self,
        fasttext_model_path: Optional[str] = None,
        primary_threshold: float = 0.65,
        secondary_threshold: float = 0.5,
        target_languages: Optional[set] = None,
        paragraph_level: bool = True,
        mixed_language_handling: str = "keep_if_primary_target",
    ):
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.target_languages = target_languages or TARGET_LANGUAGES
        self.paragraph_level = paragraph_level
        self.mixed_handling = mixed_language_handling

        self._fasttext_model = None
        self._cld3_available = False

        # Load FastText model
        if fasttext_model_path:
            self._load_fasttext(fasttext_model_path)

        # Check CLD3 availability
        try:
            import cld3
            self._cld3_available = True
        except ImportError:
            print("[lang] CLD3 not available — using FastText only")

        self.stats = {
            "total_documents": 0,
            "classified": {lang: 0 for lang in TARGET_LANGUAGES},
            "rejected_language": 0,
            "mixed_language": 0,
            "low_confidence": 0,
            "script_routed": 0,
            "paragraph_classifications": Counter(),
            "confidence_distribution": Counter(),
        }

    def classify(self, text: str) -> Dict:
        """Classify a single document's language.

        Returns: {language, confidence, is_mixed, secondary_lang, script, method}
        """
        self.stats["total_documents"] += 1

        if not text or not text.strip():
            return self._unknown_result("empty")

        # Script-based pre-routing for CJK content
        script_info = self._detect_script(text)
        if script_info:
            self.stats["script_routed"] += 1
            result = self._classify_with_script_hint(text, script_info)
            if result["confidence"] >= self.primary_threshold:
                return result

        # Paragraph-level detection
        if self.paragraph_level:
            para_results = self._classify_paragraphs(text)
            if para_results:
                return self._aggregate_paragraph_results(para_results)

        # Full document classification
        return self._classify_full(text)

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a batch of documents."""
        return [self.classify(t) for t in texts]

    def classify_documents(
        self, documents: List[Dict], text_field: str = "text"
    ) -> Tuple[List[Dict], List[Dict]]:
        """Classify and split into (target_lang_docs, rejected_docs)."""
        kept = []
        rejected = []

        for doc in documents:
            text = doc.get(text_field, doc.get("normalized_text", ""))
            result = self.classify(text)
            doc_out = dict(doc)
            doc_out["language"] = result["language"]
            doc_out["language_confidence"] = result["confidence"]
            doc_out["language_method"] = result["method"]
            doc_out["is_mixed_language"] = result["is_mixed"]

            if result["language"] in self.target_languages:
                self.stats["classified"][result["language"]] += 1
                kept.append(doc_out)
            else:
                self.stats["rejected_language"] += 1
                rejected.append(doc_out)

            # Track confidence distribution
            bucket = int(result["confidence"] * 10) / 10.0
            self.stats["confidence_distribution"][bucket] += 1

        return kept, rejected

    def is_target_language(self, result: Dict) -> bool:
        """Check if classification result indicates a target language."""
        return result["language"] in self.target_languages

    def _load_fasttext(self, model_path: str):
        """Load the FastText language identification model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FastText model not found: {model_path}\n"
                f"Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
        try:
            import fasttext
            self._fasttext_model = fasttext.load_model(model_path)
            print(f"[lang] FastText model loaded: {os.path.basename(model_path)}")
        except ImportError:
            raise ImportError("Install fasttext: pip install fasttext")

    def _classify_full(self, text: str) -> Dict:
        """Classify using FastText on full document text."""
        result = {
            "language": "unknown",
            "confidence": 0.0,
            "is_mixed": False,
            "secondary_lang": None,
            "script": None,
            "method": "none",
        }

        if not text.strip():
            return result

        # FastText
        if self._fasttext_model:
            try:
                labels, probs = self._fasttext_model.predict(
                    text.replace("\n", " ")[:10000], k=2
                )
                primary_code = LANG_CODE_MAP.get(labels[0], "unknown")
                primary_conf = probs[0]
                result["language"] = primary_code
                result["confidence"] = primary_conf
                result["method"] = "fasttext"

                if len(labels) > 1:
                    secondary_code = LANG_CODE_MAP.get(labels[1], "unknown")
                    secondary_conf = probs[1]
                    if secondary_conf > 0.2:
                        result["is_mixed"] = True
                        result["secondary_lang"] = secondary_code

                return result
            except Exception as e:
                print(f"[lang] FastText error: {e}")

        # Fallback: script-based heuristic
        script = self._detect_script_heuristic(text)
        result["language"] = script or "unknown"
        result["confidence"] = 0.3
        result["method"] = "script_heuristic"
        return result

    def _classify_paragraphs(self, text: str) -> Optional[List[Dict]]:
        """Classify language per paragraph and aggregate."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            return None

        results = []
        for para in paragraphs:
            if len(para) < 20:
                continue
            res = self._classify_full(para)
            results.append(res)
            self.stats["paragraph_classifications"][res["language"]] += 1

        return results if results else None

    def _aggregate_paragraph_results(self, para_results: List[Dict]) -> Dict:
        """Aggregate paragraph-level classifications into a document-level result."""
        lang_counts = Counter(r["language"] for r in para_results)
        total = sum(lang_counts.values())

        primary_lang, primary_count = lang_counts.most_common(1)[0]
        primary_conf = primary_count / total

        is_mixed = False
        secondary_lang = None
        if len(lang_counts) > 1:
            second_lang, second_count = lang_counts.most_common(2)[1]
            secondary_conf = second_count / total
            if secondary_conf > 0.2:
                is_mixed = True
                secondary_lang = second_lang

        if is_mixed:
            self.stats["mixed_language"] += 1

        if primary_conf < self.primary_threshold:
            self.stats["low_confidence"] += 1

        return {
            "language": primary_lang,
            "confidence": primary_conf,
            "is_mixed": is_mixed,
            "secondary_lang": secondary_lang,
            "script": None,
            "method": "paragraph_aggregate",
        }

    def _detect_script(self, text: str) -> Optional[Dict]:
        """Detect dominant script for CJK/Arabic/Cyrillic routing."""
        if not text:
            return None

        cjk_count = 0
        hiragana_count = 0
        katakana_count = 0
        hangul_count = 0
        total = len(text)

        for ch in text:
            cp = ord(ch)
            if 0x3040 <= cp <= 0x309F:
                hiragana_count += 1
            elif 0x30A0 <= cp <= 0x30FF:
                katakana_count += 1
            elif 0xAC00 <= cp <= 0xD7AF:
                hangul_count += 1
            elif any(lo <= cp <= hi for lo, hi in CJK_RANGES):
                cjk_count += 1

        script = None
        if hiragana_count + katakana_count + cjk_count > 0:
            if hiragana_count > 0 and hiragana_count >= katakana_count:
                script = "ja"
            elif hangul_count > total * 0.05:
                script = "ko"
            elif katakana_count > 0:
                script = "ja"

        if script:
            script_ratio = (hiragana_count + katakana_count + cjk_count + hangul_count) / total
            return {"script": script, "script_ratio": script_ratio}

        return None

    def _classify_with_script_hint(self, text: str, script_info: Dict) -> Dict:
        """Use script hint to narrow classification."""
        script = script_info["script"]
        result = self._classify_full(text)

        # If FastText agrees with script, boost confidence
        if result["language"] == script:
            result["confidence"] = max(result["confidence"], 0.85)
            result["method"] = "fasttext+script"
        # If FastText disagrees but script ratio is high, trust script
        elif script_info["script_ratio"] > 0.3:
            result["language"] = script
            result["confidence"] = max(result["confidence"], 0.7)
            result["method"] = "script_override"

        result["script"] = script
        return result

    def _detect_script_heuristic(self, text: str) -> Optional[str]:
        """Simple script detection as fallback."""
        info = self._detect_script(text)
        return info["script"] if info else "en"

    def _unknown_result(self, reason: str = "") -> Dict:
        self.stats["rejected_language"] += 1
        return {
            "language": "unknown",
            "confidence": 0.0,
            "is_mixed": False,
            "secondary_lang": None,
            "script": None,
            "method": reason or "unknown",
        }

    def get_stats(self) -> Dict:
        """Return classification statistics."""
        return {
            "total_documents": self.stats["total_documents"],
            "classified": dict(self.stats["classified"]),
            "rejected_language": self.stats["rejected_language"],
            "mixed_language": self.stats["mixed_language"],
            "low_confidence": self.stats["low_confidence"],
            "script_routed": self.stats["script_routed"],
        }

    def print_summary(self):
        """Print language classification summary."""
        s = self.stats
        print("\n" + "=" * 60)
        print(" LANGUAGE CLASSIFICATION SUMMARY ")
        print("=" * 60)
        print(f"  Total documents       : {s['total_documents']:,}")
        print("-" * 60)
        for lang in sorted(TARGET_LANGUAGES):
            count = s["classified"].get(lang, 0)
            pct = 100 * count / max(1, s["total_documents"])
            print(f"  {lang:<10s}              : {count:>10,} ({pct:.1f}%)")
        print("-" * 60)
        print(f"  Rejected (non-target) : {s['rejected_language']:,}")
        print(f"  Mixed language docs   : {s['mixed_language']:,}")
        print(f"  Low confidence        : {s['low_confidence']:,}")
        print(f"  Script-routed         : {s['script_routed']:,}")
        print("=" * 60)
