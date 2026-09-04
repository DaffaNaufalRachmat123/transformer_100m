"""
Heuristic Quality Scorer — compute multiple signal-based quality metrics.

Signals:
  1. Perplexity (KenLM 5-gram, per-language)
  2. Readability (Flesch-Kincaid for EN, adapted for others)
  3. Compression ratio (zlib — high = repetitive, low = random)
  4. Shannon entropy (character-level)
  5. Information density (named entity + fact density)
  6. Mean word length

These are FAST to compute and provide a baseline before the model-based
classifier. Run this first, then feed scores to the composite scorer.
"""

import math
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional, Tuple


class HeuristicQualityScorer:
    """Compute heuristic quality signals for each document."""

    def __init__(
        self,
        perplexity_threshold: float = 10000.0,
        entropy_min: float = 2.0,
        entropy_max: float = 7.0,
        compression_min: float = 1.5,
        compression_max: float = 12.0,
        kenlm_model_dir: Optional[str] = None,
    ):
        self.perplexity_threshold = perplexity_threshold
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.compression_min = compression_min
        self.compression_max = compression_max
        self.kenlm_models: Dict[str, object] = {}

        if kenlm_model_dir:
            self._load_kenlm_models(kenlm_model_dir)

    def score(self, text: str, language: str = "en") -> Dict[str, float]:
        """Compute all heuristic quality scores for a document.

        Returns dict with keys: perplexity, readability, compression_ratio,
        entropy, info_density, mean_word_length, boileplate_ratio.
        """
        if not text or not text.strip():
            return self._empty_scores()

        return {
            "perplexity": self._perplexity(text, language),
            "readability": self._readability(text, language),
            "compression_ratio": self._compression_ratio(text),
            "entropy": self._entropy(text),
            "info_density": self._info_density(text),
            "mean_word_length": self._mean_word_length(text),
            "alpha_ratio": self._alpha_ratio(text),
        }

    def is_acceptable(self, scores: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if all heuristic scores pass their thresholds."""
        failures = []

        if scores.get("perplexity", 0) > self.perplexity_threshold:
            failures.append(f"perplexity={scores['perplexity']:.0f} > {self.perplexity_threshold}")

        ent = scores.get("entropy", 0)
        if ent < self.entropy_min or ent > self.entropy_max:
            failures.append(f"entropy={ent:.2f} ∉ [{self.entropy_min}, {self.entropy_max}]")

        comp = scores.get("compression_ratio", 0)
        if comp < self.compression_min or comp > self.compression_max:
            failures.append(f"compression={comp:.2f} ∉ [{self.compression_min}, {self.compression_max}]")

        alpha = scores.get("alpha_ratio", 0)
        if alpha < 0.5:
            failures.append(f"alpha_ratio={alpha:.2f} < 0.5")

        return len(failures) == 0, failures

    def score_batch(self, texts: List[str], language: str = "en") -> List[Dict[str, float]]:
        """Score a batch of documents."""
        return [self.score(t, language) for t in texts]

    def _perplexity(self, text: str, language: str) -> float:
        """Estimate perplexity via character-level approximation (fast).

        For production, use KenLM per-language models loaded via _load_kenlm_models().
        This fallback uses character trigram probability as a proxy.
        """
        model = self.kenlm_models.get(language)
        if model is not None:
            return self._kenlm_perplexity(text, model)

        # Fast character-level approximation
        text = text.lower()
        if len(text) < 3:
            return 10000.0

        trigrams = Counter()
        total = 0
        for i in range(len(text) - 2):
            trigrams[text[i:i + 3]] += 1
            total += 1

        if total < 10:
            return 10000.0

        entropy = 0.0
        for count in trigrams.values():
            p = count / total
            entropy -= p * math.log2(p + 1e-10)

        perplexity = 2 ** entropy
        # Clamp: very low perplexity (overly simple) or very high (gibberish)
        return min(perplexity, 50000.0)

    def _kenlm_perplexity(self, text: str, model) -> float:
        """Compute true KenLM perplexity."""
        try:
            import kenlm
            score = model.score(text, bos=True, eos=True)
            # Convert log10 to perplexity
            return 10 ** (-score / (len(text.split()) + 2))
        except Exception:
            return 10000.0

    def _readability(self, text: str, language: str) -> float:
        """Compute readability score normalized to [0, 1].

        EN: Flesch-Kincaid grade level, normalized.
        Non-EN: Adapted metric based on sentence/word length ratios.
        """
        if language == "en":
            try:
                from textstat import flesch_kincaid_grade
                grade = flesch_kincaid_grade(text)
                # Normalize: grade 0-20 → score 1-0 (lower grade = more readable = higher score)
                return max(0.0, min(1.0, 1.0 - grade / 20.0))
            except Exception:
                pass

        # Adapted metric for non-English
        sentences = max(1, len(re.split(r"[.!?]+", text)))
        words = max(1, len(text.split()))
        avg_sentence_len = words / sentences
        # Complex: many long words, long sentences → lower score
        long_words = sum(1 for w in text.split() if len(w) > 8)
        long_word_ratio = long_words / max(1, words)

        score = 1.0 - min(1.0, avg_sentence_len / 40.0) * 0.5 - min(1.0, long_word_ratio / 0.3) * 0.5
        return max(0.0, min(1.0, score))

    def _compression_ratio(self, text: str) -> float:
        """zlib compression ratio. 2-8 is good; <1.5 is random; >12 is repetitive."""
        raw = text.encode("utf-8")
        if len(raw) < 100:
            return 1.0
        compressed = zlib.compress(raw, level=6)
        ratio = len(raw) / max(1, len(compressed))
        return round(ratio, 2)

    def _entropy(self, text: str) -> float:
        """Character-level Shannon entropy."""
        if len(text) < 10:
            return 0.0
        counts = Counter(text)
        total = len(text)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return round(entropy, 2)

    def _info_density(self, text: str) -> float:
        """Estimate information density: ratio of content-bearing words."""
        words = text.split()
        if not words:
            return 0.0

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "and", "but", "or", "if", "while", "because",
            "about", "up", "out", "just", "now", "it", "its", "that", "this",
            "these", "those", "he", "she", "they", "we", "you", "i", "me",
            "him", "his", "her", "my", "your", "our", "their",
        }

        content_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        return round(len(content_words) / max(1, len(words)), 2)

    def _mean_word_length(self, text: str) -> float:
        """Mean word length in characters."""
        words = text.split()
        if not words:
            return 0.0
        return round(sum(len(w) for w in words) / len(words), 2)

    def _alpha_ratio(self, text: str) -> float:
        """Ratio of alphabetic characters to total characters."""
        text = text.replace(" ", "")
        if not text:
            return 0.0
        alpha = sum(1 for c in text if c.isalpha())
        return round(alpha / len(text), 2)

    def _load_kenlm_models(self, model_dir: str):
        """Load KenLM models per language (lazy)."""
        import os
        try:
            import kenlm
        except ImportError:
            print("[heuristic] kenlm not installed; using fallback perplexity estimation")
            return

        lang_map = {"en": "en", "id": "id", "ja": "ja", "ko": "ko"}
        for lang_code, fname in lang_map.items():
            path = os.path.join(model_dir, f"{fname}.arpa")
            if os.path.exists(path):
                try:
                    self.kenlm_models[lang_code] = kenlm.Model(path)
                    print(f"[heuristic] Loaded KenLM model: {lang_code}")
                except Exception as e:
                    print(f"[heuristic] Failed to load KenLM {lang_code}: {e}")

    def _empty_scores(self):
        return {
            "perplexity": 0.0, "readability": 0.0, "compression_ratio": 1.0,
            "entropy": 0.0, "info_density": 0.0, "mean_word_length": 0.0,
            "alpha_ratio": 0.0,
        }
