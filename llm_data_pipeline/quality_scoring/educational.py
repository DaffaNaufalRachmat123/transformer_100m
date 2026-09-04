"""
Educational Value Scorer — per FineWeb-Edu approach.

Methodology:
  1. Annotate ~500K samples with Llama-3-70B on 0-5 educational value scale.
     (This is done as a pre-processing step; the annotations are stored.)
  2. Train a fastText/small BERT classifier on the annotations.
  3. Score entire corpus.

For the educational subset: keep if edu_score >= 3.
For the general corpus: use edu_score as a soft weight.
"""

import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple


class EducationalScorer:
    """Educational value classifier based on FineWeb-Edu methodology."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        edu_threshold: int = 3,
        scale: Tuple[int, int] = (0, 5),
        seed: int = 42,
    ):
        self.model_path = model_path
        self.edu_threshold = edu_threshold
        self.scale = scale
        self.seed = seed
        self._model = None

    def train_from_annotations(
        self,
        annotated_samples: List[Dict],
        save_path: str,
    ):
        """Train classifier from pre-annotated samples.

        annotated_samples: list of {"text": str, "edu_score": int (0-5)}
        """
        try:
            import fasttext
        except ImportError:
            raise ImportError("Install fasttext: pip install fasttext")

        random.seed(self.seed)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for item in annotated_samples:
                text = item["text"].replace("\n", " ").replace("\r", " ").strip()
                score = int(item["edu_score"])
                label = f"__label__edu_{score}"
                if text:
                    f.write(f"{label} {text}\n")
            train_path = f.name

        try:
            n = len(annotated_samples)
            print(f"[educational] Training edu classifier on {n:,} annotated samples...")

            model = fasttext.train_supervised(
                input=train_path,
                lr=0.5,
                epoch=10,
                wordNgrams=2,
                dim=100,
                bucket=200000,
                loss="ova",
                thread=4,
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save_model(save_path)
            self._model = model
            self.model_path = save_path
            print(f"[educational] Model saved to {save_path}")

        finally:
            os.unlink(train_path)

    def load(self, model_path: Optional[str] = None):
        """Load pre-trained educational value classifier."""
        try:
            import fasttext
        except ImportError:
            raise ImportError("Install fasttext: pip install fasttext")

        path = model_path or self.model_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Edu model not found: {path}")
        self._model = fasttext.load_model(path)
        self.model_path = path

    def score(self, text: str) -> float:
        """Predict educational value score (continuous 0.0-5.0).

        Uses weighted average of class probabilities.
        """
        if self._model is None:
            raise RuntimeError("Load or train model before scoring")

        cleaned = text.replace("\n", " ").replace("\r", " ").strip()
        if not cleaned:
            return 0.0

        try:
            labels, probs = self._model.predict(cleaned, k=6)
        except Exception:
            return 0.0

        weighted_sum = 0.0
        total_prob = 0.0
        for label, prob in zip(labels, probs):
            label = label.replace("__label__edu_", "")
            try:
                score_val = int(label)
                weighted_sum += score_val * prob
                total_prob += prob
            except ValueError:
                continue

        return weighted_sum / max(total_prob, 1e-8)

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a batch of documents for educational value."""
        return [self.score(t) for t in texts]

    def is_educational(self, score: float) -> bool:
        """Check if document meets educational threshold (edu_score >= 3)."""
        return score >= self.edu_threshold

    @staticmethod
    def generate_annotation_prompt(text: str) -> str:
        """Generate LLM annotation prompt for edu score.

        This is used to annotate training data with Llama-3-70B or similar.
        """
        return (
            "Rate the educational value of the following text on a scale of 0 to 5:\n"
            "0 = No educational value (spam, ads, navigation, boilerplate)\n"
            "1 = Very low (social media chatter, casual conversation)\n"
            "2 = Low (news, entertainment, listicles)\n"
            "3 = Moderate (general informational articles, how-to guides)\n"
            "4 = High (textbooks, academic content, detailed tutorials)\n"
            "5 = Very high (research papers, advanced textbooks, expert content)\n\n"
            f"Text:\n{text[:2000]}\n\n"
            "Respond with ONLY the integer score (0-5):"
        )

    @staticmethod
    def extract_score_from_response(response: str) -> int:
        """Extract integer score from LLM response."""
        import re
        match = re.search(r"\b([0-5])\b", response.strip())
        if match:
            return int(match.group(1))
        return 2  # default to moderate
