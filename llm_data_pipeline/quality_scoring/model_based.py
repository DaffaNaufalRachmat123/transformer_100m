"""
Model-Based Quality Classifier — per DCLM paper (highest-ROI filtering step).

Trains a fastText binary classifier:
  Positive class: Wikipedia + textbooks + curated reference content
  Negative class: random Common Crawl sample

Then scores the entire corpus by probability of "high quality" class.

Per FineWeb/DCLM findings, this single filter provides more quality
improvement than all heuristic filters combined. Threshold is ablated
at {0.3, 0.5, 0.7} on proxy model.
"""

import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple


class ModelBasedQualityClassifier:
    """fastText quality classifier: Wikipedia/curated vs random CC."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        training_samples: int = 500000,
        threshold: float = 0.5,
        seed: int = 42,
        positive_label: str = "__label__high_quality",
        negative_label: str = "__label__low_quality",
    ):
        self.model_path = model_path
        self.training_samples = training_samples
        self.threshold = threshold
        self.seed = seed
        self.positive_label = positive_label
        self.negative_label = negative_label
        self._model = None

    def train(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        save_path: str,
    ) -> bool:
        """Train a fastText binary classifier.

        positive_texts: Wikipedia, textbooks, curated content.
        negative_texts: Random Common Crawl samples.
        """
        try:
            import fasttext
        except ImportError:
            raise ImportError("Install fasttext: pip install fasttext")

        random.seed(self.seed)

        # Balance classes
        n_pos = min(len(positive_texts), self.training_samples // 2)
        n_neg = min(len(negative_texts), self.training_samples // 2)

        pos_sample = random.sample(positive_texts, n_pos) if len(positive_texts) > n_pos else positive_texts
        neg_sample = random.sample(negative_texts, n_neg) if len(negative_texts) > n_neg else negative_texts

        # Write training data in fastText format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            for text in pos_sample:
                cleaned = text.replace("\n", " ").replace("\r", " ").strip()
                if cleaned:
                    f.write(f"{self.positive_label} {cleaned}\n")
            for text in neg_sample:
                cleaned = text.replace("\n", " ").replace("\r", " ").strip()
                if cleaned:
                    f.write(f"{self.negative_label} {cleaned}\n")
            train_path = f.name

        try:
            print(f"[model_based] Training fastText on {n_pos:,} positive + {n_neg:,} negative samples...")

            model = fasttext.train_supervised(
                input=train_path,
                lr=0.5,
                epoch=5,
                wordNgrams=3,
                dim=100,
                bucket=200000,
                loss="softmax",
                thread=4,
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save_model(save_path)
            self._model = model
            self.model_path = save_path
            print(f"[model_based] Model saved to {save_path}")
            return True

        except Exception as e:
            print(f"[model_based] Training failed: {e}")
            return False
        finally:
            os.unlink(train_path)

    def load(self, model_path: Optional[str] = None):
        """Load a pre-trained fastText model."""
        try:
            import fasttext
        except ImportError:
            raise ImportError("Install fasttext: pip install fasttext")

        path = model_path or self.model_path
        if not path:
            raise ValueError("model_path required to load classifier")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        self._model = fasttext.load_model(path)
        self.model_path = path

    def score(self, text: str) -> float:
        """Return probability of 'high quality' class for a document."""
        if self._model is None:
            raise RuntimeError("Load or train model before scoring")

        cleaned = text.replace("\n", " ").replace("\r", " ").strip()
        if not cleaned:
            return 0.0

        try:
            labels, probs = self._model.predict(cleaned, k=2)
        except Exception:
            return 0.0

        for label, prob in zip(labels, probs):
            if label == self.positive_label:
                return prob
        return 0.0

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score a batch of documents."""
        return [self.score(t) for t in texts]

    def is_quality(self, score: float) -> bool:
        """Check if score passes threshold."""
        return score >= self.threshold

    def filter_corpus(
        self, documents: List[Dict], text_field: str = "text"
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split corpus into (quality_docs, low_quality_docs) by threshold."""
        high = []
        low = []
        for doc in documents:
            text = doc.get(text_field, "")
            q_score = self.score(text)
            doc["model_quality_score"] = q_score
            if q_score >= self.threshold:
                high.append(doc)
            else:
                low.append(doc)
        return high, low
