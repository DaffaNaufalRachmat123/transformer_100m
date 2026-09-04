"""
Composite Quality Scorer — combine all signals into a single quality score.

Per DATA_PIPELINE_PLAN A.6:
  quality_final = 0.4 * model_quality + 0.2 * (1/max(perplexity, 1))
                + 0.15 * readability + 0.15 * info_density
                + 0.1 * edu_score

Weights are configurable and should be ablated on proxy model.
"""

from typing import Dict, List, Optional, Tuple


DEFAULT_WEIGHTS = {
    "model_quality": 0.4,
    "perplexity_inverse": 0.2,
    "readability": 0.15,
    "info_density": 0.15,
    "edu_score": 0.1,
}


class CompositeQualityScorer:
    """Combine heuristic, model-based, and edu scores into a final quality score."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_final_score: float = 0.3,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.min_score = min_final_score

        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"[composite] Warning: weights sum to {total}, expected 1.0")

    def compute(
        self,
        heuristic_scores: Dict[str, float],
        model_quality: float,
        edu_score: float,
    ) -> float:
        """Compute the final composite quality score.

        heuristic_scores must contain: perplexity, readability, info_density
        """
        ppl = max(heuristic_scores.get("perplexity", 10000), 1.0)
        ppl_inv = 1.0 / (ppl / 100)  # scale so moderate perplexity ~0.5
        ppl_inv = max(0.0, min(1.0, ppl_inv))

        readability = heuristic_scores.get("readability", 0.0)
        info_density = heuristic_scores.get("info_density", 0.0)

        score = (
            self.weights.get("model_quality", 0) * model_quality
            + self.weights.get("perplexity_inverse", 0) * ppl_inv
            + self.weights.get("readability", 0) * readability
            + self.weights.get("info_density", 0) * info_density
            + self.weights.get("edu_score", 0) * (edu_score / 5.0)  # normalize to [0,1]
        )

        return max(0.0, min(1.0, score))

    def score_document(
        self,
        text: str,
        heuristic_scorer,
        model_scorer,
        edu_scorer,
        language: str = "en",
    ) -> Tuple[float, Dict[str, float]]:
        """Convenience: score a single document end-to-end with all scorers.

        Returns (final_score, all_scores_dict).
        """
        heuristic = heuristic_scorer.score(text, language)
        model_q = model_scorer.score(text) if model_scorer else 0.5
        edu = edu_scorer.score(text) if edu_scorer else 0.0

        final = self.compute(heuristic, model_q, edu)

        all_scores = {
            "quality_final": final,
            "model_quality_score": model_q,
            "edu_score": edu,
            **heuristic,
        }
        return final, all_scores

    def filter_corpus(
        self,
        documents: List[Dict],
        heuristic_scorer,
        model_scorer,
        edu_scorer,
        text_field: str = "text",
        language_field: str = "language",
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Filter corpus and return (pass, fail, all_scores).

        Each document in 'all_scores' has quality_final and sub-scores added.
        """
        passing = []
        failing = []
        scored = []

        for doc in documents:
            text = doc.get(text_field, "")
            lang = doc.get(language_field, "en")
            final, scores = self.score_document(text, heuristic_scorer, model_scorer, edu_scorer, lang)
            doc_with_scores = {**doc, **scores}
            scored.append(doc_with_scores)
            if final >= self.min_score:
                passing.append(doc_with_scores)
            else:
                failing.append(doc_with_scores)

        return passing, failing, scored

    def print_summary(self, scores_list: List[Dict[str, float]]):
        """Print quality score distribution summary."""
        if not scores_list:
            return

        finals = [s.get("quality_final", 0) for s in scores_list]
        model_qs = [s.get("model_quality_score", 0) for s in scores_list]
        edus = [s.get("edu_score", 0) for s in scores_list]

        n = len(finals)
        passing = sum(1 for f in finals if f >= self.min_score)

        print("\n" + "=" * 60)
        print(" QUALITY SCORING SUMMARY ")
        print("=" * 60)
        print(f"  Documents scored      : {n:,}")
        print(f"  Passing (≥{self.min_score})   : {passing:,} ({100*passing/n:.1f}%)")
        print(f"  Rejected              : {n-passing:,} ({100*(n-passing)/n:.1f}%)")
        print("-" * 60)
        print(f"  Avg quality_final     : {sum(finals)/n:.3f}")
        print(f"  Avg model_quality     : {sum(model_qs)/n:.3f}")
        print(f"  Avg edu_score         : {sum(edus)/n:.2f}")
        print("=" * 60)
