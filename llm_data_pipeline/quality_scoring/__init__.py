"""Multi-signal quality scoring per DATA_PIPELINE_PLAN A.6.

This is the HIGHEST-ROI filtering step (per DCLM findings).

Sub-modules:
  heuristic — perplexity, readability, entropy, compression ratio
  model_based — fastText classifier (Wikipedia/curated vs random CC)
  educational — FineWeb-Edu style educational value classifier
  composite — weighted combination of all signals

Composite formula:
  quality_final = 0.4 * model_quality + 0.2 * (1/perplexity)
                + 0.15 * readability + 0.15 * info_density
                + 0.1 * edu_score
"""

from .heuristic import HeuristicQualityScorer
from .model_based import ModelBasedQualityClassifier
from .educational import EducationalScorer
from .composite import CompositeQualityScorer

__all__ = [
    "HeuristicQualityScorer",
    "ModelBasedQualityClassifier",
    "EducationalScorer",
    "CompositeQualityScorer",
]
