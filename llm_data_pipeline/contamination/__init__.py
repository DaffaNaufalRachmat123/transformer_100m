"""Benchmark contamination detection per DATA_PIPELINE_PLAN A.9.

Methods (following GPT-4 contamination methodology):
  1. 13-gram exact overlap — flag any training doc sharing a 13-gram with a
     benchmark test/validation set.
  2. Embedding cosine similarity — catch paraphrased benchmark items.
  3. Full substring match — question text or answer options as substring.

Target: ZERO 13-gram overlaps with any benchmark test set.
"""

from .benchmark_loader import BenchmarkLoader
from .ngram_detector import NgramContaminationDetector
from .contamination_report import ContaminationReporter

__all__ = ["BenchmarkLoader", "NgramContaminationDetector", "ContaminationReporter"]
