"""Document filtering per DATA_PIPELINE_PLAN A.10.

Sub-modules:
  length_filter      — Min/max token count, sentence count, word count
  repetition_filter  — C4 heuristics (line dup, paragraph dup, n-gram rep, char rep)
  composition_filter — Character/word ratio filters, stop-word presence check
"""

from .length_filter import LengthFilter
from .repetition_filter import RepetitionFilter
from .composition_filter import CompositionFilter

__all__ = ["LengthFilter", "RepetitionFilter", "CompositionFilter"]
