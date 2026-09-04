"""Tokenization & sequence packing per DATA_PIPELINE_PLAN A.13.

Sub-modules:
  encoder          — BPE encode documents using trained tokenizer
  sequence_packer  — Greedy first-fit-decreasing bin packing
"""

from .encoder import TokenizationEncoder
from .sequence_packer import SequencePacker

__all__ = ["TokenizationEncoder", "SequencePacker"]
