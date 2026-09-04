"""Data mixing & proportion optimization per DATA_PIPELINE_PLAN A.11.

Sub-modules:
  domain_proportions — Compute document weights by domain + language
  upsampling         — High-quality domain upsampling (code 3x, math 5x, wiki 3x, edu 2x)
  curriculum         — Curriculum phase definitions (easy → full → quality)
"""

from .domain_proportions import DomainProportionMixer
from .upsampling import DomainUpsampler
from .curriculum import CurriculumScheduler

__all__ = ["DomainProportionMixer", "DomainUpsampler", "CurriculumScheduler"]
