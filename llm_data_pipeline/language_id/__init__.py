"""Language identification per DATA_PIPELINE_PLAN A.4.

Dual-classifier approach:
  Primary: FastText lid.176.bin (176 languages, fast, confidence ≥ 0.65)
  Secondary: CLD3 (Google's neural classifier, validates borderline cases)
  
Script-based routing for CJK, Arabic, Cyrillic scripts.
Paragraph-level detection for mixed-language documents.
"""

__all__ = ["LanguageClassifier"]
