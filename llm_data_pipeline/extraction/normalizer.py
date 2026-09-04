"""
Text Normalizer — per DATA_PIPELINE_PLAN A.3.

Normalization steps:
  - Unicode NFC normalization (NFKC for compatibility characters)
  - ftfy mojibake repair
  - Whitespace collapse (multiple spaces → single)
  - Line ending normalization (\r\n → \n)
  - Zero-width character removal (ZWNJ, ZWS except in Khmer)
  - Control character removal (except \n, \t)
  - Special handling: preserve LaTeX formulas and code blocks
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import ftfy


# Zero-width characters to remove (except those needed for Khmer script)
ZERO_WIDTH_CHARS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u200f",  # RIGHT-TO-LEFT MARK
    "\u2060",  # WORD JOINER
    "\u2061",  # FUNCTION APPLICATION
    "\u2062",  # INVISIBLE TIMES
    "\u2063",  # INVISIBLE SEPARATOR
    "\u2064",  # INVISIBLE PLUS
}

# Control characters to remove (keep \n and \t)
CONTROL_CHARS = set(chr(i) for i in range(32)) - {"\n", "\t"}

# Math patterns to preserve
MATH_PATTERNS = [
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), r"__DISPLAYMATH__\1__/DISPLAYMATH__"),
    (re.compile(r"\\begin\{equation\}"), r"__DISPLAYMATH__"),
    (re.compile(r"\\end\{equation\}"), r"__/DISPLAYMATH__"),
    (re.compile(r"\\begin\{align\}"), r"__DISPLAYMATH__"),
    (re.compile(r"\\end\{align\}"), r"__/DISPLAYMATH__"),
]

# Code block patterns
CODE_PATTERNS = [
    (re.compile(r"```(\w*)\n(.*?)```", re.DOTALL), r"__CODEBLOCK_\1__\2__/CODEBLOCK__"),
]


class TextNormalizer:
    """Normalize text after extraction."""

    def __init__(
        self,
        unicode_form: str = "NFC",
        ftfy_enabled: bool = True,
        collapse_whitespace: bool = True,
        line_ending: str = "\n",
        remove_zero_width: bool = True,
        remove_control_chars: bool = True,
        preserve_math: bool = True,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
    ):
        self.unicode_form = unicode_form
        self.ftfy_enabled = ftfy_enabled
        self.collapse_whitespace = collapse_whitespace
        self.line_ending = line_ending
        self.remove_zero_width = remove_zero_width
        self.remove_control_chars = remove_control_chars
        self.preserve_math = preserve_math
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline to a text string."""
        if not text or not text.strip():
            return ""

        # Step 0: Protect math and code blocks from normalization
        protected = []
        if self.preserve_math:
            for pattern, replacement in MATH_PATTERNS:
                text = pattern.sub(
                    lambda m: self._protect(m, protected, replacement, pattern), text
                )
        if self.preserve_code_blocks:
            text = self._protect_code_blocks(text, protected)

        # Step 1: ftfy repair (must come before Unicode normalization)
        if self.ftfy_enabled:
            try:
                text = ftfy.fix_text(text, normalization=None)
            except Exception:
                pass

        # Step 2: Unicode normalization
        try:
            text = unicodedata.normalize(self.unicode_form, text)
        except Exception:
            pass

        # Step 3: Line ending normalization
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 4: Remove zero-width characters
        if self.remove_zero_width:
            text = "".join(c for c in text if c not in ZERO_WIDTH_CHARS)

        # Step 5: Remove control characters
        if self.remove_control_chars:
            text = "".join(c for c in text if c not in CONTROL_CHARS)

        # Step 6: Collapse whitespace
        if self.collapse_whitespace:
            # Collapse multiple spaces (but not newlines)
            text = re.sub(r"[^\S\n]+", " ", text)
            # Collapse multiple newlines
            text = re.sub(r"\n{3,}", "\n\n", text)
            # Remove trailing whitespace per line
            text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
            # Remove leading whitespace per line
            text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)

        # Step 7: Restore protected blocks
        text = self._restore_protected(text, protected)

        return text.strip()

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]

    def normalize_documents(
        self, documents: List[Dict], text_field: str = "extracted_text"
    ) -> List[Dict]:
        """Normalize text in a batch of documents, adding normalized_text field."""
        results = []
        for doc in documents:
            text = doc.get(text_field, doc.get("text", ""))
            normalized = self.normalize(text)
            doc_out = dict(doc)
            doc_out["normalized_text"] = normalized
            doc_out["text"] = normalized  # set text to normalized version
            results.append(doc_out)
        return results

    def _protect(self, match, protected: List[str], replacement: str, pattern) -> str:
        """Replace a match with a placeholder and store original."""
        idx = len(protected)
        protected.append(match.group(0))
        return f"__PROTECTED_{idx}__"

    def _protect_code_blocks(self, text: str, protected: List[str]) -> str:
        """Protect code blocks from whitespace normalization."""
        for pattern, _ in CODE_PATTERNS:
            def _protect_code(m):
                idx = len(protected)
                protected.append(m.group(0))
                return f"__PROTECTED_{idx}__"
            text = pattern.sub(_protect_code, text)
        return text

    def _restore_protected(self, text: str, protected: List[str]) -> str:
        """Restore protected blocks by replacing placeholders."""
        for i, original in enumerate(protected):
            text = text.replace(f"__PROTECTED_{i}__", original)
        return text

    def get_char_stats(self, text: str) -> Dict[str, int]:
        """Get character statistics for a normalized text."""
        stats = {
            "total_chars": len(text),
            "alpha_chars": sum(1 for c in text if c.isalpha()),
            "digit_chars": sum(1 for c in text if c.isdigit()),
            "space_chars": sum(1 for c in text if c.isspace()),
            "punct_chars": sum(1 for c in text if not c.isalnum() and not c.isspace()),
            "cjk_chars": sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7a3"),
        }
        return stats
