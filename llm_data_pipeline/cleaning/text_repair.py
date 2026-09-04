"""
Text Repair — per DATA_PIPELINE_PLAN A.5.

Repairs and normalizes text content:
  - Broken encoding fix (ftfy)
  - Emoji normalization (standardize to Unicode 15.1)
  - URL cleanup (normalize, optionally remove or shorten)
  - Email address normalization
  - Phone number normalization
  - Excessive punctuation collapse (!!!!! → !)
  - Excessive whitespace/newline collapse
"""

import re
import unicodedata
from typing import Dict, List, Optional


# Emoji pattern: broad Unicode emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)

# Excessive punctuation: 3+ of same punctuation
EXCESSIVE_PUNCT = re.compile(r"([!?.,;:\-–—…])\1{2,}")

# URL pattern
URL_PATTERN = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    r"(?:/[-\w%!$&'()*+,./:;=?@~#]*)?",
    re.IGNORECASE,
)

# Email pattern
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)

# Phone pattern (international and common formats)
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?"
    r"(?:\(?\d{2,4}\)?[-.\s]?)?"
    r"\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}",
)


class TextRepairer:
    """Repair and normalize text content."""

    def __init__(
        self,
        ftfy_enabled: bool = True,
        emoji_normalize: bool = True,
        url_normalize: bool = True,
        email_normalize: bool = True,
        phone_normalize: bool = True,
        excessive_punctuation: bool = True,
        excessive_whitespace: bool = True,
        url_placeholder: str = "[URL]",
        email_placeholder: str = "[EMAIL]",
        phone_placeholder: str = "[PHONE]",
    ):
        self.ftfy_enabled = ftfy_enabled
        self.emoji_normalize = emoji_normalize
        self.url_normalize = url_normalize
        self.email_normalize = email_normalize
        self.phone_normalize = phone_normalize
        self.excessive_punct = excessive_punctuation
        self.excessive_ws = excessive_whitespace
        self.url_placeholder = url_placeholder
        self.email_placeholder = email_placeholder
        self.phone_placeholder = phone_placeholder

    def repair(self, text: str) -> str:
        """Apply all text repairs."""
        if not text or not text.strip():
            return ""

        # Step 1: ftfy encoding repair
        if self.ftfy_enabled:
            try:
                text = ftfy.fix_text(text, normalization="NFC")
            except Exception:
                pass

        # Step 2: Emoji normalization
        if self.emoji_normalize:
            text = self._normalize_emoji(text)

        # Step 3: URL normalization
        if self.url_normalize:
            text = URL_PATTERN.sub(self.url_placeholder, text)

        # Step 4: Email normalization
        if self.email_normalize:
            text = EMAIL_PATTERN.sub(self.email_placeholder, text)

        # Step 5: Phone normalization
        if self.phone_normalize:
            text = PHONE_PATTERN.sub(
                lambda m: self._replace_phone(m.group()), text
            )

        # Step 6: Excessive punctuation collapse
        if self.excessive_punct:
            text = EXCESSIVE_PUNCT.sub(r"\1", text)

        # Step 7: Whitespace normalization
        if self.excessive_ws:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

        return text.strip()

    def repair_batch(self, texts: List[str]) -> List[str]:
        """Repair a batch of texts."""
        return [self.repair(t) for t in texts]

    def repair_documents(
        self, documents: List[Dict], text_field: str = "text"
    ) -> List[Dict]:
        """Repair text in a batch of documents."""
        results = []
        for doc in documents:
            text = doc.get(text_field, "")
            repaired = self.repair(text)
            doc_out = dict(doc)
            doc_out["text"] = repaired
            doc_out["repaired_text"] = repaired
            results.append(doc_out)
        return results

    def _normalize_emoji(self, text: str) -> str:
        """Normalize emoji to standard Unicode representations."""
        def _normalize_match(match):
            emoji_char = match.group(0)
            # Apply NFC normalization to handle multi-codepoint emoji
            normalized = unicodedata.normalize("NFC", emoji_char)
            # Keep emoji but ensure consistent representation
            return normalized
        return EMOJI_PATTERN.sub(_normalize_match, text)

    def _replace_phone(self, text: str) -> str:
        """Replace phone number with placeholder if it looks like a real phone."""
        # Count digits
        digits = re.sub(r"\D", "", text)
        if 7 <= len(digits) <= 15:
            return self.phone_placeholder
        return text

    def detect_issues(self, text: str) -> Dict[str, bool]:
        """Detect potential text issues without repairing."""
        return {
            "has_mojibake": self._has_mojibake(text),
            "has_excessive_punct": bool(EXCESSIVE_PUNCT.search(text)),
            "has_urls": bool(URL_PATTERN.search(text)),
            "has_emails": bool(EMAIL_PATTERN.search(text)),
            "has_phones": bool(PHONE_PATTERN.search(text)),
        }

    def _has_mojibake(self, text: str) -> bool:
        """Heuristic check for common mojibake patterns."""
        mojibake_indicators = [
            r"Ã©", r"Ã¨", r"Ã¼",  # common UTF-8 misinterpreted as Latin-1
            r"â\u0080\u0099",      # right single quote
            r"â\u0080\u009C",      # left double quote
            r"â\u0080\u009D",      # right double quote
            r"\xef\xbf\xbd",       # replacement character
        ]
        return any(re.search(p, text) for p in mojibake_indicators)
