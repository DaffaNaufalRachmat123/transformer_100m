"""pair_ok — heuristic EN<->JA alignment check, extracted from fetch_corpus.py.

Kept dependency-free (stdlib only) so the LaBSE filter has no hidden imports
beyond sentence-transformers. The LaBSE score ranks pairs, but these cheap
heuristics run first: LaBSE happily scores "What You Probably Never Knew About
X" vs "What あなた Probably Never Knew About X" at 0.99 because the words really
are the same, even though the Japanese side is untranslated.
"""

import re

# Light offensive/spam filter; long URLs are almost always boilerplate noise.
BAD_RE = re.compile(
    r"\b(?:porn|xxx|hentai|onlyfans|casino|online ?gambling|escort ?service|viagra)\b"
    r"|http[s]?://\S{200,}",
    re.I,
)
KANA_RE = re.compile(r"[\u3040-\u30ff]")
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGITS_RE = re.compile(r"\d+")


def pair_ok(en: str, ja: str) -> bool:
    """Reject misaligned pairs. Mined bitext is full of them, and a bad pair is
    worse than no pair: it actively teaches the model to mistranslate."""
    en, ja = en.strip(), ja.strip()
    if not (8 <= len(en) <= 400) or not (4 <= len(ja) <= 300):
        return False
    if "\ufffd" in en or "\ufffd" in ja:
        return False
    if not LATIN_RE.search(en) or not CJK_RE.search(ja):
        return False
    # A "Japanese" side with no kana is usually a bare noun list or Chinese.
    if not KANA_RE.search(ja):
        return False
    # An English side carrying CJK means the columns are swapped or merged.
    if len(CJK_RE.findall(en)) > 2:
        return False
    # Japanese is roughly 2x denser than English per character. Anything far
    # outside that band is an alignment error, not a translation.
    ratio = len(en) / max(1, len(ja))
    if not (0.7 <= ratio <= 6.0):
        return False
    # Numbers survive translation. If they disagree, the sentences do not match.
    de, dj = set(DIGITS_RE.findall(en)), set(DIGITS_RE.findall(ja))
    if de and dj and not (de & dj):
        return False
    if len(de) >= 2 and not dj:
        return False
    return not (BAD_RE.search(en) or BAD_RE.search(ja))