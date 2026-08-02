#!/usr/bin/env python3
"""
Clean mC4 Indonesian corpus by removing:
- Lines under 150 chars (53% of data — dates, names, fragments, URLs)
- Adult content
- Gambling content
- URL-dominated lines (>10 URLs)
- Non-Indonesian text (Thai, Arabic, CJK characters)
- Comment/timestamp-only lines

Preserves:
- Indonesian text with actual content
"""

import re
import sys
import os
from collections import Counter

ADULT_KEYWORDS = [
    "bugil", "telanjang", "porno", "bokep", "ngentot", "sange",
    "memek", "kontol", "jilbab", "abg", "toge", "bispak", "lonte",
    "pelacur", "jablay", "vcs", "colmek", "open bo", "cewek",
    "ML", "tante", "girang", "sange", "colmek", "crot",
]

GAMBLING_KEYWORDS = [
    "poker", "judi", "slot", "togel", "toto", "kasino", "taruhan",
    "betting", "bandar", "capsa", "domino", "sicbo", "roulette",
    "baccarat", "dadu", "tembak ikan",
]

SEO_SPAM_KEYWORDS = [
    "murah", "diskon", "promo", "gratis ongkir", "harga", "beli",
    "jual", "cod", "bayar di tempat", "shopee", "tokopedia",
    "lazada", "bukalapak", "blibli",
]

def count_indonesian_chars(text):
    """Count Latin alphabet characters (Indonesian uses Latin)."""
    return len(re.findall(r'[a-zA-Z]', text))


def count_non_latin(text):
    """Count non-Latin characters (CJK, Thai, Arabic, etc.)."""
    return len(re.findall(r'[^\x00-\x7F\u0100-\u024F]', text))


def count_urls(text):
    return len(re.findall(r'https?://[^\s]+', text))


def is_too_short(text):
    return len(text) < 150


def is_adult(text):
    lower = text.lower()
    return sum(1 for kw in ADULT_KEYWORDS if kw in lower) >= 1


def is_gambling(text):
    lower = text.lower()
    return sum(1 for kw in GAMBLING_KEYWORDS if kw in lower) >= 1


def is_seo_spam(text):
    """SEO spam: short text with many shopping keywords."""
    lower = text.lower()
    if len(text) > 500:
        return False
    matches = sum(1 for kw in SEO_SPAM_KEYWORDS if kw in lower)
    return matches >= 3


def is_url_dominated(text):
    url_count = count_urls(text)
    if url_count > 10:
        return True
    url_chars = sum(len(u) for u in re.findall(r'https?://[^\s]+', text))
    return url_chars > len(text) * 0.3


def is_non_indonesian(text):
    """Check if text has significant non-Latin characters (CJK, Thai, Arabic)."""
    non_latin = count_non_latin(text)
    return non_latin > len(text) * 0.3


def is_comment_only(text):
    """Check if line is just a blog comment (date + name + short text)."""
    if len(text) > 200:
        return False
    # Pattern: "Name, Date at Time" or "Name Date"
    patterns = [
        r'^\w+\s+\d{1,2},\s*\d{4}\s+at\s+\d{1,2}:\d{2}',
        r'^\w+\s+\w+\s+\d{1,2},\s*\d{4}',
        r'^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}',
    ]
    for pat in patterns:
        if re.match(pat, text):
            return True
    return False


def classify_line(text):
    if is_too_short(text):
        return False, "too_short"
    if is_non_indonesian(text):
        return False, "non_indonesian"
    if is_adult(text):
        return False, "adult"
    if is_gambling(text):
        return False, "gambling"
    if is_url_dominated(text):
        return False, "url_dominated"
    if is_seo_spam(text):
        return False, "seo_spam"
    if is_comment_only(text):
        return False, "comment_only"
    return True, "clean"


def main():
    input_path = "./data/corpus_final/indonesian/mc4_id.txt"
    output_path = "./data/corpus_final/indonesian/mc4_id_clean.txt"

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Cleaning: {input_path}")
    print(f"Output:   {output_path}")
    print()

    stats = Counter()
    total_lines = 0
    kept_lines = 0
    total_chars_orig = 0
    total_chars_clean = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            text = line.strip()
            total_lines += 1
            total_chars_orig += len(text)

            if not text:
                stats["empty"] += 1
                continue

            keep, reason = classify_line(text)
            stats[reason] += 1

            if keep:
                fout.write(line)
                kept_lines += 1
                total_chars_clean += len(text)

            if total_lines % 1000000 == 0:
                print(f"  Processed {total_lines:,} lines... "
                      f"(kept {kept_lines:,}, {kept_lines/total_lines*100:.1f}%)")

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  Total lines:        {total_lines:>10,}")
    print(f"  Kept lines:         {kept_lines:>10,}  ({kept_lines/total_lines*100:5.1f}%)")
    print(f"  Removed lines:      {total_lines-kept_lines:>10,}  ({(total_lines-kept_lines)/total_lines*100:5.1f}%)")
    print()
    print(f"  Original size:      {total_chars_orig/1e9:>10.2f} GB")
    print(f"  Clean size:         {total_chars_clean/1e9:>10.2f} GB")
    print(f"  Size reduction:     {(1 - total_chars_clean/total_chars_orig)*100:>10.1f}%")
    print()
    print("  Removal breakdown:")
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        if reason != "clean":
            pct = count / total_lines * 100
            print(f"    {reason:<20} {count:>8,}  ({pct:5.1f}%)")
    print(f"    {'clean (kept)':<20} {stats['clean']:>8,}  ({stats['clean']/total_lines*100:5.1f}%)")


if __name__ == "__main__":
    main()