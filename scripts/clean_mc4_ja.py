#!/usr/bin/env python3
"""
Clean mC4 Japanese corpus by removing:
- Adult/porn content
- SEO spam (keyword-stuffed product listings)
- Gambling content
- URL-dominated pages
- Non-Japanese dominant text
- Korean characters

Preserves:
- Clean Japanese text (blogs, news, articles, forums)
- Mixed Japanese-English (product descriptions, tech articles)
- Lines with kanji (Chinese characters overlap with Japanese)
"""

import re
import sys
import os
from collections import Counter

# ── Keyword lists ──

ADULT_KEYWORDS = [
    "エロ", "無修正", "アダルト", "AV女優", "風俗", "セックス",
    "hentai", "裏ビデオ", "ハプニングバー", "ハプバー", "出会い系",
    "援交", "ソープ", "デリヘル", "裏", "ピンサロ", "ホテヘル",
    "性感", "マッサージ", "人妻", "不倫", "浮気", "寝取られ",
    "エロゲ", "エロ漫画", "アダルトビデオ", "無料動画", "ライブチャット",
    "JK", "風俗店", "性感エステ", "メンズエステ",
]

GAMBLING_KEYWORDS = [
    "パチンコ", "パチスロ", "スロット", "競馬", "競輪", "競艇",
    "ボートレース", "カジノ", "ギャンブル", "攻略法", "必勝法",
    "パチン", "スロット", "大当たり", "出玉", "換金",
    "スロット", "パチンコ", "確変", "潜伏", "天井",
]

SEO_SPAM_KEYWORDS = [
    "送料無料", "最安値", "激安", "人気", "ランキング", "おすすめ",
    "限定品", "驚きの値段", "高品質", "全国無料", "品質保証",
    "定番人気", "価格.com", "価格比較", "購入はこちら", "今すぐ購入",
    "カートに入れる", "お気に入り", "在庫あり", "即日発送",
]

# ── Detection functions ──

def count_japanese_chars(text):
    """Count hiragana + katakana + kanji characters."""
    return len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))


def count_korean_chars(text):
    """Count hangul characters."""
    return len(re.findall(r'[\uac00-\ud7af]', text))


def count_urls(text):
    """Count URLs in text."""
    return len(re.findall(r'https?://[^\s]+', text))


def count_english_chars(text):
    """Count ASCII alphabet characters."""
    return len(re.findall(r'[a-zA-Z]', text))


def is_adult(text):
    """Check if text contains adult content."""
    matches = sum(1 for kw in ADULT_KEYWORDS if kw in text)
    return matches >= 1


def is_gambling(text):
    """Check if text contains gambling content."""
    matches = sum(1 for kw in GAMBLING_KEYWORDS if kw in text)
    return matches >= 1


def is_seo_spam(text):
    """Check if text is SEO spam (keyword stuffing)."""
    matches = sum(1 for kw in SEO_SPAM_KEYWORDS if kw in text)
    # Need at least 3 SEO keywords to classify as spam
    return matches >= 3


def is_url_dominated(text):
    """Check if text is dominated by URLs."""
    url_count = count_urls(text)
    if url_count > 10:
        return True
    url_chars = sum(len(u) for u in re.findall(r'https?://[^\s]+', text))
    return url_chars > len(text) * 0.3


def is_non_japanese(text):
    """Check if text is mostly non-Japanese."""
    ja_chars = count_japanese_chars(text)
    total = len(text)
    if total == 0:
        return True
    # If less than 10% Japanese characters, treat as non-Japanese
    return ja_chars < total * 0.1


def has_korean(text):
    """Check if text contains Korean characters."""
    return count_korean_chars(text) > 5


def is_too_short(text):
    """Check if text is too short to be useful."""
    return len(text) < 50


def is_too_long(text):
    """Check if text is too long (causes gradient issues)."""
    return len(text) > 5000


def is_pure_chinese(text):
    """Check if text is Chinese without Japanese kana."""
    cn_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    ja_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    if cn_count == 0:
        return False
    # If Hanzi count > 50 and no kana → pure Chinese
    if cn_count > 50 and ja_count == 0:
        return True
    # If Hanzi dominates and kana is negligible → Chinese
    if cn_count > 100 and ja_count < cn_count * 0.05:
        return True
    return False


def has_html_entities(text):
    """Check for HTML entities that indicate dirty data."""
    return bool(re.search(r'&nbsp;|&amp;|&lt;|&gt;|&quot;|&#\d+;', text))


def is_ecommerce_junk(text):
    """Check if text is low-quality ecommerce listing."""
    length = len(text)
    if length > 500:
        return False
    ecomm_signals = ["円〜", "レビュー対象商品", "ショッピング", "通販",
                      "カート", "決済", "購入"]
    signals = sum(1 for kw in ecomm_signals if kw in text)
    return signals >= 2


def classify_line(text):
    """Classify a line. Returns (should_keep, reason)."""
    if is_too_short(text):
        return False, "too_short"
    if is_too_long(text):
        return False, "too_long"
    if is_non_japanese(text):
        return False, "non_japanese"
    if is_pure_chinese(text):
        return False, "pure_chinese"
    if has_korean(text):
        return False, "korean_chars"
    if has_html_entities(text):
        return False, "html_entities"
    if is_adult(text):
        return False, "adult"
    if is_gambling(text):
        return False, "gambling"
    if is_url_dominated(text):
        return False, "url_dominated"
    if is_seo_spam(text):
        return False, "seo_spam"
    if is_ecommerce_junk(text):
        return False, "ecommerce"
    return True, "clean"


# ── Main ──

def main():
    input_path = "./data/corpus_final/japanese/mc4_ja.txt"
    output_path = "./data/corpus_final/japanese/mc4_ja_clean.txt"
    
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
                continue
            
            keep, reason = classify_line(text)
            stats[reason] += 1
            
            if keep:
                fout.write(line)
                kept_lines += 1
                total_chars_clean += len(text)
            
            if total_lines % 50000 == 0:
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