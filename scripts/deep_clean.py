#!/usr/bin/env python3
"""Deep clean all EN and JA files with strict pattern matching."""
import re, os

# ============================================================
# EN: Multi-keyword approach — require 2+ weak signals or 1 strong signal
# ============================================================

STRONG_ADULT_EN = [
    r'\bporn\b', r'\bhentai\b', r'\bonlyfans\b', r'\bcamgirl\b',
    r'\bhardcore\b', r'\bblowjob\b', r'\bgangbang\b', r'\bcreampie\b',
    r'\bbdsm\b', r'\bswingers\b', r'\bthreesome\b', r'\bvoyeur\b',
    r'\bupskirt\b', r'\bprostitute\b', r'\bbrothel\b', r'\bstripper\b',
    r'\bescort\s+service', r'\badult\s+video', r'\badult\s+film',
    r'\bnaked\b', r'\bnude\b', r'\borganism\b', r'\bmasturbat',
]
WEAK_ADULT_EN = [
    r'\bsexual\b', r'\bsex\b', r'\berotic\b', r'\bxxx\b',
    r'\bfetish\b', r'\bincest\b', r'\banal\b',
]

STRONG_GAMBLE_EN = [
    r'\bcasino\b', r'\bpoker\b', r'\bblackjack\b', r'\broulette\b',
    r'\bslot\s+machine', r'\bsportsbook\b', r'\bjackpot\b',
    r'\breal\s+money\s+gambl', r'\bonline\s+casino',
]
WEAK_GAMBLE_EN = [
    r'\bbetting\b', r'\bgambling\b', r'\blottery\b', r'\bbingo\b',
]

def has_strong(text, patterns):
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE)) >= 1
def has_weak(text, patterns):
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE)) >= 2

def is_en_adult(text):
    if has_strong(text, STRONG_ADULT_EN): return True
    if has_weak(text, WEAK_ADULT_EN): return True
    return False

def is_en_gambling(text):
    if has_strong(text, STRONG_GAMBLE_EN): return True
    if has_weak(text, WEAK_GAMBLE_EN): return True
    return False

# ============================================================
# JA: Similar approach
# ============================================================

STRONG_ADULT_JA = [
    "エロ", "無修正", "アダルト", "風俗", "セックス", "hentai",
    "裏ビデオ", "援交", "ソープ", "デリヘル", "寝取られ",
    "エロゲ", "エロ漫画", "ライブチャット", "性感エステ",
    "メンズエステ", "ハプニングバー", "JKビジネス",
    "AV女優", "ピンサロ", "ホテヘル",
]
WEAK_ADULT_JA = [
    "人妻", "不倫", "浮気", "出会い系",
]

STRONG_GAMBLE_JA = [
    "パチンコ", "パチスロ", "競馬", "競輪", "競艇",
    "カジノ", "ギャンブル", "大当たり", "出玉", "換金",
]
WEAK_GAMBLE_JA = [
    "スロット", "攻略", "必勝",
]

def is_ja_adult(text):
    if sum(1 for kw in STRONG_ADULT_JA if kw in text) >= 1: return True
    if sum(1 for kw in WEAK_ADULT_JA if kw in text) >= 2: return True
    return False

def is_ja_gambling(text):
    if sum(1 for kw in STRONG_GAMBLE_JA if kw in text) >= 1: return True
    if sum(1 for kw in WEAK_GAMBLE_JA if kw in text) >= 2: return True
    return False

# ============================================================
# Cleaner
# ============================================================

def clean_file(label, path, lang):
    if not os.path.exists(path):
        print(f"  [{label}] MISSING")
        return
    
    out = path + ".deepclean"
    total = kept = 0
    removed = {"adult":0, "gambling":0, "short":0, "repl":0, "url":0}
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f, \
         open(out, 'w', encoding='utf-8') as o:
        for line in f:
            total += 1
            text = line.rstrip('\n\r')
            l = len(text)
            
            if l == 0: continue
            if lang == "EN" and l < 80:
                removed["short"] += 1; continue
            if lang == "JA" and l < 20:
                removed["short"] += 1; continue
            if '\ufffd' in text:
                removed["repl"] += 1; continue
            
            urls = re.findall(r'https?://[^\s]+', text)
            if len(urls) > 10 or sum(len(u) for u in urls) > len(text) * 0.3:
                removed["url"] += 1; continue
            
            if lang == "EN":
                if is_en_adult(text): removed["adult"] += 1; continue
                if is_en_gambling(text): removed["gambling"] += 1; continue
            else:
                if is_ja_adult(text): removed["adult"] += 1; continue
                if is_ja_gambling(text): removed["gambling"] += 1; continue
            
            o.write(line)
            kept += 1
    
    os.rename(path, path + ".prev")
    os.rename(out, path)
    removed_total = sum(removed.values())
    print(f"  [{label}] {total:,} -> {kept:,} ({kept/max(total,1)*100:.1f}%)  removed={removed_total:,}")
    for k,v in removed.items():
        if v > 0: print(f"    {k}: {v:,}")

print("DEEP CLEANING ALL DATA FILES\n")
clean_file("EN fineweb_edu", "data/corpus_final/english/fineweb_edu.txt", "EN")
clean_file("EN wiki_en", "data/corpus_final/english/wiki_en.txt", "EN")
clean_file("EN openwebtext", "data/corpus_final/english/openwebtext.txt", "EN")
clean_file("JA mc4_ja", "data/corpus_final/japanese/mc4_ja.txt", "JA")
clean_file("JA wiki_ja", "data/corpus_final/japanese/wiki_ja.txt", "JA")
print("\nDONE")