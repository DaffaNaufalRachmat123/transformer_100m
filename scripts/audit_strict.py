#!/usr/bin/env python3
"""Final audit with strict pattern matching (same as deep_clean)."""
import re, os

files = [
    ("EN", "data/corpus_final/english/fineweb_edu.txt"),
    ("EN", "data/corpus_final/english/wiki_en.txt"),
    ("EN", "data/corpus_final/english/openwebtext.txt"),
    ("JA", "data/corpus_final/japanese/mc4_ja.txt"),
    ("JA", "data/corpus_final/japanese/wiki_ja.txt"),
]

# EN strict patterns
STRONG_ADULT_EN = ['porn','hentai','onlyfans','camgirl','hardcore','blowjob',
    'gangbang','creampie','bdsm','swingers','threesome','voyeur','upskirt',
    'brothel','stripper','escort service','adult video','adult film','naked','nude']
WEAK_ADULT_EN = ['masturbat','sexual','erotic','incest','fetish','anal']

for lang, path in files:
    if not os.path.exists(path):
        print(f"\n[{lang}] {path} - MISSING"); continue
    
    size = os.path.getsize(path)
    lines = total_len = 0
    min_l, max_l = 1e9, 0
    adult_good = gamble_good = 0
    repl_chars = url_bad = 0
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            lines += 1
            text = line.rstrip('\n\r')
            l = len(text)
            total_len += l
            if l > 0:
                if l < min_l: min_l = l
                if l > max_l: max_l = l
            
            if '\ufffd' in text: repl_chars += 1
            
            if lang == "EN":
                lower = text.lower()
                strong = any(k in lower for k in STRONG_ADULT_EN)
                weak = sum(1 for k in WEAK_ADULT_EN if k in lower)
                if strong or weak >= 3: adult_good += 1
                g = sum(1 for k in ['casino','blackjack','roulette','slot machine','sportsbook','jackpot','betting','gambling','lottery'] if k in lower)
                if g >= 2: gamble_good += 1
            else:
                strong = ['エロ','無修正','風俗','セックス','hentai','裏ビデオ','援交','ソープ','デリヘル','寝取られ','エロゲ','エロ漫画','ライブチャット','性感エステ','AV女優','ピンサロ']
                if any(k in text for k in strong): adult_good += 1
                if any(k in text for k in ['パチンコ','パチスロ','カジノ','ギャンブル','大当たり']): gamble_good += 1
            
            cs = text.count('https://')
            if cs > 10 and cs * 40 > l * 0.3: url_bad += 1
    
    avg = total_len / max(lines, 1)
    issues = adult_good + gamble_good + repl_chars + url_bad
    score = 100 - (issues/max(lines,1)*100)
    
    print(f"\n[{'EN' if lang=='EN' else 'JA'}] {os.path.basename(path):30s} | "
          f"{size/1e6:5.0f}MB | {lines:>10,}L | {min_l:>4}-{max_l:<5} avg={avg:3.0f} | "
          f"adult={adult_good} gamble={gamble_good} repl={repl_chars} url={url_bad} | SCORE={score:.1f}%")

print(f"\n{'='*70}")
total_mb = sum(os.path.getsize(p) for _, p in files if os.path.exists(p))/1e6
total_lines = 0
for _, p in files:
    if os.path.exists(p): total_lines += sum(1 for _ in open(p,'rb'))
print(f"TOTAL: {total_mb:.0f}MB, {total_lines:,} lines")
