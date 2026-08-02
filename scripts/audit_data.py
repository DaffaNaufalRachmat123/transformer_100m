#!/usr/bin/env python3
"""Deep audit of all EN and JA training data files."""
import re, os

files = [
    ("EN", "data/corpus_final/english/fineweb_edu.txt"),
    ("EN", "data/corpus_final/english/wiki_en.txt"),
    ("EN", "data/corpus_final/english/openwebtext.txt"),
    ("JA", "data/corpus_final/japanese/mc4_ja.txt"),
    ("JA", "data/corpus_final/japanese/wiki_ja.txt"),
]

ADULT_EN = ["porn","xxx","sexual","orgasm","masturb","erotic","nude","naked",
    "stripper","escort","brothel","prostitut","hentai","onlyfans",
    "adult video","hardcore","blowjob","anal","swingers","threesome",
    "bdsm","fetish","incest","voyeur","upskirt","gangbang"]
GAMBLE_EN = ["casino","poker","blackjack","roulette","slot","sportsbook",
    "betting","gambling","lottery","bingo","jackpot","real money"]
SPAM_EN = ["buy now","click here","limited offer","discount code",
    "free trial","order now","best price","subscribe now",
    "risk free","make money","wire transfer","act now",
    "satisfaction guaranteed","don't miss","100% free"]
ADULT_JA = ["エロ","無修正","アダルト","AV","風俗","セックス","hentai",
    "裏ビデオ","出会い系","援交","ソープ","デリヘル","人妻","不倫",
    "寝取られ","エロゲ","エロ漫画","ライブチャット","性感"]
GAMBLE_JA = ["パチンコ","パチスロ","スロット","競馬","競輪","競艇",
    "カジノ","ギャンブル","攻略","必勝","大当たり","換金","出玉"]
SPAM_JA = ["送料無料","最安値","激安","購入はこちら","今すぐ購入",
    "カートに入れる","即日発送","限定品","品質保証","全国無料"]

for lang, path in files:
    if not os.path.exists(path):
        print(f"\n[{lang}] {path} - MISSING")
        continue
    
    size = os.path.getsize(path)
    lines = empty = nulls = repl_chars = 0
    min_l, max_l = 1e9, 0
    total_len = 0
    adult = gamble = spam = 0
    urls = htmls = 0
    cn_contam = 0
    samples = []
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            lines += 1
            text = line.rstrip('\n\r')
            l = len(text)
            total_len += l
            if l == 0: empty += 1
            else:
                if l < min_l: min_l = l
                if l > max_l: max_l = l
            
            if '\x00' in text: nulls += 1
            if '\ufffd' in text: repl_chars += 1
            
            if lang == "EN":
                lower = text.lower()
                if sum(1 for kw in ADULT_EN if kw in lower) >= 1: adult += 1
                if sum(1 for kw in GAMBLE_EN if kw in lower) >= 2: gamble += 1
                if sum(1 for kw in SPAM_EN if kw in lower) >= 3: spam += 1
            else:
                if sum(1 for kw in ADULT_JA if kw in text) >= 1: adult += 1
                if sum(1 for kw in GAMBLE_JA if kw in text) >= 1: gamble += 1
                if sum(1 for kw in SPAM_JA if kw in text) >= 3: spam += 1
                cn = len(re.findall(r'[\u4e00-\u9fff]', text))
                ja = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
                if cn > 50 and ja == 0: cn_contam += 1
            
            if re.findall(r'https?://[^\s]+', text): urls += 1
            if re.search(r'&nbsp;|&amp;|&lt;|&gt;|&quot;|&#\d+;', text): htmls += 1
            
            if i % 500000 == 0 and len(samples) < 3:
                samples.append(text[:150])
    
    avg = total_len / max(lines, 1)
    
    print(f"\n{'='*70}")
    print(f"[{lang}] {os.path.basename(path)}")
    print(f"  Size: {size/1e6:.0f}MB  Lines: {lines:,}  Empty: {empty}")
    print(f"  Len: min={min_l}  max={max_l}  avg={avg:.0f}")
    print(f"  Null bytes: {nulls}  Replace chars: {repl_chars}")
    print(f"  Adult: {adult} ({adult/max(lines,1)*100:.2f}%)")
    print(f"  Gambling: {gamble} ({gamble/max(lines,1)*100:.2f}%)")
    print(f"  Spam/Ads: {spam} ({spam/max(lines,1)*100:.2f}%)")
    print(f"  URLs: {urls} ({urls/max(lines,1)*100:.2f}%)")
    print(f"  HTML entities: {htmls}")
    if lang == "JA":
        print(f"  CN contamination: {cn_contam} ({cn_contam/max(lines,1)*100:.2f}%)")
    
    issues = adult + gamble + spam + repl_chars + nulls + cn_contam
    score = 100 - (issues / max(lines,1) * 100)
    print(f"  QUALITY SCORE: {score:.1f}% clean")
    
    if samples:
        print(f"  SAMPLES:")
        for s in samples:
            print(f"    {s}")
