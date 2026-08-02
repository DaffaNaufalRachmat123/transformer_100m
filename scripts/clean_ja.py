#!/usr/bin/env python3
import re, os
from collections import Counter

ADULT_JA = ["エロ","無修正","アダルト","AV","風俗","セックス","hentai",
    "裏ビデオ","出会い系","援交","ソープ","デリヘル","人妻","不倫",
    "浮気","寝取られ","エロゲ","エロ漫画","ライブチャット",
    "性感エステ","メンズエステ","JKビジネス","ハプニングバー"]
GAMBLING_JA = ["パチンコ","パチスロ","スロット","競馬","競輪","競艇",
    "カジノ","ギャンブル","攻略法","必勝法","大当たり","換金"]
SPAM_JA = ["送料無料","最安値","激安","ランキング","おすすめ",
    "購入はこちら","今すぐ購入","カートに入れる","即日発送",
    "限定品","驚きの値段","品質保証","全国無料"]

def is_pure_chinese(text):
    cn = len(re.findall(r'[\u4e00-\u9fff]', text))
    ja = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    return cn > 50 and ja == 0

def has_adult(text):
    return sum(1 for kw in ADULT_JA if kw in text) >= 1
def has_gambling(text):
    return sum(1 for kw in GAMBLING_JA if kw in text) >= 1
def has_spam(text):
    m = sum(1 for kw in SPAM_JA if kw in text)
    return (len(text) < 500 and m >= 3) or m >= 5

def classify(text):
    if not text: return False
    if len(text) < 20: return False
    if len(text) > 5000: return False
    if '\ufffd' in text: return False
    if is_pure_chinese(text): return False
    if has_adult(text): return False
    if has_gambling(text): return False
    if has_spam(text): return False
    urls = re.findall(r'https?://[^\s]+', text)
    if len(urls) > 15 or sum(len(u) for u in urls) > len(text) * 0.3:
        return False
    return True

print("CLEANING: wiki_ja.txt")
input_path = "data/corpus_final/japanese/wiki_ja.txt"
output_path = input_path + ".clean"
stats = Counter()
total = kept = 0
total_chars = clean_chars = 0

with open(input_path,'r',encoding='utf-8',errors='replace') as f, \
     open(output_path,'w',encoding='utf-8') as o:
    for line in f:
        text = line.rstrip('\n\r')
        total += 1
        total_chars += len(text)
        if classify(text):
            o.write(line); kept += 1
            clean_chars += len(text)
        if total % 3000000 == 0:
            print(f"  {total/1e6:.1f}M lines, kept {kept/total*100:.1f}%")

print(f"  Total: {total:,} -> Kept: {kept:,} ({kept/total*100:.1f}%)")
print(f"  Size: {total_chars/1e9:.2f}GB -> {clean_chars/1e9:.2f}GB")

os.rename(input_path, input_path + ".orig")
os.rename(output_path, input_path)
print(f"  -> wiki_ja.txt (cleaned)\n")
