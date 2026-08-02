#!/usr/bin/env python3
import re, os
from collections import Counter

ADULT_EN = ["porn","xxx","sexual","orgasm","masturb","erotic","nude","naked",
    "stripper","escort","brothel","prostitut","hentai","onlyfans",
    "adult video","hardcore","blowjob","anal","swingers","threesome",
    "bdsm","fetish","incest"]
GAMBLING_EN = ["casino","poker","blackjack","roulette","slot machine",
    "sportsbook","betting","gambling","lottery","bingo","jackpot"]
SPAM_EN = ["buy now","click here","limited offer","discount code",
    "free trial","order now","best price","subscribe now",
    "risk free","make money online","wire transfer"]

def has_adult(text):
    return sum(1 for kw in ADULT_EN if kw in text.lower()) >= 1
def has_gambling(text):
    return sum(1 for kw in GAMBLING_EN if kw in text.lower()) >= 2
def has_spam(text):
    m = sum(1 for kw in SPAM_EN if kw in text.lower())
    return (len(text) < 300 and m >= 2) or m >= 4
def classify(text):
    if not text: return False
    if len(text) < 80: return False
    if '\ufffd' in text: return False
    urls = re.findall(r'https?://[^\s]+', text)
    if len(urls) > 10 or sum(len(u) for u in urls) > len(text) * 0.3: return False
    if has_adult(text): return False
    if has_gambling(text): return False
    if has_spam(text): return False
    return True

def clean_file(label, input_path):
    print(f"CLEANING: {label}")
    output_path = input_path + ".clean"
    total = kept = 0
    with open(input_path,'r',encoding='utf-8',errors='replace') as f, \
         open(output_path,'w',encoding='utf-8') as o:
        for line in f:
            total += 1
            if classify(line.rstrip('\n\r')):
                o.write(line); kept += 1
            if total % 1000000 == 0:
                print(f"  {total/1e6:.0f}M lines, kept {kept/total*100:.1f}%")
    print(f"  {total:,} -> {kept:,} ({kept/total*100:.1f}%)")
    os.rename(input_path, input_path + ".orig")
    os.rename(output_path, input_path)
    print(f"  -> {input_path} (cleaned)\n")

clean_file("openwebtext.txt", "data/corpus_final/english/openwebtext.txt")
