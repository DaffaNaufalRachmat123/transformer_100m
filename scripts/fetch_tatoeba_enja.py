#!/usr/bin/env python3
"""fetch_tatoeba_enja.py — Download clean EN<->JA parallel pairs from Tatoeba.

Joins jpn_sentences + eng_sentences via jpn-eng links, filters to short,
clean, high-quality pairs, and writes SFT-format JSONL (both directions).

Replaces the broken Tatoeba parser in prepare_translation_enja.py.

Usage:
    python3 scripts/fetch_tatoeba_enja.py [--max_pairs 15000] [--out sft_data/sft_clean/sft_translation_tatoeba.jsonl]

Requirements: network access. Uses only stdlib (urllib, bz2, gzip).
"""

import argparse
import bz2
import json
import io
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = "https://downloads.tatoeba.org/exports/per_language"
JPN_SENT = f"{BASE}/jpn/jpn_sentences.tsv.bz2"
ENG_SENT = f"{BASE}/eng/eng_sentences.tsv.bz2"
JPN_ENG_LINKS = f"{BASE}/jpn/jpn-eng_links.tsv.bz2"

SYSTEMS_EN2JA = [
    "You are a professional English-to-Japanese translator.",
    "You are a translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "Translate the following text accurately.",
]
SYSTEMS_JA2EN = [
    "You are a professional Japanese-to-English translator.",
    "You are a translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "あなたはプロの翻訳者です。",
]


def is_good_enja(en, ja, min_len=3, max_len=80):
    """Filter for clean, short, accurate-enough translation pairs."""
    if not en or not ja:
        return False
    if len(en) < min_len or len(ja) < min_len:
        return False
    if len(en) > max_len or len(ja) > max_len:
        return False
    # Japanese must contain kana/kanji
    if not any('\u3040' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f'
               or '\uff00' <= c <= '\uffef' for c in ja):
        return False
    # English must be mostly ASCII
    ascii_ratio = sum(1 for c in en if ord(c) < 128) / max(len(en), 1)
    if ascii_ratio < 0.8:
        return False
    # No control chars
    if any(ord(c) < 32 for c in en + ja):
        return False
    return True


def load_bz2_lines(url):
    """Stream-decompress a .bz2 file and yield lines."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    text = bz2.decompress(data).decode("utf-8", errors="replace")
    for line in text.split("\n"):
        yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=15000)
    ap.add_argument("--out", default="sft_data/sft_clean/sft_translation_tatoeba.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load jpn sentences: id -> text
    print("[1/4] Loading jpn sentences...")
    jpn = {}
    for line in load_bz2_lines(JPN_SENT):
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                jpn[int(parts[0])] = parts[2]
            except ValueError:
                pass
    print(f"  jpn sentences: {len(jpn)}")

    # 2. Load eng sentences: id -> text
    print("[2/4] Loading eng sentences...")
    eng = {}
    for line in load_bz2_lines(ENG_SENT):
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                eng[int(parts[0])] = parts[2]
            except ValueError:
                pass
    print(f"  eng sentences: {len(eng)}")

    # 3. Join via links
    print("[3/4] Joining links...")
    pairs = set()  # (en, ja) dedup
    for line in load_bz2_lines(JPN_ENG_LINKS):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            jid, eid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        ja = jpn.get(jid)
        en = eng.get(eid)
        if ja and en and is_good_enja(en, ja):
            pairs.add((en.strip(), ja.strip()))
    print(f"  raw good pairs: {len(pairs)}")

    # 4. Write SFT format (both directions)
    print("[4/4] Writing SFT JSONL...")
    rng = random.Random(args.seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]

    with open(out_path, "w", encoding="utf-8") as f:
        for en, ja in pairs:
            for direction, sysp in (("en-ja", SYSTEMS_EN2JA), ("ja-en", SYSTEMS_JA2EN)):
                sp = rng.choice(sysp)
                if direction == "en-ja":
                    user = f"Translate to Japanese: {en}"
                    asst = ja
                else:
                    user = f"Translate to English: {ja}"
                    asst = en
                text = f"<|system|>{sp}<|user|>{user}<|assistant|>{asst}<|end|>"
                f.write(json.dumps({"text": text, "lang": direction,
                                    "task": "translation"}, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(pairs)} unique pairs -> {len(pairs)*2} rows")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()