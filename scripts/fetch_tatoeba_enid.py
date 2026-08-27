#!/usr/bin/env python3
"""fetch_tatoeba_enid.py — Download clean EN<->ID parallel pairs from Tatoeba.

Mirrors fetch_tatoeba_enja.py but joins ind (Indonesian) + eng sentences via
ind-eng links, and emits both en-id and id-en SFT rows.

Usage:
    python3 scripts/fetch_tatoeba_enid.py [--max_pairs 15000] [--out sft_data/sft_clean/sft_translation_enid.jsonl]

Requirements: network access. Uses only stdlib (urllib, bz2).
"""

import argparse
import bz2
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = "https://downloads.tatoeba.org/exports/per_language"
IND_SENT = f"{BASE}/ind/ind_sentences.tsv.bz2"
ENG_SENT = f"{BASE}/eng/eng_sentences.tsv.bz2"
IND_ENG_LINKS = f"{BASE}/ind/ind-eng_links.tsv.bz2"

SYSTEMS_EN2ID = [
    "You are a professional English-to-Indonesian translator.",
    "You are a translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "Translate the following text accurately.",
]
SYSTEMS_ID2EN = [
    "You are a professional Indonesian-to-English translator.",
    "You are a translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "Kamu adalah penerjemah profesional.",
]
SYSTEMS_JA2ID = [
    "Kamu adalah penerjemah profesional Jepang-Indonesia.",
    "Kamu adalah penerjemah. Terjemahkan dengan akurat dan alami.",
]
SYSTEMS_ID2JA = [
    "あなたはプロのインドネシア語翻訳者です。",
    "あなたは翻訳者です。正確で自然に翻訳してください。",
]


def is_good_enid(en, iid, min_len=3, max_len=90):
    """Filter for clean, short translation pairs."""
    if not en or not iid:
        return False
    if len(en) < min_len or len(iid) < min_len:
        return False
    if len(en) > max_len or len(iid) > max_len:
        return False
    # Both are Latin-script: require mostly ASCII, no control chars
    for s in (en, iid):
        ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(len(s), 1)
        if ascii_ratio < 0.75:
            return False
        if any(ord(c) < 32 for c in s):
            return False
    return True


def load_bz2_lines(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    text = bz2.decompress(data).decode("utf-8", errors="replace")
    for line in text.split("\n"):
        yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=15000)
    ap.add_argument("--out", default="sft_data/sft_clean/sft_translation_enid.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading ind sentences...")
    ind = {}
    for line in load_bz2_lines(IND_SENT):
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                ind[int(parts[0])] = parts[2]
            except ValueError:
                pass
    print(f"  ind sentences: {len(ind)}")

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

    print("[3/4] Joining links...")
    pairs = set()
    for line in load_bz2_lines(IND_ENG_LINKS):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            ind_id, eng_id = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if ind_id in ind and eng_id in eng:
            en, iid = eng[eng_id], ind[ind_id]
            if is_good_enid(en, iid):
                pairs.add((en, iid))
    print(f"  clean pairs: {len(pairs)}")

    print("[4/4] Writing SFT JSONL...")
    rng = random.Random(args.seed)
    ordered = sorted(pairs)
    rng.shuffle(ordered)
    ordered = ordered[: args.max_pairs * 2]

    rows = []
    for en, iid in ordered:
        sys_e2i = rng.choice(SYSTEMS_EN2ID)
        rows.append({
            "text": f"<|system|>{sys_e2i}<|user|>Please translate this to Indonesian: {en}<|assistant|>{iid}<|end|>",
            "lang": "en-id", "task": "translation",
        })
        sys_i2e = rng.choice(SYSTEMS_ID2EN)
        rows.append({
            "text": f"<|system|>{sys_i2e}<|user|>Terjemahkan ke bahasa Inggris: {iid}<|assistant|>{en}<|end|>",
            "lang": "id-en", "task": "translation",
        })
    rng.shuffle(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done: {len(rows)} translation rows -> {out_path}")


if __name__ == "__main__":
    main()
