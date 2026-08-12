#!/usr/bin/env python3
"""fetch_parallel_enja.py — Download a large, high-quality EN<->JA parallel corpus.

Pulls from multiple open sources, dedups, applies a strict quality filter, and
writes a clean tab-separated (EN \t JA) file that SFT formatters consume.

Sources (all open, no auth):
  1. Tatoeba full export  — clean, high-quality short sentences (~1M raw pairs)
  2. OPUS OpenSubtitles   — subtitle dialogue (short, conversational)
  3. OPUS News-Commentary — clean news commentary
  4. OPUS KFTT            — Kyoto Free Translation Task (high quality)
  5. OPUS wikimedia       — Wikipedia article pairs

Downloads are streamed + incrementally filtered so memory stays bounded.

Usage:
    python3 scripts/fetch_parallel_enja.py \
        [--sources tatoeba,opensubtitles,newscommentary,kftt,wikimedia] \
        [--min_len 3] [--max_len 120] \
        [--out data/parallel_enja.tsv]

Output: data/parallel_enja.tsv  (one pair per line: EN\tJA, deduped, filtered)
"""

import argparse
import bz2
import gzip
import io
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language"
TATOEBA_JPN = f"{TATOEBA_BASE}/jpn/jpn_sentences.tsv.bz2"
TATOEBA_ENG = f"{TATOEBA_BASE}/eng/eng_sentences.tsv.bz2"
TATOEBA_LINKS = f"{TATOEBA_BASE}/jpn/jpn-eng_links.tsv.bz2"

OPUS_BASE = "https://object.pouta.csc.fi"
OPUS_CORPORA = {
    "opensubtitles": ("OPUS-OpenSubtitles/v2018/moses/en-ja.txt.zip", 1_500_000),
    "newscommentary": ("OPUS-News-Commentary/v16/moses/en-ja.txt.zip", 300_000),
    "kftt": ("OPUS-KFTT/v1.0/moses/en-ja.txt.zip", 400_000),
    "wikimedia": ("OPUS-wikimedia/v20230407/moses/en-ja.txt.zip", 1_000_000),
}


def has_ja(s):
    return any('\u3040' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f'
               or '\uff00' <= c <= '\uffef' for c in s)


def good_pair(en, ja, min_len, max_len):
    """Strict filter: both sides present, correct writing systems, sane length."""
    en = en.strip()
    ja = ja.strip()
    if not en or not ja:
        return None
    if len(en) < min_len or len(ja) < min_len:
        return None
    if len(en) > max_len or len(ja) > max_len:
        return None
    # Target must actually contain Japanese script (kana/kanji).
    if not has_ja(ja):
        return None
    # English side must be mostly ASCII (avoid mojibake / CJK leakage into EN).
    if not en:
        return None
    ascii_ratio = sum(1 for c in en if ord(c) < 128) / len(en)
    if ascii_ratio < 0.8:
        return None
    # Reject placeholder / template / code-like text.
    if re.search(r'%[0-9]|@action|(^|\s)\{\{', en + ja):
        return None
    # Reject control chars / tabs / newlines.
    if any(c in en + ja for c in ('\t', '\n', '\r', '\x00')):
        return None
    # Reject obvious non-translation (identical text on both sides).
    if en.lower().strip() == ja.lower().strip():
        return None
    return (en, ja)


def _stream_bz2(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    text = bz2.decompress(data).decode("utf-8", errors="replace")
    return text.split("\n")


def fetch_tatoeba(out_f, seen, min_len, max_len):
    """Full Tatoeba: join jpn_sentences x eng_sentences via jpn-eng links."""
    print("[tatoeba] loading jpn sentences...")
    jpn = {}
    for line in _stream_bz2(TATOEBA_JPN):
        p = line.split("\t")
        if len(p) >= 3:
            try:
                jpn[int(p[0])] = p[2].strip()
            except ValueError:
                pass
    print(f"[tatoeba]   jpn: {len(jpn)}")

    print("[tatoeba] loading eng sentences...")
    eng = {}
    for line in _stream_bz2(TATOEBA_ENG):
        p = line.split("\t")
        if len(p) >= 3:
            try:
                eng[int(p[0])] = p[2].strip()
            except ValueError:
                pass
    print(f"[tatoeba]   eng: {len(eng)}")

    print("[tatoeba] joining links...")
    n = 0
    for line in _stream_bz2(TATOEBA_LINKS):
        p = line.split("\t")
        if len(p) < 2:
            continue
        try:
            jid, eid = int(p[0]), int(p[1])
        except ValueError:
            continue
        ja = jpn.get(jid)
        en = eng.get(eid)
        if ja and en:
            pair = good_pair(en, ja, min_len, max_len)
            if pair:
                key = (pair[0].lower(), pair[1])
                if key not in seen:
                    seen.add(key)
                    out_f.write(f"{pair[0]}\t{pair[1]}\n")
                    n += 1
    print(f"[tatoeba]   kept: {n}")


def _moses_pairs(zip_url):
    """Yield (en, ja) from an OPUS .txt.zip (file names en-ja.en / en-ja.ja)."""
    req = urllib.request.Request(zip_url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    en_name = [n for n in zf.namelist() if n.endswith(".en")][0]
    ja_name = [n for n in zf.namelist() if n.endswith(".ja")][0]
    en_lines = zf.read(en_name).decode("utf-8", errors="replace").split("\n")
    ja_lines = zf.read(ja_name).decode("utf-8", errors="replace").split("\n")
    for en, ja in zip(en_lines, ja_lines):
        yield en, ja


def fetch_opus(name, out_f, seen, min_len, max_len):
    rel, cap = OPUS_CORPORA[name]
    url = f"{OPUS_BASE}/{rel}"
    print(f"[opus:{name}] downloading {url}")
    n = 0
    for en, ja in _moses_pairs(url):
        pair = good_pair(en, ja, min_len, max_len)
        if pair:
            key = (pair[0].lower(), pair[1])
            if key not in seen:
                seen.add(key)
                out_f.write(f"{pair[0]}\t{pair[1]}\n")
                n += 1
                if n >= cap:
                    break
    print(f"[opus:{name}]   kept: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources",
                    default="tatoeba,opensubtitles,newscommentary,kftt,wikimedia")
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--out", default="data/parallel_enja.tsv")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for name in args.sources.split(","):
            name = name.strip().lower()
            if not name:
                continue
            if name == "tatoeba":
                fetch_tatoeba(f, seen, args.min_len, args.max_len)
            elif name in OPUS_CORPORA:
                fetch_opus(name, f, seen, args.min_len, args.max_len)
            else:
                print(f"[warn] unknown source: {name}")
            total = len(seen)

    print(f"\nDone: {total} unique EN<->JA pairs -> {out_path}")


if __name__ == "__main__":
    main()