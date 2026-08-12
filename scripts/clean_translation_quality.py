#!/usr/bin/env python3
"""clean_translation_quality.py — Deep quality filter for SFT & DPO translation data.

Removes translation rows that are actually "ngaco":
  - JA answer for en-ja that contains NO Japanese (untranslated)
  - EN answer for ja-en that contains NO Latin (untranslated)
  - mojibake / encoding-garbled rows (many 0x80-0xBF bytes)
  - overly short answers (< 3 chars)
  - identical source/target (placeholder like %1 (%2), code, etc.)

Reads & rewrites in place:
  sft_data/sft_translation_focused/train.jsonl
  sft_data/dpo_enja/dpo_translation_enja.jsonl

Usage:
  python3 scripts/clean_translation_quality.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SFT = ROOT / "sft_data" / "sft_translation_focused" / "train.jsonl"
DPO = ROOT / "sft_data" / "dpo_enja" / "dpo_translation_enja.jsonl"


def has_ja(s):
    return any('\u3040' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f'
               or '\uff00' <= c <= '\uffef' for c in s)


def has_lat(s):
    return bool(re.search(r'[A-Za-z]{2,}', s))


def is_mojibake(s):
    weird = sum(1 for c in s if '\x80' <= c <= '\xbf')
    return weird > 5


def looks_placeholder(s):
    # placeholder / config / code-style text unlikely to be a real sentence
    return bool(re.search(r'%[0-9]', s))  # %1, %2, %s etc.


def looks_code_or_ui(s):
    # Only clear UI-action tags / raw code tokens that are NOT a natural
    # translation. Technical words like "function"/"import" inside a valid
    # translated sentence are legitimate and must be kept.
    return bool(re.search(r'@action|%[0-9]', s))


def clean_sft(lines):
    kept, dropped = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        t = r.get("text", "")
        parts = t.split("<|assistant|>")
        if len(parts) < 2:
            dropped += 1
            continue
        asst = parts[1].split("<|end|>")[0].strip()
        user = t.split("<|user|>")[1].split("<|assistant|>")[0] if "<|user|>" in t else ""
        lang = r.get("lang")
        bad = False
        if not asst or len(asst) < 3:
            bad = True
        elif lang == "en-ja" and not has_ja(asst):
            bad = True
        elif lang == "ja-en" and not has_lat(asst):
            bad = True
        if is_mojibake(asst) or is_mojibake(t):
            bad = True
        # Only strip code/UI/placeholder from translation rows; general chat
        # (lang=en/ja/id) legitimately contains code/python/braces and is kept.
        if lang in ("en-ja", "ja-en"):
            if looks_placeholder(user):
                bad = True
            if looks_code_or_ui(user):
                bad = True
        if bad:
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped


def clean_dpo(lines):
    kept, dropped = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        chosen = r.get("chosen", "")
        rejected = r.get("rejected", "")
        lang = r.get("lang")
        bad = False
        if not chosen or not rejected:
            bad = True
        elif lang == "en-ja":
            if not has_ja(chosen) or not has_ja(rejected):
                bad = True
        elif lang == "ja-en":
            if not has_lat(chosen) or not has_lat(rejected):
                bad = True
        if is_mojibake(chosen) or is_mojibake(rejected):
            bad = True
        if bad:
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped


def write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def main():
    for name, path, fn in [("SFT", SFT, clean_sft), ("DPO", DPO, clean_dpo)]:
        if not path.exists():
            print(f"[{name}] skip (not found): {path}")
            continue
        lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
        orig = len(lines)
        kept, dropped = fn(lines)
        write(path, kept)
        print(f"[{name}] {orig} -> {len(kept)} (dropped {dropped})  {path}")


if __name__ == "__main__":
    main()