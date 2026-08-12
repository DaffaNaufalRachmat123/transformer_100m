#!/usr/bin/env python3
"""build_sft_translation.py — Convert large clean parallel TSV into SFT JSONL.

Reads data/parallel_enja.tsv (EN \t JA pairs, deduped + filtered by
fetch_parallel_enja.py) and emits SFT-format rows in BOTH directions with
varied system prompts and user templates.

Key design for a small (204M) model:
  - Dedup by normalized English text (subtitle corpora repeat EN lines a lot);
    keep the first (highest-confidence) JA translation per EN.
  - Keep sentences in a model-friendly length band (word count), so the model
    sees clean, learnable translation patterns rather than trivia.
  - Stratified sampling keeps a balanced mix of short and medium sentences.
  - Writes a held-out eval split of NOVEL sentences (removed from train) so
    generalization can be measured.

Usage:
    python3 scripts/build_sft_translation.py \
        [--in data/parallel_enja.tsv] \
        [--out sft_data/sft_clean/sft_translation_large.jsonl] \
        [--eval_out sft_data/sft_clean/sft_translation_eval.jsonl] \
        [--max_pairs 300000] [--eval_fraction 0.05] [--seed 42]
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_EN2JA = [
    "You are a professional English-to-Japanese translator.",
    "You are a translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "Translate the following text accurately.",
]
SYS_JA2EN = [
    "You are a professional Japanese-to-English translator.",
    "You are a translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "あなたはプロの翻訳者です。",
]
TMPL_EN2JA = [
    "Translate to Japanese: {t}",
    "日本語に翻訳してください: {t}",
    "Translate: {t}",
    "日本語に翻訳: {t}",
]
TMPL_JA2EN = [
    "Translate to English: {t}",
    "英語に翻訳してください: {t}",
    "英語に翻訳: {t}",
    "Translate: {t}",
]


def has_ja(s):
    return any('\u3040' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f'
               or '\uff00' <= c <= '\uffef' for c in s)


def word_count(s):
    return len(re.findall(r"[A-Za-z0-9']+", s))


def good_for_model(en, ja):
    """Keep sentences a small model can actually learn to translate well."""
    if not en or not ja:
        return False
    ew = word_count(en)
    if ew < 3 or ew > 18:
        return False
    # Japanese length sanity (chars) — roughly proportional to EN words.
    if len(ja) < 4 or len(ja) > 80:
        return False
    if not has_ja(ja):
        return False
    return True


def render(en, ja, rng):
    """Return (en2ja_json, ja2en_json) with varied templates."""
    sp = rng.choice(SYS_EN2JA)
    up = rng.choice(TMPL_EN2JA).format(t=en)
    t1 = f"<|system|>{sp}<|user|>{up}<|assistant|>{ja}<|end|>"
    sp = rng.choice(SYS_JA2EN)
    up = rng.choice(TMPL_JA2EN).format(t=ja)
    t2 = f"<|system|>{sp}<|user|>{up}<|assistant|>{en}<|end|>"
    return (
        json.dumps({"text": t1, "lang": "en-ja", "task": "translation"}, ensure_ascii=False),
        json.dumps({"text": t2, "lang": "ja-en", "task": "translation"}, ensure_ascii=False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="data/parallel_enja.tsv")
    ap.add_argument("--out", default="sft_data/sft_clean/sft_translation_large.jsonl")
    ap.add_argument("--eval_out", default="sft_data/sft_clean/sft_translation_eval.jsonl")
    ap.add_argument("--max_pairs", type=int, default=300000)
    ap.add_argument("--eval_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    in_path = ROOT / args.infile
    out_path = ROOT / args.out
    eval_path = ROOT / args.eval_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load pairs, dedup by normalized EN, keep model-friendly lengths.
    seen_en = set()
    pairs = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            en, ja = parts[0].strip(), parts[1].strip()
            key = re.sub(r"[^a-z0-9]+", " ", en.lower()).strip()
            if not key or key in seen_en:
                continue
            seen_en.add(key)
            if good_for_model(en, ja):
                pairs.append((en, ja))
    print(f"  loaded+dedup model-friendly pairs: {len(pairs)}")

    # 2. Stratified sample to max_pairs by EN word-count bucket.
    buckets = {}
    for p in pairs:
        b = min(word_count(p[0]) // 3, 5)  # buckets 0..5
        buckets.setdefault(b, []).append(p)
    target = int(args.max_pairs / max(len(buckets), 1))
    sampled = []
    for b in sorted(buckets):
        pool = buckets[b]
        rng.shuffle(pool)
        sampled.extend(pool[:target])
    rng.shuffle(sampled)
    print(f"  after stratified sampling: {len(sampled)}")

    # 3. Split held-out eval (novel sentences excluded from train).
    n_eval = max(1, int(len(sampled) * args.eval_fraction))
    eval_pairs = sampled[:n_eval]
    train_pairs = sampled[n_eval:]

    # 4. Write train (both directions).
    with open(out_path, "w", encoding="utf-8") as f:
        n = 0
        for en, ja in train_pairs:
            e2j, j2e = render(en, ja, rng)
            f.write(e2j + "\n")
            f.write(j2e + "\n")
            n += 2
    print(f"  train rows written: {n} ({n//2} unique pairs) -> {out_path}")

    # 5. Write held-out eval (both directions, marked lang with eval flag).
    with open(eval_path, "w", encoding="utf-8") as f:
        n = 0
        for en, ja in eval_pairs:
            e2j, j2e = render(en, ja, rng)
            f.write(e2j + "\n")
            f.write(j2e + "\n")
            n += 2
    print(f"  eval rows written: {n} ({n//2} unique pairs) -> {eval_path}")

    print(f"\nDone. Train unique pairs: {len(train_pairs)}, Eval unique pairs: {len(eval_pairs)}")


if __name__ == "__main__":
    main()