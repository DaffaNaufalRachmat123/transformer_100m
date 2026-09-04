#!/usr/bin/env python3
"""build_translation_bank.py — High-frequency sentence bank (T2 track).

Mines the most common, short, conversational sentence pairs from the raw
multi-source parallel data. These pairs are the "closed world" of everyday
translation — the track that can reach ~100% exact-match through injection +
SFT + RL (same formula as facts v11).

Selection score per pair:
  + appears in MULTIPLE distinct sources (strongest commonality signal)
  + short (both sides <= 80 chars)
  + passes junk filters
  + sentence-final punctuation or interjection (natural utterances)

Outputs:
  sft_data/sft_translation_bank/train.jsonl   (both directions per pair)
  sft_data/sft_translation_bank/eval.jsonl    (held-out bank pairs, 15%)
  data/corpus_inject_bank/bank_{enid,enja,jaid}.txt  (pretrain injection)

Usage:
  python3 scripts/build_translation_bank.py [--per_pair 60000] [--exposures 200]
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_parallel_v2 import text_junk

ROOT = Path(__file__).resolve().parent.parent

MAX_CHARS = 80

SYS_POOL = [
    "You are a professional translator. Translate accurately and naturally.",
    "You are a multilingual translation assistant.",
    "Kamu adalah asisten penerjemah multibahasa.",
]
PROMPTS = {
    ("en", "id"): "Terjemahkan ke bahasa Indonesia: {src}",
    ("id", "en"): "Translate to English: {src}",
    ("en", "ja"): "日本語に翻訳: {src}",
    ("ja", "en"): "Translate to English: {src}",
    ("ja", "id"): "Terjemahkan ke bahasa Indonesia: {src}",
    ("id", "ja"): "日本語に翻訳: {src}",
}


def natural_utterance(a, b):
    """Short conversational sentence with terminal punctuation/interjection."""
    for s in (a, b):
        if not (3 <= len(s) <= MAX_CHARS):
            return False
    # at least one side ends like a natural utterance
    ok_end = any(s.rstrip().endswith((".", "?", "!", "。", "？", "！"))
                 for s in (a, b))
    return ok_end


def load_pair_sources(pair_dir):
    """pair_dir with 3-col tsvs -> {(a, b): set(sources)}"""
    by_pair = defaultdict(set)
    d = ROOT / pair_dir
    if not d.exists():
        return by_pair
    for f in sorted(d.glob("*.tsv")):
        src = f.stem
        for line in open(f, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            a, b = p[0].strip(), p[1].strip()
            by_pair[(a, b)].add(src)
    return by_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pair", type=int, default=60_000)
    ap.add_argument("--eval_frac", type=float, default=0.15)
    ap.add_argument("--exposures", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ── 1. en-id: multi-source mining from raw dir
    enid = load_pair_sources("data/parallel_raw/en-id")
    # ── 2. ja-id: tatoeba only (NLLB distill will extend later)
    jaid = load_pair_sources("data/parallel_raw/ja-id")
    # ── 3. en-ja: from the cleaned big TSV (single source; score by
    #        naturalness only)
    enja = {}
    p = ROOT / "data" / "parallel_en_ja.tsv"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            q = line.rstrip("\n").split("\t")
            if len(q) < 2:
                continue
            enja[(q[0].strip(), q[1].strip())] = {"clean"}

    banks = {}
    for name, pool, cap in (("en-id", enid, args.per_pair),
                            ("ja-id", jaid, args.per_pair),
                            ("en-ja", enja, args.per_pair)):
        scored = []
        for (a, b), srcs in pool.items():
            if not natural_utterance(a, b):
                continue
            if text_junk(a, b):
                continue
            n_src = len(srcs)
            # score: source agreement dominates; shorter is better
            score = n_src * 1000 + (MAX_CHARS * 2 - len(a) - len(b))
            scored.append((score, a, b))
        scored.sort(key=lambda x: -x[0])
        # keep the top slice; drop near-duplicate sources of same EN side
        seen_a = set()
        sel = []
        for score, a, b in scored:
            ka = a.casefold()
            if ka in seen_a:
                continue
            seen_a.add(ka)
            sel.append((a, b))
            if len(sel) >= cap:
                break
        banks[name] = sel
        print(f"{name}: {len(pool)} pool -> {len(sel)} bank pairs "
              f"(multi-source share: "
              f"{sum(1 for s, a, b in scored[:len(sel)] if s >= 1000)/max(len(sel),1):.0%})")

    # ── 4. split train/eval by pair, render SFT rows + injection corpus
    out_dir = ROOT / "sft_data" / "sft_translation_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    inj_dir = ROOT / "data" / "corpus_inject_bank"
    inj_dir.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows = [], []
    stats = {}
    for name, sel in banks.items():
        rng.shuffle(sel)
        n_eval = int(len(sel) * args.eval_frac)
        eval_set = sel[:n_eval]
        train_set = sel[n_eval:]
        la, lb = name.split("-")
        pair_key = name.replace("-", "")

        inj_lines = []
        for a, b in train_set:
            for (src, tgt), (sa, sb), lang in (
                ((la, lb), (a, b), f"{la}-{lb}"),
                ((lb, la), (b, a), f"{lb}-{la}"),
            ):
                train_rows.append({
                    "text": f"<|system|>{rng.choice(SYS_POOL)}<|user|>"
                            f"{PROMPTS[(src, tgt)].format(src=sa)}<|assistant|>{sb}<|end|>",
                    "lang": lang, "task": "translation_bank", "pair": name,
                })
            # injection render (both directions concatenated, like mixed bucket)
            inj_lines.append(f"{a} {b}")
            inj_lines.append(f"{b} {a}")
        for a, b in eval_set:
            train_rows.append({  # keep schema uniform; eval written below
                "text": f"<|system|>{SYS_POOL[0]}<|user|>"
                        f"{PROMPTS[(la, lb)].format(src=a)}<|assistant|>{b}<|end|>",
                "lang": f"{la}-{lb}", "task": "translation_bank", "pair": name})
            eval_rows.append({
                "text": f"<|system|>{SYS_POOL[0]}<|user|>"
                        f"{PROMPTS[(la, lb)].format(src=a)}<|assistant|>{b}<|end|>",
                "lang": f"{la}-{lb}", "task": "translation_bank", "pair": name})

        stats[name] = {"train_pairs": len(train_set), "eval_pairs": len(eval_set)}

        # injection file with exposures
        base = inj_lines
        total = list(base)
        for _ in range(args.exposures - 1):
            shuf = list(base)
            rng.shuffle(shuf)
            total.extend(shuf)
        rng.shuffle(total)
        with open(inj_dir / f"bank_{pair_key}.txt", "w", encoding="utf-8") as f:
            for l in total:
                f.write(l + "\n")
        stats[name]["injection_lines"] = len(total)

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(ROOT / "logs" / "translation_bank_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nBank SFT: {len(train_rows)} train + {len(eval_rows)} eval -> {out_dir}")
    print(f"Injection corpus (x{args.exposures}) -> {inj_dir}")
    print(f"Stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
