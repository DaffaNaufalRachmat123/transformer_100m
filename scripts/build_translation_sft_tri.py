#!/usr/bin/env python3
"""build_translation_sft_tri.py — Trilingual (6-direction) translation SFT set.

Inputs (2-col TSVs: side_a \t side_b):
  data/parallel_en_ja.tsv   (existing 1.96M pool)
  data/parallel_en_id.tsv   (filter_parallel_v2 output)
  data/parallel_ja_id.tsv   (filter_parallel_v2 output)

Directions and user-prompt styles follow the existing pipeline:
  en->ja "日本語に翻訳:", ja->en "Translate to English:",
  en->id "Terjemahkan ke bahasa Indonesia:", id->en "Translate to English:",
  ja->id "Terjemahkan ke bahasa Indonesia:", id->ja "日本語に翻訳:"

Eval split is taken at the PAIR level before training rows are rendered, and
additionally screened with a fuzzy signature so near-duplicates of eval
sentences cannot appear in train.

Output:
  sft_data/sft_translation_tri/train.jsonl
  sft_data/sft_translation_tri/eval.jsonl
  logs/translation_tri_stats.json
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_POOL = [
    "You are a multilingual translation assistant.",
    "You are a professional translator. Translate accurately and naturally.",
    "You are a helpful translation engine. Output only the translation.",
    "Kamu adalah asisten penerjemah multibahasa. Terjemahkan dengan akurat.",
]

PROMPTS = {
    ("en", "ja"): "日本語に翻訳: {src}",
    ("ja", "en"): "Translate to English: {src}",
    ("en", "id"): "Terjemahkan ke bahasa Indonesia: {src}",
    ("id", "en"): "Translate to English: {src}",
    ("ja", "id"): "Terjemahkan ke bahasa Indonesia: {src}",
    ("id", "ja"): "日本語に翻訳: {src}",
}

MAX_CHARS = 140


def bucket_of(s):
    n = len(s)
    if n < 45:
        return "short"
    if n < 95:
        return "medium"
    return "long"


def sig(s):
    """Cheap fuzzy signature: sorted set of lowercased 5-grams (first 24)."""
    t = re.sub(r"\s+", " ", s.casefold())
    grams = sorted({t[i:i+5] for i in range(max(len(t) - 4, 0))})
    return tuple(grams[:24])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enja", default="data/parallel_en_ja.tsv")
    ap.add_argument("--enid", default="data/parallel_en_id.tsv")
    ap.add_argument("--jaid", default="data/parallel_ja_id.tsv")
    ap.add_argument("--out_dir", default="sft_data/sft_translation_tri")
    ap.add_argument("--cap_enja", type=int, default=400_000)
    ap.add_argument("--cap_enid", type=int, default=400_000)
    ap.add_argument("--cap_jaid", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--eval_pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pools = {}
    for pair, path in (("en-ja", args.enja), ("en-id", args.enid), ("ja-id", args.jaid)):
        p = ROOT / path
        if not p.exists():
            print(f"WARN: {p} missing — pair {pair} skipped")
            continue
        pairs = []
        for line in open(p, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not a or not b or len(a) > MAX_CHARS or len(b) > MAX_CHARS:
                continue
            pairs.append((a, b))
        rng.shuffle(pairs)
        pools[pair] = pairs
        print(f"{pair}: {len(pairs)} usable pairs from {path}")

    # ── Eval split first (pair level), then fuzzy-screen train against eval
    eval_pairs = {}
    train_pairs = {}
    eval_sigs = set()
    for pair, pairs in pools.items():
        ev = []
        for a, b in pairs:
            if len(ev) >= args.eval_pairs:
                break
            sa, sb = sig(a), sig(b)
            if sa in eval_sigs or sb in eval_sigs:
                continue
            eval_sigs.add(sa)
            eval_sigs.add(sb)
            ev.append((a, b))
        eval_pairs[pair] = ev
        rest = [p for p in pairs if p not in set(ev)]
        train_pairs[pair] = rest

    caps = {"en-ja": args.cap_enja, "en-id": args.cap_enid, "ja-id": args.cap_jaid}
    train_rows, eval_rows = [], []
    for pair, pairs in train_pairs.items():
        cap = caps.get(pair, 0)
        if cap and len(pairs) > cap:
            pairs = pairs[:cap]
        la, lb = pair.split("-")
        for a, b in pairs:
            sys_p = rng.choice(SYS_POOL)
            train_rows.append({
                "text": f"<|system|>{sys_p}<|user|>{PROMPTS[(la, lb)].format(src=a)}<|assistant|>{b}<|end|>",
                "lang": f"{la}-{lb}", "task": "translation", "pair": pair,
                "bucket": bucket_of(a),
            })
            sys_p2 = rng.choice(SYS_POOL)
            train_rows.append({
                "text": f"<|system|>{sys_p2}<|user|>{PROMPTS[(lb, la)].format(src=b)}<|assistant|>{a}<|end|>",
                "lang": f"{lb}-{la}", "task": "translation", "pair": pair,
                "bucket": bucket_of(b),
            })
    for pair, pairs in eval_pairs.items():
        la, lb = pair.split("-")
        for a, b in pairs:
            eval_rows.append({
                "text": f"<|system|>{SYS_POOL[0]}<|user|>{PROMPTS[(la, lb)].format(src=a)}<|assistant|>{b}<|end|>",
                "lang": f"{la}-{lb}", "task": "translation", "pair": pair,
                "bucket": bucket_of(a),
            })
            eval_rows.append({
                "text": f"<|system|>{SYS_POOL[0]}<|user|>{PROMPTS[(lb, la)].format(src=b)}<|assistant|>{a}<|end|>",
                "lang": f"{lb}-{la}", "task": "translation", "pair": pair,
                "bucket": bucket_of(b),
            })

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_by_lang": dict(Counter(r["lang"] for r in train_rows)),
        "eval_by_lang": dict(Counter(r["lang"] for r in eval_rows)),
        "train_by_bucket": dict(Counter(r["bucket"] for r in train_rows)),
        "train_by_pair": dict(Counter(r["pair"] for r in train_rows)),
    }
    with open(ROOT / "logs" / "translation_tri_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone: {len(train_rows)} train + {len(eval_rows)} eval rows -> {out_dir}")
    print(f"  by lang: {stats['train_by_lang']}")
    print(f"  by bucket: {stats['train_by_bucket']}")


if __name__ == "__main__":
    main()
