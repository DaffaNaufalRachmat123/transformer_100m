#!/usr/bin/env python3
"""build_dpo_tri.py — DPO preference pairs for 3 language pairs.

Chosen   = the real parallel translation.
Rejected = (a) 60% mismatched translation: a correct-looking translation of a
              DIFFERENT source sentence in the same pair (hard negative), or
          (b) 40% untranslated copy of the source.

Rows follow the existing DPO schema: {system, prompt, chosen, rejected, lang, task}.

Usage:
  python3 scripts/build_dpo_tri.py [--per_pair 20000]
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_POOL = [
    "You are a professional translator. Provide natural, idiomatic translations.",
    "You are a multilingual translation assistant.",
    "Kamu adalah penerjemah profesional. Berikan terjemahan yang alami dan akurat.",
]

PROMPTS = {
    "en-ja": "Translate from EN to JA:\n{src}",
    "ja-en": "Translate from JA to EN:\n{src}",
    "en-id": "Translate from EN to ID:\n{src}",
    "id-en": "Translate from ID to EN:\n{src}",
    "ja-id": "Translate from JA to ID:\n{src}",
    "id-ja": "Translate from ID to JA:\n{src}",
}

MAX_CHARS = 140


def has_ja(s):
    return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in s)


def direction_ok(src, tgt, lang):
    """Sanity: target must be in the target language."""
    if lang.endswith("ja"):
        return has_ja(tgt)
    return re.search(r"[A-Za-z]{2,}", tgt) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enja", default="data/parallel_en_ja.tsv")
    ap.add_argument("--enid", default="data/parallel_en_id.tsv")
    ap.add_argument("--jaid", default="data/parallel_ja_id.tsv")
    ap.add_argument("--jaid_nllb", default="data/parallel_ja_id_nllb.tsv",
                    help="synthetic NLLB-distilled ja-id pairs (optional)")
    ap.add_argument("--per_pair", type=int, default=20000)
    ap.add_argument("--out", default="sft_data/dpo_tri/dpo_translation_tri.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pools = {}
    for pair, path in (("en-ja", args.enja), ("en-id", args.enid), ("ja-id", args.jaid)):
        p = ROOT / path
        if not p.exists():
            print(f"WARN: {p} missing — {pair} skipped")
            continue
        pairs = []
        for line in open(p, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a and b and len(a) <= MAX_CHARS and len(b) <= MAX_CHARS:
                pairs.append((a, b))
        rng.shuffle(pairs)
        pools[pair] = pairs
        print(f"{pair}: {len(pairs)} pairs")

    # optional synthetic ja-id pool
    p = ROOT / args.jaid_nllb
    if p.exists():
        extra = []
        for line in open(p, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                a, b = parts[0].strip(), parts[1].strip()
                if a and b and len(a) <= MAX_CHARS and len(b) <= MAX_CHARS:
                    extra.append((a, b))
        rng.shuffle(extra)
        pools["ja-id"] = pools.get("ja-id", []) + extra
        print(f"ja-id: +{len(extra)} synthetic (NLLB) pairs")

    rows = []
    stats = Counter()
    for pair, pairs in pools.items():
        n = min(args.per_pair, len(pairs))
        sample = pairs[:n]
        for i, (a, b) in enumerate(sample):
            la, lb = pair.split("-")
            # alternate direction per pair so both directions are covered
            lang = f"{la}-{lb}" if i % 2 == 0 else f"{lb}-{la}"
            src, chosen = (a, b) if i % 2 == 0 else (b, a)
            if not direction_ok(src, chosen, lang):
                continue
            rejected = None
            if rng.random() < 0.60:
                # mismatched translation from a neighbouring pair (hard negative)
                for j in ((i + 7) % len(sample), (i + 13) % len(sample), (i + 29) % len(sample)):
                    other = sample[j]
                    cand = other[1] if i % 2 == 0 else other[0]
                    if cand != chosen and direction_ok(src, cand, lang):
                        rejected = cand
                        stats[f"{pair}:mismatch"] += 1
                        break
            if rejected is None:
                rejected = src  # untranslated copy
                stats[f"{pair}:copy"] += 1
            if rejected == chosen:
                continue
            rows.append({
                "system": rng.choice(SYS_POOL),
                "prompt": PROMPTS[lang].format(src=src),
                "chosen": chosen,
                "rejected": rejected,
                "lang": lang,
                "task": "translation_quality",
            })

    rng.shuffle(rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(rows)} DPO rows -> {out_path}")
    print(f"  by lang: {dict(Counter(r['lang'] for r in rows))}")
    print(f"  stats: {dict(stats)}")


if __name__ == "__main__":
    main()
