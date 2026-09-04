#!/usr/bin/env python3
"""build_mixed_bucket.py — Assemble the pretrain 'mixed' bucket:
fact-injection corpora + parallel translation pairs rendered as bilingual
concatenated documents (one doc per line).

Parallel render: "EN sentence. ID sentence." (and the reverse direction) —
simple concatenation is a proven cross-lingual alignment signal in pretraining.
The loader samples files by size with replacement, so no manual oversampling
is needed.

Output (data/corpus_1b/mixed/):
  facts_en.txt / facts_ja.txt / facts_id.txt   (copied from corpus_inject_facts)
  parallel_enid.txt / parallel_enja.txt / parallel_jaid.txt

Usage:
  python3 scripts/build_mixed_bucket.py [--pairs_per_dir 400000]
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render_pairs(in_path, out_path, rng, max_pairs, max_chars=140):
    n = 0
    lines = []
    for line in open(in_path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 2:
            continue
        a, b = p[0].strip(), p[1].strip()
        if not a or not b or len(a) > max_chars or len(b) > max_chars:
            continue
        # both directions interleaved
        lines.append(f"{a} {b}")
        lines.append(f"{b} {a}")
        n += 1
        if n >= max_pairs:
            break
    rng.shuffle(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_per_dir", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mixed = ROOT / "data" / "corpus_1b" / "mixed"
    mixed.mkdir(parents=True, exist_ok=True)

    # 1. fact injection copies
    import shutil
    total_inject = 0
    for lang in ("en", "ja", "id"):
        src = ROOT / "data" / "corpus_inject_facts" / f"facts_{lang}.txt"
        dst = mixed / f"facts_{lang}.txt"
        if src.exists():
            shutil.copyfile(src, dst)
            total_inject += dst.stat().st_size
            print(f"copied facts_{lang}.txt ({dst.stat().st_size/1e6:.0f}MB)")

    # 2. parallel pairs
    stats = {}
    for pair, fname in (("en_id", "parallel_enid.txt"),
                        ("en_ja", "parallel_enja.txt"),
                        ("ja_id", "parallel_jaid.txt")):
        src = ROOT / "data" / f"parallel_{pair}.tsv"
        if not src.exists():
            print(f"WARN: {src} missing — skipped")
            continue
        out = mixed / fname
        n = render_pairs(src, out, rng, args.pairs_per_dir)
        stats[pair] = {"docs": n, "bytes": out.stat().st_size}
        print(f"{fname}: {n} docs ({out.stat().st_size/1e6:.0f}MB)")

    with open(ROOT / "logs" / "mixed_bucket_stats.json", "w") as f:
        json.dump({"inject_bytes": total_inject, "parallel": stats}, f, indent=2)
    print(f"\nmixed bucket ready: {mixed}")


if __name__ == "__main__":
    main()
