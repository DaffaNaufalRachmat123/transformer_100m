#!/usr/bin/env python3
"""prepare_sft_smoke_mix.py — Build the single-dir SFT mix for the smoke run.

Merges all SFT sets into sft_data/sft_smoke/train.jsonl with per-source caps,
language-balanced reservoir sampling. Smoke target: ~1.7M rows ≈ 160M
tokens/epoch (2 epochs ≈ 320M tokens — enough for the format/memorization
signal without paying the full 3-epoch 750M-token bill).

--vocab6: filter for the EXISTING newline-stripping tokenizer (vocab_6).
Drops every row whose answer contains a multi-line Python block (stripped
newlines make such code untrainable/garbage); adds the legacy single-line
coding v1 set. Used by the $15 vast.ai smoke (204M recipe-proof).

Usage:
  python3 scripts/prepare_sft_smoke_mix.py [--vocab6]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (path, cap) — None cap = take all
SOURCES = [
    ("sft_data/sft_facts_v11/train.jsonl",        400_000),
    ("sft_data/sft_translation_tri/train.jsonl",   800_000),  # of 2.0M
    ("sft_data/sft_translation_bank/train.jsonl",  None),     # 233K all
    ("sft_data/sft_math/train.jsonl",              40_000),   # of 80K
    ("sft_data/sft_math_v2/train.jsonl",           80_000),   # of 120K
    ("sft_data/sft_math_pot/train.jsonl",          None),     # 60K all
    ("sft_data/sft_coding_v3/train.jsonl",         None),     # 9K all
    ("sft_data/sft_code_explain/train.jsonl",      None),     # 40K all
    ("sft_data/sft_code_bugfix/train.jsonl",       None),     # 30K all
]

# vocab6 mode: add legacy single-line coding set, drop multi-line-code rows
SOURCES_V6_EXTRA = [("sft_data/sft_coding/train.jsonl", None)]  # 312 rows


def v6_ok(r):
    """Keep row only if its answer survives the newline-stripping tokenizer.
    Multi-line code is untrainable with vocab_6 (Python needs newlines);
    plain-text answers (math CoT etc.) are fine — newlines just merge."""
    t = r.get("text", "")
    if "<|assistant|>" not in t:
        return False
    ans = t.split("<|assistant|>", 1)[1].split("<|end|>")[0]
    # fenced code (PoT): keep only if the code body is single-line
    if "```" in ans:
        for fence in ans.split("```"):
            if fence.startswith("python"):
                body = fence[len("python"):].strip("\n")
                if "\n" in body.strip():
                    return False
    # raw code answers (coding tasks have no fences): any multi-line code
    # (def/input/print/import) cannot survive newline stripping
    if "\n" in ans:
        for marker in ("print(", "input(", "def ", "import ", "for ", "if "):
            if marker in ans:
                return False
    return True

OUT = ROOT / "sft_data" / "sft_smoke"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab6", action="store_true",
                    help="filter for the newline-stripping vocab_6 tokenizer")
    args = ap.parse_args()

    rng = random.Random(42)
    out_dir = ROOT / "sft_data" / ("sft_smoke_v6" if args.vocab6 else "sft_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = list(SOURCES)
    if args.vocab6:
        sources += SOURCES_V6_EXTRA
    all_rows = []
    for path, cap in sources:
        p = ROOT / path
        if not p.exists():
            print(f"WARN missing: {path}")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if args.vocab6:
            before = len(rows)
            rows = [r for r in rows if v6_ok(r)]
            dropped = before - len(rows)
        else:
            dropped = 0
        if cap is not None and len(rows) > cap:
            # language-balanced reservoir: cap per (lang) proportionally
            by_lang = defaultdict(list)
            for r in rows:
                by_lang[r.get("lang", "?")].append(r)
            take = []
            n_langs = len(by_lang)
            per = cap // n_langs
            extra = cap - per * n_langs
            for i, (lang, group) in enumerate(sorted(by_lang.items())):
                k = per + (1 if i < extra else 0)
                if len(group) > k:
                    take.extend(rng.sample(group, k))
                else:
                    take.extend(group)
            rows = take
        all_rows.extend(rows)
        note = f" (v6 filter dropped {dropped})" if args.vocab6 else ""
        print(f"{path:<48} {len(rows):>8,} rows{note}")
    rng.shuffle(all_rows)
    out_path = out_dir / "train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # token estimate
    est = 0
    for r in all_rows[::50]:
        est += len(r["text"]) / 3.4  # ~3.4 chars/token blended
    est *= 50
    print(f"\nTOTAL {len(all_rows):,} rows -> {out_path}")
    print(f"est ~{est/1e6:.0f}M tokens/epoch (2 epochs ≈ {2*est/1e6:.0f}M)")


if __name__ == "__main__":
    main()
