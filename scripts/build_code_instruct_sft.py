#!/usr/bin/env python3
"""build_code_instruct_sft.py — widen the coding SFT pool.

The mixer is quota-proportional, so the smallest family caps the whole dataset:
coding had 9,006 unique rows against a 0.075 share, which pinned the mixed set
to 120,080 rows even though 400,000 were requested. MBPP's train split is only
374 problems, so rephrasing cannot fix that by itself.

These two sets are genuine instruction/solution pairs and are large enough to
lift the cap outright. They are English, which is also the language HumanEval
and MBPP are written in.

DECONTAMINATION is not optional here. Magicoder-Evol-Instruct in particular is
known to overlap HumanEval, and a leaked problem turns the benchmark into a
memorisation test. Every candidate is checked against the held-out sets by
normalised n-gram overlap and dropped on a hit.

Usage:
  python3 scripts/build_code_instruct_sft.py --max_rows 120000
"""

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_EN = ("You are a coding assistant. Write concise, correct programs and "
          "explain them briefly.")

SPECS = [
    # name, repo, config, fields -> (instruction, response)
    ("oss_instruct", "ise-uiuc/Magicoder-OSS-Instruct-75K", None, ("problem", "solution")),
    ("evol_instruct", "ise-uiuc/Magicoder-Evol-Instruct-110K", None, ("instruction", "response")),
]

WORD_RE = re.compile(r"[a-z0-9_]+")
DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ASSERT_RE = re.compile(r"^\s*assert\b.*$", re.M)


def norm_ngrams(text, n=10):
    w = WORD_RE.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def load_heldout():
    """Distinctive markers of the held-out problems.

    Matching whole prompts by n-gram does not work: the shared text is prose
    boilerplate like "your task is to write a function that", which flagged
    ~10% of candidates about HTML forms and robot arms as HumanEval leaks.
    Function names and assertion lines are what actually identify a problem.
    """
    names, asserts = set(), collections.Counter()
    nprob = 0
    for rel in ("eval_data/humaneval.jsonl", "eval_data/mbpp_heldout.jsonl"):
        p = ROOT / rel
        if not p.exists():
            print(f"  [WARN] {rel} missing — decontamination is incomplete")
            continue
        n = 0
        for line in p.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            nprob += 1
            for fn in DEF_RE.findall(str(d.get("prompt", ""))):
                # Generic names ('add', 'f') collide by chance; distinctive ones
                # like has_close_elements do not.
                if len(fn) >= 10 or ("_" in fn and len(fn) >= 8):
                    names.add(fn)
            per = set()
            for a in ASSERT_RE.findall(str(d.get("tests", ""))):
                per |= norm_ngrams(a, n=10)
            asserts.update(per)
        print(f"  decontam: {rel} -> {n} problems")
    # Every HumanEval test is "def check(candidate): assert candidate(...)", so
    # n-grams shared by several problems are that scaffolding, not a fingerprint.
    uniq = {g for g, c in asserts.items() if c == 1}
    print(f"  dropped {len(asserts)-len(uniq):,} boilerplate assertion n-grams "
          f"seen in >1 problem")
    return names, uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sft_data/sft_code_instruct")
    ap.add_argument("--max_rows", type=int, default=120000)
    ap.add_argument("--min_chars", type=int, default=120)
    ap.add_argument("--max_chars", type=int, default=6000)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERR] pip install datasets", file=sys.stderr)
        return 1

    print("loading held-out sets for decontamination")
    heldout_names, heldout_asserts = load_heldout()
    name_re = (re.compile(r"\b(" + "|".join(sorted(map(re.escape, heldout_names), key=len, reverse=True)) + r")\b")
               if heldout_names else None)
    print(f"  {len(heldout_names):,} distinctive function names, "
          f"{len(heldout_asserts):,} assertion n-grams")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    leaked = 0
    seen = set()
    stats = {}

    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for name, repo, cfg, (kq, ka) in SPECS:
            if rows >= args.max_rows:
                break
            kept = 0
            try:
                ds = load_dataset(repo, cfg, split="train", streaming=True)
            except Exception as e:
                print(f"  [WARN] {name}: {type(e).__name__}: {str(e)[:80]}")
                continue
            for d in ds:
                if rows >= args.max_rows:
                    break
                q, a = d.get(kq), d.get(ka)
                if not isinstance(q, str) or not isinstance(a, str):
                    continue
                q, a = q.strip(), a.strip()
                if not (args.min_chars <= len(q) + len(a) <= args.max_chars):
                    continue
                key = q[:120]
                if key in seen:
                    continue
                seen.add(key)
                blob = q + " " + a
                if (name_re and name_re.search(blob)) or \
                        (norm_ngrams(blob, n=10) & heldout_asserts):
                    leaked += 1
                    continue
                f.write(json.dumps({
                    "task": name,
                    "lang": "en",
                    "text": f"<|system|>{SYS_EN}<|user|>{q}<|assistant|>{a}<|end|>",
                    "has_think": False,
                    "source": name,
                    "family": "coding",
                }, ensure_ascii=False) + "\n")
                rows += 1
                kept += 1
                if kept % 10000 == 0:
                    print(f"  {name}: {kept:,} kept ({leaked:,} leaked so far)", flush=True)
            stats[name] = kept
            print(f"  {name}: {kept:,} rows")

    (out_dir / "stats.json").write_text(json.dumps(
        {"train": rows, "per_source": stats, "decontam_dropped": leaked},
        indent=2), encoding="utf-8")
    print(f"\n[done] {out_dir}  {rows:,} rows, {leaked:,} dropped as held-out leaks")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
