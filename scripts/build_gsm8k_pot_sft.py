#!/usr/bin/env python3
"""build_gsm8k_pot_sft.py — turn GSM8K's calculator annotations into PoT rows.

sft_think_math is Program-of-Thought: the <|think|> span holds Python and the
answer follows it. That style is deliberate — a 492M model cannot do reliable
multi-step mental arithmetic, but it can emit three lines of Python whose
execution is exact. Adding natural-language CoT sets (MetaMath, orca-math)
would mix two answer styles, so this expands the PoT pool instead.

GSM8K annotates every arithmetic step inline: "Natalia sold 48/2 = <<48/2=24>>24
clips". 98.7% of the train split carries these, and they convert to Python
mechanically. Nothing is invented — the expressions are the dataset's own.

Every generated program is EXECUTED and kept only if it reproduces the official
"#### N" answer, so a mis-parse becomes a dropped row rather than a wrong label.

Usage:
  python3 scripts/build_gsm8k_pot_sft.py
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CALC_RE = re.compile(r"<<([^>=]+)=([^>]+)>>")
FINAL_RE = re.compile(r"####\s*([-\d.,]+)")
SAFE_RE = re.compile(r"^[0-9+\-*/(). ]+$")

SYS_EN = ("You are a math tutor. Write a short Python program that computes the "
          "answer, then state the result.")


def to_number(s):
    s = str(s).replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def build_program(answer_full):
    """Calculator steps -> Python. Returns (code, value) or None."""
    steps = CALC_RE.findall(answer_full)
    if not steps or len(steps) > 12:
        return None
    lines, env, last = [], {}, None
    for i, (expr, result) in enumerate(steps):
        expr = expr.strip()
        if not SAFE_RE.match(expr):
            return None
        var = f"v{i+1}"
        try:
            val = eval(expr, {"__builtins__": {}}, {})       # literals only
        except Exception:
            return None
        lines.append(f"{var} = {expr}")
        env[var] = val
        last = var
    if last is None:
        return None
    lines.append(f"print({last})")
    return "\n".join(lines), env[last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gsm8k/train.jsonl")
    ap.add_argument("--out", default="sft_data/sft_gsm8k_pot")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = ROOT / args.src
    if not src.exists():
        print(f"[ERR] missing {src}")
        return 1

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    total = kept = no_calc = mismatch = 0
    rows = []
    for line in src.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        total += 1
        q = str(d.get("question", "")).strip()
        full = str(d.get("answer_full", ""))
        gold = to_number(d.get("answer")) if d.get("answer") is not None else None
        if gold is None:
            m = FINAL_RE.search(full)
            gold = to_number(m.group(1)) if m else None
        if not q or gold is None:
            continue
        built = build_program(full)
        if built is None:
            no_calc += 1
            continue
        code, val = built
        # The program has to actually reproduce the official answer.
        if to_number(val) != gold:
            mismatch += 1
            continue
        rows.append({
            "text": f"<|system|>{SYS_EN}<|user|>{q}<|assistant|>"
                    f"<|think|>{code}<|/think|>Therefore, the answer is {gold}.<|end|>",
            "lang": "en",
            "task": "gsm8k_pot",
            "expected_output": str(gold),
            "has_think": True,
            "source": "gsm8k_pot",
            "family": "math",
        })
        kept += 1

    rng.shuffle(rows)
    n_eval = min(400, len(rows) // 50)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in rows[n_eval:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "eval.jsonl").open("w", encoding="utf-8") as f:
        for r in rows[:n_eval]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "stats.json").write_text(json.dumps(
        {"train": len(rows) - n_eval, "eval": n_eval, "read": total,
         "dropped_no_calc": no_calc, "dropped_answer_mismatch": mismatch},
        indent=2), encoding="utf-8")

    print(f"[done] {out_dir}")
    print(f"  read {total:,}  kept {kept:,} ({100*kept/max(total,1):.1f}%)  "
          f"no-calc {no_calc:,}  answer-mismatch {mismatch:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
