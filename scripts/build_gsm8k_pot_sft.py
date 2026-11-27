#!/usr/bin/env python3
"""build_gsm8k_pot_sft.py — turn GSM8K's calculator annotations into evaluated PoT rows (Option B1).

Each arithmetic step in <|think|> carries its evaluated value as an inline Python
comment (`v1 = 48 / 2  # = 24`), so standalone C++/GGUF inference computes step by
step in context without requiring an external Python interpreter at runtime, while
remaining 100% valid executable Python.
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
          "answer step by step, then state the result.")


def fmt_num(val):
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    return str(val)


def to_number(s):
    s = str(s).replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return int(round(f)) if abs(f - round(f)) < 1e-9 else round(f, 6)
    except ValueError:
        return None


def build_program(answer_full, trace_values=True):
    """Calculator steps -> Python with inline evaluated trace comments."""
    steps = CALC_RE.findall(answer_full)
    if not steps or len(steps) > 12:
        return None
    lines, env, last = [], {}, None
    for i, (expr, _result) in enumerate(steps):
        expr = re.sub(r"\s+", " ", expr.strip())
        if not SAFE_RE.match(expr):
            return None
        var = f"v{i+1}"
        try:
            val = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None
        v_str = fmt_num(val)
        if trace_values:
            lines.append(f"{var} = {expr}  # = {v_str}")
        else:
            lines.append(f"{var} = {expr}")
        env[var] = val
        last = var
    if last is None:
        return None
    final_str = fmt_num(env[last])
    if trace_values:
        lines.append(f"print({last})  # = {final_str}")
    else:
        lines.append(f"print({last})")
    return "\n".join(lines), env[last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/gsm8k/train.jsonl")
    ap.add_argument("--out", default="sft_data/sft_gsm8k_pot")
    ap.add_argument("--no_trace_values", action="store_true")
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
        q = str(d.get("question", "")).replace("\r\n", "\n").strip()
        full = str(d.get("answer_full", "")).replace("\r\n", "\n")
        gold = to_number(d.get("answer")) if d.get("answer") is not None else None
        if gold is None:
            m = FINAL_RE.search(full)
            gold = to_number(m.group(1)) if m else None
        if not q or gold is None:
            continue
        built = build_program(full, trace_values=not args.no_trace_values)
        if built is None:
            no_calc += 1
            continue
        code, val = built
        if to_number(fmt_num(val)) != gold:
            mismatch += 1
            continue
        gold_str = fmt_num(gold)
        rows.append({
            "text": f"<|system|>{SYS_EN}<|user|>{q}<|assistant|>"
                    f"<|think|>{code}<|/think|>Therefore, the answer is {gold_str}.<|end|>",
            "lang": "en",
            "task": "gsm8k_pot",
            "expected_output": gold_str,
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
         "trace_values": not args.no_trace_values,
         "dropped_no_calc": no_calc, "dropped_answer_mismatch": mismatch},
        indent=2), encoding="utf-8")

    print(f"[GSM8K-PoT-B1] read={total:,} kept={kept:,} -> train={len(rows)-n_eval:,} eval={n_eval:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
