#!/usr/bin/env python3
"""Add reasoning spans to coding and math SFT rows, and widen the coding set.

Thinking is applied only to coding and math. For facts and translation a
reasoning span costs tokens and invites the model to narrate a justification it
cannot actually perform, which is how small models learn to sound confident
while being wrong; those tasks keep a direct answer.

The trace is derived from the problem itself (declared I/O, the assertions that
must hold, the arithmetic actually performed), never invented, so it cannot
teach reasoning that contradicts the target.

  python3 scripts/build_thinking_sft.py --out sft_data/sft_coding_think
"""
import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

THINK, THINK_END = "<|think|>", "<|/think|>"

PLAN_EN = [
    "Read the input, then {step}.",
    "The task: {step}. Read input first.",
    "Plan: parse the input, {step}, then print the result.",
]
PLAN_JA = [
    "入力を読み、{step}。",
    "手順：入力を解析し、{step}、結果を出力する。",
]

FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def unfence(text):
    """Return the code inside a markdown fence, or the text unchanged."""
    m = FENCE_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def turn_parts(text):
    """Split a rendered chat row into system/user/assistant pieces."""
    m = re.match(r"<\|system\|>(.*?)<\|user\|>(.*?)<\|assistant\|>(.*)", text,
                 re.DOTALL)
    if not m:
        return None
    sys_, user, ans = m.groups()
    for stop in ("<|end|>", "<|endoftext|>"):
        i = ans.find(stop)
        if i >= 0:
            ans = ans[:i]
    return sys_, user, ans.strip()


def describe_code(code, lang):
    """One-line summary of what the program does, from its AST, in `lang`.

    Both languages are generated from the same AST facts rather than
    translating one into the other, so a Japanese row never carries English
    fragments -- mixing scripts inside an answer is exactly what teaches the
    model to ignore the prompt language.
    """
    try:
        tree = ast.parse(unfence(code))
    except SyntaxError:
        return None
    reads = len([n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "input"])
    loops = len([n for n in ast.walk(tree)
                 if isinstance(n, (ast.For, ast.While))])
    conds = len([n for n in ast.walk(tree) if isinstance(n, ast.If)])
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    bits = []
    if reads:
        bits.append(f"read {reads} input value{'s' if reads > 1 else ''}"
                    if lang == "en" else f"入力を{reads}個読み取り")
    if funcs:
        bits.append(f"define {funcs[0]}()" if lang == "en"
                    else f"{funcs[0]}()を定義し")
    if loops:
        bits.append("loop over the values" if lang == "en" else "値をループ処理し")
    if conds:
        bits.append("branch on a condition" if lang == "en" else "条件で分岐し")
    if not bits:
        bits.append("compute the result directly" if lang == "en" else "直接計算する")
    return (", ".join(bits) if lang == "en" else "".join(bits))


def trace_for_code(user, ans, lang, rng):
    """Reasoning grounded in the reference solution's structure."""
    step = describe_code(ans, lang)
    if not step:
        return None
    tmpl = rng.choice(PLAN_EN if lang == "en" else PLAN_JA)
    return tmpl.format(step=step)


def trace_for_assert(user, tests, lang):
    """For MBPP-style rows the assertions are the specification."""
    if not tests:
        return None
    first = next((l.strip() for l in tests.split("\n") if l.strip().startswith("assert")),
                 None)
    if not first:
        return None
    if lang == "ja":
        return f"満たすべき条件：{first}"
    return f"The solution must satisfy: {first}"


def split_math(ans):
    """Move a PoT program into the reasoning span, leaving only the conclusion.

    Keeping the program in both halves would train the model to emit the same
    code twice; the program IS the reasoning, so it belongs inside the span.
    """
    m = FENCE_RE.search(ans)
    if not m:
        return None, None
    body = m.group(1).strip()
    if len(body.split("\n")) > 12:
        return None, None
    conclusion = (ans[:m.start()] + ans[m.end():]).strip()
    if not conclusion:
        return None, None
    return body, conclusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_root", default="sft_data")
    ap.add_argument("--out_code", default="sft_data/sft_think_code")
    ap.add_argument("--out_math", default="sft_data/sft_think_math")
    ap.add_argument("--sources", default="sft_coding_v3,sft_coding_mbpp,"
                                         "sft_code_bugfix,sft_math_pot")
    ap.add_argument("--langs", default="en,ja")
    ap.add_argument("--eval_frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    keep = {x.strip() for x in a.langs.split(",") if x.strip()}
    root = Path(a.sft_root)
    rows, stats, seen = [], Counter(), set()

    for name in [s.strip() for s in a.sources.split(",") if s.strip()]:
        f = root / name / "train.jsonl"
        if not f.exists():
            stats[f"missing:{name}"] += 1
            continue
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lang = str(rec.get("lang", "")).lower()
            if lang not in keep:
                stats["drop_lang"] += 1
                continue
            parts = turn_parts(rec.get("text", ""))
            if not parts:
                stats["unparsable"] += 1
                continue
            sys_, user, ans = parts
            if not ans:
                stats["empty_answer"] += 1
                continue

            if rec.get("tests"):
                trace = trace_for_assert(user, rec["tests"], lang)
            elif "math" in name:
                trace, ans = split_math(ans)
            else:
                trace = trace_for_code(user, ans, lang, rng)
            if not trace or not ans:
                stats["no_trace"] += 1
                continue

            text = (f"<|system|>{sys_}<|user|>{user}<|assistant|>"
                    f"{THINK}{trace}{THINK_END}{ans}<|end|>")
            h = hashlib.sha1(text.encode()).digest()
            if h in seen:
                stats["dup"] += 1
                continue
            seen.add(h)

            out = dict(rec)
            out["text"] = text
            out["task"] = rec.get("task", "code")
            out["has_think"] = True
            out["source"] = name
            out["family"] = "math" if "math" in name else "coding"
            rows.append(out)
            stats[f"kept:{name}"] += 1

    rng.shuffle(rows)
    for fam, out_path in (("coding", a.out_code), ("math", a.out_math)):
        part = [r for r in rows if r["family"] == fam]
        n_eval = int(len(part) * a.eval_frac)
        ev, tr = part[:n_eval], part[n_eval:]
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, chunk in (("train", tr), ("eval", ev)):
            with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out_dir / "stats.json").write_text(
            json.dumps({"train": len(tr), "eval": len(ev),
                        "lang": dict(Counter(r["lang"] for r in tr).most_common())},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[THINK] {fam:7s} train {len(tr):6d} / eval {len(ev):4d} -> {out_dir}")

    for k, v in stats.most_common(12):
        print(f"    {k:26s} {v}")


if __name__ == "__main__":
    main()
