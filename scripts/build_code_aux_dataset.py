#!/usr/bin/env python3
"""build_code_aux_dataset.py — Code-explanation (C2) + bug-fix (C3) datasets.

C2 code-explanation:
  Procedural functions (deterministic templates: compute stats, string
  transforms, small algorithms) each rendered with a 3-language explanation
  of what the code does. Exec-verified: running the function on a sample
  input must produce the documented output.

C3 bug-fix:
  (buggy_code, failing_test, fixed_code) triples. Buggy variants are
  systematic mutations (swapped operators, off-by-one, wrong variable,
  swapped branch) applied to correct templates; the mutation is guaranteed
  to break at least one test (exec-checked).

Outputs:
  sft_data/sft_code_explain/train.jsonl   (~40K rows)
  sft_data/sft_code_bugfix/train.jsonl    (~30K rows)
  rl_data/code_bugfix.jsonl               (2K RL prompts, exec-verified)

Explanation formats are single-line-friendly where possible; multi-line code
is fine (newline tokenizer lands before training).
"""

import argparse
import json
import random
import re
import io
from contextlib import redirect_stdout
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS = {
    "en": "You are a coding assistant.",
    "id": "Kamu adalah asisten coding.",
    "ja": "あなたはコーディングアシスタントです。",
}


# ───────────────────────── code templates (name, code, doc, tests) ─────────
def t_stats(rng):
    vals = [rng.randint(-50, 50) for _ in range(rng.randint(5, 9))]
    code = (f"def summarize(nums):\n"
            f"    total = sum(nums)\n"
            f"    avg = total / len(nums)\n"
            f"    return total, round(avg, 2)\n")
    doc = {
        "en": "Define a function summarize(nums) that returns the sum of the numbers and their average rounded to two decimals.",
        "id": "Definisikan fungsi summarize(nums) yang mengembalikan jumlah bilangan dan rata-ratanya dibulatkan dua desimal.",
        "ja": "数値の合計と小数第2位に丸めた平均を返す関数summarize(nums)を定義してください。",
    }
    tests = [(str(vals), str((sum(vals), round(sum(vals)/len(vals), 2))))]
    return code, doc, tests


def t_norm(rng):
    s = " ".join(rng.choice(["Hello", "WORLD", "Foo", "BaR", "TeSt"]) for _ in range(rng.randint(2, 4)))
    code = (f"def norm(text):\n"
            f"    words = text.lower().split()\n"
            f"    return '-'.join(sorted(words))\n")
    doc = {
        "en": "Define norm(text) that lowercases the text, splits it into words, sorts them, and joins them with hyphens.",
        "id": "Definisikan norm(text) yang mengubah teks ke huruf kecil, memecah menjadi kata, mengurutkan, lalu menggabungkan dengan tanda hubung.",
        "ja": "テキストを小文字化して単語に分割し、並べ替えてハイフンでつなぐ関数norm(text)を定義してください。",
    }
    expected = "-".join(sorted(s.lower().split()))
    tests = [(f'"{s}"', expected)]
    return code, doc, tests


def t_filter(rng):
    vals = [rng.randint(-30, 30) for _ in range(rng.randint(6, 10))]
    code = (f"def keep_positive(nums):\n"
            f"    return [n for n in nums if n > 0]\n")
    doc = {
        "en": "Define keep_positive(nums) that returns only the positive numbers, preserving order.",
        "id": "Definisikan keep_positive(nums) yang mengembalikan hanya bilangan positif, menjaga urutan.",
        "ja": "正の数だけを順序を保って返す関数keep_positive(nums)を定義してください。",
    }
    tests = [(str(vals), str([v for v in vals if v > 0]))]
    return code, doc, tests


def t_lookup(rng):
    keys = rng.sample(["a", "b", "c", "d", "e", "f"], rng.randint(3, 5))
    vals = [rng.randint(1, 99) for _ in keys]
    pairs = ", ".join(f'"{k}": {v}' for k, v in zip(keys, vals))
    code = (f"def best(scores):\n"
            f"    return max(scores, key=scores.get)\n")
    doc = {
        "en": "Define best(scores) that returns the key with the highest value in the dict.",
        "id": "Definisikan best(scores) yang mengembalikan key dengan nilai tertinggi dalam dict.",
        "ja": "辞書の中で最も値が大きいキーを返す関数best(scores)を定義してください。",
    }
    tests = [(f"{{{pairs}}}", max(zip(vals, keys))[1])]
    return code, doc, tests


def t_digits(rng):
    n = rng.randint(100, 99999)
    code = (f"def digit_sum(n):\n"
            f"    return sum(int(d) for d in str(abs(n)))\n")
    doc = {
        "en": "Define digit_sum(n) that adds up all decimal digits of |n|.",
        "id": "Definisikan digit_sum(n) yang menjumlahkan semua digit desimal dari |n|.",
        "ja": "|n|の各桁の数字を合計する関数digit_sum(n)を定義してください。",
    }
    tests = [(str(n), str(sum(int(d) for d in str(n))))]
    return code, doc, tests


TEMPLATES = [t_stats, t_norm, t_filter, t_lookup, t_digits]


def run_snippet(stmt):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(stmt, {})
        return True, buf.getvalue().strip()
    except Exception:
        return False, None


def call_fn(code, fn, arg):
    stmt = code + f"\nprint({fn}({arg}))"
    return run_snippet(stmt)


# ───────────────────────── bug mutations ────────────────────────────────────
def mutate(code, rng):
    """Apply one systematic bug; return mutated code or None."""
    choices = []
    # swapped comparison
    if "> 0" in code:
        choices.append(("cmp", code.replace("> 0", ">= 0", 1)))
    # off-by-one in slicing
    if "[1]" in code:
        choices.append(("off", code.replace("[1]", "[0]", 1)))
    # wrong operator
    if "sum(" in code and "+" not in code.split("sum(")[1][:20]:
        pass
    if "sorted(" in code:
        choices.append(("sort", code.replace("sorted(", "sorted(", 1).replace("return '-'.join(sorted(words))", "return '-'.join(sorted(words, reverse=True))", 1)))
    if "round(avg, 2)" in code:
        choices.append(("rnd", code.replace("round(avg, 2)", "round(avg, 1)", 1)))
    if "abs(n)" in code:
        choices.append(("abs", code.replace("abs(n)", "n", 1)))
    if "max(scores" in code:
        choices.append(("mx", code.replace("max(scores", "min(scores", 1)))
    if not choices:
        return None, None
    return rng.choice(choices)


MUT_DOC = {
    "cmp": {
        "en": "The function has a bug: it keeps zeros even though only strictly positive numbers should be kept. Fix the comparison.",
        "id": "Fungsi tersebut punya bug: nilai nol tetap disertakan padahal hanya bilangan positif sejati yang harus diambil. Perbaiki perbandingannya.",
        "ja": "この関数にはバグがあります：正の数のみを残すべきなのに0も残ります。比較を修正してください。",
    },
    "sort": {
        "en": "The function has a bug: it sorts the words in descending order instead of ascending. Fix the sort.",
        "id": "Fungsi tersebut punya bug: kata diurutkan menurun padahal seharusnya menaik. Perbaiki pengurutannya.",
        "ja": "この関数にはバグがあります：単語を降順に並べています。昇順になるよう修正してください。",
    },
    "rnd": {
        "en": "The function has a bug: it rounds the average to one decimal instead of two. Fix the rounding.",
        "id": "Fungsi tersebut punya bug: rata-rata dibulatkan ke satu desimal padahal seharusnya dua. Perbaiki pembulatannya.",
        "ja": "この関数にはバグがあります：平均を小数第1位に丸めています。第2位に丸めるよう修正してください。",
    },
    "abs": {
        "en": "The function has a bug: it forgets to take the absolute value before summing digits, so negative inputs break it. Fix it.",
        "id": "Fungsi tersebut punya bug: lupa mengambil nilai mutlak sebelum menjumlahkan digit, sehingga input negatif error. Perbaiki.",
        "ja": "この関数にはバグがあります：絶対値を取らずに桁を合計するため、負の入力で壊れます。修正してください。",
    },
    "mx": {
        "en": "The function has a bug: it returns the key with the LOWEST value instead of the highest. Fix it.",
        "id": "Fungsi tersebut punya bug: mengembalikan key dengan nilai TERKECIL padahal seharusnya terbesar. Perbaiki.",
        "ja": "この関数にはバグがあります：最も値が小さいキーを返します。最大のキーを返すよう修正してください。",
    },
    "off": {
        "en": "The function has a bug: it uses index 0 where it should use index 1. Fix the index.",
        "id": "Fungsi tersebut punya bug: memakai indeks 0 padahal seharusnya 1. Perbaiki indeksnya.",
        "ja": "この関数にはバグがあります：インデックス1を使うべきところで0を使っています。修正してください。",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_explain", type=int, default=40_000)
    ap.add_argument("--n_bugfix", type=int, default=30_000)
    ap.add_argument("--n_rl", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    langs = ["en"] * 4 + ["id"] * 4 + ["ja"] * 2

    # ── C2: explanations (exec-verified per template instance)
    explain_rows = []
    i = 0
    while len(explain_rows) < args.n_explain:
        t = TEMPLATES[i % len(TEMPLATES)]
        i += 1
        code, doc, tests = t(rng)
        arg, expected = tests[0]
        # discover fn name
        m = re.match(r"def (\w+)\(", code)
        fn = m.group(1)
        ok, out = call_fn(code, fn, arg)
        if not ok or out != expected:
            continue
        lang = rng.choice(langs)
        # answer = explanation + code (model explains then writes code)
        body = f"{doc[lang]}\n\n```python\n{code.rstrip()}\n```"
        explain_rows.append({
            "text": f"<|system|>{SYS[lang]}<|user|>{doc[lang]}<|assistant|>{body}<|end|>",
            "lang": lang, "task": f"explain_{fn}",
        })
    rng.shuffle(explain_rows)

    # ── C3: bug-fix pairs (mutation must break >=1 test)
    bug_rows, rl_rows = [], []
    i = 0
    seen = set()
    while len(bug_rows) < args.n_bugfix:
        t = TEMPLATES[i % len(TEMPLATES)]
        i += 1
        code, doc, tests = t(rng)
        arg, expected = tests[0]
        m = re.match(r"def (\w+)\(", code)
        fn = m.group(1)
        kind, buggy = mutate(code, rng)
        if kind is None:
            continue
        # buggy must FAIL, correct must PASS
        ok_good, out_good = call_fn(code, fn, arg)
        ok_bad, out_bad = call_fn(buggy, fn, arg)
        if not ok_good or out_good != expected or ok_bad and out_bad == expected:
            continue
        lang = rng.choice(langs)
        h = (fn, kind, arg)
        if h in seen:
            continue
        seen.add(h)
        q = {
            "text_prompt": (
                f"{MUT_DOC[kind][lang]}\n\n```python\n{buggy.rstrip()}\n```"),
        }
        body = f"```python\n{code.rstrip()}\n```"
        row = {
            "text": f"<|system|>{SYS[lang]}<|user|>{q['text_prompt']}<|assistant|>{body}<|end|>",
            "lang": lang, "task": f"bugfix_{fn}_{kind}",
        }
        bug_rows.append(row)
        if len(rl_rows) < args.n_rl and rng.random() < 0.07:
            rl_rows.append({
                "task": f"bugfix_{fn}_{kind}", "lang": lang,
                "prompt": q["text_prompt"], "mock_input": " ",
                "expected_output": expected,
                "call_hint": f"{fn}({arg})",
            })

    rng.shuffle(bug_rows)
    rng.shuffle(rl_rows)

    for name, rows in (("sft_code_explain", explain_rows),
                       ("sft_code_bugfix", bug_rows)):
        d = ROOT / "sft_data" / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(ROOT / "rl_data" / "code_bugfix.jsonl", "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"explain: {len(explain_rows)} rows ({dict(Counter(r['task'] for r in explain_rows))})")
    print(f"bugfix : {len(bug_rows)} rows")
    print(f"rl     : {len(rl_rows)} rows -> rl_data/code_bugfix.jsonl")


if __name__ == "__main__":
    main()
