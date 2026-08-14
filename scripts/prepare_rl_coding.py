#!/usr/bin/env python3
"""
prepare_rl_coding.py — Build a small RL coding dataset (EN + JA).

Each example is a deterministic "write a program" task that reads from stdin
and prints one answer, so the reward can be computed by executing the generated
program and comparing stdout to the expected output (rule-based verifier).

Output: JSONL with fields:
    lang           : "en" | "ja"
    prompt         : instruction (chat format handled at train time)
    mock_input     : stdin content for the verifier
    expected_output: expected stdout (stripped)

Usage:
    python3 scripts/prepare_rl_coding.py --out rl_data/coding.jsonl --n 2000
"""
import argparse
import json
import random
import sys


def sum_of_two(rng):
    a, b = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
    inp = f"{a} {b}"
    exp = str(a + b)
    return inp, exp, lambda: True


def sum_of_list(rng):
    n = rng.randint(2, 8)
    vals = [rng.randint(-100, 100) for _ in range(n)]
    inp = str(n) + "\n" + "\n".join(map(str, vals))
    exp = str(sum(vals))
    return inp, exp, lambda: True


def count_even(rng):
    n = rng.randint(2, 8)
    vals = [rng.randint(-100, 100) for _ in range(n)]
    inp = str(n) + "\n" + "\n".join(map(str, vals))
    exp = str(sum(1 for v in vals if v % 2 == 0))
    return inp, exp, lambda: True


def max_of_nums(rng):
    n = rng.randint(2, 8)
    vals = [rng.randint(-1000, 1000) for _ in range(n)]
    inp = str(n) + "\n" + "\n".join(map(str, vals))
    exp = str(max(vals))
    return inp, exp, lambda: True


def arithmetic(rng):
    a, b = rng.randint(-100, 100), rng.randint(-100, 100)
    c = rng.randint(-100, 100)
    inp = f"{a} {b} {c}"
    exp = str(a * b + c)
    return inp, exp, lambda: True


def compare(rng):
    a, b = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
    if a == b:
        b = a + 1
    inp = f"{a} {b}"
    exp = "greater" if a > b else "smaller"
    return inp, exp, lambda: True


def absolute_diff(rng):
    a, b = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
    inp = f"{a} {b}"
    exp = str(abs(a - b))
    return inp, exp, lambda: True


TASKS = {
    "sum_of_two": sum_of_two,
    "sum_of_list": sum_of_list,
    "count_even": count_even,
    "max_of_nums": max_of_nums,
    "arithmetic": arithmetic,
    "compare": compare,
    "absolute_diff": absolute_diff,
}

EN_PROMPTS = {
    "sum_of_two": "Write a Python program. It reads two integers a and b from standard input (space separated) and prints their sum. Output only the number.",
    "sum_of_list": "Write a Python program. The first line of input is an integer n, followed by n integers. Print the sum of all n integers. Output only the number.",
    "count_even": "Write a Python program. The first line of input is an integer n, followed by n integers. Print how many of them are even. Output only the number.",
    "max_of_nums": "Write a Python program. The first line of input is an integer n, followed by n integers. Print the maximum of them. Output only the number.",
    "arithmetic": "Write a Python program. It reads three integers a, b and c from standard input (space separated) and prints a*b + c. Output only the number.",
    "compare": "Write a Python program. It reads two integers a and b (space separated). Print 'greater' if a > b, otherwise print 'smaller'. Output only the word.",
    "absolute_diff": "Write a Python program. It reads two integers a and b (space separated) and prints their absolute difference. Output only the number.",
}

JA_PROMPTS = {
    "sum_of_two": "Pythonのプログラムを書いてください。標準入力から2つの整数aとb（スペース区切り）を読み、その和を出力します。数字だけを出力してください。",
    "sum_of_list": "Pythonのプログラムを書いてください。最初の行に整数n、続いてn個の整数が与えられます。n個の整数の合計を出力してください。数字だけを出力してください。",
    "count_even": "Pythonのプログラムを書いてください。最初の行に整数n、続いてn個の整数が与えられます。そのうち偶数の個数を出力してください。数字だけを出力してください。",
    "max_of_nums": "Pythonのプログラムを書いてください。最初の行に整数n、続いてn個の整数が与えられます。その最大値を出力してください。数字だけを出力してください。",
    "arithmetic": "Pythonのプログラムを書いてください。標準入力から3つの整数a、b、c（スペース区切り）を読み、a*b + c を出力します。数字だけを出力してください。",
    "compare": "Pythonのプログラムを書いてください。標準入力から2つの整数aとb（スペース区切り）を読みます。a > b なら 'greater'、それ以外なら 'smaller' と出力してください。単語だけを出力してください。",
    "absolute_diff": "Pythonのプログラムを書いてください。標準入力から2つの整数aとb（スペース区切り）を読み、その差の絶対値を出力します。数字だけを出力してください。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rl_data/coding.jsonl")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    task_names = list(TASKS.keys())

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            task = rng.choice(task_names)
            lang = rng.choice(["en", "ja"])
            inp, exp, _ok = TASKS[task](rng)
            prompt = EN_PROMPTS[task] if lang == "en" else JA_PROMPTS[task]
            rec = {
                "lang": lang,
                "prompt": prompt,
                "mock_input": inp,
                "expected_output": exp,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[rl-data] Wrote {args.n} examples -> {args.out}")
    for t in task_names:
        print(f"  task: {t}  (en+ja)")


if __name__ == "__main__":
    main()