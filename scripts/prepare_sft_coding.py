#!/usr/bin/env python3
"""
prepare_sft_coding.py — Build a coding SFT dataset (EN + JA) in the chat
format used by Stage 2 (SFT).

This is the "preparation" step before RL: it teaches the 204M policy to write
Python programs (read stdin -> print the answer), so that Stage 5 RL has a
base it can amplify. The assistant response is a *reference solution*; the
verifier-style tasks (prompt + mock_input + expected_output) are emitted to a
separate file for RL evaluation.

Outputs:
  --out_sft   coding SFT JSONL (chat format, for train_sft_cuda --stage 2)
  --out_rl    RL JSONL (prompt + mock_input + expected_output, boolish rewards)

Usage:
    python3 scripts/prepare_sft_coding.py --n 4000
"""
import argparse
import json
import os
import random


def _read_ints(rng):
    n = rng.randint(2, 8)
    vals = [rng.randint(-100, 100) for _ in range(n)]
    return n, vals


def sum_of_two(rng):
    a, b = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
    inp = f"{a} {b}"
    exp = str(a + b)
    sol = f"a, b = map(int, input().split())\nprint(a + b)"
    return inp, exp, sol


def sum_of_list(rng):
    n, vals = _read_ints(rng)
    inp = str(n) + "\n" + "\n".join(map(str, vals))
    exp = str(sum(vals))
    sol = ("n = int(input())\n"
           "vals = [int(input()) for _ in range(n)]\n"
           "print(sum(vals))")
    return inp, exp, sol


def count_even(rng):
    n, vals = _read_ints(rng)
    inp = str(n) + "\n" + "\n".join(map(str, vals))
    exp = str(sum(1 for v in vals if v % 2 == 0))
    sol = ("n = int(input())\n"
           "vals = [int(input()) for _ in range(n)]\n"
           "print(sum(1 for v in vals if v % 2 == 0))")
    return inp, exp, sol


def max_of_nums(rng):
    n, vals = _read_ints(rng)
    inp = str(n) + "\n" + "\n".join(map(str, vals))
    exp = str(max(vals))
    sol = ("n = int(input())\n"
           "vals = [int(input()) for _ in range(n)]\n"
           "print(max(vals))")
    return inp, exp, sol


def arithmetic(rng):
    a, b, c = rng.randint(-100, 100), rng.randint(-100, 100), rng.randint(-100, 100)
    inp = f"{a} {b} {c}"
    exp = str(a * b + c)
    sol = "a, b, c = map(int, input().split())\nprint(a * b + c)"
    return inp, exp, sol


def compare(rng):
    a, b = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
    if a == b:
        b = a + 1
    inp = f"{a} {b}"
    exp = "greater" if a > b else "smaller"
    sol = ("a, b = map(int, input().split())\n"
           "print('greater' if a > b else 'smaller')")
    return inp, exp, sol


def absolute_diff(rng):
    a, b = rng.randint(-1000, 1000), rng.randint(-1000, 1000)
    inp = f"{a} {b}"
    exp = str(abs(a - b))
    sol = "a, b = map(int, input().split())\nprint(abs(a - b))"
    return inp, exp, sol


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
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_sft", default="sft_data/sft_coding/train.jsonl")
    ap.add_argument("--out_rl", default="rl_data/coding.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    task_names = list(TASKS.keys())

    os.makedirs(os.path.dirname(args.out_sft) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_rl) or ".", exist_ok=True)

    system_en = "You are a coding assistant. Write short, correct Python programs."
    system_ja = "あなたはコーディングアシスタントです。簡潔で正しいPythonプログラムを書いてください。"

    with open(args.out_sft, "w", encoding="utf-8") as fs, \
         open(args.out_rl, "w", encoding="utf-8") as fr:
        for _ in range(args.n):
            task = rng.choice(task_names)
            lang = rng.choice(["en", "ja"])
            inp, exp, sol = TASKS[task](rng)
            prompt = EN_PROMPTS[task] if lang == "en" else JA_PROMPTS[task]
            system = system_en if lang == "en" else system_ja
            text = (f"<|system|>{system}<|user|>{prompt}<|assistant|>{sol}<|end|>")
            fs.write(json.dumps({"text": text, "lang": lang, "task": task},
                                ensure_ascii=False) + "\n")
            fr.write(json.dumps({"lang": lang, "prompt": prompt,
                                 "mock_input": inp, "expected_output": exp},
                                ensure_ascii=False) + "\n")

    print(f"[sft-coding] Wrote {args.n} examples -> {args.out_sft}")
    print(f"[rl-coding]  Wrote {args.n} examples -> {args.out_rl}")
    for t in task_names:
        print(f"  task: {t}  (en+ja)")


if __name__ == "__main__":
    main()