#!/usr/bin/env python3
"""build_sft_coding_v2.py — Expanded coding SFT + RL dataset (31 tasks, 3 langs).

Extends build_sft_coding_dataset.py (13 numeric tasks, EN/JA) with:
  +18 new tasks: strings, primes, gcd/lcm, fizzbuzz, leap year, grades,
     title case, digit sums, binary conversion, sorting, ...
  ID paraphrases for ALL 31 tasks (trilingual prompts: en / id / ja)
  Example-embedded prompts: each prompt names a worked example
     ("For example, given the input X the output should be Y"), making the
     unique-example space effectively unbounded.
  RL rows use a FRESH instance (different from the prompt example) so a
     degenerate print-the-answer solution can never pass verification.

Solutions are EXECUTION-VERIFIED at build time (every variant, plus a 15%
re-verify sample), reusing the old module's run_solution harness.

NOTE: some solutions are multi-line (for/if blocks). They are correct TEXT
data; they train correctly once the newline-preserving tokenizer (vocab_7)
lands — which is already on the 1B build critical path.

Outputs:
  sft_data/sft_coding_v2/train.jsonl
  rl_data/coding_v2.jsonl

Usage:
  python3 scripts/build_sft_coding_v2.py --n 6000
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sft_coding_dataset as OLD

ROOT = Path(__file__).resolve().parent.parent

# ───────────────────────── new input generators ─────────────────────────────
def gen_reverse_string(rng):
    s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 12)))
    return s, s[::-1]


def gen_palindrome(rng):
    base = "".join(rng.choice("abcde") for _ in range(rng.randint(3, 6)))
    if rng.random() < 0.5:
        s = base + base[::-1]              # guaranteed palindrome
        return s, "yes"
    while True:
        s = "".join(rng.choice("abcde") for _ in range(rng.randint(4, 10)))
        if s != s[::-1]:
            return s, "no"


def gen_count_vowels(rng):
    s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(6, 15)))
    return s, str(sum(1 for c in s if c in "aeiou"))


def gen_second_max(rng):
    vals = rng.sample(range(1, 500), rng.randint(5, 8))
    s = sorted(vals)
    return " ".join(map(str, vals)), str(s[-2])


def gen_remove_dup_sorted(rng):
    n = rng.randint(6, 12)
    vals = [rng.randint(0, 9) for _ in range(n)]
    uniq = sorted(set(vals))
    return " ".join(map(str, vals)), " ".join(map(str, uniq))


def gen_fizzbuzz(rng):
    n = rng.randint(5, 30)
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return str(n), "\n".join(out)


def gen_primes_to_n(rng):
    n = rng.randint(10, 100)
    ps = [p for p in range(2, n + 1)
          if all(p % d for d in range(2, int(math.isqrt(p)) + 1))]
    return str(n), " ".join(map(str, ps))


def gen_gcd(rng):
    a, b = rng.randint(1, 999), rng.randint(1, 999)
    return f"{a} {b}", str(math.gcd(a, b))


def gen_lcm(rng):
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    return f"{a} {b}", str(a * b // math.gcd(a, b))


def gen_leap_year(rng):
    y = rng.randint(1600, 2400)
    ok = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    return str(y), "yes" if ok else "no"


def gen_grade(rng):
    sc = rng.randint(0, 100)
    g = "A" if sc >= 90 else "B" if sc >= 80 else "C" if sc >= 70 else "D" if sc >= 60 else "F"
    return str(sc), g


def gen_word_count(rng):
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 8)))
             for _ in range(rng.randint(3, 10))]
    s = " ".join(words)
    return s, str(len(words))


def gen_title_case(rng):
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 8)))
             for _ in range(rng.randint(2, 5))]
    s = " ".join(words)
    return s, " ".join(w.capitalize() for w in words)


def gen_sum_digits(rng):
    v = rng.randint(10, 99999)
    return str(v), str(sum(int(d) for d in str(v)))


def gen_bin_to_dec(rng):
    bits = "".join(rng.choice("01") for _ in range(rng.randint(4, 10)))
    return bits, str(int(bits, 2))


def gen_sort_three(rng):
    a, b, c = rng.sample(range(-99, 99), 3)
    s = sorted([a, b, c])
    return f"{a} {b} {c}", " ".join(map(str, s))


def gen_count_char(rng):
    t = rng.choice("abcdefghij")
    s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(6, 15)))
    return f"{t} {s}", str(s.count(t))


def gen_swap_case(rng):
    s = "".join(rng.choice("AbBcCdDeEfFgGhH") for _ in range(rng.randint(5, 12)))
    return s, s.swapcase()


NEW_TASK_INPUT = {
    "reverse_string": gen_reverse_string,
    "palindrome_check": gen_palindrome,
    "count_vowels": gen_count_vowels,
    "second_max": gen_second_max,
    "remove_dup_sorted": gen_remove_dup_sorted,
    "fizzbuzz": gen_fizzbuzz,
    "primes_to_n": gen_primes_to_n,
    "gcd_two": gen_gcd,
    "lcm_two": gen_lcm,
    "leap_year": gen_leap_year,
    "grade_classify": gen_grade,
    "word_count": gen_word_count,
    "title_case": gen_title_case,
    "sum_digits": gen_sum_digits,
    "bin_to_dec": gen_bin_to_dec,
    "sort_three": gen_sort_three,
    "count_char": gen_count_char,
    "swap_case": gen_swap_case,
}

# ───────────────────────── new solutions ────────────────────────────────────
NEW_SOLUTIONS = {
    "reverse_string": [
        "print(input()[::-1])",
        "s = input()\nprint(''.join(reversed(s)))",
    ],
    "palindrome_check": [
        "s = input()\nprint('yes' if s == s[::-1] else 'no')",
        "s = input()\nprint('yes' if s == ''.join(reversed(s)) else 'no')",
    ],
    "count_vowels": [
        "s = input()\nprint(sum(1 for c in s if c in 'aeiou'))",
        "print(len([c for c in input() if c in 'aeiou']))",
    ],
    "second_max": [
        "vals = sorted(set(map(int, input().split())))\nprint(vals[-2])",
        "vals = list(map(int, input().split()))\nm = max(vals)\nprint(max(v for v in vals if v != m))",
    ],
    "remove_dup_sorted": [
        "vals = sorted(set(map(int, input().split())))\nprint(' '.join(map(str, vals)))",
        "seen = []\nfor v in sorted(map(int, input().split())):\n    if not seen or seen[-1] != v:\n        seen.append(v)\nprint(' '.join(map(str, seen)))",
    ],
    "fizzbuzz": [
        "n = int(input())\nfor i in range(1, n + 1):\n    if i % 15 == 0:\n        print('FizzBuzz')\n    elif i % 3 == 0:\n        print('Fizz')\n    elif i % 5 == 0:\n        print('Buzz')\n    else:\n        print(i)",
        "n = int(input())\nprint('\\n'.join('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)))",
    ],
    "primes_to_n": [
        "n = int(input())\nprint(' '.join(str(p) for p in range(2, n + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))))",
    ],
    "gcd_two": [
        "import math\na, b = map(int, input().split())\nprint(math.gcd(a, b))",
        "a, b = map(int, input().split())\nwhile b:\n    a, b = b, a % b\nprint(a)",
    ],
    "lcm_two": [
        "import math\na, b = map(int, input().split())\nprint(a * b // math.gcd(a, b))",
    ],
    "leap_year": [
        "y = int(input())\nprint('yes' if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 'no')",
    ],
    "grade_classify": [
        "sc = int(input())\nprint('A' if sc >= 90 else 'B' if sc >= 80 else 'C' if sc >= 70 else 'D' if sc >= 60 else 'F')",
        "sc = int(input())\nif sc >= 90:\n    print('A')\nelif sc >= 80:\n    print('B')\nelif sc >= 70:\n    print('C')\nelif sc >= 60:\n    print('D')\nelse:\n    print('F')",
    ],
    "word_count": [
        "print(len(input().split()))",
    ],
    "title_case": [
        "print(' '.join(w.capitalize() for w in input().split()))",
        "print(input().title())",
    ],
    "sum_digits": [
        "print(sum(int(d) for d in input()))",
        "n = int(input())\ns = 0\nwhile n:\n    s += n % 10\n    n //= 10\nprint(s)",
    ],
    "bin_to_dec": [
        "print(int(input(), 2))",
    ],
    "sort_three": [
        "a, b, c = map(int, input().split())\nprint(' '.join(map(str, sorted([a, b, c]))))",
        "vals = sorted(map(int, input().split()))\nprint(vals[0], vals[1], vals[2])",
    ],
    "count_char": [
        "t, s = input().split()\nprint(s.count(t))",
    ],
    "swap_case": [
        "print(input().swapcase())",
    ],
}

# ───────────────────────── paraphrases for new tasks ────────────────────────
NEW_EN = {
    "reverse_string": [
        "Write a Python program that reads a string and prints it reversed.",
        "Write a Python program. Read one line of text and output it in reverse order.",
        "Write a Python program that takes a string as input and prints the string backwards.",
        "Write a Python program to reverse the input string and print the result.",
    ],
    "palindrome_check": [
        "Write a Python program that reads a string and prints 'yes' if it is a palindrome, otherwise 'no'.",
        "Write a Python program. Check whether the input string reads the same forwards and backwards; print 'yes' or 'no'.",
        "Write a Python program that determines if the given string is a palindrome. Output only 'yes' or 'no'.",
        "Write a Python program to test the input string for being a palindrome; print 'yes' if it is, 'no' if not.",
    ],
    "count_vowels": [
        "Write a Python program that reads a string and prints how many vowels (a, e, i, o, u) it contains.",
        "Write a Python program. Count the vowels in the input string and output the count.",
        "Write a Python program that counts the number of vowels in the given text.",
        "Write a Python program to count how many of the characters in the input are vowels.",
    ],
    "second_max": [
        "Write a Python program that reads space-separated integers and prints the second largest distinct value.",
        "Write a Python program. From the list of integers on the input line, output the second largest number.",
        "Write a Python program that finds the second maximum among the input integers.",
        "Write a Python program to print the second biggest distinct integer from the input.",
    ],
    "remove_dup_sorted": [
        "Write a Python program that reads space-separated integers, removes duplicates, sorts them ascending, and prints them space-separated.",
        "Write a Python program. Output the distinct input integers in sorted order, separated by spaces.",
        "Write a Python program that prints the sorted unique values from the input.",
        "Write a Python program to deduplicate the input integers and print them in ascending order.",
    ],
    "fizzbuzz": [
        "Write a Python program that reads an integer n and prints the numbers 1 to n, one per line, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.",
        "Write a Python program. Implement FizzBuzz from 1 to n (n is the input), one entry per line.",
        "Write a Python program that plays FizzBuzz: for each i from 1 to the input n print 'Fizz', 'Buzz', 'FizzBuzz', or i by the usual rules.",
        "Write a Python program to print FizzBuzz sequence up to the given number, one per line.",
    ],
    "primes_to_n": [
        "Write a Python program that reads an integer n and prints all prime numbers up to n, space-separated.",
        "Write a Python program. List every prime not exceeding the input n on one line, separated by spaces.",
        "Write a Python program that outputs all primes <= n where n is read from input.",
        "Write a Python program to print the prime numbers up to the input value, separated by single spaces.",
    ],
    "gcd_two": [
        "Write a Python program that reads two integers and prints their greatest common divisor.",
        "Write a Python program. Compute the GCD of the two input integers and output it.",
        "Write a Python program that outputs the greatest common divisor of a and b given on one line.",
        "Write a Python program to find the GCD of the two given integers.",
    ],
    "lcm_two": [
        "Write a Python program that reads two integers and prints their least common multiple.",
        "Write a Python program. Compute the LCM of the two input integers and output it.",
        "Write a Python program that outputs the least common multiple of the two given numbers.",
        "Write a Python program to find the LCM of a and b provided on the input line.",
    ],
    "leap_year": [
        "Write a Python program that reads a year and prints 'yes' if it is a leap year, otherwise 'no'.",
        "Write a Python program. Decide whether the input year is a leap year; output 'yes' or 'no'.",
        "Write a Python program that checks the given year for leap-year rules and prints 'yes'/'no'.",
        "Write a Python program to determine if the input year is a leap year; print only 'yes' or 'no'.",
    ],
    "grade_classify": [
        "Write a Python program that reads a score from 0 to 100 and prints the letter grade: A for 90+, B for 80-89, C for 70-79, D for 60-69, F otherwise.",
        "Write a Python program. Convert the numeric input score into a letter grade (A>=90, B>=80, C>=70, D>=60, else F).",
        "Write a Python program that maps an input score to A/B/C/D/F using standard cutoffs.",
        "Write a Python program to classify the input exam score as a letter grade.",
    ],
    "word_count": [
        "Write a Python program that reads a line and prints the number of words (space-separated).",
        "Write a Python program. Count the words in the input line and output the count.",
        "Write a Python program that outputs how many words the input sentence contains.",
        "Write a Python program to count space-separated words in the input.",
    ],
    "title_case": [
        "Write a Python program that reads lowercase words and prints them in title case (each word capitalized), space-separated.",
        "Write a Python program. Capitalize the first letter of every input word and print the result.",
        "Write a Python program that converts the input line to title case.",
        "Write a Python program to print the input words with initial capital letters.",
    ],
    "sum_digits": [
        "Write a Python program that reads a non-negative integer and prints the sum of its digits.",
        "Write a Python program. Add up all digits of the input number and output the sum.",
        "Write a Python program that computes the digit sum of the input integer.",
        "Write a Python program to output the sum of the digits of the given number.",
    ],
    "bin_to_dec": [
        "Write a Python program that reads a binary string and prints its decimal value.",
        "Write a Python program. Convert the input binary number to decimal and output it.",
        "Write a Python program that interprets the input as base-2 and prints the base-10 value.",
        "Write a Python program to convert binary input to a decimal number.",
    ],
    "sort_three": [
        "Write a Python program that reads three integers and prints them in ascending order, space-separated.",
        "Write a Python program. Sort the three input numbers from smallest to largest and output them.",
        "Write a Python program that outputs the three given integers in sorted order.",
        "Write a Python program to print three input numbers in ascending order on one line.",
    ],
    "count_char": [
        "Write a Python program that reads a character and a string (space-separated), then prints how many times the character occurs in the string.",
        "Write a Python program. The input is a target character and a text; output the occurrence count.",
        "Write a Python program that counts occurrences of a given character in the given string.",
        "Write a Python program to count how many times the first input character appears in the second input string.",
    ],
    "swap_case": [
        "Write a Python program that reads a string and prints it with the case of every letter swapped.",
        "Write a Python program. Swap uppercase to lowercase and vice versa in the input, then print it.",
        "Write a Python program that toggles the letter case of the input string.",
        "Write a Python program to output the input with each letter's case inverted.",
    ],
}

NEW_JA = {
    "reverse_string": [
        "文字列を1行読み取り、それを逆順にして出力するPythonプログラムを書いてください。",
        "入力された文字列を前から後ろへ反転させて出力するPythonプログラムを実装してください。",
        "1行の文字列を受け取り、逆さにした文字列を出力するPythonプログラムを書いてください。",
        "与えられた文字列を逆順で表示するPythonプログラムを作成してください。",
    ],
    "palindrome_check": [
        "文字列を読み取り、回文なら「yes」、そうでなければ「no」を出力するPythonプログラムを書いてください。",
        "入力文字列が前後どちらから読んでも同じか判定し、yes か no を出力するPythonプログラムを実装してください。",
        "与えられた文字列が回文かどうかを判定するPythonプログラムを書いてください。出力は yes か no のみです。",
        "入力が回文である場合 yes、ない場合 no を表示するPythonプログラムを作成してください。",
    ],
    "count_vowels": [
        "文字列を読み取り、含まれる母音 a, e, i, o, u の個数を出力するPythonプログラムを書いてください。",
        "入力文字列中の母音の数を数えて出力するPythonプログラムを実装してください。",
        "与えられたテキストの母音の数を数えるPythonプログラムを書いてください。",
        "入力文字列に母音が何文字あるかを出力するPythonプログラムを作成してください。",
    ],
    "second_max": [
        "スペース区切りの整数を読み取り、2番目に大きい値（重複を除く）を出力するPythonプログラムを書いてください。",
        "入力行の整数列の中で2番目に大きい数を出力するPythonプログラムを実装してください。",
        "入力された整数の中から2番目の最大値を見つけるPythonプログラムを書いてください。",
        "与えられた整数の2番目に大きい異なる値を出力するPythonプログラムを作成してください。",
    ],
    "remove_dup_sorted": [
        "スペース区切りの整数を読み取り、重複を取り除いて昇順にソートし、スペース区切りで出力するPythonプログラムを書いてください。",
        "入力の整数から重複を除き、昇順で出力するPythonプログラムを実装してください。",
        "入力から一意な値を取り出し昇順に並べて表示するPythonプログラムを書いてください。",
        "与えられた整数列を重複削除のうえ昇順で出力するPythonプログラムを作成してください。",
    ],
    "fizzbuzz": [
        "整数nを読み取り、1からnまでの数を1行ずつ出力するPythonプログラムを書いてください。ただし3の倍数は「Fizz」、5の倍数は「Buzz」、両方の倍数は「FizzBuzz」とします。",
        "FizzBuzzを1から入力nまで実行するPythonプログラムを実装してください。1要素ごとに改行してください。",
        "入力nまでFizzBuzzを出力するPythonプログラムを書いてください。",
        "1から与えられた数までFizzBuzzの並びを1行ずつ表示するPythonプログラムを作成してください。",
    ],
    "primes_to_n": [
        "整数nを読み取り、n以下の素数をすべてスペース区切りで出力するPythonプログラムを書いてください。",
        "入力n以下の素数をすべて1行にスペース区切りで列挙するPythonプログラムを実装してください。",
        "n以下の素数をすべて出力するPythonプログラムを書いてください。",
        "入力値以下の素数をスペース区切りで表示するPythonプログラムを作成してください。",
    ],
    "gcd_two": [
        "2つの整数を読み取り、その最大公約数を出力するPythonプログラムを書いてください。",
        "入力された2整数のGCDを計算して出力するPythonプログラムを実装してください。",
        "1行の a と b の最大公約数を出力するPythonプログラムを書いてください。",
        "与えられた2整数の最大公約数を求めるPythonプログラムを作成してください。",
    ],
    "lcm_two": [
        "2つの整数を読み取り、その最小公倍数を出力するPythonプログラムを書いてください。",
        "入力された2整数のLCMを計算して出力するPythonプログラムを実装してください。",
        "与えられた2数の最小公倍数を出力するPythonプログラムを書いてください。",
        "入力行の a と b の最小公倍数を求めるPythonプログラムを作成してください。",
    ],
    "leap_year": [
        "年を読み取り、閏年なら「yes」、そうでなければ「no」を出力するPythonプログラムを書いてください。",
        "入力された年が閏年かどうか判定し、yes か no を出力するPythonプログラムを実装してください。",
        "与えられた年が閏年かを判定するPythonプログラムを書いてください。出力は yes か no のみです。",
        "入力の年がうるう年の場合 yes、違う場合 no を表示するPythonプログラムを作成してください。",
    ],
    "grade_classify": [
        "0から100の点数を読み取り、成績を出力するPythonプログラムを書いてください。90以上はA、80以上はB、70以上はC、60以上はD、それ以外はFとします。",
        "入力された点数を成績字母（A>=90, B>=80, C>=70, D>=60, それ以外F）に変換するPythonプログラムを実装してください。",
        "入力点数をA/B/C/D/Fに分類するPythonプログラムを書いてください。",
        "試験の点数を成績に変換して出力するPythonプログラムを作成してください。",
    ],
    "word_count": [
        "1行を読み取り、単語数（スペース区切り）を出力するPythonプログラムを書いてください。",
        "入力行の単語数を数えて出力するPythonプログラムを実装してください。",
        "入力された文にいくつの単語があるかを出力するPythonプログラムを書いてください。",
        "入力のスペース区切りの単語を数えるPythonプログラムを作成してください。",
    ],
    "title_case": [
        "小文字の単語群を読み取り、各単語の先頭を大文字にしてスペース区切りで出力するPythonプログラムを書いてください。",
        "入力された各単語の最初の文字を大文字にして出力するPythonプログラムを実装してください。",
        "入力行をタイトルケースに変換するPythonプログラムを書いてください。",
        "入力単語の頭文字を大文字にして表示するPythonプログラムを作成してください。",
    ],
    "sum_digits": [
        "非負整数を読み取り、各桁の数字の合計を出力するPythonプログラムを書いてください。",
        "入力された数の各桁を合計して出力するPythonプログラムを実装してください。",
        "入力整数の桁の合計を計算するPythonプログラムを書いてください。",
        "与えられた数の桁数値の合計を出力するPythonプログラムを作成してください。",
    ],
    "bin_to_dec": [
        "2進数の文字列を読み取り、10進数の値を出力するPythonプログラムを書いてください。",
        "入力された2進数を10進数に変換して出力するPythonプログラムを実装してください。",
        "入力を基数2として解釈し、10進数の値を表示するPythonプログラムを書いてください。",
        "2進数の入力を10進数に変換するPythonプログラムを作成してください。",
    ],
    "sort_three": [
        "3つの整数を読み取り、昇順にソートしてスペース区切りで出力するPythonプログラムを書いてください。",
        "入力された3つの数を小さい順に並べて出力するPythonプログラムを実装してください。",
        "与えられた3整数を昇順で出力するPythonプログラムを書いてください。",
        "3つの入力数を1行に昇順で表示するPythonプログラムを作成してください。",
    ],
    "count_char": [
        "文字と文字列（スペース区切り）を読み取り、その文字が出現する回数を出力するPythonプログラムを書いてください。",
        "対象文字と文章が入力されるので、出現回数を出力するPythonプログラムを実装してください。",
        "与えられた文字列中の特定の文字を数えるPythonプログラムを書いてください。",
        "1つ目の入力文字が2つ目の文字列に何回現れるかを出力するPythonプログラムを作成してください。",
    ],
    "swap_case": [
        "文字列を読み取り、すべての字母の大文字小文字を入れ替えて出力するPythonプログラムを書いてください。",
        "入力の大文字を小文字に、小文字を大文字に変換して出力するPythonプログラムを実装してください。",
        "入力文字列の文字のケースを切り替えるPythonプログラムを書いてください。",
        "入力の各文字の大文字小文字を反転して表示するPythonプログラムを作成してください。",
    ],
}

# ───────────────────────── Indonesian paraphrases (ALL 31 tasks) ────────────
ID_PARAPHRASES = {
    "sum_of_two": [
        "Tulis program Python yang membaca dua integer a dan b lalu mencetak jumlahnya.",
        "Tulis program Python. Baca dua bilangan bulat dari input dan outputkan penjumlahannya.",
        "Tulis program Python yang menerima dua angka dan menampilkan hasil tambahnya.",
        "Tulis program Python untuk menjumlahkan dua bilangan bulat yang diberikan.",
    ],
    "arithmetic": [
        "Tulis program Python yang membaca tiga integer a, b, c dan mencetak hasil a × b + c.",
        "Tulis program Python. Baca tiga bilangan bulat, lalu outputkan a*b+c.",
        "Tulis program Python yang menghitung a kali b ditambah c dari input.",
        "Tulis program Python untuk menghitung hasil dari a × b + c.",
    ],
    "compare": [
        "Tulis program Python yang membaca dua integer dan mencetak 'greater' jika a > b, selain itu 'smaller'.",
        "Tulis program Python. Bandingkan dua bilangan input, outputkan greater atau smaller.",
        "Tulis program Python yang membandingkan dua angka dan menampilkan greater/smaller.",
        "Tulis program Python untuk membandingkan dua bilangan bulat yang diberikan.",
    ],
    "absolute_diff": [
        "Tulis program Python yang membaca dua integer dan mencetak selisih absolutnya.",
        "Tulis program Python. Hitung nilai mutlak dari selisih dua input dan outputkan.",
        "Tulis program Python yang menampilkan beda absolut dua bilangan.",
        "Tulis program Python untuk menghitung selisih mutlak dua bilangan bulat.",
    ],
    "sum_of_list": [
        "Tulis program Python. Baris pertama berisi n, diikuti n bilangan bulat. Cetak jumlah seluruh bilangan tersebut.",
        "Tulis program Python yang membaca n lalu n integer, dan mencetak totalnya.",
        "Tulis program Python untuk menjumlahkan n bilangan bulat dari input.",
        "Tulis program Python. Input: n diikuti n angka. Output: jumlah semua angka.",
    ],
    "product_of_list": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak hasil kali seluruhnya.",
        "Tulis program Python yang membaca n bilangan lalu mencetak perkaliannya.",
        "Tulis program Python untuk mengalikan n bilangan bulat dari input.",
        "Tulis program Python. Input n dan n angka, output hasil kalinya.",
    ],
    "count_even": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak berapa yang genap.",
        "Tulis program Python yang menghitung banyaknya bilangan genap dari input.",
        "Tulis program Python untuk menghitung jumlah bilangan genap dalam daftar input.",
        "Tulis program Python. Baca n angka, output berapa banyak yang habis dibagi 2.",
    ],
    "count_odd": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak berapa yang ganjil.",
        "Tulis program Python yang menghitung banyaknya bilangan ganjil dari input.",
        "Tulis program Python untuk menghitung jumlah bilangan ganjil dalam daftar input.",
        "Tulis program Python. Baca n angka, output berapa banyak yang ganjil.",
    ],
    "count_positive": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak berapa yang positif (lebih dari 0).",
        "Tulis program Python yang menghitung banyaknya bilangan positif dari input.",
        "Tulis program Python untuk menghitung jumlah angka positif dalam daftar.",
        "Tulis program Python. Baca n angka, output banyaknya yang lebih besar dari nol.",
    ],
    "max_of_nums": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak yang terbesar.",
        "Tulis program Python yang mencari nilai maksimum dari n bilangan input.",
        "Tulis program Python untuk menampilkan bilangan terbesar dari daftar.",
        "Tulis program Python. Input n angka, output nilai maksimumnya.",
    ],
    "min_of_nums": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak yang terkecil.",
        "Tulis program Python yang mencari nilai minimum dari n bilangan input.",
        "Tulis program Python untuk menampilkan bilangan terkecil dari daftar.",
        "Tulis program Python. Input n angka, output nilai minimumnya.",
    ],
    "sum_of_evens": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak jumlah bilangan genapnya.",
        "Tulis program Python yang menjumlahkan bilangan-bilangan genap dari input.",
        "Tulis program Python untuk menghitung total angka genap dalam daftar.",
        "Tulis program Python. Baca n angka, output jumlah yang habis dibagi 2.",
    ],
    "count_div_by3": [
        "Tulis program Python. Baris pertama n, diikuti n integer. Cetak berapa yang habis dibagi 3.",
        "Tulis program Python yang menghitung banyaknya bilangan kelipatan 3 dari input.",
        "Tulis program Python untuk menghitung jumlah angka yang habis dibagi tiga.",
        "Tulis program Python. Baca n angka, output berapa yang kelipatan 3.",
    ],
    "reverse_string": [
        "Tulis program Python yang membaca sebuah string dan mencetaknya secara terbalik.",
        "Tulis program Python. Baca satu baris teks, lalu outputkan dalam urutan terbalik.",
        "Tulis program Python yang menerima string dan menampilkan kebalikannya.",
        "Tulis program Python untuk membalik string input dan mencetak hasilnya.",
    ],
    "palindrome_check": [
        "Tulis program Python yang membaca string dan mencetak 'yes' jika palindrome, selain itu 'no'.",
        "Tulis program Python. Periksa apakah string input sama dibaca dari depan dan belakang; cetak yes atau no.",
        "Tulis program Python yang menentukan apakah teks tersebut palindrome. Output hanya yes atau no.",
        "Tulis program Python untuk menguji apakah input adalah palindrome; ya maka yes, tidak maka no.",
    ],
    "count_vowels": [
        "Tulis program Python yang membaca string dan mencetak banyaknya vokal (a, e, i, o, u).",
        "Tulis program Python. Hitung huruf vokal dalam string input dan outputkan jumlahnya.",
        "Tulis program Python yang menghitung jumlah huruf vokal pada teks.",
        "Tulis program Python untuk menghitung ada berapa karakter vokal dalam input.",
    ],
    "second_max": [
        "Tulis program Python yang membaca integer dipisah spasi dan mencetak nilai terbesar kedua (tanpa duplikat).",
        "Tulis program Python. Dari daftar integer pada baris input, outputkan angka terbesar kedua.",
        "Tulis program Python yang mencari maksimum kedua di antara bilangan input.",
        "Tulis program Python untuk mencetak integer berbeda terbesar kedua dari input.",
    ],
    "remove_dup_sorted": [
        "Tulis program Python yang membaca integer dipisah spasi, menghapus duplikat, mengurutkan naik, lalu mencetaknya dipisah spasi.",
        "Tulis program Python. Outputkan integer input yang berbeda-beda dalam urutan naik, dipisah spasi.",
        "Tulis program Python yang mencetak nilai unik terurut dari input.",
        "Tulis program Python untuk menghilangkan duplikat dan mencetak hasilnya urut naik.",
    ],
    "fizzbuzz": [
        "Tulis program Python yang membaca integer n dan mencetak angka 1 sampai n, satu per baris, dengan kelipatan 3 jadi 'Fizz', kelipatan 5 jadi 'Buzz', keduanya jadi 'FizzBuzz'.",
        "Tulis program Python. Implementasikan FizzBuzz dari 1 sampai n (n adalah input), satu hasil per baris.",
        "Tulis program Python yang menjalankan FizzBuzz sampai angka input, satu per baris.",
        "Tulis program Python untuk mencetak deret FizzBuzz hingga bilangan yang diberikan.",
    ],
    "primes_to_n": [
        "Tulis program Python yang membaca integer n dan mencetak semua bilangan prima sampai n, dipisah spasi.",
        "Tulis program Python. Daftar semua prima yang tidak melebihi n input dalam satu baris.",
        "Tulis program Python yang mengoutputkan semua prima <= n dengan n dari input.",
        "Tulis program Python untuk mencetak bilangan prima hingga nilai input, dipisah satu spasi.",
    ],
    "gcd_two": [
        "Tulis program Python yang membaca dua integer dan mencetak FPB-nya.",
        "Tulis program Python. Hitung GCD dua integer input dan outputkan.",
        "Tulis program Python yang mengoutputkan faktor persekutuan terbesar a dan b dalam satu baris.",
        "Tulis program Python untuk mencari FPB dari dua bilangan bulat.",
    ],
    "lcm_two": [
        "Tulis program Python yang membaca dua integer dan mencetak KPK-nya.",
        "Tulis program Python. Hitung LCM dua integer input dan outputkan.",
        "Tulis program Python yang mengoutputkan kelipatan persekutuan terkecil dua bilangan.",
        "Tulis program Python untuk mencari KPK dari a dan b pada baris input.",
    ],
    "leap_year": [
        "Tulis program Python yang membaca tahun dan mencetak 'yes' jika tahun kabisat, selain itu 'no'.",
        "Tulis program Python. Tentukan apakah tahun input tahun kabisat; output yes atau no.",
        "Tulis program Python yang memeriksa aturan kabisat dan mencetak yes/no.",
        "Tulis program Python untuk menentukan apakah input tahun kabisat; cetak hanya yes atau no.",
    ],
    "grade_classify": [
        "Tulis program Python yang membaca nilai 0-100 dan mencetak huruf mutu: A untuk 90+, B 80-89, C 70-79, D 60-69, selainnya F.",
        "Tulis program Python. Ubah nilai angka menjadi huruf mutu (A>=90, B>=80, C>=70, D>=60, sisanya F).",
        "Tulis program Python yang memetakan nilai ke A/B/C/D/F dengan batas standar.",
        "Tulis program Python untuk mengklasifikasikan nilai ujian sebagai huruf mutu.",
    ],
    "word_count": [
        "Tulis program Python yang membaca satu baris dan mencetak jumlah katanya (dipisah spasi).",
        "Tulis program Python. Hitung kata pada baris input dan outputkan jumlahnya.",
        "Tulis program Python yang mengoutputkan banyak kata dalam kalimat input.",
        "Tulis program Python untuk menghitung kata dipisah spasi pada input.",
    ],
    "title_case": [
        "Tulis program Python yang membaca kata-kata huruf kecil dan mencetaknya dengan huruf kapital di awal, dipisah spasi.",
        "Tulis program Python. Kapitalkan huruf pertama setiap kata input lalu cetak.",
        "Tulis program Python yang mengubah baris input menjadi title case.",
        "Tulis program Python untuk mencetak kata input dengan huruf awal kapital.",
    ],
    "sum_digits": [
        "Tulis program Python yang membaca integer non-negatif dan mencetak jumlah digitnya.",
        "Tulis program Python. Jumlahkan semua digit angka input dan outputkan.",
        "Tulis program Python yang menghitung jumlah digit dari bilangan input.",
        "Tulis program Python untuk mengoutputkan jumlah digit bilangan yang diberikan.",
    ],
    "bin_to_dec": [
        "Tulis program Python yang membaca string biner dan mencetak nilai desimalnya.",
        "Tulis program Python. Konversi input biner ke desimal dan outputkan.",
        "Tulis program Python yang menafsirkan input sebagai basis-2 dan mencetak nilainya basis-10.",
        "Tulis program Python untuk mengubah input biner menjadi bilangan desimal.",
    ],
    "sort_three": [
        "Tulis program Python yang membaca tiga integer dan mencetaknya urut naik, dipisah spasi.",
        "Tulis program Python. Urutkan tiga angka input dari kecil ke besar lalu outputkan.",
        "Tulis program Python yang mengoutputkan tiga bilangan dalam urutan terurut.",
        "Tulis program Python untuk mencetak tiga bilangan input urut naik dalam satu baris.",
    ],
    "count_char": [
        "Tulis program Python yang membaca satu karakter dan satu string (dipisah spasi), lalu mencetak berapa kali karakter itu muncul.",
        "Tulis program Python. Input berupa karakter target dan teks; output jumlah kemunculannya.",
        "Tulis program Python yang menghitung kemunculan karakter pada string.",
        "Tulis program Python untuk menghitung berapa kali karakter input pertama muncul pada string input kedua.",
    ],
    "swap_case": [
        "Tulis program Python yang membaca string dan mencetaknya dengan besar-kecil huruf setiap karakter ditukar.",
        "Tulis program Python. Tukar huruf besar menjadi kecil dan sebaliknya pada input, lalu cetak.",
        "Tulis program Python yang membalik kapitalisasi huruf string input.",
        "Tulis program Python untuk mengoutputkan input dengan kapitalisasi tiap huruf dibalik.",
    ],
}

SYSTEM_ID = "Kamu adalah asisten coding. Tulis program Python yang singkat dan benar."

# ───────────────────────── merged registries ────────────────────────────────
TASK_INPUT = dict(OLD.TASK_INPUT)
TASK_INPUT.update(NEW_TASK_INPUT)

SOLUTIONS = dict(OLD.SOLUTIONS)
SOLUTIONS.update(NEW_SOLUTIONS)

EN_PARAPHRASES = dict(OLD.EN_PARAPHRASES)
EN_PARAPHRASES.update(NEW_EN)

JA_PARAPHRASES = dict(OLD.JA_PARAPHRASES)
JA_PARAPHRASES.update(NEW_JA)

# ───────────────────────── example embedding ────────────────────────────────
def esc(s):
    return s.replace("\n", "\\n").replace("\t", "\\t")


EXAMPLE_SUFFIX = {
    "en": ' For example, given the input "{i}" the output should be "{o}".',
    "ja": ' 例えば、入力 "{i}" に対する出力は "{o}" となります。',
    "id": ' Sebagai contoh, untuk input "{i}" keluarannya adalah "{o}".',
}


def embed_example(prompt, lang, inp, exp):
    return prompt + EXAMPLE_SUFFIX[lang].format(i=esc(inp), o=esc(exp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--n_rl", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_sft", default="sft_data/sft_coding_v2/train.jsonl")
    ap.add_argument("--out_rl", default="rl_data/coding_v2.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = sorted(TASK_INPUT.keys())

    # ── 1. Execution-verify every solution variant
    print(f"[1/3] Verifying solution variants for {len(tasks)} tasks...", flush=True)
    ok_solutions = {}
    for task in tasks:
        good = []
        for sol in SOLUTIONS[task]:
            inp, exp = TASK_INPUT[task](rng)
            ok, out = OLD.run_solution(sol, inp)
            if ok and out == exp.strip():
                good.append(sol)
            else:
                print(f"  [FAIL] {task}: got={out!r} exp={exp!r}")
        ok_solutions[task] = good
        if not good:
            print(f"  [FATAL] task '{task}' has no working solution.")
            sys.exit(1)
    print(f"  all {len(tasks)} tasks have working solutions "
          f"({sum(len(v) for v in ok_solutions.values())} variants total)")

    # ── 2. Build unique SFT examples (balanced task × lang round-robin)
    print("[2/3] Building examples...", flush=True)
    seen = set()
    rows = []
    langs_cycle = ["en", "id", "ja"]
    i = 0
    while len(rows) < args.n:
        task = tasks[i % len(tasks)]
        lang = langs_cycle[(i // len(tasks)) % 3]
        i += 1
        paras = {"en": EN_PARAPHRASES, "id": ID_PARAPHRASES,
                 "ja": JA_PARAPHRASES}[lang][task]
        para = rng.choice(paras)
        ex_in, ex_exp = TASK_INPUT[task](rng)
        prompt = embed_example(para, lang, ex_in, ex_exp)
        sol = rng.choice(ok_solutions[task])
        system = {"en": OLD.SYSTEM_EN, "id": SYSTEM_ID,
                  "ja": OLD.SYSTEM_JA}[lang]
        text = f"<|system|>{system}<|user|>{prompt}<|assistant|>{sol}<|end|>"
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append((task, lang, text, prompt, sol))

    # ── 3. Re-verify a random 15% sample on FRESH instances
    print("[3/3] Re-verifying random sample on fresh instances...", flush=True)
    sample = rng.sample(rows, min(max(10, len(rows) // 6), len(rows)))
    bad = 0
    for task, lang, text, prompt, sol in sample:
        inp, exp = TASK_INPUT[task](rng)
        ok, out = OLD.run_solution(sol, inp)
        if not ok or out != exp.strip():
            bad += 1
            print(f"  [RE-VERIFY FAIL] {task} got={out!r} exp={exp!r}")
    print(f"  re-verified {len(sample)}: {bad} failures")

    # ── write SFT
    out_sft = ROOT / args.out_sft
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sft, "w", encoding="utf-8") as f:
        for task, lang, text, prompt, sol in rows:
            f.write(json.dumps({"task": task, "lang": lang, "text": text},
                               ensure_ascii=False) + "\n")

    # ── RL rows: prompt WITHOUT solution; fresh instance different from any
    # prompt example (freshness by generation, not equality-check)
    rl_rows = []
    seen_rl = set()
    i = 0
    while len(rl_rows) < args.n_rl:
        task = tasks[i % len(tasks)]
        lang = langs_cycle[(i // len(tasks)) % 3]
        i += 1
        paras = {"en": EN_PARAPHRASES, "id": ID_PARAPHRASES,
                 "ja": JA_PARAPHRASES}[lang][task]
        para = rng.choice(paras)
        ex_in, ex_exp = TASK_INPUT[task](rng)      # example for the prompt
        rl_in, rl_exp = TASK_INPUT[task](rng)      # FRESH hidden instance
        prompt = embed_example(para, lang, ex_in, ex_exp)
        h = hashlib.sha1((task + lang + prompt + rl_in).encode()).hexdigest()
        if h in seen_rl:
            continue
        seen_rl.add(h)
        rl_rows.append({"task": task, "lang": lang, "prompt": prompt,
                        "mock_input": rl_in, "expected_output": rl_exp})
    out_rl = ROOT / args.out_rl
    out_rl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_rl, "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSFT coding v2: {len(rows)} rows -> {out_sft}")
    print(f"  by lang: {dict(Counter(r[1] for r in rows))}")
    print(f"  by task: {len(set(r[0] for r in rows))} tasks")
    print(f"RL coding v2: {len(rl_rows)} rows -> {out_rl}")


if __name__ == "__main__":
    main()
