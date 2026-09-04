#!/usr/bin/env python3
"""build_sft_coding_v3.py — Coding dataset v3: v2's 31 tasks + 15 new
archetypes (46 total), trilingual prompts, exec-verified solutions,
example-embedded prompts, RL rows with hidden fresh instances.

New archetypes: multiplication-table row, diagonal sum, word frequency,
CSV parsing, rotate list, transpose matrix, set intersection, second
smallest, cumulative sums, vowel replace, anagram check, power by loop,
char frequency dict, median, count words longer than k.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_sft_coding_dataset as OLD
import build_sft_coding_v2 as V2

ROOT = Path(__file__).resolve().parent.parent


# ───────────────────────── 15 new generators ────────────────────────────────
def g_mult_table_row(rng):
    row = rng.randint(2, 12)
    return str(row), " ".join(str(row * i) for i in range(1, 13))


def g_diag_sum(rng):
    n = rng.randint(3, 6)
    m = [[rng.randint(-9, 99) for _ in range(n)] for _ in range(n)]
    inp = str(n) + "\n" + "\n".join(" ".join(map(str, r)) for r in m)
    return inp, str(sum(m[i][i] for i in range(n)))


def g_word_freq(rng):
    words = [rng.choice(["kucing", "anjing", "burung", "ikan", "ular", "ayam"])
             for _ in range(rng.randint(8, 20))]
    w = rng.choice(words)
    inp = w + "\n" + " ".join(words)
    return inp, str(words.count(w))


def g_csv_parse(rng):
    name = rng.choice(["andi", "budi", "citra", "dewi", "eko"])
    a, b = rng.randint(1, 100), rng.randint(1, 100)
    inp = f"{name},{a},{b}"
    return inp, str(a + b)


def g_rotate(rng):
    k = rng.randint(1, 5)
    vals = [rng.randint(0, 99) for _ in range(rng.randint(5, 9))]
    n = len(vals)
    rot = vals[k % n:] + vals[:k % n]
    return f"{k} " + " ".join(map(str, vals)), " ".join(map(str, rot))


def g_transpose(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    m = [[rng.randint(0, 9) for _ in range(c)] for _ in range(r)]
    inp = f"{r} {c}\n" + "\n".join(" ".join(map(str, row)) for row in m)
    t = [[m[i][j] for i in range(r)] for j in range(c)]
    return inp, "\n".join(" ".join(map(str, row)) for row in t)


def g_set_inter(rng):
    a = sorted(rng.sample(range(0, 50), rng.randint(4, 9)))
    b = sorted(rng.sample(range(0, 50), rng.randint(4, 9)))
    inter = sorted(set(a) & set(b))
    return (" ".join(map(str, a)) + "\n" + " ".join(map(str, b)),
            " ".join(map(str, inter)) if inter else "empty")


def g_second_small(rng):
    vals = rng.sample(range(1, 300), rng.randint(5, 9))
    s = sorted(set(vals))
    return " ".join(map(str, vals)), str(s[1])


def g_cumsum(rng):
    vals = [rng.randint(1, 50) for _ in range(rng.randint(5, 9))]
    out, acc = [], 0
    for v in vals:
        acc += v
        out.append(acc)
    return " ".join(map(str, vals)), " ".join(map(str, out))


def g_vowel_replace(rng):
    s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(6, 14)))
    return s, s.replace("a", "*").replace("e", "*").replace("i", "*").replace("o", "*").replace("u", "*")


def g_anagram(rng):
    base = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(4, 7)))
    perm = list(base)
    rng.shuffle(perm)
    perm = "".join(perm)
    if rng.random() < 0.5:
        return f"{base} {perm}", "yes" if sorted(base) == sorted(perm) else "no"
    other = list(base)
    rng.shuffle(other)
    other = "".join(other)
    if other != perm and sorted(other) != sorted(base):
        return f"{base} {other}", "no"
    return f"{base} {perm}", "yes" if sorted(base) == sorted(perm) else "no"


def g_power(rng):
    b = rng.randint(2, 9)
    e = rng.randint(2, 9)
    return f"{b} {e}", str(b ** e)


def g_char_freq_dict(rng):
    s = "".join(rng.choice("abcde") for _ in range(rng.randint(8, 16)))
    counts = {c: s.count(c) for c in sorted(set(s))}
    out = ",".join(f"{c}:{n}" for c, n in counts.items())
    return s, out


def g_median(rng):
    n = rng.randint(4, 8)
    vals = [rng.randint(0, 99) for _ in range(n)]
    s = sorted(vals)
    med = (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
    med = int(med) if med == int(med) else med
    return " ".join(map(str, vals)), str(med)


def g_longer_than(rng):
    k = rng.randint(3, 6)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 9)))
             for _ in range(rng.randint(5, 10))]
    return f"{k} " + " ".join(words), str(sum(1 for w in words if len(w) > k))


NEW_INPUT = {
    "mult_table_row": g_mult_table_row,
    "diag_sum": g_diag_sum,
    "word_freq": g_word_freq,
    "csv_parse": g_csv_parse,
    "rotate_list": g_rotate,
    "transpose": g_transpose,
    "set_intersection": g_set_inter,
    "second_smallest": g_second_small,
    "cumulative_sum": g_cumsum,
    "vowel_replace": g_vowel_replace,
    "anagram_check": g_anagram,
    "power_loop": g_power,
    "char_freq_dict": g_char_freq_dict,
    "median_value": g_median,
    "longer_than_k": g_longer_than,
}

NEW_SOLUTIONS = {
    "mult_table_row": [
        "r = int(input())\nprint(' '.join(str(r * i) for i in range(1, r * 0 + 13)))".replace("r * 0 + 13", "13"),
        "r = int(input())\nout = []\nfor i in range(1, 13):\n    out.append(str(r * i))\nprint(' '.join(out))",
    ],
    "diag_sum": [
        "n = int(input())\nm = [list(map(int, input().split())) for _ in range(n)]\nprint(sum(m[i][i] for i in range(n)))",
    ],
    "word_freq": [
        "w = input()\nwords = input().split()\nprint(words.count(w))",
    ],
    "csv_parse": [
        "name, a, b = input().split(',')\nprint(int(a) + int(b))",
    ],
    "rotate_list": [
        "k, *vals = map(int, input().split())\nn = len(vals)\nk %= n\nprint(' '.join(map(str, vals[k:] + vals[:k])))",
    ],
    "transpose": [
        "r, c = map(int, input().split())\nm = [list(map(int, input().split())) for _ in range(r)]\nfor j in range(c):\n    print(' '.join(str(m[i][j]) for i in range(r)))",
    ],
    "set_intersection": [
        "a = set(map(int, input().split()))\nb = set(map(int, input().split()))\ninter = sorted(a & b)\nprint(' '.join(map(str, inter)) if inter else 'empty')",
    ],
    "second_smallest": [
        "vals = sorted(set(map(int, input().split())))\nprint(vals[1])",
    ],
    "cumulative_sum": [
        "vals = list(map(int, input().split()))\nout = []\ns = 0\nfor v in vals:\n    s += v\n    out.append(s)\nprint(' '.join(map(str, out)))",
    ],
    "vowel_replace": [
        "s = input()\nfor v in 'aeiou':\n    s = s.replace(v, '*')\nprint(s)",
    ],
    "anagram_check": [
        "a, b = input().split()\nprint('yes' if sorted(a) == sorted(b) else 'no')",
    ],
    "power_loop": [
        "b, e = map(int, input().split())\nresult = 1\nfor _ in range(e):\n    result *= b\nprint(result)",
    ],
    "char_freq_dict": [
        "s = input()\ncounts = {}\nfor c in s:\n    counts[c] = counts.get(c, 0) + 1\nprint(','.join(f'{c}:{n}' for c, n in sorted(counts.items())))",
    ],
    "median_value": [
        "import statistics\nvals = list(map(int, input().split()))\nm = statistics.median(vals)\nprint(int(m) if m == int(m) else m)",
    ],
    "longer_than_k": [
        "k, *words = input().split()\nk = int(k)\nprint(sum(1 for w in words if len(w) > k))",
    ],
}

P = {
    "en": {
        "mult_table_row": ["Write a Python program that reads an integer r and prints the first 12 multiples of r, space-separated.",
                           "Write a Python program. Output r×1 through r×12 for the input r, separated by spaces."],
        "diag_sum": ["Write a Python program that reads an n×n matrix (first line n, then n rows) and prints the sum of the main diagonal.",
                     "Write a Python program. Sum the diagonal elements of the input square matrix."],
        "word_freq": ["Write a Python program: first line a word, second line a sentence; print how often the word appears.",
                      "Write a Python program. Count occurrences of the first-line word in the second-line text."],
        "csv_parse": ["Write a Python program that reads 'name,a,b' CSV and prints a+b.",
                      "Write a Python program. Parse the comma-separated input and output the sum of the two numbers."],
        "rotate_list": ["Write a Python program: first number k, then a list; rotate the list left by k and print it.",
                        "Write a Python program. Left-rotate the input list by k (given first) and output the result."],
        "transpose": ["Write a Python program that reads a matrix (first line r c) and prints its transpose.",
                      "Write a Python program. Transpose the input r×c matrix and print it row by row."],
        "set_intersection": ["Write a Python program: two lines of integers; print the sorted intersection, or 'empty'.",
                             "Write a Python program. Output the sorted common elements of the two input lines, or 'empty'."],
        "second_smallest": ["Write a Python program that prints the second smallest distinct integer from the input.",
                            "Write a Python program. From the input integers output the second smallest unique value."],
        "cumulative_sum": ["Write a Python program that prints the running (cumulative) sums of the input integers.",
                           "Write a Python program. Output the prefix sums of the input numbers, space-separated."],
        "vowel_replace": ["Write a Python program that replaces every vowel in the input string with '*'.",
                          "Write a Python program. Substitute all vowels in the input with asterisks and print it."],
        "anagram_check": ["Write a Python program that reads two words and prints 'yes' if they are anagrams.",
                          "Write a Python program. Determine whether the two input words are anagrams; print yes/no."],
        "power_loop": ["Write a Python program that reads b and e and prints b to the power e using a loop (no **).",
                       "Write a Python program. Compute b^e with repeated multiplication and print it."],
        "char_freq_dict": ["Write a Python program that prints each distinct character of the input with its count as 'c:n', comma-separated in sorted order.",
                           "Write a Python program. Output sorted character frequencies of the input in c:n form."],
        "median_value": ["Write a Python program that prints the median of the input integers.",
                         "Write a Python program. Compute and print the median of the input numbers."],
        "longer_than_k": ["Write a Python program: first an integer k, then words; print how many words are longer than k.",
                          "Write a Python program. Count input words longer than the first-line k."],
    },
    "id": {
        "mult_table_row": ["Tulis program Python yang membaca integer r dan mencetak 12 kelipatan pertama r, dipisah spasi.",
                           "Tulis program Python. Output r×1 sampai r×12 untuk input r, dipisah spasi."],
        "diag_sum": ["Tulis program Python yang membaca matriks n×n (baris pertama n, lalu n baris) dan mencetak jumlah diagonal utamanya.",
                     "Tulis program Python. Jumlahkan elemen diagonal matriks persegi input."],
        "word_freq": ["Tulis program Python: baris pertama sebuah kata, baris kedua kalimat; cetak berapa kali kata itu muncul.",
                      "Tulis program Python. Hitung kemunculan kata baris pertama pada teks baris kedua."],
        "csv_parse": ["Tulis program Python yang membaca CSV 'nama,a,b' dan mencetak a+b.",
                      "Tulis program Python. Parsing input dipisah koma lalu output jumlah dua angkanya."],
        "rotate_list": ["Tulis program Python: angka pertama k, lalu daftar; rotasi daftar ke kiri sebanyak k lalu cetak.",
                        "Tulis program Python. Rotasi kiri daftar input sebanyak k (diberikan lebih dulu) lalu output."],
        "transpose": ["Tulis program Python yang membaca matriks (baris pertama r c) dan mencetak transposenya.",
                      "Tulis program Python. Transpos matriks r×c input lalu cetak per baris."],
        "set_intersection": ["Tulis program Python: dua baris integer; cetak irisan terurut, atau 'empty'.",
                             "Tulis program Python. Output elemen umum terurut dari dua baris input, atau 'empty'."],
        "second_smallest": ["Tulis program Python yang mencetak integer berbeda terkecil kedua dari input.",
                            "Tulis program Python. Dari integer input, output nilai unik terkecil kedua."],
        "cumulative_sum": ["Tulis program Python yang mencetak jumlah kumulatif (running sum) dari integer input.",
                           "Tulis program Python. Output jumlah prefix dari angka input, dipisah spasi."],
        "vowel_replace": ["Tulis program Python yang mengganti setiap vokal di string input dengan '*'.",
                          "Tulis program Python. Ganti semua vokal input dengan asterisk lalu cetak."],
        "anagram_check": ["Tulis program Python yang membaca dua kata dan mencetak 'yes' jika keduanya anagram.",
                          "Tulis program Python. Tentukan apakah dua kata input anagram; cetak yes/no."],
        "power_loop": ["Tulis program Python yang membaca b dan e lalu mencetak b pangkat e dengan loop (tanpa **).",
                       "Tulis program Python. Hitung b^e dengan perkalian berulang lalu cetak."],
        "char_freq_dict": ["Tulis program Python yang mencetak tiap karakter berbeda input dengan jumlahnya sebagai 'c:n', dipisah koma, terurut.",
                           "Tulis program Python. Output frekuensi karakter input terurut dalam bentuk c:n."],
        "median_value": ["Tulis program Python yang mencetak median dari integer input.",
                         "Tulis program Python. Hitung dan cetak median angka input."],
        "longer_than_k": ["Tulis program Python: pertama integer k, lalu kata-kata; cetak berapa kata yang lebih panjang dari k.",
                          "Tulis program Python. Hitung kata input yang lebih panjang dari k baris pertama."],
    },
    "ja": {
        "mult_table_row": ["整数rを読み、rの最初の12個の倍数をスペース区切りで出力するPythonプログラムを書いてください。",
                           "Pythonプログラムを書いてください。入力rに対しr×1からr×12までを出力します。"],
        "diag_sum": ["n×n行列（1行目n、続いてn行）を読み、主対角成分の和を出力するPythonプログラムを書いてください。",
                     "Pythonプログラムを書いてください。入力正方行列の対角成分の和を求めます。"],
        "word_freq": ["1行目に単語、2行目に文を読み、その単語の出現回数を出力するPythonプログラムを書いてください。",
                      "Pythonプログラムを書いてください。1行目の単語が2行目の文中に何回現れるか数えます。"],
        "csv_parse": ["'名前,a,b'形式のCSVを読み、a+bを出力するPythonプログラムを書いてください。",
                      "Pythonプログラムを書いてください。カンマ区切りの入力を解析し2数の和を出力します。"],
        "rotate_list": ["最初にk、続いてリストを読み、リストを左にk回転して出力するPythonプログラムを書いてください。",
                        "Pythonプログラムを書いてください。入力リストをkだけ左回転させ出力します。"],
        "transpose": ["行列（1行目にr c）を読み、転置行列を出力するPythonプログラムを書いてください。",
                      "Pythonプログラムを書いてください。入力r×c行列の転置を行ごとに出力します。"],
        "set_intersection": ["2行の整数を読み、昇順の共通要素を、なければ'empty'を出力するPythonプログラムを書いてください。",
                             "Pythonプログラムを書いてください。2行の入力の共通要素を昇順で出力します。"],
        "second_smallest": ["入力の中の2番目に小さい異なる整数を出力するPythonプログラムを書いてください。",
                            "Pythonプログラムを書いてください。入力整数の重複を除いた2番目に小さい値を出力します。"],
        "cumulative_sum": ["入力整数の累積和を出力するPythonプログラムを書いてください。",
                           "Pythonプログラムを書いてください。入力数列の累積和をスペース区切りで出力します。"],
        "vowel_replace": ["入力文字列のすべての母音を'*'に置換して出力するPythonプログラムを書いてください。",
                          "Pythonプログラムを書いてください。入力の母音をすべてアスタリスクに置き換えます。"],
        "anagram_check": ["2つの単語を読み、アナグラムなら'yes'を出力するPythonプログラムを書いてください。",
                          "Pythonプログラムを書いてください。2単語がアナグラムか判定します。"],
        "power_loop": ["bとeを読み、**を使わずループでbのe乗を出力するPythonプログラムを書いてください。",
                       "Pythonプログラムを書いてください。繰り返し乗算でb^eを計算します。"],
        "char_freq_dict": ["入力の各文字を出現回数とともに'c:n'形式・昇順・カンマ区切りで出力するPythonプログラムを書いてください。",
                           "Pythonプログラムを書いてください。入力の文字頻度を昇順でc:n形式に出力します。"],
        "median_value": ["入力整数の中央値を出力するPythonプログラムを書いてください。",
                         "Pythonプログラムを書いてください。入力数値の中央値を計算し出力します。"],
        "longer_than_k": ["最初に整数k、続いて単語群を読み、kより長い単語数を出力するPythonプログラムを書いてください。",
                          "Pythonプログラムを書いてください。1行目のkより長い単語を数えます。"],
    },
}


def esc(s):
    return s.replace("\n", "\\n").replace("\t", "\\t")


EXAMPLE_SUFFIX = {
    "en": ' For example, given the input "{i}" the output should be "{o}".',
    "ja": ' 例えば、入力 "{i}" に対する出力は "{o}" となります。',
    "id": ' Sebagai contoh, untuk input "{i}" keluarannya adalah "{o}".',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9_000)
    ap.add_argument("--n_rl", type=int, default=4_500)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out_sft", default="sft_data/sft_coding_v3/train.jsonl")
    ap.add_argument("--out_rl", default="rl_data/coding_v3.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    TASK_INPUT = dict(V2.TASK_INPUT)
    TASK_INPUT.update(NEW_INPUT)
    SOLUTIONS = dict(V2.SOLUTIONS)
    SOLUTIONS.update(NEW_SOLUTIONS)
    EN = dict(V2.EN_PARAPHRASES)
    JA = dict(V2.JA_PARAPHRASES)
    IDP = dict(V2.ID_PARAPHRASES)
    for t in NEW_INPUT:
        EN[t] = P["en"][t]
        JA[t] = P["ja"][t]
        IDP[t] = P["id"][t]

    tasks = sorted(TASK_INPUT.keys())
    print(f"[v3] {len(tasks)} tasks total")

    # verify all solutions
    ok_sol = {}
    for t in tasks:
        good = []
        for sol in SOLUTIONS[t]:
            inp, exp = TASK_INPUT[t](rng)
            ok, out = OLD.run_solution(sol, inp)
            if ok and out == exp.strip():
                good.append(sol)
            else:
                print(f"  [FAIL] {t}: got={out!r} exp={exp!r}")
        ok_sol[t] = good
        if not good:
            sys.exit(f"FATAL: {t}")
    print(f"[v3] all solutions verified ({sum(map(len, ok_sol.values()))} variants)")

    SYSTEM = {"en": OLD.SYSTEM_EN, "id": V2.SYSTEM_ID, "ja": OLD.SYSTEM_JA}

    def make_prompt(task, lang, rng):
        para = rng.choice({"en": EN, "id": IDP, "ja": JA}[lang][task])
        ex_in, ex_exp = TASK_INPUT[task](rng)
        return para + EXAMPLE_SUFFIX[lang].format(i=esc(ex_in), o=esc(ex_exp))

    rows, seen = [], set()
    i = 0
    langs_cycle = ["en", "id", "ja"]
    while len(rows) < args.n:
        task = tasks[i % len(tasks)]
        lang = langs_cycle[(i // len(tasks)) % 3]
        i += 1
        prompt = make_prompt(task, lang, rng)
        sol = rng.choice(ok_sol[task])
        text = f"<|system|>{SYSTEM[lang]}<|user|>{prompt}<|assistant|>{sol}<|end|>"
        h = hashlib.sha1(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        rows.append((task, lang, text, prompt, sol))

    # re-verify 10%
    sample = rng.sample(rows, min(len(rows) // 10, 900))
    bad = 0
    for task, lang, text, prompt, sol in sample:
        inp, exp = TASK_INPUT[task](rng)
        ok, out = OLD.run_solution(sol, inp)
        if not ok or out != exp.strip():
            bad += 1
    print(f"[v3] re-verify {len(sample)}: {bad} failures")

    rl_rows, seen_rl = [], set()
    i = 0
    while len(rl_rows) < args.n_rl:
        task = tasks[i % len(tasks)]
        lang = langs_cycle[(i // len(tasks)) % 3]
        i += 1
        prompt = make_prompt(task, lang, rng)
        rl_in, rl_exp = TASK_INPUT[task](rng)
        h = hashlib.sha1((task + lang + prompt + rl_in).encode()).hexdigest()
        if h in seen_rl:
            continue
        seen_rl.add(h)
        rl_rows.append({"task": task, "lang": lang, "prompt": prompt,
                        "mock_input": rl_in, "expected_output": rl_exp})

    out_sft = ROOT / args.out_sft
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sft, "w", encoding="utf-8") as f:
        for task, lang, text, prompt, sol in rows:
            f.write(json.dumps({"task": task, "lang": lang, "text": text},
                               ensure_ascii=False) + "\n")
    out_rl = ROOT / args.out_rl
    with open(out_rl, "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[v3] SFT {len(rows)} rows ({dict(Counter(r[1] for r in rows))}), "
          f"RL {len(rl_rows)} rows")
    print(f"  {len(tasks)} task archetypes")


if __name__ == "__main__":
    main()
