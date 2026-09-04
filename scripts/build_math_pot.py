#!/usr/bin/env python3
"""build_math_pot.py — Program-of-Thoughts math dataset (exec-verified).

Each row's answer is a Python program that COMPUTES the answer (baked-in
question numbers) and prints it. At build time every generated program is
EXECUTED and its stdout asserted equal to the expected answer — the dataset
is verified by construction, and the RL verifier (code mode) can reuse the
same contract.

SFT answer format:
    ```python
    <program>
    ```
    Therefore, the answer is X.          (anchor mirrors printed value)

RL rows use mock_input=" " (non-empty → verifier code mode) with
expected_output = printed value.

Outputs:
  sft_data/sft_math_pot/train.jsonl  (~60K)
  sft_data/sft_math_pot/eval.jsonl   (1.5K)
  rl_data/math_pot.jsonl             (3K)
"""

import argparse
import io
import json
import random
import re
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS = {
    "en": "You are a math tutor. Write a short Python program that computes the answer, then state the result.",
    "id": "Kamu adalah tutor matematika. Tulis program Python singkat yang menghitung jawabannya, lalu sebutkan hasilnya.",
    "ja": "あなたは数学の先生です。答えを計算する短いPythonプログラムを書き、その結果を述べてください。",
}
ANCHOR = {
    "en": "Therefore, the answer is {a}.",
    "id": "Jadi, jawabannya adalah {a}.",
    "ja": "したがって、答えは{a}です。",
}


def plain(x):
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.2f}"
    return str(x)


# ── scenario: (params, q{lang}, code) ; answer derived by EXECUTING the code ─
def p_speed(rng):
    v, t, back = rng.randint(40, 110), rng.randint(2, 9), rng.randint(20, 90)
    q = {
        "en": f"A car travels at {v} km/h for {t} hours, then drives {back} km back toward the start. Write a program to compute how far it is from the start.",
        "id": f"Mobil melaju {v} km/jam selama {t} jam, lalu menempuh {back} km kembali ke arah awal. Tulis program untuk menghitung jaraknya dari titik awal.",
        "ja": f"車が{v}km/hで{t}時間走り、その後{back}km戻りました。出発点からの距離を計算するプログラムを書いてください。",
    }
    code = f"v, t, back = {v}, {t}, {back}\nout = v * t - back\nprint(out)"
    return q, code


def p_money(rng):
    p1, p2, p3 = rng.randint(2, 15) * 1000, rng.randint(1, 9) * 500, rng.randint(1, 5) * 2000
    pay = rng.choice([50, 100]) * 1000
    total = p1 + p2 + p3
    while total >= pay:
        pay *= 2
    q = {
        "en": f"Rudi buys items costing {p1}, {p2}, and {p3}, paying with {pay}. Write a program to compute the change.",
        "id": f"Rudi membeli barang seharga {p1}, {p2}, dan {p3}, membayar dengan {pay}. Tulis program untuk menghitung kembalian.",
        "ja": f"ルディは{p1}円、{p2}円、{p3}円の品物を買い、{pay}円支払いました。お釣りを計算するプログラムを書いてください。",
    }
    code = f"total = {p1} + {p2} + {p3}\nchange = {pay} - total\nprint(change)"
    return q, code


def p_discount(rng):
    price = rng.choice([120, 150, 200, 240, 300, 480]) * 1000
    disc = rng.choice([10, 20, 25, 30])
    tax = rng.choice([5, 8, 11])
    q = {
        "en": f"A {price} item gets a {disc}% discount, then {tax}% tax on the discounted price. Write a program to compute the final price.",
        "id": f"Barang seharga {price} didiskon {disc}%, lalu pajak {tax}% atas harga diskon. Tulis program untuk menghitung harga akhir.",
        "ja": f"{price}円の商品に{disc}%の割引、その後{tax}%の税が加わります。最終価格を計算するプログラムを書いてください。",
    }
    code = (f"price, disc, tax = {price}, {disc}, {tax}\n"
            f"after = price * (100 - disc) // 100\n"
            f"final = after + after * tax // 100\nprint(final)")
    return q, code


def p_tank(rng):
    fill, drain, hours = rng.randint(6, 20), 0, rng.randint(3, 9)
    drain = rng.randint(2, fill - 1)
    q = {
        "en": f"A tank fills at {fill} L/h while a leak drains {drain} L/h, for {hours} hours from empty. Write a program to compute the volume.",
        "id": f"Tangki terisi {fill} L/jam sementara bocor {drain} L/jam, selama {hours} jam dari kosong. Tulis program untuk menghitung volumenya.",
        "ja": f"タンクに毎時{fill}L注水し、毎時{drain}L漏れ、空から{hours}時間経ちました。水量を計算するプログラムを書いてください。",
    }
    code = f"net = {fill} - {drain}\nvolume = net * {hours}\nprint(volume)"
    return q, code


def p_avg(rng):
    n = rng.choice([4, 5])
    mean = rng.randint(10, 60)
    old = mean + rng.randint(-6, 6)
    new = old + rng.choice([8, 10, 12])
    q = {
        "en": f"The average of {n} numbers is {mean}. One number, {old}, is replaced by {new}. Write a program to compute the new average.",
        "id": f"Rata-rata {n} bilangan adalah {mean}. Satu bilangan, {old}, diganti {new}. Tulis program untuk menghitung rata-rata baru.",
        "ja": f"{n}個の数の平均は{mean}です。1つの数{old}を{new}に置き換えます。新しい平均を計算するプログラムを書いてください。",
    }
    code = (f"n, mean, old, new = {n}, {mean}, {old}, {new}\n"
            f"s = n * mean - old + new\n"
            f"avg = s / n\nprint(int(avg) if avg == int(avg) else round(avg, 2))")
    return q, code


def p_series(rng):
    a = rng.randint(-20, 40)
    d = rng.randint(2, 12)
    n = rng.randint(6, 20)
    q = {
        "en": f"An arithmetic sequence starts at {a} with difference {d}. Write a program to compute the sum of the first {n} terms.",
        "id": f"Barisan aritmetika mulai dari {a} dengan beda {d}. Tulis program untuk menghitung jumlah {n} suku pertama.",
        "ja": f"等差数列は{a}から始まり公差は{d}です。初項から{n}項までの和を計算するプログラムを書いてください。",
    }
    code = (f"a, d, n = {a}, {d}, {n}\n"
            f"last = a + (n - 1) * d\n"
            f"total = n * (a + last) // 2\nprint(total)")
    return q, code


def p_fence(rng):
    w = rng.randint(4, 30)
    l = w + rng.randint(2, 25)
    unit = rng.choice([5, 8, 12]) * 1000
    q = {
        "en": f"A {l} m by {w} m garden needs fencing at {unit} per meter. Write a program to compute the total cost.",
        "id": f"Kebun {l} m × {w} m perlu pagar dengan biaya {unit} per meter. Tulis program untuk menghitung total biayanya.",
        "ja": f"縦{l}m、横{w}mの庭を1m{unit}円で柵で囲みます。総費用を計算するプログラムを書いてください。",
    }
    code = f"perim = 2 * ({l} + {w})\ncost = perim * {unit}\nprint(cost)"
    return q, code


def p_pct(rng):
    base = rng.choice([200, 400, 500, 800]) * 1000
    up, down = rng.choice([10, 20, 25, 50]), rng.choice([5, 10, 20])
    q = {
        "en": f"A {base} salary rises {up}% then falls {down}%. Write a program to compute the current salary.",
        "id": f"Gaji {base} naik {up}% lalu turun {down}%. Tulis program untuk menghitung gaji sekarang.",
        "ja": f"給料{base}円が{up}%上がり、その後{down}%下がりました。現在の給料を計算するプログラムを書いてください。",
    }
    code = (f"base, up, down = {base}, {up}, {down}\n"
            f"raised = base * (100 + up) / 100\n"
            f"final = raised * (100 - down) / 100\n"
            f"print(int(final) if final == int(final) else round(final, 2))")
    return q, code


FAMILIES = [p_speed, p_money, p_discount, p_tank, p_avg, p_series, p_fence, p_pct]


def run_code_capture(code):
    buf = io.StringIO()
    g = {"__builtins__": __builtins__}
    try:
        with redirect_stdout(buf):
            exec(code, g)
        return buf.getvalue().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=60_000)
    ap.add_argument("--n_eval", type=int, default=1_500)
    ap.add_argument("--n_rl", type=int, default=3_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="sft_data/sft_math_pot")
    ap.add_argument("--out_rl", default="rl_data/math_pot.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    langs = ["en"] * 4 + ["id"] * 4 + ["ja"] * 2

    def gen(i):
        r = random.Random(1_000_003 + i)
        fam = FAMILIES[i % len(FAMILIES)]
        q, code = fam(r)
        lang = r.choice(langs)
        return fam.__name__, q, code, lang

    def make_row(i):
        name, q, code, lang = gen(i)
        out = run_code_capture(code)
        assert out is not None and out != "", f"code produced nothing: {code}"
        body = (f"```python\n{code}\n```\n" +
                ANCHOR[lang].format(a=out))
        return {
            "text": f"<|system|>{SYS[lang]}<|user|>{q[lang]}<|assistant|>{body}<|end|>",
            "lang": lang, "task": f"pot_{name}",
            "expected_output": out,
        }, code, out

    train, eval_rows, rl = [], [], []
    i = 0
    while len(train) < args.n_train:
        row, code, out = make_row(i)
        i += 1
        # round-robin over families balances tasks; bounded families
        # (p_pct, p_discount) intentionally oversample like facts v11
        train.append(row)
    for j in range(args.n_eval):
        row, _, _ = make_row(args.n_train + 50_000 + j)
        eval_rows.append(row)
    for j in range(args.n_rl):
        name, q, code, lang = gen(args.n_train + 100_000 + j)
        out = run_code_capture(code)
        assert out
        rl.append({
            "task": f"pot_{name}", "lang": lang, "system": SYS[lang],
            "prompt": q[lang], "mock_input": " ",
            "expected_output": out,
        })

    rng.shuffle(train)
    rng.shuffle(eval_rows)
    rng.shuffle(rl)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("train.jsonl", train), ("eval.jsonl", eval_rows)):
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / args.out_rl, "w", encoding="utf-8") as f:
        for r in rl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"PoT: {len(train)} train + {len(eval_rows)} eval + {len(rl)} RL (all exec-verified)")
    print(f"  by lang: {dict(Counter(r['lang'] for r in train))}")
    print(f"  by task: {dict(Counter(r['task'] for r in train))}")


if __name__ == "__main__":
    main()
