#!/usr/bin/env python3
"""build_math_dataset.py — Synthetic trilingual math SFT + RL data with CoT.

10 task families, all VERIFIED BY CONSTRUCTION (answers computed by the same
generator that renders the question — no external answer key to drift).

Response format (the anchor line is ALWAYS last — the RL verifier extracts it):
    EN: "<short steps>\nTherefore, the answer is X."
    ID: "<langkah singkat>\nJadi, jawabannya adalah X."
    JA: "<短い手順>\nしたがって、答えはXです。"

Outputs:
  sft_data/sft_math/train.jsonl     (~80K rows, langs: en 0.4 / id 0.4 / ja 0.2)
  sft_data/sft_math/eval.jsonl      (held-out, by instance-seed)
  rl_data/math.jsonl                (prompts + expected answer; mock_input "")

RL contract (load_rl_data compatible): {task, lang, prompt, mock_input,
expected_output, system}. Math rows carry a math-tutor system prompt via the
optional "system" field (supported by train_cuda_unified.cu).
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS = {
    "en": "You are a careful math tutor. Solve the problem step by step, then give the final answer on the last line.",
    "id": "Kamu adalah tutor matematika yang teliti. Kerjakan soal langkah demi langkah, lalu berikan jawaban akhir pada baris terakhir.",
    "ja": "あなたは丁寧な数学の先生です。順を追って解き、最終的な答えを最後の行に書いてください。",
}

ANCHOR = {
    "en": "Therefore, the answer is {a}.",
    "id": "Jadi, jawabannya adalah {a}.",
    "ja": "したがって、答えは{a}です。",
}


def fmt_num(x):
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        x = int(round(x))
    return f"{x:,}" if isinstance(x, int) and abs(x) >= 10000 else str(x)


def plain(x):
    """Canonical machine form used for expected_output (no thousand commas)."""
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x)


# ───────────────────────── task family generators ──────────────────────────
# Each returns dict: {params..., q: {en,id,ja}, steps: {en,id,ja}, ans}
# steps use {s0},{s1}... placeholders replaced with step strings per lang.

def t_arith_chain(rng):
    a, b, c, d = (rng.randint(3, 99) for _ in range(4))
    e = rng.randint(2, 9)
    ops = rng.sample([("+", lambda x, y: x + y), ("-", lambda x, y: x - y),
                      ("×", lambda x, y: x * y)], 2)
    # left-to-right chain with explicit parens: ((a op1 b) op2 c) op3 d
    o3 = rng.choice([("+", lambda x, y: x + y), ("-", lambda x, y: x - y)])
    x1 = ops[0][1](a, b); x2 = ops[1][1](x1, c); x3 = o3[1](x2, d)
    expr = f"((({a} {ops[0][0]} {b}) {ops[1][0]} {c}) {o3[0]} {d})"
    q = {
        "en": f"Calculate: {expr}",
        "id": f"Hitunglah: {expr}",
        "ja": f"計算してください：{expr}",
    }
    steps = {
        "en": f"({a} {ops[0][0]} {b}) = {fmt_num(x1)}; then {fmt_num(x1)} {ops[1][0]} {c} = {fmt_num(x2)}; then {fmt_num(x2)} {o3[0]} {d} = {fmt_num(x3)}.",
        "id": f"({a} {ops[0][0]} {b}) = {fmt_num(x1)}; lalu {fmt_num(x1)} {ops[1][0]} {c} = {fmt_num(x2)}; lalu {fmt_num(x2)} {o3[0]} {d} = {fmt_num(x3)}.",
        "ja": f"({a} {ops[0][0]} {b}) = {fmt_num(x1)}、次に {fmt_num(x1)} {ops[1][0]} {c} = {fmt_num(x2)}、次に {fmt_num(x2)} {o3[0]} {d} = {fmt_num(x3)}。",
    }
    return {"q": q, "steps": steps, "ans": x3, "task": "arith_chain"}


NAMES = {
    "en": ["A store", "A warehouse", "A school", "A bakery", "A factory"],
    "id": ["Sebuah toko", "Sebuah gudang", "Sebuah sekolah", "Sebuah toko roti", "Sebuah pabrik"],
    "ja": ["ある店", "ある倉庫", "ある学校", "あるパン屋", "ある工場"],
}
ITEMS = {
    "en": ["boxes", "packages", "cartons", "bundles"],
    "id": ["kotak", "paket", "kardus", "ikatan"],
    "ja": ["箱", "荷物", "カートン", "束"],
}
CONTENTS = {
    "en": ["pens", "notebooks", "bottles", "chocolates"],
    "id": ["pena", "buku tulis", "botol", "cokelat"],
    "ja": ["ペン", "ノート", "ボトル", "チョコレート"],
}


def t_word_multistep(rng):
    lang = "en"
    per, n, sold = rng.randint(6, 24), rng.randint(8, 30), rng.randint(10, 150)
    total = per * n
    if sold >= total:
        sold = total - rng.randint(1, 9)
    left = total - sold
    q = {
        "en": f"{rng.choice(NAMES['en'])} has {n} {rng.choice(ITEMS['en'])}, each containing {per} {rng.choice(CONTENTS['en'])}. After selling {sold} of them individually, how many are left?",
        "id": f"{rng.choice(NAMES['id'])} memiliki {n} {rng.choice(ITEMS['id'])}, setiap {rng.choice(ITEMS['id'])} berisi {per} {rng.choice(CONTENTS['id'])}. Setelah menjual {sold} buah secara satuan, berapa banyak yang tersisa?",
        "ja": f"{rng.choice(NAMES['ja'])}には{rng.choice(ITEMS['ja'])}が{n}個あり、それぞれに{rng.choice(CONTENTS['ja'])}が{per}個入っています。{sold}個単体で売れた後、残りはいくつですか？",
    }
    steps = {
        "en": f"Total = {n} × {per} = {fmt_num(total)}. Left = {fmt_num(total)} − {sold} = {fmt_num(left)}.",
        "id": f"Total = {n} × {per} = {fmt_num(total)}. Sisa = {fmt_num(total)} − {sold} = {fmt_num(left)}.",
        "ja": f"合計 = {n} × {per} = {fmt_num(total)}。残り = {fmt_num(total)} − {sold} = {fmt_num(left)}。",
    }
    return {"q": q, "steps": steps, "ans": left, "task": "word_multistep"}


def t_percent_price(rng):
    price = rng.choice([40, 60, 80, 120, 150, 200, 250, 300, 400, 500]) * 1000
    pct = rng.choice([10, 15, 20, 25, 30, 40, 50])
    disc = price * pct // 100
    final = price - disc
    q = {
        "en": f"An item costs {fmt_num(price)} and gets a {pct}% discount. What is the final price?",
        "id": f"Sebuah barang berharga {fmt_num(price)} didiskon {pct}%. Berapa harga akhirnya?",
        "ja": f"商品の価格は{fmt_num(price)}円で、{pct}%の割引があります。最終価格はいくらですか？",
    }
    steps = {
        "en": f"Discount = {pct}% × {fmt_num(price)} = {fmt_num(disc)}. Final = {fmt_num(price)} − {fmt_num(disc)} = {fmt_num(final)}.",
        "id": f"Diskon = {pct}% × {fmt_num(price)} = {fmt_num(disc)}. Akhir = {fmt_num(price)} − {fmt_num(disc)} = {fmt_num(final)}.",
        "ja": f"割引額 = {pct}% × {fmt_num(price)} = {fmt_num(disc)}。最終価格 = {fmt_num(price)} − {fmt_num(disc)} = {fmt_num(final)}。",
    }
    return {"q": q, "steps": steps, "ans": final, "task": "percent_price"}


def t_ratio_share(rng):
    total = rng.choice([60, 90, 120, 150, 180, 200, 240, 300])
    r1, r2, r3 = rng.sample([1, 2, 3, 4, 5], 3)
    unit = total // (r1 + r2 + r3)
    while unit * (r1 + r2 + r3) != total:
        total = rng.choice([60, 90, 120, 150, 180, 200, 240, 300])
        r1, r2, r3 = rng.sample([1, 2, 3, 4, 5], 3)
        unit = total // (r1 + r2 + r3)
    biggest = max(r1, r2, r3) * unit
    q = {
        "en": f"Divide {total} candies in the ratio {r1} : {r2} : {r3}. How many are in the largest share?",
        "id": f"Bagilah {total} permen dengan perbandingan {r1} : {r2} : {r3}. Berapa banyak bagian terbesar?",
        "ja": f"{total}個の飴を{r1}：{r2}：{r3}に分けます。最も多い部分はいくつですか？",
    }
    steps = {
        "en": f"Total parts = {r1}+{r2}+{r3} = {r1+r2+r3}; one part = {total} ÷ {r1+r2+r3} = {unit}. Largest = {max(r1,r2,r3)} × {unit} = {fmt_num(biggest)}.",
        "id": f"Total bagian = {r1}+{r2}+{r3} = {r1+r2+r3}; satu bagian = {total} ÷ {r1+r2+r3} = {unit}. Terbesar = {max(r1,r2,r3)} × {unit} = {fmt_num(biggest)}.",
        "ja": f"合計部分 = {r1}+{r2}+{r3} = {r1+r2+r3}、1部分 = {total} ÷ {r1+r2+r3} = {unit}。最大 = {max(r1,r2,r3)} × {unit} = {fmt_num(biggest)}。",
    }
    return {"q": q, "steps": steps, "ans": biggest, "task": "ratio_share"}


def t_second_max(rng):
    n = rng.randint(5, 8)
    vals = rng.sample(range(1, 200), n)
    s = sorted(vals)
    ans = s[-2]
    lst_en = ", ".join(map(str, vals))
    lst_ja = "、".join(map(str, vals))
    q = {
        "en": f"From the numbers {lst_en}, what is the second largest?",
        "id": f"Dari angka {lst_en}, berapa angka terbesar kedua?",
        "ja": f"数字{lst_ja}の中で2番目に大きい数は何ですか？",
    }
    steps = {
        "en": f"The largest is {s[-1]}; the next largest is {ans}.",
        "id": f"Yang terbesar adalah {s[-1]}; berikutnya adalah {ans}.",
        "ja": f"最大は{s[-1]}、次に大きいのは{ans}です。",
    }
    return {"q": q, "steps": steps, "ans": ans, "task": "second_max"}


def t_sort_asc(rng):
    vals = rng.sample(range(-99, 999), 5)
    s = sorted(vals)
    ans = " ".join(map(str, s))
    lst_en = ", ".join(map(str, vals))
    lst_ja = "、".join(map(str, vals))
    q = {
        "en": f"Sort these numbers in ascending order: {lst_en}. Give them separated by single spaces.",
        "id": f"Urutkan angka berikut dari terkecil ke terbesar: {lst_en}. Tulis dipisahkan satu spasi.",
        "ja": f"これらの数字を昇順に並べてください：{lst_ja}。半角スペースで区切ってください。",
    }
    steps = {
        "en": f"Arranged from smallest to largest: {ans}.",
        "id": f"Disusun dari terkecil ke terbesar: {ans}.",
        "ja": f"小さい順に並べると：{ans}。",
    }
    return {"q": q, "steps": steps, "ans": ans, "task": "sort_asc"}


UNIT_CONV = [
    # (unit_en, unit_id, unit_ja, factor)
    ("hours to minutes", "jam menjadi menit", "時間を分に", 60),
    ("minutes to seconds", "menit menjadi detik", "分を秒に", 60),
    ("kilometers to meters", "kilometer menjadi meter", "キロメートルをメートルに", 1000),
    ("kilograms to grams", "kilogram menjadi gram", "キログラムをグラムに", 1000),
]


def t_unit_convert(rng):
    desc_en, desc_id, desc_ja, factor = rng.choice(UNIT_CONV)
    v = rng.randint(2, 99)
    ans = v * factor
    q = {
        "en": f"Convert {v} {desc_en}.",
        "id": f"Konversikan {v} {desc_id}.",
        "ja": f"{v}{desc_ja}変換してください。",
    }
    steps = {
        "en": f"{v} × {fmt_num(factor)} = {fmt_num(ans)}.",
        "id": f"{v} × {fmt_num(factor)} = {fmt_num(ans)}.",
        "ja": f"{v} × {fmt_num(factor)} = {fmt_num(ans)}。",
    }
    return {"q": q, "steps": steps, "ans": ans, "task": "unit_convert"}


def t_remainder(rng):
    b = rng.randint(3, 19)
    a = rng.randint(20, 999)
    ans = a % b
    q = {
        "en": f"What is the remainder when {a} is divided by {b}?",
        "id": f"Berapa sisa pembagian {a} dibagi {b}?",
        "ja": f"{a}を{b}で割った余りはいくつですか？",
    }
    steps = {
        "en": f"{a} ÷ {b} = {a//b} remainder {ans}.",
        "id": f"{a} ÷ {b} = {a//b} sisa {ans}.",
        "ja": f"{a} ÷ {b} = {a//b} 余り {ans}。",
    }
    return {"q": q, "steps": steps, "ans": ans, "task": "remainder"}


def t_average(rng):
    n = rng.randint(4, 6)
    mean = rng.randint(5, 80)
    vals = [mean + rng.randint(-9, 9) for _ in range(n - 1)]
    vals.append(mean * n - sum(vals))  # force exact mean
    ans = sum(vals) // n
    lst_en = ", ".join(map(str, vals))
    lst_ja = "、".join(map(str, vals))
    q = {
        "en": f"What is the average of {lst_en}?",
        "id": f"Berapa rata-rata dari {lst_en}?",
        "ja": f"{lst_ja}の平均はいくつですか？",
    }
    steps = {
        "en": f"Sum = {fmt_num(sum(vals))}; count = {n}; average = {fmt_num(sum(vals))} ÷ {n} = {ans}.",
        "id": f"Jumlah = {fmt_num(sum(vals))}; banyak = {n}; rata-rata = {fmt_num(sum(vals))} ÷ {n} = {ans}.",
        "ja": f"合計 = {fmt_num(sum(vals))}、個数 = {n}、平均 = {fmt_num(sum(vals))} ÷ {n} = {ans}。",
    }
    return {"q": q, "steps": steps, "ans": ans, "task": "average"}


def t_missing_operand(rng):
    b = rng.randint(2, 12)
    x = rng.randint(2, 30)
    c = b * x
    q = {
        "en": f"{b} × ? = {c}. What number replaces the question mark?",
        "id": f"{b} × ? = {c}. Angka berapa yang menggantikan tanda tanya?",
        "ja": f"{b} × ? = {c}。？に入る数はいくつですか？",
    }
    steps = {
        "en": f"? = {c} ÷ {b} = {x}.",
        "id": f"? = {c} ÷ {b} = {x}.",
        "ja": f"？ = {c} ÷ {b} = {x}。",
    }
    return {"q": q, "steps": steps, "ans": x, "task": "missing_operand"}


TASKS = [t_arith_chain, t_word_multistep, t_percent_price, t_ratio_share,
         t_second_max, t_sort_asc, t_unit_convert, t_remainder, t_average,
         t_missing_operand]


def render(prob, lang):
    body = prob["steps"][lang]
    if lang == "ja":
        ans_disp = fmt_num(prob["ans"]) if not isinstance(prob["ans"], str) else prob["ans"]
        anchor = ANCHOR["ja"].format(a=ans_disp)
    else:
        ans_disp = fmt_num(prob["ans"]) if not isinstance(prob["ans"], str) else prob["ans"]
        anchor = ANCHOR[lang].format(a=ans_disp)
    return f"{body}\n{anchor}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=80_000)
    ap.add_argument("--n_eval", type=int, default=2_000)
    ap.add_argument("--n_rl", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_sft_dir", default="sft_data/sft_math")
    ap.add_argument("--out_rl", default="rl_data/math.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    langs = ["en"] * 4 + ["id"] * 4 + ["ja"] * 2

    # Families with bounded parameter space can run out of unique questions —
    # they are allowed to oversample (intentional repetition, like facts v11).
    UNBOUNDED = {"arith_chain", "word_multistep", "second_max", "sort_asc",
                 "remainder", "average"}
    seen_q = set()

    def gen_one(i):
        """Round-robin over task families for a balanced mix."""
        while True:
            prob = TASKS[i % len(TASKS)](rng)
            lang = rng.choice(langs)
            if prob["task"] in UNBOUNDED:
                key = (prob["task"], lang, prob["q"][lang])
                if key in seen_q:
                    continue
                seen_q.add(key)
            return prob, lang

    # ── SFT train
    rows = []
    for i in range(args.n_train):
        prob, lang = gen_one(i)
        ans_txt = render(prob, lang)
        rows.append({
            "text": f"<|system|>{SYS[lang]}<|user|>{prob['q'][lang]}<|assistant|>{ans_txt}<|end|>",
            "lang": lang, "task": f"math_{prob['task']}",
            "expected_output": plain(prob["ans"]),
        })
    rng.shuffle(rows)

    # ── eval (held-out instances — strict no-repeat, unbounded only)
    eval_rows = []
    seen_eval = set()
    i = 0
    while len(eval_rows) < args.n_eval:
        prob = TASKS[i % len(TASKS)](rng)
        i += 1
        if prob["task"] not in UNBOUNDED:
            continue
        lang = rng.choice(langs)
        key = (prob["task"], lang, prob["q"][lang])
        if key in seen_eval or key in seen_q:
            continue
        seen_eval.add(key)
        ans_txt = render(prob, lang)
        eval_rows.append({
            "text": f"<|system|>{SYS[lang]}<|user|>{prob['q'][lang]}<|assistant|>{ans_txt}<|end|>",
            "lang": lang, "task": f"math_{prob['task']}",
            "expected_output": plain(prob["ans"]),
        })
    rng.shuffle(eval_rows)

    # ── RL prompts (fresh instances; mock_input empty; per-row system)
    rl_rows = []
    seen_rl = set()
    i = 0
    while len(rl_rows) < args.n_rl:
        prob = TASKS[i % len(TASKS)](rng)
        i += 1
        lang = rng.choice(langs)
        key = (prob["task"], lang, prob["q"][lang])
        if key in seen_rl:
            continue
        seen_rl.add(key)
        rl_rows.append({
            "task": f"math_{prob['task']}",
            "lang": lang,
            "system": SYS[lang],
            "prompt": prob["q"][lang],
            "mock_input": "",
            "expected_output": plain(prob["ans"]),
        })
    rng.shuffle(rl_rows)

    out_dir = ROOT / args.out_sft_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rl_path = ROOT / args.out_rl
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rl_path, "w", encoding="utf-8") as f:
        for r in rl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SFT math: {len(rows)} train + {len(eval_rows)} eval -> {out_dir}")
    print(f"  by lang: {dict(Counter(r['lang'] for r in rows))}")
    print(f"  by task: {dict(Counter(r['task'] for r in rows))}")
    print(f"RL math: {len(rl_rows)} prompts -> {rl_path}")


if __name__ == "__main__":
    main()
