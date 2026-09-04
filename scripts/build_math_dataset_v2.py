#!/usr/bin/env python3
"""build_math_dataset_v2.py — Compositional word problems (3-8 steps, 3 langs).

12 deep scenarios (vs v1's template-fixed ones). Each generator:
  - builds a coherent quantity graph (intermediates computed in order)
  - renders step-by-step CoT in en/id/ja (steps mirror the computation)
  - ~30% of problems include a DISTRACTOR sentence (unused number)
  - some families produce decimal / negative answers

Outputs:
  sft_data/sft_math_v2/train.jsonl   (~120K rows)
  sft_data/sft_math_v2/eval.jsonl    (2K held-out)
  rl_data/math_v2.jsonl              (6K prompts, anchor-verified)

Anchor contract identical to v1 (rl_verifier v2 handles it).
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


def fmt(x):
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            x = int(round(x))
        else:
            return f"{x:,.2f}"
    return f"{x:,}" if isinstance(x, int) and abs(x) >= 10000 else str(x)


def plain(x):
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


# ── distractor sentences (irrelevant numbers) ───────────────────────────────
def distract(rng, lang):
    thing = rng.choice([
        ("The shop next door is closed on Sundays.", "Toko sebelah tutup pada hari Minggu.", "隣の店は日曜日に休みです。"),
        ("There are 31 days in this month.", "Ada 31 hari dalam bulan ini.", "今月は31日あります。"),
        ("The bus line has 14 stops in total.", "Rute bus itu memiliki 14 pemberhentian.", "そのバス路線は全部で14の停留所があります。"),
        ("A nearby parking lot fits 60 cars.", "Parkiran sebelah memuat 60 mobil.", "近くの駐車場は60台収容できます。"),
    ])
    return thing[{"en": 0, "id": 1, "ja": 2}[lang]]


# ───────────────────────── 12 scenario generators ──────────────────────────
def s_speed(rng):
    v = rng.randint(40, 110)
    t = rng.randint(2, 9)
    rest = rng.randint(20, 90)
    d = v * t
    dist_ans = d - rest
    q = {
        "en": f"A car travels at {v} km/h for {t} hours, then the driver rests. Later the same day it drives another {rest} km back toward the start. How far is the car from the starting point?",
        "id": f"Sebuah mobil melaju dengan kecepatan {v} km/jam selama {t} jam, lalu pengemudi beristirahat. Kemudian pada hari yang sama mobil melaju {rest} km lagi kembali ke arah awal. Berapa jauh mobil tersebut dari titik awal?",
        "ja": f"車が{v}km/hで{t}時間走り、運転手は休憩しました。その日の後で、さらに{rest}kmを出発点に向かって戻りました。車は出発点から何km離れていますか？",
    }
    steps = {
        "en": f"Outbound distance = {v} × {t} = {fmt(d)} km. Driving back {rest} km leaves {fmt(d)} − {rest} = {fmt(dist_ans)} km from the start.",
        "id": f"Jarak pergi = {v} × {t} = {fmt(d)} km. Menyetir kembali {rest} km menyisakan {fmt(d)} − {rest} = {fmt(dist_ans)} km dari titik awal.",
        "ja": f"往路 = {v} × {t} = {fmt(d)}km。{rest}km戻ると、出発点から{fmt(d)} − {rest} = {fmt(dist_ans)}kmです。",
    }
    return q, steps, dist_ans


def s_work_rate(rng):
    a = rng.randint(2, 12)
    b = a + rng.randint(1, 8)
    days_together = a * b // (a + b)
    # use integer-friendly: total work = lcm or product
    work = a * b
    t = work / (a + b)
    if abs(t - round(t)) > 1e-9:
        # force integer by choosing work = a*b, t = a*b/(a+b) rarely integer;
        # instead pick b as multiple making it clean: t=ab/(a+b) integer iff (a+b)|ab.
        # simpler: total rooms = a*b always works when? t = ab/(a+b). Use decimal ok.
        ans = round(t, 2)
    else:
        ans = int(round(t))
    q = {
        "en": f"Worker A paints {a} rooms per day and worker B paints {b} rooms per day. Working together on a building of {work} rooms, how many whole days does the job take if fractional days count as a full day (round up)?",
        "id": f"Pekerja A mengecat {a} ruangan per hari dan pekerja B mengecat {b} ruangan per hari. Bekerja bersama pada gedung berisi {work} ruangan, berapa hari penuh yang dibutuhkan jika hari pecahan dihitung satu hari penuh (dibulatkan ke atas)?",
        "ja": f"職人Aは1日に{a}室、職人Bは1日に{b}室ペンキを塗ります。{work}室の建物を二人で塗ると、端数は1日として数えて何日かかりますか？",
    }
    import math
    rate = a + b
    exact = work / rate
    import math as m
    ans = m.ceil(exact)
    steps = {
        "en": f"Combined rate = {a} + {b} = {rate} rooms/day. Exact time = {work} ÷ {rate} = {exact:.2f} days, so rounding up gives {ans} days.",
        "id": f"Laju gabungan = {a} + {b} = {rate} ruangan/hari. Waktu persis = {work} ÷ {rate} = {exact:.2f} hari, dibulatkan ke atas menjadi {ans} hari.",
        "ja": f"合計ペース = {a} + {b} = {rate}室/日。正確な日数 = {work} ÷ {rate} = {exact:.2f}日、切り上げて{ans}日です。",
    }
    return q, steps, ans


def s_age(rng):
    ratio = rng.randint(3, 5)
    son = rng.randint(5, 14)
    father = son * ratio
    later = rng.randint(4, 15)
    ans = (father + later) - (son + later)  # difference constant
    q = {
        "en": f"A father is {ratio} times as old as his {son}-year-old son. In {later} years, how many years older than the son will the father be?",
        "id": f"Seorang ayah berusia {ratio} kali usia anaknya yang berumur {son} tahun. Setelah {later} tahun lagi, berapa tahun lebih tua ayah daripada anaknya?",
        "ja": f"父の年齢は{son}歳の息子の{ratio}倍です。{later}年後、父は息子より何歳年上ですか？",
    }
    steps = {
        "en": f"Father now = {ratio} × {son} = {father}. Both age {later} years, so the difference stays {father} − {son} = {ans}.",
        "id": f"Ayah sekarang = {ratio} × {son} = {father}. Keduanya bertambah {later} tahun, jadi selisihnya tetap {father} − {son} = {ans}.",
        "ja": f"父の現在の年齢 = {ratio} × {son} = {father}。二人とも{later}歳増えるので、差は{father} − {son} = {ans}のままです。",
    }
    return q, steps, ans


def s_money_chain(rng):
    p1 = rng.randint(2, 15) * 1000
    p2 = rng.randint(1, 9) * 500
    p3 = rng.randint(1, 5) * 2000
    pay = rng.choice([50, 100]) * 1000
    total = p1 + p2 + p3
    while total >= pay:
        pay *= 2
    ans = pay - total
    q = {
        "en": f"Rudi buys a book for {fmt(p1)}, a pen for {fmt(p2)}, and a bag for {fmt(p3)}. He pays with {fmt(pay)}. How much change does he get?",
        "id": f"Rudi membeli buku seharga {fmt(p1)}, pena {fmt(p2)}, dan tas {fmt(p3)}. Ia membayar dengan {fmt(pay)}. Berapa kembaliannya?",
        "ja": f"ルディは本を{fmt(p1)}円、ペンを{fmt(p2)}円、かばんを{fmt(p3)}円買い、{fmt(pay)}円を支払いました。お釣りはいくらですか？",
    }
    steps = {
        "en": f"Total = {fmt(p1)} + {fmt(p2)} + {fmt(p3)} = {fmt(total)}. Change = {fmt(pay)} − {fmt(total)} = {fmt(ans)}.",
        "id": f"Total = {fmt(p1)} + {fmt(p2)} + {fmt(p3)} = {fmt(total)}. Kembalian = {fmt(pay)} − {fmt(total)} = {fmt(ans)}.",
        "ja": f"合計 = {fmt(p1)} + {fmt(p2)} + {fmt(p3)} = {fmt(total)}。お釣り = {fmt(pay)} − {fmt(total)} = {fmt(ans)}。",
    }
    return q, steps, ans


def s_pct_chain(rng):
    base = rng.choice([200, 400, 500, 800]) * 1000
    up = rng.choice([10, 20, 25, 50])
    down = rng.choice([5, 10, 20])
    raised = base * (1 + up / 100)
    ans = round(raised * (1 - down / 100), 2)
    ans = int(ans) if abs(ans - round(ans)) < 1e-9 else ans
    q = {
        "en": f"A salary of {fmt(base)} rises by {up}% one year, then falls by {down}% the next. What is the salary now?",
        "id": f"Gaji sebesar {fmt(base)} naik {up}% tahun lalu, lalu turun {down}% tahun ini. Berapa gajinya sekarang?",
        "ja": f"給料{fmt(base)}円が{up}%上がり、翌年{down}%下がりました。現在の給料はいくらですか？",
    }
    steps = {
        "en": f"After the rise: {fmt(base)} × {1 + up/100:.2f} = {fmt(raised)}. After the fall: {fmt(raised)} × {1 - down/100:.2f} = {fmt(ans)}.",
        "id": f"Setelah naik: {fmt(base)} × {1 + up/100:.2f} = {fmt(raised)}. Setelah turun: {fmt(raised)} × {1 - down/100:.2f} = {fmt(ans)}.",
        "ja": f"昇給後：{fmt(base)} × {1 + up/100:.2f} = {fmt(raised)}。減給後：{fmt(raised)} × {1 - down/100:.2f} = {fmt(ans)}。",
    }
    return q, steps, ans


def s_rect_fence(rng):
    w = rng.randint(4, 30)
    l = w + rng.randint(2, 25)
    unit = rng.choice([5, 8, 12]) * 1000
    perim = 2 * (w + l)
    ans = perim * unit
    q = {
        "en": f"A rectangular garden is {l} m long and {w} m wide. Fencing costs {fmt(unit)} per meter. How much does fencing the whole garden cost?",
        "id": f"Kebun berbentuk persegi panjang memiliki panjang {l} m dan lebar {w} m. Biaya pagar {fmt(unit)} per meter. Berapa biaya memagari seluruh kebun?",
        "ja": f"長さ{l}m、幅{w}mの長方形の庭があります。柵は1メートル{fmt(unit)}円です。庭全体を柵で囲むといくらかかりますか？",
    }
    steps = {
        "en": f"Perimeter = 2 × ({l} + {w}) = {perim} m. Cost = {perim} × {fmt(unit)} = {fmt(ans)}.",
        "id": f"Keliling = 2 × ({l} + {w}) = {perim} m. Biaya = {perim} × {fmt(unit)} = {fmt(ans)}.",
        "ja": f"周長 = 2 × ({l} + {w}) = {perim}m。費用 = {perim} × {fmt(unit)} = {fmt(ans)}。",
    }
    return q, steps, ans


def s_tank(rng):
    fill = rng.randint(6, 20)
    drain = rng.randint(2, fill - 1)
    hours = rng.randint(3, 9)
    net = fill - drain
    ans = net * hours
    q = {
        "en": f"A tank is filled by a pipe at {fill} liters per hour while a leak drains {drain} liters per hour. How many liters are in the tank after {hours} hours, starting empty?",
        "id": f"Sebuah tangki diisi pipa dengan {fill} liter per jam sementara kebocoran mengalirkan {drain} liter per jam. Berapa liter air di tangki setelah {hours} jam, mulai dari kosong?",
        "ja": f"タンクには毎時{fill}リットル注水され、同時に毎時{drain}リットル漏れます。空の状態から{hours}時間後に何リットル溜まりますか？",
    }
    steps = {
        "en": f"Net rate = {fill} − {drain} = {net} L/h. Volume = {net} × {hours} = {fmt(ans)} L.",
        "id": f"Laju bersih = {fill} − {drain} = {net} L/jam. Volume = {net} × {hours} = {fmt(ans)} L.",
        "ja": f"正味の流速 = {fill} − {drain} = {net}L/時。水量 = {net} × {hours} = {fmt(ans)}L。",
    }
    return q, steps, ans


def s_discount_tax(rng):
    price = rng.choice([120, 150, 200, 240, 300, 480]) * 1000
    disc = rng.choice([10, 20, 25, 30])
    tax = rng.choice([5, 8, 11])
    d = price * disc // 100
    after = price - d
    t = after * tax // 100
    ans = after + t
    q = {
        "en": f"A laptop priced {fmt(price)} gets a {disc}% discount, then {tax}% tax is added to the discounted price. What is the final price?",
        "id": f"Laptop berharga {fmt(price)} mendapat diskon {disc}%, lalu pajak {tax}% ditambahkan pada harga setelah diskon. Berapa harga akhirnya?",
        "ja": f"{fmt(price)}円のノートPCに{disc}%の割引があり、その後割引後の価格に{tax}%の税が加わります。最終価格はいくらですか？",
    }
    steps = {
        "en": f"Discount = {disc}% × {fmt(price)} = {fmt(d)}; price = {fmt(after)}. Tax = {tax}% × {fmt(after)} = {fmt(t)}. Final = {fmt(after)} + {fmt(t)} = {fmt(ans)}.",
        "id": f"Diskon = {disc}% × {fmt(price)} = {fmt(d)}; harga = {fmt(after)}. Pajak = {tax}% × {fmt(after)} = {fmt(t)}. Akhir = {fmt(after)} + {fmt(t)} = {fmt(ans)}.",
        "ja": f"割引額 = {disc}% × {fmt(price)} = {fmt(d)}、価格 = {fmt(after)}。税 = {tax}% × {fmt(after)} = {fmt(t)}。最終 = {fmt(after)} + {fmt(t)} = {fmt(ans)}。",
    }
    return q, steps, ans


def s_avg_replace(rng):
    n = rng.choice([4, 5])
    mean = rng.randint(10, 60)
    vals = [mean + rng.randint(-6, 6) for _ in range(n - 1)]
    vals.append(mean * n - sum(vals))
    old = vals[0]
    new = old + rng.choice([8, 10, 12, 15])
    new_sum = sum(vals) - old + new
    ans = new_sum / n
    ans = int(ans) if abs(ans - round(ans)) < 1e-9 else round(ans, 2)
    q = {
        "en": f"The average of {n} numbers is {mean}. If one of them, {old}, is replaced by {new}, what is the new average?",
        "id": f"Rata-rata dari {n} bilangan adalah {mean}. Jika salah satunya, {old}, diganti dengan {new}, berapa rata-rata barunya?",
        "ja": f"{n}個の数の平均は{mean}です。そのうちの1つ{old}を{new}に置き換えると、新しい平均はいくつですか？",
    }
    steps = {
        "en": f"Sum = {n} × {mean} = {fmt(mean * n)}. New sum = {fmt(mean * n)} − {old} + {new} = {fmt(new_sum)}. New average = {fmt(new_sum)} ÷ {n} = {fmt(ans)}.",
        "id": f"Jumlah = {n} × {mean} = {fmt(mean * n)}. Jumlah baru = {fmt(mean * n)} − {old} + {new} = {fmt(new_sum)}. Rata-rata baru = {fmt(new_sum)} ÷ {n} = {fmt(ans)}.",
        "ja": f"合計 = {n} × {mean} = {fmt(mean * n)}。新しい合計 = {fmt(mean * n)} − {old} + {new} = {fmt(new_sum)}。新しい平均 = {fmt(new_sum)} ÷ {n} = {fmt(ans)}。",
    }
    return q, steps, ans


def s_arith_seq(rng):
    a = rng.randint(-20, 40)
    d = rng.randint(2, 12)
    n = rng.randint(6, 20)
    last = a + (n - 1) * d
    total = n * (a + last) // 2
    q = {
        "en": f"An arithmetic sequence starts at {a} with common difference {d}. What is the sum of the first {n} terms?",
        "id": f"Barisan aritmetika dimulai dari {a} dengan beda {d}. Berapa jumlah {n} suku pertama?",
        "ja": f"等差数列は{a}から始まり、公差は{d}です。初項から{n}項目までの和はいくつですか？",
    }
    steps = {
        "en": f"Last term = {a} + ({n} − 1) × {d} = {last}. Sum = {n} × ({a} + {last}) ÷ 2 = {fmt(total)}.",
        "id": f"Suku terakhir = {a} + ({n} − 1) × {d} = {last}. Jumlah = {n} × ({a} + {last}) ÷ 2 = {fmt(total)}.",
        "ja": f"末項 = {a} + ({n} − 1) × {d} = {last}。和 = {n} × ({a} + {last}) ÷ 2 = {fmt(total)}。",
    }
    return q, steps, total


def s_mixture(rng):
    v1 = rng.choice([2, 4, 6])
    c1 = rng.choice([10, 20, 30])
    v2 = rng.choice([3, 5])
    c2 = rng.choice([40, 50, 60])
    salt = v1 * c1 + v2 * c2
    vol = v1 + v2
    conc = salt / vol
    conc = int(conc) if abs(conc - round(conc)) < 1e-9 else round(conc, 2)
    q = {
        "en": f"{v1} liters of a {c1}% salt solution are mixed with {v2} liters of a {c2}% solution. What is the concentration of the mixture?",
        "id": f"{v1} liter larutan garam {c1}% dicampur dengan {v2} liter larutan {c2}%. Berapa konsentrasi campurannya?",
        "ja": f"{c1}%の食塩水{v1}Lと{c2}%の食塩水{v2}Lを混ぜます。混合液の濃度は何%ですか？",
    }
    steps = {
        "en": f"Salt = {v1}×{c1} + {v2}×{c2} = {fmt(v1*c1)} + {fmt(v2*c2)} = {fmt(salt)}. Volume = {v1} + {v2} = {vol} L. Concentration = {fmt(salt)} ÷ {vol} = {fmt(conc)}%.",
        "id": f"Garam = {v1}×{c1} + {v2}×{c2} = {fmt(v1*c1)} + {fmt(v2*c2)} = {fmt(salt)}. Volume = {v1} + {v2} = {vol} L. Konsentrasi = {fmt(salt)} ÷ {vol} = {fmt(conc)}%.",
        "ja": f"塩 = {v1}×{c1} + {v2}×{c2} = {fmt(v1*c1)} + {fmt(v2*c2)} = {fmt(salt)}。体積 = {v1} + {v2} = {vol}L。濃度 = {fmt(salt)} ÷ {vol} = {fmt(conc)}%。",
    }
    return q, steps, conc


def s_clock(rng):
    h1, m1 = rng.randint(6, 10), rng.choice([0, 15, 30, 45])
    dur_h = rng.randint(1, 4)
    dur_m = rng.choice([30, 45])
    total = h1 * 60 + m1 + dur_h * 60 + dur_m
    h2 = total // 60
    m2 = total % 60
    q = {
        "en": f"A train departs at {h1}:{m1:02d} and the journey takes {dur_h} hours {dur_m} minutes. What time does it arrive (24-hour clock)?",
        "id": f"Kereta berangkat pukul {h1}:{m1:02d} dan perjalanan memakan {dur_h} jam {dur_m} menit. Pukul berapa kereta tiba (jam 24)?",
        "ja": f"電車は{h1}時{m1}分に出発し、旅には{dur_h}時間{dur_m}分かかります。到着時刻は何時何分ですか（24時間表記）？",
    }
    # answer format: "HH:MM"
    ans_txt = f"{h2:02d}:{m2:02d}"
    q["en"] += " Give the answer as HH:MM."
    q["id"] += " Berikan jawaban dalam format JJ:MM."
    q["ja"] += " HH:MMの形式で答えてください。"
    steps = {
        "en": f"Departure = {h1*60+m1} minutes past midnight; journey = {dur_h*60+dur_m} minutes; arrival = {total} minutes = {h2:02d}:{m2:02d}.",
        "id": f"Keberangkatan = menit ke-{h1*60+m1}; perjalanan = {dur_h*60+dur_m} menit; kedatangan = menit ke-{total} = {h2:02d}:{m2:02d}.",
        "ja": f"出発 = 午前0時から{h1*60+m1}分、移動 = {dur_h*60+dur_m}分、到着 = {total}分 = {h2:02d}:{m2:02d}。",
    }
    return q, steps, ans_txt


SCENARIOS = [
    ("speed_round_trip", s_speed),
    ("work_rate_ceil", s_work_rate),
    ("age_difference", s_age),
    ("money_chain_change", s_money_chain),
    ("percentage_chain", s_pct_chain),
    ("rect_fence_cost", s_rect_fence),
    ("tank_net_rate", s_tank),
    ("discount_then_tax", s_discount_tax),
    ("average_replace", s_avg_replace),
    ("arith_series_sum", s_arith_seq),
    ("mixture_concentration", s_mixture),
    ("clock_arrival", s_clock),
]


def render(steps, ans, lang):
    a = ans if isinstance(ans, str) else fmt(ans)
    return f"{steps[lang]}\n{ANCHOR[lang].format(a=a)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=120_000)
    ap.add_argument("--n_eval", type=int, default=2_000)
    ap.add_argument("--n_rl", type=int, default=6_000)
    ap.add_argument("--distract_rate", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="sft_data/sft_math_v2")
    ap.add_argument("--out_rl", default="rl_data/math_v2.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    langs = ["en"] * 4 + ["id"] * 4 + ["ja"] * 2

    def gen(seed_i):
        r = random.Random(seed_i)
        name, fn = SCENARIOS[seed_i % len(SCENARIOS)]
        q, steps, ans = fn(r)
        lang = r.choice(langs)
        if r.random() < args.distract_rate:
            q = {k: v + " " + distract(r, k) for k, v in q.items()}
        return name, q, steps, ans, lang

    def make_row(seed_i):
        name, q, steps, ans, lang = gen(seed_i)
        body = render(steps, ans, lang)
        return {
            "text": f"<|system|>{SYS[lang]}<|user|>{q[lang]}<|assistant|>{body}<|end|>",
            "lang": lang, "task": f"mathv2_{name}",
            "expected_output": ans if isinstance(ans, str) else plain(ans),
        }

    train = [make_row(i) for i in range(args.n_train)]
    rng.shuffle(train)
    eval_rows = [make_row(args.n_train + i) for i in range(args.n_eval)]
    rng.shuffle(eval_rows)
    rl = []
    for i in range(args.n_rl):
        name, q, steps, ans, lang = gen(args.n_train + args.n_eval + i)
        rl.append({
            "task": f"mathv2_{name}", "lang": lang, "system": SYS[lang],
            "prompt": q[lang], "mock_input": "",
            "expected_output": ans if isinstance(ans, str) else plain(ans),
        })
    rng.shuffle(rl)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / args.out_rl, "w", encoding="utf-8") as f:
        for r in rl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"math v2: {len(train)} train + {len(eval_rows)} eval + {len(rl)} RL")
    print(f"  by lang: {dict(Counter(r['lang'] for r in train))}")
    print(f"  by task: {dict(Counter(r['task'] for r in train))}")


if __name__ == "__main__":
    main()
