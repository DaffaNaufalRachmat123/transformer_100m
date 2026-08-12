#!/usr/bin/env python3
"""build_facts_ja.py — Build clean, verified JA fact Q&A + bilingual fact pairs.

Facts are hand-curated (verified) so quality is guaranteed — no noise from
web scraping. Each fact is emitted:
  - as a JA Q&A instruction (lang=ja)
  - as an EN Q&A instruction (lang=en)
  - as a translation pair (en-ja / ja-en) where the fact sentence is a
    natural translation (e.g. "The capital of Japan is Tokyo." <-> "日本の首都は東京です。")

Output: sft_data/sft_clean/sft_facts_curated.jsonl
"""

import json
import random
from pathlib import Path

RNG = random.Random(7)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sft_data" / "sft_clean" / "sft_facts_curated.jsonl"

SYS = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
SYS_EN = "You are a helpful, accurate multilingual assistant."

# (ja_question, ja_answer, en_question, en_answer, en_sentence, ja_sentence)
# en_sentence = natural EN translation of ja_sentence (and vice versa).
FACTS = [
    # ── Capital ─────────────────────────────────────────────────────────────
    ("日本の首都は？", "日本の首都は東京です。",
     "What is the capital of Japan?", "The capital of Japan is Tokyo.",
     "The capital of Japan is Tokyo.", "日本の首都は東京です。"),
    ("フランスの首都は？", "フランスの首都はパリです。",
     "What is the capital of France?", "The capital of France is Paris.",
     "The capital of France is Paris.", "フランスの首都はパリです。"),
    ("イギリスの首都は？", "イギリスの首都はロンドンです。",
     "What is the capital of the United Kingdom?", "The capital of the United Kingdom is London.",
     "The capital of the United Kingdom is London.", "イギリスの首都はロンドンです。"),
    ("アメリカの首都は？", "アメリカの首都はワシントンD.C.です。",
     "What is the capital of the United States?", "The capital of the United States is Washington, D.C.",
     "The capital of the United States is Washington, D.C.", "アメリカの首都はワシントンD.C.です。"),
    ("中国の首都は？", "中国の首都は北京です。",
     "What is the capital of China?", "The capital of China is Beijing.",
     "The capital of China is Beijing.", "中国の首都は北京です。"),
    ("韓国の首都は？", "韓国の首都はソウルです。",
     "What is the capital of South Korea?", "The capital of South Korea is Seoul.",
     "The capital of South Korea is Seoul.", "韓国の首都はソウルです。"),
    ("インドの首都は？", "インドの首都はニューデリーです。",
     "What is the capital of India?", "The capital of India is New Delhi.",
     "The capital of India is New Delhi.", "インドの首都はニューデリーです。"),
    ("インドネシアの首都は？", "インドネシアの首都はジャカルタです。",
     "What is the capital of Indonesia?", "The capital of Indonesia is Jakarta.",
     "The capital of Indonesia is Jakarta.", "インドネシアの首都はジャカルタです。"),
    ("ドイツの首都は？", "ドイツの首都はベルリンです。",
     "What is the capital of Germany?", "The capital of Germany is Berlin.",
     "The capital of Germany is Berlin.", "ドイツの首都はベルリンです。"),
    ("イタリアの首都は？", "イタリアの首都はローマです。",
     "What is the capital of Italy?", "The capital of Italy is Rome.",
     "The capital of Italy is Rome.", "イタリアの首都はローマです。"),
    ("ロシアの首都は？", "ロシアの首都はモスクワです。",
     "What is the capital of Russia?", "The capital of Russia is Moscow.",
     "The capital of Russia is Moscow.", "ロシアの首都はモスクワです。"),
    ("スペインの首都は？", "スペインの首都はマドリードです。",
     "What is the capital of Spain?", "The capital of Spain is Madrid.",
     "The capital of Spain is Madrid.", "スペインの首都はマドリードです。"),
    ("カナダの首都は？", "カナダの首都はオタワです。",
     "What is the capital of Canada?", "The capital of Canada is Ottawa.",
     "The capital of Canada is Ottawa.", "カナダの首都はオタワです。"),
    ("オーストラリアの首都は？", "オーストラリアの首都はキャンベラです。",
     "What is the capital of Australia?", "The capital of Australia is Canberra.",
     "The capital of Australia is Canberra.", "オーストラリアの首都はキャンベラです。"),
    ("ブラジルの首都は？", "ブラジルの首都はブラジリアです。",
     "What is the capital of Brazil?", "The capital of Brazil is Brasília.",
     "The capital of Brazil is Brasília.", "ブラジルの首都はブラジリアです。"),

    # ── Geography / landmarks ───────────────────────────────────────────────
    ("日本で一番高い山は？", "日本で一番高い山は富士山です。",
     "What is the tallest mountain in Japan?", "The tallest mountain in Japan is Mount Fuji.",
     "The tallest mountain in Japan is Mount Fuji.", "日本で一番高い山は富士山です。"),
    ("世界で一番高い山は？", "世界で一番高い山はエベレストです。",
     "What is the tallest mountain in the world?", "The tallest mountain in the world is Mount Everest.",
     "The tallest mountain in the world is Mount Everest.", "世界で一番高い山はエベレストです。"),
    ("日本で一番長い川は？", "日本で一番長い川は信濃川です。",
     "What is the longest river in Japan?", "The longest river in Japan is the Shinano River.",
     "The longest river in Japan is the Shinano River.", "日本で一番長い川は信濃川です。"),
    ("世界で一番長い川は？", "世界で一番長い川はナイル川です。",
     "What is the longest river in the world?", "The longest river in the world is the Nile.",
     "The longest river in the world is the Nile.", "世界で一番長い川はナイル川です。"),
    ("地球の衛星は？", "地球の衛星は月です。",
     "What is the moon of the Earth?", "The moon of the Earth is the Moon.",
     "The Moon is the Earth's satellite.", "地球の衛星は月です。"),
    ("太陽系で一番大きな惑星は？", "太陽系で一番大きな惑星は木星です。",
     "What is the largest planet in the Solar System?", "The largest planet in the Solar System is Jupiter.",
     "The largest planet in the solar system is Jupiter.", "太陽系で一番大きな惑星は木星です。"),
    ("太陽系で一番小さい惑星は？", "太陽系で一番小さい惑星は水星です。",
     "What is the smallest planet in the Solar System?", "The smallest planet in the Solar System is Mercury.",
     "The smallest planet in the solar system is Mercury.", "太陽系で一番小さい惑星は水星です。"),
    ("太陽系で一番暑い惑星は？", "太陽系で一番暑い惑星は金星です。",
     "What is the hottest planet in the Solar System?", "The hottest planet in the Solar System is Venus.",
     "The hottest planet in the solar system is Venus.", "太陽系で一番暑い惑星は金星です。"),
    ("日本の首都・東京の人口は？", "東京の人口はおよそ1400万人です。",
     "What is the population of Tokyo?", "The population of Tokyo is approximately 14 million.",
     "The population of Tokyo is approximately 14 million.", "東京の人口はおよそ1400万人です。"),

    # ── Science / general ───────────────────────────────────────────────────
    ("水の化学式は？", "水の化学式はH2Oです。",
     "What is the chemical formula of water?", "The chemical formula of water is H2O.",
     "The chemical formula of water is H2O.", "水の化学式はH2Oです。"),
    ("二酸化炭素の化学式は？", "二酸化炭素の化学式はCO2です。",
     "What is the chemical formula of carbon dioxide?", "The chemical formula of carbon dioxide is CO2.",
     "The chemical formula of carbon dioxide is CO2.", "二酸化炭素の化学式はCO2です。"),
    ("酸素の化学記号は？", "酸素の化学記号はOです。",
     "What is the chemical symbol for oxygen?", "The chemical symbol for oxygen is O.",
     "The chemical symbol for oxygen is O.", "酸素の化学記号はOです。"),
    ("水は何度で沸騰しますか？", "水は100度で沸騰します。",
     "At what temperature does water boil?", "Water boils at 100 degrees Celsius.",
     "Water boils at 100 degrees Celsius.", "水は100度で沸騰します。"),
    ("水は何度で凍りますか？", "水は0度で凍ります。",
     "At what temperature does water freeze?", "Water freezes at 0 degrees Celsius.",
     "Water freezes at 0 degrees Celsius.", "水は0度で凍ります。"),
    ("1キロメートルは何メートルですか？", "1キロメートルは1000メートルです。",
     "How many meters are in a kilometer?", "There are 1000 meters in a kilometer.",
     "There are 1000 meters in one kilometer.", "1キロメートルは1000メートルです。"),
    ("1年は何ヶ月ですか？", "1年は12ヶ月です。",
     "How many months are in a year?", "There are 12 months in a year.",
     "There are twelve months in a year.", "1年は12ヶ月です。"),
    ("1週間は何日ですか？", "1週間は7日です。",
     "How many days are in a week?", "There are 7 days in a week.",
     "There are seven days in a week.", "1週間は7日です。"),
    ("人間の足の数は？", "人間の足は2本です。",
     "How many legs does a human have?", "A human has two legs.",
     "A human has two legs.", "人間の足は2本です。"),
]

# Extra country facts (population / currency) — expandable, verified.
MORE = [
    ("日本の通貨は？", "日本の通貨は円（JPY）です。",
     "What is the currency of Japan?", "The currency of Japan is the yen (JPY).",
     "The currency of Japan is the yen.", "日本の通貨は円です。"),
    ("アメリカの通貨は？", "アメリカの通貨はドル（USD）です。",
     "What is the currency of the United States?", "The currency of the United States is the dollar (USD).",
     "The currency of the United States is the dollar.", "アメリカの通貨はドルです。"),
    ("中国の人口は？", "中国の人口はおよそ14億人です。",
     "What is the population of China?", "The population of China is approximately 1.4 billion.",
     "The population of China is about 1.4 billion.", "中国の人口はおよそ14億人です。"),
    ("インドの人口は？", "インドの人口はおよそ14億人です。",
     "What is the population of India?", "The population of India is approximately 1.4 billion.",
     "The population of India is about 1.4 billion.", "インドの人口はおよそ14億人です。"),
    # ── More countries / capitals ───────────────────────────────────────────
    ("タイの首都は？", "タイの首都はバンコクです。",
     "What is the capital of Thailand?", "The capital of Thailand is Bangkok.",
     "The capital of Thailand is Bangkok.", "タイの首都はバンコクです。"),
    ("ベトナムの首都は？", "ベトナムの首都はハノイです。",
     "What is the capital of Vietnam?", "The capital of Vietnam is Hanoi.",
     "The capital of Vietnam is Hanoi.", "ベトナムの首都はハノイです。"),
    ("フィリピンの首都は？", "フィリピンの首都はマニラです。",
     "What is the capital of the Philippines?", "The capital of the Philippines is Manila.",
     "The capital of the Philippines is Manila.", "フィリピンの首都はマニラです。"),
    ("マレーシアの首都は？", "マレーシアの首都はクアラルンプールです。",
     "What is the capital of Malaysia?", "The capital of Malaysia is Kuala Lumpur.",
     "The capital of Malaysia is Kuala Lumpur.", "マレーシアの首都はクアラルンプールです。"),
    ("メキシコの首都は？", "メキシコの首都はメキシコシティです。",
     "What is the capital of Mexico?", "The capital of Mexico is Mexico City.",
     "The capital of Mexico is Mexico City.", "メキシコの首都はメキシコシティです。"),
    ("エジプトの首都は？", "エジプトの首都はカイロです。",
     "What is the capital of Egypt?", "The capital of Egypt is Cairo.",
     "The capital of Egypt is Cairo.", "エジプトの首都はカイロです。"),
    ("トルコの首都は？", "トルコの首都はアンカラです。",
     "What is the capital of Turkey?", "The capital of Turkey is Ankara.",
     "The capital of Turkey is Ankara.", "トルコの首都はアンカラです。"),
    ("スイスの首都は？", "スイスの首都はベルンです。",
     "What is the capital of Switzerland?", "The capital of Switzerland is Bern.",
     "The capital of Switzerland is Bern.", "スイスの首都はベルンです。"),
    ("オランダの首都は？", "オランダの首都はアムステルダムです。",
     "What is the capital of the Netherlands?", "The capital of the Netherlands is Amsterdam.",
     "The capital of the Netherlands is Amsterdam.", "オランダの首都はアムステルダムです。"),
    ("オーストリアの首都は？", "オーストリアの首都はウィーンです。",
     "What is the capital of Austria?", "The capital of Austria is Vienna.",
     "The capital of Austria is Vienna.", "オーストリアの首都はウィーンです。"),
    ("ポーランドの首都は？", "ポーランドの首都はワルシャワです。",
     "What is the capital of Poland?", "The capital of Poland is Warsaw.",
     "The capital of Poland is Warsaw.", "ポーランドの首都はワルシャワです。"),
    ("スウェーデンの首都は？", "スウェーデンの首都はストックホルムです。",
     "What is the capital of Sweden?", "The capital of Sweden is Stockholm.",
     "The capital of Sweden is Stockholm.", "スウェーデンの首都はストックホルムです。"),
    ("ノルウェーの首都は？", "ノルウェーの首都はオスロです。",
     "What is the capital of Norway?", "The capital of Norway is Oslo.",
     "The capital of Norway is Oslo.", "ノルウェーの首都はオスロです。"),
    ("デンマークの首都は？", "デンマークの首都はコペンハーゲンです。",
     "What is the capital of Denmark?", "The capital of Denmark is Copenhagen.",
     "The capital of Denmark is Copenhagen.", "デンマークの首都はコペンハーゲンです。"),
    ("ギリシャの首都は？", "ギリシャの首都はアテネです。",
     "What is the capital of Greece?", "The capital of Greece is Athens.",
     "The capital of Greece is Athens.", "ギリシャの首都はアテネです。"),
    ("アルゼンチンの首都は？", "アルゼンチンの首都はブエノスアイレスです。",
     "What is the capital of Argentina?", "The capital of Argentina is Buenos Aires.",
     "The capital of Argentina is Buenos Aires.", "アルゼンチンの首都はブエノスアイレスです。"),
    ("南アフリカの首都は？", "南アフリカには3つの首都があります。行政首都はプレトリアです。",
     "What is the capital of South Africa?", "South Africa has three capitals; the administrative capital is Pretoria.",
     "The administrative capital of South Africa is Pretoria.", "南アフリカの行政首都はプレトリアです。"),
    ("ニュージーランドの首都は？", "ニュージーランドの首都はウェリントンです。",
     "What is the capital of New Zealand?", "The capital of New Zealand is Wellington.",
     "The capital of New Zealand is Wellington.", "ニュージーランドの首都はウェリントンです。"),
    # ── More geography / science ────────────────────────────────────────────
    ("世界で一番人口が多い国は？", "世界で一番人口が多い国はインドです。",
     "What is the most populous country in the world?", "The most populous country in the world is India.",
     "The most populous country in the world is India.", "世界で一番人口が多い国はインドです。"),
    ("世界で一番大きな国は？", "世界で一番大きな国はロシアです。",
     "What is the largest country in the world?", "The largest country in the world is Russia.",
     "The largest country in the world is Russia.", "世界で一番大きな国はロシアです。"),
    ("世界で一番大きな海は？", "世界で一番大きな海は太平洋です。",
     "What is the largest ocean in the world?", "The largest ocean in the world is the Pacific Ocean.",
     "The largest ocean in the world is the Pacific Ocean.", "世界で一番大きな海は太平洋です。"),
    ("日本で一番大きな都市は？", "日本で一番大きな都市は東京です。",
     "What is the largest city in Japan?", "The largest city in Japan is Tokyo.",
     "The largest city in Japan is Tokyo.", "日本で一番大きな都市は東京です。"),
    ("世界で一番高いビルは？", "世界で一番高いビルはブルジュ・ハリファです。",
     "What is the tallest building in the world?", "The tallest building in the world is Burj Khalifa.",
     "The tallest building in the world is Burj Khalifa.", "世界で一番高いビルはブルジュ・ハリファです。"),
    ("イギリスの通貨は？", "イギリスの通貨はポンド（GBP）です。",
     "What is the currency of the United Kingdom?", "The currency of the United Kingdom is the pound (GBP).",
     "The currency of the United Kingdom is the pound.", "イギリスの通貨はポンドです。"),
    ("欧州連合の通貨は？", "ユーロは欧州連合の多くの国で使われています。",
     "What is the currency of the European Union?", "The euro is used by many countries in the European Union.",
     "The euro is used by many countries in the EU.", "ユーロは欧州連合の多くの国で使われています。"),
    ("地球の直径は？", "地球の直径はおよそ1万2700キロメートルです。",
     "What is the diameter of the Earth?", "The diameter of the Earth is about 12,700 kilometers.",
     "The diameter of the Earth is about 12,700 kilometers.", "地球の直径はおよそ1万2700キロメートルです。"),
    ("太陽は何の星ですか？", "太陽は恒星です。",
     "What type of star is the Sun?", "The Sun is a star.",
     "The Sun is a star.", "太陽は恒星です。"),
    ("1年は何日ですか？", "1年は365日です。",
     "How many days are in a year?", "There are 365 days in a year.",
     "There are 365 days in a year.", "1年は365日です。"),
    ("1時間は何分ですか？", "1時間は60分です。",
     "How many minutes are in an hour?", "There are 60 minutes in an hour.",
     "There are 60 minutes in an hour.", "1時間は60分です。"),
    ("1分は何秒ですか？", "1分は60秒です。",
     "How many seconds are in a minute?", "There are 60 seconds in a minute.",
     "There are 60 seconds in a minute.", "1分は60秒です。"),
    ("1ダースは何個ですか？", "1ダースは12個です。",
     "How many items are in a dozen?", "There are 12 items in a dozen.",
     "There are twelve items in a dozen.", "1ダースは12個です。"),
    ("円の周囲の長さを何と呼びますか？", "円の周囲の長さを円周と呼びます。",
     "What is the circumference of a circle called?", "The distance around a circle is called the circumference.",
     "The distance around a circle is the circumference.", "円の周囲の長さを円周と呼びます。"),
    ("人間の心臓は1日に何回くらい鼓動しますか？", "人間の心臓は1日におよそ10万回鼓動します。",
     "How many times does the human heart beat per day?", "The human heart beats about 100,000 times per day.",
     "The human heart beats about 100,000 times a day.", "人間の心臓は1日におよそ10万回鼓動します。"),
    ("血液の色は？", "血液は赤いです。",
     "What color is blood?", "Blood is red.",
     "Blood is red.", "血液は赤いです。"),
    ("空の色は？", "晴れた日の空は青いです。",
     "What color is the sky?", "The sky is blue on a clear day.",
     "The sky is blue on a clear day.", "晴れた日の空は青いです。"),
    ("地球上で最も大きな動物は？", "地球上で最も大きな動物はシロナガスクジラです。",
     "What is the largest animal on Earth?", "The largest animal on Earth is the blue whale.",
     "The largest animal on Earth is the blue whale.", "地球上で最も大きな動物はシロナガスクジラです。"),
    ("日本で最も人気のある魚は？", "日本で最も人気のある魚はマグロです。",
     "What is the most popular fish in Japan?", "The most popular fish in Japan is tuna.",
     "The most popular fish in Japan is tuna.", "日本で最も人気のある魚はマグロです。"),
    ("日本語の「ありがとう」は英語で何と言いますか？", "「ありがとう」は英語で「Thank you」と言います。",
     "How do you say 'arigatou' in English?", "'Arigatou' means 'thank you' in English.",
     "'Arigatou' means 'thank you'.", "「ありがとう」は英語で「Thank you」と言います。"),
    ("日本語の「さようなら」は英語で何と言いますか？", "「さようなら」は英語で「Goodbye」と言います。",
     "How do you say 'sayonara' in English?", "'Sayonara' means 'goodbye' in English.",
     "'Sayonara' means 'goodbye'.", "「さようなら」は英語で「Goodbye」と言います。"),
]


def main():
    FACTS.extend(MORE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ja_q, ja_a, en_q, en_a, en_sent, ja_sent in FACTS:
        # JA Q&A
        rows.append({"text": f"<|system|>{SYS}<|user|>{ja_q}<|assistant|>{ja_a}<|end|>",
                     "lang": "ja", "task": "instruction"})
        # EN Q&A
        rows.append({"text": f"<|system|>{SYS_EN}<|user|>{en_q}<|assistant|>{en_a}<|end|>",
                     "lang": "en", "task": "instruction"})
        # EN->JA translation
        rows.append({"text": f"<|system|>You are a professional English-to-Japanese translator.<|user|>Translate to Japanese: {en_sent}<|assistant|>{ja_sent}<|end|>",
                     "lang": "en-ja", "task": "translation"})
        # JA->EN translation
        rows.append({"text": f"<|system|>You are a professional Japanese-to-English translator.<|user|>Translate to English: {ja_sent}<|assistant|>{en_sent}<|end|>",
                     "lang": "ja-en", "task": "translation"})

    RNG.shuffle(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"facts curated: {len(rows)} rows")
    print(dict(Counter(r['lang'] for r in rows)))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()