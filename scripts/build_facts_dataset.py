#!/usr/bin/env python3
"""build_facts_dataset.py — Build a high-quality Q&A fact dataset for SFT.

Goal: fix the "language confusion" problem (EN questions answered in JA and
vice versa) and improve simple fact answering. Sources, in priority order:

  1. sft_facts_curated.jsonl  — hand-curated, verified EN+JA fact Q&A (gold).
  2. sft_ja.jsonl             — strict-filtered OBJECTIVE JA facts only
                               (capitals, units, chemistry, planets, rivers,
                               mountains, population). Rejects opinion/context/
                               reading-comprehension questions and noisy answers.
  3. sft_en_facts.jsonl       — strict-filtered + format-normalized EN facts.

Output is balanced between EN and JA so the model learns to answer in the
language of the question.

Output: sft_data/sft_facts_focused/train.jsonl
"""

import json
import re
import random
from pathlib import Path

RNG = random.Random(2024)
ROOT = Path(__file__).resolve().parent.parent
SFT_CLEAN = ROOT / "sft_data" / "sft_clean"
FACT_OVERRIDES = ROOT / "data" / "fact_overrides.json"

# Mode determines output dir + EN/JA ratio:
#   all        : balanced 50/50 -> sft_facts_large/       (combined)
#   ja_only    : 100% JA       -> sft_facts_ja/           (Stage 1)
#   en_dominant: ~80% EN/20% JA-> sft_facts_en/           (Stage 2)
#   strict     : clean EN/JA fact QA with explicit category prompts only
MODE = "all"
OUT_DIR = ROOT / "sft_data" / "sft_facts_large"
OUT_FILE = OUT_DIR / "train.jsonl"
EN_SHARE = 1.0    # fraction of EN rows in en_dominant mode (JA shares the rest)
STRICT_EVAL_FILE = "eval.jsonl"

SYS_EN = "You are a helpful, accurate multilingual assistant."
SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
SYS_ID = "Kamu adalah asisten AI multibahasa yang membantu, jujur, dan akurat."

# ── Template augmentation (EN + JA) per category ──────────────────────────
# Each category has list of (q_template, a_template) using {sub}/{val}.
# TEMPLATES_PER_FACT controls how many variants each unique fact expands to.
TEMPLATES_PER_FACT = 30

# (q_en, a_en, q_ja, a_ja) template lists per category. Only the "answer"
# templates differ per category; question templates are generic so the model
# learns many natural phrasings.
EN_Q_TMPLS = [
    "What is {sub}?",
    "Can you tell me {sub}?",
    "Do you know {sub}?",
    "What do you know about {sub}?",
    "Tell me about {sub}.",
    "What can you say about {sub}?",
    "I have a question about {sub}.",
    "Could you tell me about {sub}?",
    "Please tell me about {sub}.",
    "I would like to know {sub}.",
    "What can you tell me about {sub}?",
    "I am curious about {sub}.",
    "Would you tell me about {sub}?",
    "Can you explain {sub}?",
    "What is known about {sub}?",
    "Tell me what you know about {sub}.",
    "Please explain {sub}.",
    "What do you think {sub} is?",
    "Can you tell me what {sub} is?",
    "Do you happen to know {sub}?",
    "What is the answer to {sub}?",
    "Could you help me with {sub}?",
    "Can you share what {sub} is?",
    "Would you know {sub}?",
    "What is {sub}, if I may ask?",
    "Could you clarify {sub}?",
    "What do you know of {sub}?",
    "Please share {sub} with me.",
    "Can you describe {sub}?",
    "What is {sub} according to you?",
]
# Per category: (a_en, a_ja) built from sub/val.
CAT_ANSWERS = {
    "capital": ("The capital of {sub} is {val}.", "{sub}の首都は{val}です。"),
    "capital0": ("{val} is the capital of {sub}.", "{val}は{sub}の首都です。"),
    "largest_city": ("The largest city of {sub} is {val}.", "{sub}で一番大きい都市は{val}です。"),
    "currency": ("The currency of {sub} is {val}.", "{sub}の通貨は{val}です。"),
    "official_language": ("An official language of {sub} is {val}.", "{sub}の公用語は{val}です。"),
    "population": ("The population of {sub} is about {val}.", "{sub}の人口はおよそ{val}人です。"),
    "highest_point": ("The highest point of {sub} is {val}.", "{sub}で一番高い場所は{val}です。"),
    "continent_of": ("{sub} is in {val}.", "{sub}は{val}にあります。"),
    "head_of_state": ("The head of state of {sub} is {val}.", "{sub}の国家元首は{val}です。"),
    "area": ("The area of {sub} is about {val} square kilometers.", "{sub}の面積はおよそ{val}平方キロメートルです。"),
    "element_symbol": ("The chemical symbol of {sub} is {val}.", "{sub}の化学記号は{val}です。"),
    "element_number": ("The atomic number of {sub} is {val}.", "{sub}の原子番号は{val}です。"),
    "mountain_country": ("{sub} is in {val}.", "{sub}は{val}にあります。"),
    "river_country": ("{sub} is in {val}.", "{sub}は{val}にあります。"),
    "planet_kind": ("{sub} is a planet.", "{sub}は惑星です。"),
    "language_country": ("{sub} is spoken in {val}.", "{sub}は{val}で話されています。"),
    "person_birthplace": ("{sub} was born in {val}.", "{sub}は{val}で生まれました。"),
    "person_occupation": ("{sub} is a {val}.", "{sub}は{val}です。"),
}
CAT_ANSWERS_BINDING = {
    "capital": ("For {sub}, the capital is {val}.", "{sub}については、首都は{val}です。"),
    "currency": ("For {sub}, the currency is {val}.", "{sub}については、通貨は{val}です。"),
    "official_language": ("For {sub}, an official language is {val}.", "{sub}については、公用語は{val}です。"),
    "population": ("For {sub}, the population is about {val}.", "{sub}については、人口はおよそ{val}人です。"),
    "highest_point": ("For {sub}, the highest point is {val}.", "{sub}については、一番高い場所は{val}です。"),
    "continent_of": ("For {sub}, the continent is {val}.", "{sub}については、大陸は{val}です。"),
    "area": ("For {sub}, the area is about {val} square kilometers.", "{sub}については、面積はおよそ{val}平方キロメートルです。"),
    "element_symbol": ("For {sub}, the chemical symbol is {val}.", "{sub}については、化学記号は{val}です。"),
    "element_number": ("For {sub}, the atomic number is {val}.", "{sub}については、原子番号は{val}です。"),
}
ANTI_CONFUSION_Q = {
    "capital": {
        "en": [
            "I am asking for the capital of {sub}. What is it?",
            "Give only the capital city of {sub}.",
            "What is {sub}'s capital, not its continent?",
            "Answer the capital question for {sub}.",
        ],
        "ja": [
            "{sub}の首都について質問しています。答えは何ですか？",
            "{sub}の首都だけを答えてください。",
            "{sub}の大陸ではなく、首都を教えてください。",
            "{sub}の首都の質問に答えてください。",
        ],
    },
    "currency": {
        "en": [
            "I am asking for the currency of {sub}. What is it?",
            "Give only the currency of {sub}.",
            "What is {sub}'s currency, not its continent?",
            "Answer the currency question for {sub}.",
        ],
        "ja": [
            "{sub}の通貨について質問しています。答えは何ですか？",
            "{sub}の通貨だけを答えてください。",
            "{sub}の大陸ではなく、通貨を教えてください。",
            "{sub}の通貨の質問に答えてください。",
        ],
    },
    "population": {
        "en": [
            "I am asking for the population of {sub}. What is it?",
            "Give only the population of {sub}.",
            "What is {sub}'s population, not its currency?",
            "Tell me how many people live in {sub}.",
        ],
        "ja": [
            "{sub}の人口について質問しています。答えは何ですか？",
            "{sub}の人口だけを答えてください。",
            "{sub}の通貨ではなく、人口を教えてください。",
            "{sub}には何人くらい住んでいますか？",
        ],
    },
    "area": {
        "en": [
            "I am asking for the area of {sub}. What is it?",
            "Give only the area of {sub} in square kilometers.",
            "What is {sub}'s area, not its population?",
            "Answer the area question for {sub}.",
        ],
        "ja": [
            "{sub}の面積について質問しています。答えは何ですか？",
            "{sub}の面積だけを平方キロメートルで答えてください。",
            "{sub}の人口ではなく、面積を教えてください。",
            "{sub}の面積の質問に答えてください。",
        ],
    },
    "highest_point": {
        "en": [
            "I am asking for the highest point of {sub}. What is it?",
            "Give only the highest point of {sub}.",
            "What is {sub}'s highest point, not its area?",
            "Give the highest point of {sub}, not its area or continent.",
            "Answer the highest-point question for {sub}.",
        ],
        "ja": [
            "{sub}の最高地点について質問しています。答えは何ですか？",
            "{sub}の最高地点だけを答えてください。",
            "{sub}の面積ではなく、最高地点を教えてください。",
            "{sub}の面積や大陸ではなく、最高地点を教えてください。",
            "{sub}の最高地点の質問に答えてください。",
        ],
    },
    "official_language": {
        "en": [
            "I am asking for an official language of {sub}. What language is it?",
            "Give only an official language of {sub}, not its capital.",
            "What language is official in {sub}, not its continent?",
            "Answer with an official language of {sub}, not a city or symbol.",
        ],
        "ja": [
            "{sub}の公用語について質問しています。何語ですか？",
            "{sub}の首都ではなく、公用語だけを答えてください。",
            "{sub}の大陸ではなく、公用語を教えてください。",
            "都市名や記号ではなく、{sub}の公用語を答えてください。",
        ],
    },
    "element_symbol": {
        "en": [
            "I am asking for the chemical symbol of {sub}. What is it?",
            "Give only the chemical symbol of {sub}.",
            "What is {sub}'s chemical symbol, not its atomic number?",
            "Answer the chemical-symbol question for {sub}.",
        ],
        "ja": [
            "{sub}の化学記号について質問しています。答えは何ですか？",
            "{sub}の化学記号だけを答えてください。",
            "{sub}の原子番号ではなく、化学記号を教えてください。",
            "{sub}の化学記号の質問に答えてください。",
        ],
    },
    "element_number": {
        "en": [
            "I am asking for the atomic number of {sub}. What is it?",
            "Give only the atomic number of {sub}.",
            "What is {sub}'s atomic number, not its chemical symbol?",
            "Answer the atomic-number question for {sub}.",
        ],
        "ja": [
            "{sub}の原子番号について質問しています。答えは何ですか？",
            "{sub}の原子番号だけを答えてください。",
            "{sub}の化学記号ではなく、原子番号を教えてください。",
            "{sub}の原子番号の質問に答えてください。",
        ],
    },
}
BINDING_Q = {
    "capital": ("For {sub}, what is the capital?", "{sub}については、首都は何ですか？"),
    "currency": ("For {sub}, what is the currency?", "{sub}については、通貨は何ですか？"),
    "official_language": ("For {sub}, what is an official language?", "{sub}については、公用語は何ですか？"),
    "population": ("For {sub}, what is the population?", "{sub}については、人口は何人ですか？"),
    "highest_point": ("For {sub}, what is the highest point?", "{sub}については、一番高い場所はどこですか？"),
    "continent_of": ("For {sub}, what is the continent?", "{sub}については、大陸はどこですか？"),
    "area": ("For {sub}, what is the area?", "{sub}については、面積はどれくらいですか？"),
    "element_symbol": ("For {sub}, what is the chemical symbol?", "{sub}については、化学記号は何ですか？"),
    "element_number": ("For {sub}, what is the atomic number?", "{sub}については、原子番号は何ですか？"),
}
REQUEST_LABELS = {
    "capital": ("the capital", "首都"),
    "currency": ("the currency", "通貨"),
    "official_language": ("an official language", "公用語"),
    "population": ("the population", "人口"),
    "highest_point": ("the highest point", "最高地点"),
    "continent_of": ("the continent", "大陸"),
    "area": ("the area", "面積"),
    "element_symbol": ("the chemical symbol", "化学記号"),
    "element_number": ("the atomic number", "原子番号"),
}
CONTRAST_DISTRACTORS = {
    "highest_point": ["area", "continent_of", "capital"],
    "official_language": ["capital", "continent_of", "currency"],
    "currency": ["capital", "official_language", "continent_of"],
    "area": ["population", "highest_point"],
    "population": ["area", "currency"],
}

# ── Indonesian (trilingual V10) ────────────────────────────────────────────
STRICT_ID_Q = {
    "capital": [
        "Apa ibu kota {sub}?",
        "Kota apa yang menjadi ibu kota {sub}?",
        "Sebutkan ibu kota {sub}.",
        "Apakah kamu tahu ibu kota {sub}?",
        "Tolong beri tahu ibu kota {sub}.",
        "Aku ingin tahu ibu kota {sub}.",
        "Bisa sebutkan ibu kota {sub}?",
        "Ibu kota {sub} adalah kota apa?",
    ],
    "currency": [
        "Apa mata uang {sub}?",
        "Mata uang apa yang digunakan di {sub}?",
        "Sebutkan mata uang {sub}.",
        "Apakah kamu tahu mata uang {sub}?",
        "Tolong beri tahu mata uang {sub}.",
        "Aku ingin tahu mata uang {sub}.",
        "Bisa sebutkan mata uang {sub}?",
        "Uang apa yang dipakai di {sub}?",
    ],
    "official_language": [
        "Apa bahasa resmi {sub}?",
        "Bahasa apa yang resmi di {sub}?",
        "Sebutkan bahasa resmi {sub}.",
        "Apakah kamu tahu bahasa resmi {sub}?",
        "Tolong beri tahu bahasa resmi {sub}.",
        "Aku ingin tahu bahasa resmi {sub}.",
        "Bisa sebutkan bahasa resmi {sub}?",
        "Bahasa apa yang digunakan secara resmi di {sub}?",
    ],
    "population": [
        "Berapa populasi {sub}?",
        "Seberapa besar populasi {sub}?",
        "Sebutkan populasi {sub}.",
        "Apakah kamu tahu populasi {sub}?",
        "Tolong beri tahu populasi {sub}.",
        "Aku ingin tahu populasi {sub}.",
        "Bisa sebutkan populasi {sub}?",
        "Kira-kira ada berapa penduduk di {sub}?",
    ],
    "highest_point": [
        "Apa titik tertinggi di {sub}?",
        "Tempat apa yang menjadi titik tertinggi {sub}?",
        "Sebutkan titik tertinggi {sub}.",
        "Apakah kamu tahu titik tertinggi {sub}?",
        "Tolong beri tahu titik tertinggi {sub}.",
        "Aku ingin tahu titik tertinggi {sub}.",
        "Bisa sebutkan titik tertinggi {sub}?",
        "Apa tempat tertinggi di {sub}?",
    ],
    "continent_of": [
        "Di benua apa {sub} berada?",
        "{sub} berada di benua apa?",
        "Sebutkan benua {sub}.",
        "Apakah kamu tahu {sub} ada di benua mana?",
        "Tolong beri tahu benua {sub}.",
        "Aku ingin tahu {sub} ada di benua apa.",
        "Bisa sebutkan benua tempat {sub} berada?",
        "{sub} terletak di benua apa?",
    ],
    "area": [
        "Berapa luas {sub}?",
        "Seberapa luas wilayah {sub}?",
        "Sebutkan luas {sub}.",
        "Apakah kamu tahu luas {sub}?",
        "Tolong beri tahu luas {sub}.",
        "Aku ingin tahu luas {sub}.",
        "Bisa sebutkan luas {sub}?",
        "Berapa kilometer persegi luas {sub}?",
    ],
    "element_symbol": [
        "Apa simbol kimia {sub}?",
        "Sebutkan simbol kimia {sub}.",
        "Apakah kamu tahu simbol kimia {sub}?",
        "Simbol kimia apa yang mewakili {sub}?",
        "Tolong beri tahu simbol kimia {sub}.",
        "Aku ingin tahu simbol kimia {sub}.",
        "Bisa sebutkan simbol kimia {sub}?",
        "Lambang kimia apa untuk {sub}?",
    ],
    "element_number": [
        "Berapa nomor atom {sub}?",
        "Sebutkan nomor atom {sub}.",
        "Apakah kamu tahu nomor atom {sub}?",
        "{sub} memiliki nomor atom berapa?",
        "Tolong beri tahu nomor atom {sub}.",
        "Aku ingin tahu nomor atom {sub}.",
        "Bisa sebutkan nomor atom {sub}?",
        "Nomor berapa yang diberikan kepada unsur {sub}?",
    ],
}
CAT_ANSWERS_ID = {
    "capital": "Ibu kota {sub} adalah {val}.",
    "currency": "Mata uang {sub} adalah {val}.",
    "official_language": "Bahasa resmi {sub} adalah {val}.",
    "population": "Populasi {sub} adalah sekitar {val}.",
    "highest_point": "Titik tertinggi di {sub} adalah {val}.",
    "continent_of": "{sub} berada di {val}.",
    "area": "Luas {sub} adalah sekitar {val} kilometer persegi.",
    "element_symbol": "Simbol kimia {sub} adalah {val}.",
    "element_number": "Nomor atom {sub} adalah {val}.",
}
CAT_ANSWERS_ID_BINDING = {
    "capital": "Untuk {sub}, ibu kotanya adalah {val}.",
    "currency": "Untuk {sub}, mata uangnya adalah {val}.",
    "official_language": "Untuk {sub}, bahasa resminya adalah {val}.",
    "population": "Untuk {sub}, populasinya adalah sekitar {val}.",
    "highest_point": "Untuk {sub}, titik tertingginya adalah {val}.",
    "continent_of": "Untuk {sub}, benuanya adalah {val}.",
    "area": "Untuk {sub}, luasnya adalah sekitar {val} kilometer persegi.",
    "element_symbol": "Untuk {sub}, simbol kimianya adalah {val}.",
    "element_number": "Untuk {sub}, nomor atomnya adalah {val}.",
}
BINDING_Q_ID = {
    "capital": "Untuk {sub}, apa ibu kotanya?",
    "currency": "Untuk {sub}, apa mata uangnya?",
    "official_language": "Untuk {sub}, apa bahasa resminya?",
    "population": "Untuk {sub}, berapa populasinya?",
    "highest_point": "Untuk {sub}, apa titik tertingginya?",
    "continent_of": "Untuk {sub}, di benua apa ia berada?",
    "area": "Untuk {sub}, berapa luasnya?",
    "element_symbol": "Untuk {sub}, apa simbol kimianya?",
    "element_number": "Untuk {sub}, berapa nomor atomnya?",
}
REQUEST_LABELS_ID = {
    "capital": "ibu kota",
    "currency": "mata uang",
    "official_language": "bahasa resmi",
    "population": "populasi",
    "highest_point": "titik tertinggi",
    "continent_of": "benua",
    "area": "luas",
    "element_symbol": "simbol kimia",
    "element_number": "nomor atom",
}
ANSWER_LABELS_ID = {
    "capital": "ibu kotanya",
    "currency": "mata uangnya",
    "official_language": "bahasa resminya",
    "population": "populasinya",
    "highest_point": "titik tertingginya",
    "continent_of": "benuanya",
    "area": "luasnya",
    "element_symbol": "simbol kimianya",
    "element_number": "nomor atomnya",
}
ANTI_CONFUSION_Q_ID = {
    "capital": [
        "Aku bertanya tentang ibu kota {sub}. Apa jawabannya?",
        "Sebutkan ibu kota {sub} saja.",
        "Bukan benuanya, sebutkan ibu kota {sub}.",
        "Jawab pertanyaan tentang ibu kota {sub}.",
    ],
    "currency": [
        "Aku bertanya tentang mata uang {sub}. Apa jawabannya?",
        "Sebutkan mata uang {sub} saja.",
        "Bukan benuanya, sebutkan mata uang {sub}.",
        "Jawab pertanyaan tentang mata uang {sub}.",
    ],
    "population": [
        "Aku bertanya tentang populasi {sub}. Apa jawabannya?",
        "Sebutkan populasi {sub} saja.",
        "Bukan mata uangnya, sebutkan populasi {sub}.",
        "Berapa jumlah penduduk {sub}?",
    ],
    "area": [
        "Aku bertanya tentang luas {sub}. Apa jawabannya?",
        "Sebutkan luas {sub} saja dalam kilometer persegi.",
        "Bukan populasinya, sebutkan luas {sub}.",
        "Jawab pertanyaan tentang luas {sub}.",
    ],
    "highest_point": [
        "Aku bertanya tentang titik tertinggi {sub}. Apa jawabannya?",
        "Sebutkan titik tertinggi {sub} saja.",
        "Bukan ibu kotanya, sebutkan titik tertinggi {sub}.",
        "Jawab pertanyaan tentang titik tertinggi {sub}.",
    ],
    "element_symbol": [
        "Aku bertanya tentang simbol kimia {sub}. Apa jawabannya?",
        "Sebutkan simbol kimia {sub} saja.",
        "Bukan nomor atomnya, sebutkan simbol kimia {sub}.",
        "Jawab pertanyaan tentang simbol kimia {sub}.",
    ],
    "element_number": [
        "Aku bertanya tentang nomor atom {sub}. Apa jawabannya?",
        "Sebutkan nomor atom {sub} saja.",
        "Bukan simbol kimianya, sebutkan nomor atom {sub}.",
        "Jawab pertanyaan tentang nomor atom {sub}.",
    ],
}

# Strict fact QA uses only explicit category prompts. Generic prompts like
# "Tell me about X" are valid for summaries, but poison single-attribute SFT
# because the same entity can map to many different attributes.
STRICT_CATEGORIES = {
    "capital", "currency", "official_language", "population", "highest_point",
    "continent_of", "area", "element_symbol", "element_number",
}
STRICT_DROP_CATEGORIES = {
    "head_of_state", "largest_city", "capital0", "mountain_country",
    "river_country", "planet_kind", "language_country", "person_birthplace",
    "person_occupation",
}
OVERSAMPLE_WEIGHTS = {
    # Hard categories from recent eval get more exposure. Area/continent are
    # intentionally lower because the model tends to over-answer with them.
    "element_symbol": 4.0,
    "highest_point": 4.0,
    "population": 4.0,
    "currency": 3.0,
    "official_language": 3.0,
    "capital": 2.0,
    "element_number": 2.0,
    "continent_of": 1.0,
    "area": 1.0,
}
OVERSAMPLE_WEIGHTS_V2 = {
    "element_symbol": 5.0,
    "population": 5.0,
    "capital": 4.0,
    "highest_point": 4.0,
    "currency": 3.0,
    "official_language": 3.0,
    "element_number": 3.0,
    "area": 1.2,
    "continent_of": 1.0,
}
OVERSAMPLE_LANG_MULT_V2 = {
    "en": {
        "population": 1.5,
        "capital": 1.5,
        "area": 1.3,
        "element_symbol": 1.5,
        "element_number": 1.3,
    },
    "ja": {},
}
OVERSAMPLE_WEIGHTS_V3_EN = {
    "element_symbol": 6.0,
    "population": 6.0,
    "capital": 5.0,
    "element_number": 4.0,
    "highest_point": 4.0,
    "currency": 3.5,
    "official_language": 3.0,
    "area": 1.5,
    "continent_of": 0.8,
}
OVERSAMPLE_LANG_MULT_V3_EN = {
    "en": {
        "population": 2.0,
        "capital": 2.0,
        "element_symbol": 2.0,
        "element_number": 1.7,
        "area": 1.6,
        "highest_point": 1.4,
        "currency": 1.3,
    },
    "ja": {
        "continent_of": 0.8,
        "area": 0.9,
    },
}
OVERSAMPLE_WEIGHTS_V4_EN = {
    "element_symbol": 6.0,
    "population": 6.0,
    "capital": 5.0,
    "element_number": 4.0,
    "highest_point": 4.0,
    "currency": 3.5,
    "official_language": 3.0,
    "area": 1.5,
    "continent_of": 0.8,
}
OVERSAMPLE_LANG_MULT_V4_EN = {
    "en": {
        "population": 2.0,
        "capital": 2.0,
        "element_symbol": 2.0,
        "element_number": 1.7,
        "area": 1.6,
        "highest_point": 1.4,
        "currency": 1.3,
    },
    "ja": {},
}
HARD_EN_WEIGHTS = {
    "element_symbol": 7.0,
    "capital": 6.0,
    "population": 5.0,
    "area": 5.0,
    "highest_point": 4.0,
    "element_number": 3.0,
    "currency": 2.0,
    "official_language": 2.0,
    "continent_of": 0.5,
}
VALUE_FIX_WEIGHTS = {
    "element_symbol": 5.0,
    "capital": 5.0,
    "population": 5.0,
    "area": 4.0,
    "highest_point": 4.0,
    "currency": 3.0,
    "official_language": 3.0,
    "element_number": 3.0,
    "continent_of": 2.0,
}
V5_WEIGHTS = {
    "element_symbol": 5.0,
    "capital": 5.0,
    "population": 5.0,
    "area": 3.0,
    "highest_point": 4.0,
    "currency": 3.0,
    "official_language": 3.0,
    "element_number": 3.0,
    "continent_of": 1.0,
}
V5_LANG_MULT = {
    "en": {
        "capital": 1.3,
        "population": 1.3,
        "element_symbol": 1.2,
        "area": 1.2,
    },
    "ja": {
        "capital": 1.0,
        "population": 1.0,
        "area": 1.0,
    },
}
V6_EN_REPAIR_WEIGHTS = {
    "official_language": 7.0,
    "highest_point": 7.0,
    "currency": 4.5,
    "element_symbol": 5.0,
    "population": 4.0,
    "capital": 2.5,
    "element_number": 3.0,
    "area": 1.2,
    "continent_of": 0.5,
}
V6_EN_REPAIR_LANG_MULT = {
    "en": {
        "official_language": 2.6,
        "highest_point": 2.4,
        "currency": 1.8,
        "element_symbol": 1.3,
        "area": 0.75,
        "capital": 0.85,
        "continent_of": 0.55,
    },
    "ja": {
        "population": 1.15,
        "capital": 1.15,
        "element_symbol": 1.10,
        "element_number": 1.10,
        "highest_point": 1.10,
        "continent_of": 1.05,
    },
}
V8_BALANCED_REPAIR_WEIGHTS = {
    "official_language": 6.0,
    "highest_point": 6.5,
    "currency": 4.0,
    "element_symbol": 4.5,
    "population": 4.0,
    "capital": 3.0,
    "element_number": 3.0,
    "area": 1.6,
    "continent_of": 0.7,
}
V8_BALANCED_REPAIR_LANG_MULT = {
    "en": {
        "official_language": 2.0,
        "highest_point": 2.1,
        "currency": 1.5,
        "element_symbol": 1.15,
        "population": 0.95,
        "capital": 0.80,
        "area": 0.75,
        "continent_of": 0.60,
    },
    "ja": {
        "population": 1.25,
        "capital": 1.25,
        "element_symbol": 1.20,
        "element_number": 1.20,
        "highest_point": 1.20,
        "continent_of": 1.15,
        "area": 1.10,
        "official_language": 1.05,
    },
}
# V9: tuned for the 3.85B-token checkpoint. Key changes vs V8:
#  - EN keeps the repair boost for the weakest categories.
#  - JA gets a currency/official_language/area boost (V8 under-served them).
#  - JA binding share raised globally: honest-eval showed JA subject-binding
#    at only 10% (model drops the subject from the answer).
V9_FULL_WEIGHTS = {
    "official_language": 6.0,
    "highest_point": 6.0,
    "currency": 4.5,
    "element_symbol": 4.5,
    "population": 4.0,
    "capital": 3.0,
    "element_number": 3.0,
    "area": 1.8,
    "continent_of": 0.7,
}
V9_FULL_LANG_MULT = {
    "en": {
        "official_language": 2.0,
        "highest_point": 2.1,
        "currency": 1.5,
        "element_symbol": 1.15,
        "population": 1.0,
        "capital": 0.85,
        "area": 0.8,
        "continent_of": 0.6,
    },
    "ja": {
        "population": 1.2,
        "capital": 1.2,
        "element_symbol": 1.2,
        "element_number": 1.2,
        "highest_point": 1.15,
        "continent_of": 1.1,
        "area": 1.3,
        "official_language": 1.3,
        "currency": 1.3,
    },
}
STRICT_EN_Q = {
    "capital": [
        "What is the capital of {sub}?",
        "Which city is the capital of {sub}?",
        "Tell me the capital of {sub}.",
        "Do you know the capital of {sub}?",
        "Please tell me the capital city of {sub}.",
        "Can you give the capital of {sub}?",
        "I want to know the capital of {sub}.",
        "What city serves as the capital of {sub}?",
    ],
    "currency": [
        "What is the currency of {sub}?",
        "Which currency is used in {sub}?",
        "Tell me the currency of {sub}.",
        "Do you know the currency of {sub}?",
        "Please tell me the currency used in {sub}.",
        "Can you give the currency of {sub}?",
        "I want to know the currency of {sub}.",
        "What money is used in {sub}?",
    ],
    "official_language": [
        "What is an official language of {sub}?",
        "Which language is official in {sub}?",
        "Tell me an official language of {sub}.",
        "Do you know an official language of {sub}?",
        "Please tell me an official language of {sub}.",
        "Can you give an official language of {sub}?",
        "I want to know an official language of {sub}.",
        "What language is officially used in {sub}?",
    ],
    "population": [
        "What is the population of {sub}?",
        "How large is the population of {sub}?",
        "Tell me the population of {sub}.",
        "Do you know the population of {sub}?",
        "Please tell me the population of {sub}.",
        "Can you give the population of {sub}?",
        "I want to know the population of {sub}.",
        "About how many people live in {sub}?",
    ],
    "highest_point": [
        "What is the highest point of {sub}?",
        "Which place is the highest point of {sub}?",
        "Tell me the highest point of {sub}.",
        "Do you know the highest point of {sub}?",
        "Please tell me the highest point of {sub}.",
        "Can you give the highest point of {sub}?",
        "I want to know the highest point of {sub}.",
        "What is the highest place in {sub}?",
    ],
    "continent_of": [
        "Which continent is {sub} in?",
        "What continent is {sub} in?",
        "Tell me which continent {sub} is in.",
        "Do you know which continent {sub} is in?",
        "Please tell me the continent of {sub}.",
        "Can you give the continent of {sub}?",
        "I want to know which continent {sub} is in.",
        "On which continent is {sub} located?",
    ],
    "area": [
        "What is the area of {sub}?",
        "How large is the area of {sub}?",
        "Tell me the area of {sub}.",
        "Do you know the area of {sub}?",
        "Please tell me the area of {sub}.",
        "Can you give the area of {sub}?",
        "I want to know the area of {sub}.",
        "How many square kilometers is {sub}?",
    ],
    "element_symbol": [
        "What is the chemical symbol of {sub}?",
        "Tell me the chemical symbol of {sub}.",
        "Do you know the chemical symbol of {sub}?",
        "Which chemical symbol represents {sub}?",
        "Please tell me the chemical symbol of {sub}.",
        "Can you give the chemical symbol of {sub}?",
        "I want to know the chemical symbol of {sub}.",
        "What symbol represents the element {sub}?",
    ],
    "element_number": [
        "What is the atomic number of {sub}?",
        "Tell me the atomic number of {sub}.",
        "Do you know the atomic number of {sub}?",
        "Which atomic number does {sub} have?",
        "Please tell me the atomic number of {sub}.",
        "Can you give the atomic number of {sub}?",
        "I want to know the atomic number of {sub}.",
        "What number is assigned to the element {sub}?",
    ],
}
STRICT_JA_Q = {
    "capital": [
        "{sub}の首都はどこですか？",
        "{sub}の首都を教えてください。",
        "{sub}の首都は何ですか？",
        "{sub}の首都をご存知ですか？",
        "{sub}の首都について教えてください。",
        "{sub}の首都を知りたいです。",
        "{sub}の首都を答えてください。",
        "{sub}で首都とされる都市はどこですか？",
    ],
    "currency": [
        "{sub}の通貨は何ですか？",
        "{sub}で使われている通貨は何ですか？",
        "{sub}の通貨を教えてください。",
        "{sub}の通貨をご存知ですか？",
        "{sub}の通貨について教えてください。",
        "{sub}の通貨を知りたいです。",
        "{sub}の通貨を答えてください。",
        "{sub}で使うお金は何ですか？",
    ],
    "official_language": [
        "{sub}の公用語は何ですか？",
        "{sub}で公用語とされている言語は何ですか？",
        "{sub}の公用語を教えてください。",
        "{sub}の公用語をご存知ですか？",
        "{sub}の公用語について教えてください。",
        "{sub}の公用語を知りたいです。",
        "{sub}の公用語を答えてください。",
        "{sub}で公式に使われる言語は何ですか？",
    ],
    "population": [
        "{sub}の人口は何人ですか？",
        "{sub}の人口はどれくらいですか？",
        "{sub}の人口を教えてください。",
        "{sub}の人口をご存知ですか？",
        "{sub}の人口について教えてください。",
        "{sub}の人口を知りたいです。",
        "{sub}の人口を答えてください。",
        "{sub}にはおよそ何人が住んでいますか？",
    ],
    "highest_point": [
        "{sub}で一番高い場所はどこですか？",
        "{sub}の最高地点はどこですか？",
        "{sub}で一番高い場所を教えてください。",
        "{sub}の最高地点をご存知ですか？",
        "{sub}で一番高い場所について教えてください。",
        "{sub}の最高地点を知りたいです。",
        "{sub}の最高地点を答えてください。",
        "{sub}で最も高い地点はどこですか？",
    ],
    "continent_of": [
        "{sub}はどの大陸にありますか？",
        "{sub}がある大陸はどこですか？",
        "{sub}はどの大陸に属しますか？",
        "{sub}の大陸を教えてください。",
        "{sub}の大陸について教えてください。",
        "{sub}が属する大陸を知りたいです。",
        "{sub}がある大陸を答えてください。",
        "{sub}は何大陸にありますか？",
    ],
    "area": [
        "{sub}の面積はどれくらいですか？",
        "{sub}の面積は何平方キロメートルですか？",
        "{sub}の面積を教えてください。",
        "{sub}の面積をご存知ですか？",
        "{sub}の面積について教えてください。",
        "{sub}の面積を知りたいです。",
        "{sub}の面積を答えてください。",
        "{sub}は何平方キロメートルありますか？",
    ],
    "element_symbol": [
        "{sub}の化学記号は何ですか？",
        "{sub}の化学記号を教えてください。",
        "{sub}を表す化学記号は何ですか？",
        "{sub}の化学記号をご存知ですか？",
        "{sub}の化学記号について教えてください。",
        "{sub}の化学記号を知りたいです。",
        "{sub}の化学記号を答えてください。",
        "元素{sub}を表す記号は何ですか？",
    ],
    "element_number": [
        "{sub}の原子番号は何ですか？",
        "{sub}の原子番号を教えてください。",
        "{sub}は原子番号何番ですか？",
        "{sub}の原子番号をご存知ですか？",
        "{sub}の原子番号について教えてください。",
        "{sub}の原子番号を知りたいです。",
        "{sub}の原子番号を答えてください。",
        "元素{sub}に割り当てられた番号は何ですか？",
    ],
}
# Category-specific question seed (used to build the EN question).
CAT_Q_SUBJECT = {
    "capital": "the capital of {sub}",
    "largest_city": "the largest city of {sub}",
    "currency": "the currency of {sub}",
    "official_language": "an official language of {sub}",
    "population": "the population of {sub}",
    "highest_point": "the highest point of {sub}",
    "continent_of": "which continent {sub} is in",
    "head_of_state": "the head of state of {sub}",
    "area": "the area of {sub}",
    "element_symbol": "the chemical symbol of {sub}",
    "element_number": "the atomic number of {sub}",
    "mountain_country": "which country {sub} is in",
    "river_country": "which country {sub} is in",
    "planet_kind": "{sub}",
    "language_country": "which country {sub} is spoken in",
    "person_birthplace": "where {sub} was born",
    "person_occupation": "what {sub} is known as",
}
# JA question templates (generic ask).
JA_Q_TMPLS = [
    "{sub}について教えてください。",
    "{sub}は何ですか？",
    "{sub}について知っていますか？",
    "{sub}についてどう思いますか？",
    "{sub}を教えてください。",
    "{sub}について説明してください。",
    "{sub}はどういうものですか？",
    "{sub}について話してください。",
    "{sub}についてどうですか？",
    "{sub}について言ってください。",
    "{sub}について教えてもらえますか？",
    "{sub}は何でしょうか？",
    "{sub}について詳しく教えてください。",
    "{sub}について知りたいです。",
    "{sub}は正確には何ですか？",
    "{sub}について共有できますか？",
    "{sub}をご存知ですか？",
    "{sub}について何か言えますか？",
    "{sub}について興味があります。",
    "{sub}を教えてもらえませんか？",
    "{sub}についての答えは何ですか？",
    "{sub}を説明してもらえますか？",
    "{sub}について何が知られていますか？",
    "{sub}が何か教えてください。",
    "{sub}についての答えを教えてください。",
    "{sub}を説明してください。",
    "{sub}は何だと思いますか？",
    "{sub}を教えてもらえますか？",
    "{sub}は何ですか、ご存知の範囲で？",
    "{sub}について示してもらえますか？",
]
# JA category-specific question tail (Xの首都は? etc.)
JA_CAT_Q = {
    "capital": "{sub}の首都はどこですか？", "capital0": "{sub}の首都はどこですか？",
    "largest_city": "{sub}で一番大きい都市はどこですか？",
    "currency": "{sub}の通貨は何ですか？",
    "official_language": "{sub}の公用語は何ですか？",
    "population": "{sub}の人口は何人ですか？",
    "highest_point": "{sub}で一番高い場所はどこですか？",
    "continent_of": "{sub}はどの大陸にありますか？",
    "head_of_state": "{sub}の国家元首は誰ですか？",
    "area": "{sub}の面積はどれくらいですか？",
    "element_symbol": "{sub}の化学記号は何ですか？",
    "element_number": "{sub}の原子番号は何ですか？",
    "mountain_country": "{sub}はどの国にありますか？",
    "river_country": "{sub}はどの国にありますか？",
    "planet_kind": "{sub}は何ですか？",
    "language_country": "{sub}はどの国で話されていますか？",
    "person_birthplace": "{sub}はどこで生まれましたか？",
    "person_occupation": "{sub}は何で知られていますか？",
}


def render_fact_templates(cat, sub_en, sub_ja, val_en, val_ja):
    """Expand one (sub,val) bilingual fact into TEMPLATES_PER_FACT variants."""
    ans = CAT_ANSWERS.get(cat)
    if not ans:
        return []
    a_en_t, a_ja_t = ans
    subject_en = CAT_Q_SUBJECT.get(cat, "{sub}").format(sub=sub_en, val=val_en)
    subject_ja = JA_CAT_Q.get(cat, "{sub}の首都は？").format(sub=sub_ja, val=val_ja)
    a_en = a_en_t.format(sub=sub_en, val=val_en)
    a_ja = a_ja_t.format(sub=sub_ja, val=val_ja)
    out = []
    # Build a diverse set of EN question phrasings — all are QUESTIONS built
    # from the category subject phrase (e.g. "the capital of Japan").
    en_phrasings = [t.format(sub=subject_en) for t in EN_Q_TMPLS]
    # Japanese: category-specific question first, then generic JA questions.
    ja_phrasings = [subject_ja] + [t.format(sub=sub_ja) for t in JA_Q_TMPLS]
    for i in range(TEMPLATES_PER_FACT):
        q_en = en_phrasings[i % len(en_phrasings)]
        q_ja = ja_phrasings[i % len(ja_phrasings)]
        out.append({"q_en": q_en, "a_en": a_en, "q_ja": q_ja, "a_ja": a_ja})
    return out

# ── Objective-fact patterns (JA) ──────────────────────────────────────────
JA_OBJ_PATTERNS = [
    r"首都", r"何分", r"何日", r"何秒", r"何ヶ月", r"何個", r"何度",
    r"化学式", r"分子式", r"通貨", r"人口", r"一番高い山", r"一番長い川",
    r"一番大きい", r"一番小さい", r"太陽系", r"惑星", r"地球の衛星",
    r"1キロ", r"1メートル", r"1ダース", r"1年は", r"1週間は", r"1時間は",
    r"沸騰", r"凍り", r"血液", r"大陸", r"一番高いビル", r"一番大きい国",
    r"一番人口が多い国", r"一番大きい海",
]
# Subjective / context / reading-comprehension signals → REJECT (JA)
JA_REJECT = [
    "一番好き", "最適", "最高", "最強", "ベスト", "おすすめ", "なぜ", "何故",
    "理由", "比較", "違い", "識別", "この", "その", "踏まえ", "パラグラフ",
    "グループ", "関係", "当てはまらない", "どのように", "歴史上最高",
    "ついて説明", "まとめて", "挙げて", "それぞれ", "教えてくださいください",
    "Sで始まる", "Cで始まる", "新行政首都", "何という", "どのくらい",
    # extraction / village-specific / obscure signals → REJECT
    "抽出せよ", "抽出して", "男性人口", "女性人口", "準拠", "暗号通貨", "村は",
    "村の人口", "の人口は", "人口を抽出", "何というスコア", "優勝しているのでしょうか",
]

# ── Objective-fact patterns (EN) ──────────────────────────────────────────
EN_OBJ_PATTERNS = [
    r"capital of", r"population of", r"tallest mountain", r"largest",
    r"longest", r"currency of", r"how many", r"smallest planet",
    r"largest planet", r"chemical formula", r"boiling point", r"freezing point",
    r"how many days", r"how many seconds", r"how many minutes",
    r"how many months", r"largest ocean", r"largest country",
    r"most populous country", r"tallest building", r"largest animal",
]
# Subjective / list-selection / math -heavy signals → REJECT (EN)
EN_REJECT = [
    "prime factor", "longest word in the given", "sort", "rank",
    "list of", "respectively", "find the", "given a list", "name a country",
    "which country is the second", "producer", "identify", "comparison",
    "difference between", "explain why", "why is", "best", "favorite",
    "recommend", "largest prime",
    # non-fact / task-style templates → REJECT
    "remove the words", "put the following words", "in order", "true or false",
    "guess ", "answer a question", "answer the following", "answer the given",
    "tell me the name", "state the capital", "construct a question",
    "fill in the blank", "choose the correct", "output the name",
    "a country with a population", "match the following", "reorder",
    "shortest to longest", "which of the following",
]


def has_ja(s):
    return any('\u3040' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f'
               or '\uff00' <= c <= '\uffef' for c in s)


def has_lat(s):
    return bool(re.search(r'[A-Za-z]{2,}', s))


def clean_question(u):
    """Normalize stray prefixes like 'Question:' / 'Answer'."""
    u = u.strip()
    u = re.sub(r'^(Question\s*:\s*|Answer\s*:\s*|Find the name of\s*|Name\s*)', '', u, flags=re.I)
    return u.strip()


def has_qid(*parts):
    return any(re.search(r'\bQ\d{4,}\b', p or "") for p in parts)


def clean_numeric_value(v):
    v = v.strip().replace(",", "")
    try:
        x = float(v)
    except ValueError:
        return v.strip()
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}".rstrip("0").rstrip(".")


def load_fact_overrides():
    base = {
        "drop_facts": [],
        "subject_overrides": [],
        "value_overrides": [],
        "aliases": [],
    }
    if not FACT_OVERRIDES.exists():
        return base
    try:
        data = json.loads(FACT_OVERRIDES.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to load fact overrides: {FACT_OVERRIDES}: {exc}") from exc
    for key, default in base.items():
        value = data.get(key, default)
        data[key] = value if isinstance(value, list) else default
    return data


def override_entry_matches(entry, cat, sub_en, sub_ja, val_en, val_ja):
    # comment-only entries (or entries with no matchable field) must never
    # match anything — otherwise they silently drop the entire dataset.
    matchable = {
        "category": cat,
        "subject_en": sub_en,
        "subject_ja": sub_ja,
        "value_en": val_en,
        "value_ja": val_ja,
    }
    has_constraint = any(
        entry.get(key) is not None and str(entry[key]).strip() != ""
        for key in matchable
    )
    if not has_constraint:
        return False
    for key, current in matchable.items():
        expected = entry.get(key)
        if expected is None or str(expected).strip() == "":
            continue
        if key in {"category", "subject_en", "value_en"}:
            if str(expected).casefold() != str(current).casefold():
                return False
        else:
            if str(expected).strip() != str(current).strip():
                return False
    return True


def apply_fact_overrides(cat, sub_en, sub_ja, val_en, val_ja, overrides, sub_id="", val_id=""):
    for entry in overrides["drop_facts"]:
        if override_entry_matches(entry, cat, sub_en, sub_ja, val_en, val_ja):
            return None

    for entry in overrides["subject_overrides"]:
        if override_entry_matches(entry, cat, sub_en, sub_ja, val_en, val_ja):
            sub_en = (entry.get("new_subject_en") or sub_en).strip()
            sub_ja = (entry.get("new_subject_ja") or sub_ja).strip()
            sub_id = (entry.get("new_subject_id") or sub_id).strip()

    for entry in overrides["value_overrides"]:
        if override_entry_matches(entry, cat, sub_en, sub_ja, val_en, val_ja):
            val_en = (entry.get("new_value_en") or val_en).strip()
            val_ja = (entry.get("new_value_ja") or val_ja).strip()
            val_id = (entry.get("new_value_id") or val_id).strip()

    aliases_en, aliases_ja = [], []
    for entry in overrides["aliases"]:
        if override_entry_matches(entry, cat, sub_en, sub_ja, val_en, val_ja):
            aliases_en.extend(str(alias).strip() for alias in entry.get("aliases_en", []) if str(alias).strip())
            aliases_ja.extend(str(alias).strip() for alias in entry.get("aliases_ja", []) if str(alias).strip())

    return {
        "category": cat,
        "subject_en": sub_en,
        "subject_ja": sub_ja,
        "value_en": val_en,
        "value_ja": val_ja,
        "subject_id": sub_id,
        "value_id": val_id,
        "aliases_en": sorted(set(aliases_en)),
        "aliases_ja": sorted(set(aliases_ja)),
    }


def invalid_strict_fact(cat, sub_en, sub_ja, val_en, val_ja):
    if cat in STRICT_DROP_CATEGORIES or cat not in STRICT_CATEGORIES:
        return True
    if not sub_en.strip() or not sub_ja.strip() or not val_en.strip() or not val_ja.strip():
        return True
    if has_qid(sub_en, sub_ja, val_en, val_ja):
        return True
    bad_values = {"St", "Washington, D", "Donald J", "Surangel Whipps Jr"}
    if val_en.strip() in bad_values:
        return True
    if "�" in (sub_en + sub_ja + val_en + val_ja):
        return True
    # Country/entity facts should have readable labels. Element symbols are the
    # only category where a one- or two-letter value is expected.
    if cat != "element_symbol" and len(val_en.strip()) <= 2:
        return True
    return False


def render_strict_fact(fact):
    cat = fact["category"]
    sub_en = fact["subject_en"]
    sub_ja = fact["subject_ja"]
    val_en = fact["value_en"]
    val_ja = fact["value_ja"]
    sub_id = fact.get("subject_id", "")
    val_id = fact.get("value_id", "")
    ans = CAT_ANSWERS.get(cat)
    if not ans:
        return []
    a_en_t, a_ja_t = ans
    if cat in {"area", "population", "element_number"}:
        val_en = clean_numeric_value(val_en)
        val_ja = val_en
        val_id = val_en
    a_en = a_en_t.format(sub=sub_en, val=val_en)
    a_ja = a_ja_t.format(sub=sub_ja, val=val_ja)
    out = []
    fact_id = f"{cat}::{sub_en.casefold()}"
    base_fields = {
        "category": cat,
        "subject_en": sub_en, "subject_ja": sub_ja,
        "value_en": val_en, "value_ja": val_ja,
        "subject_id": sub_id, "value_id": val_id,
        "aliases_en": fact.get("aliases_en", []), "aliases_ja": fact.get("aliases_ja", []),
        "source": "wikidata_strict", "fact_id": fact_id,
    }
    for i, q in enumerate(STRICT_EN_Q[cat]):
        row = dict(base_fields)
        row.update({
            "text": f"<|system|>{SYS_EN}<|user|>{q.format(sub=sub_en)}<|assistant|>{a_en}<|end|>",
            "lang": "en", "task": "fact_qa", "template_id": f"{cat}_en_{i:02d}",
        })
        out.append(row)
    for i, q in enumerate(STRICT_JA_Q[cat]):
        row = dict(base_fields)
        row.update({
            "text": f"<|system|>{SYS_JA}<|user|>{q.format(sub=sub_ja)}<|assistant|>{a_ja}<|end|>",
            "lang": "ja", "task": "fact_qa", "template_id": f"{cat}_ja_{i:02d}",
        })
        out.append(row)
    if sub_id and val_id and cat in CAT_ANSWERS_ID:
        a_id = CAT_ANSWERS_ID[cat].format(sub=sub_id, val=val_id)
        for i, q in enumerate(STRICT_ID_Q[cat]):
            row = dict(base_fields)
            row.update({
                "text": f"<|system|>{SYS_ID}<|user|>{q.format(sub=sub_id)}<|assistant|>{a_id}<|end|>",
                "lang": "id", "task": "fact_qa", "template_id": f"{cat}_id_{i:02d}",
            })
            out.append(row)
    return out


def render_fact_variant(fact, lang, variant="normal", subject_facts=None):
    cat = fact["category"]
    sub_en = fact["subject_en"]
    sub_ja = fact["subject_ja"]
    val_en = fact["value_en"]
    val_ja = fact["value_ja"]
    sub_id = fact.get("subject_id", "")
    val_id = fact.get("value_id", "")
    if cat in {"area", "population", "element_number"}:
        val_en = clean_numeric_value(val_en)
        val_ja = val_en
        val_id = val_en

    if lang == "id":
        # Indonesian render: requires trilingual labels. Fall back to EN when
        # the fact has no ID labels (bilingual-only facts).
        if not (sub_id and val_id and cat in CAT_ANSWERS_ID):
            return render_fact_variant(fact, "en", variant, subject_facts)
        ans_id_t = CAT_ANSWERS_ID_BINDING[cat] if variant in {"binding", "contrast"} else CAT_ANSWERS_ID[cat]
        if variant == "binding":
            q = BINDING_Q_ID[cat]
        elif variant == "contrast":
            q = make_contrast_question(fact, lang, subject_facts)
            if q is None:
                q = BINDING_Q_ID[cat]
        elif variant == "anti" and cat in ANTI_CONFUSION_Q_ID:
            q = RNG.choice(ANTI_CONFUSION_Q_ID[cat])
        else:
            q = RNG.choice(STRICT_ID_Q[cat])
        text = f"<|system|>{SYS_ID}<|user|>{q.format(sub=sub_id)}<|assistant|>{ans_id_t.format(sub=sub_id, val=val_id)}<|end|>"
        return {
            "text": text,
            "lang": "id",
            "task": "fact_qa",
            "category": cat,
            "subject_en": sub_en,
            "subject_ja": sub_ja,
            "value_en": val_en,
            "value_ja": val_ja,
            "subject_id": sub_id,
            "value_id": val_id,
            "aliases_en": fact.get("aliases_en", []),
            "aliases_ja": fact.get("aliases_ja", []),
            "source": "wikidata_strict",
            "fact_id": f"{cat}::{sub_en.casefold()}",
            "template_id": f"{cat}_{lang}_{variant}",
            "variant": variant,
        }

    ans_map = CAT_ANSWERS_BINDING if variant in {"binding", "contrast"} else CAT_ANSWERS
    a_en_t, a_ja_t = ans_map[cat]
    if variant == "binding":
        q = BINDING_Q[cat][0 if lang == "en" else 1]
    elif variant == "contrast":
        q = make_contrast_question(fact, lang, subject_facts)
        if q is None:
            q = BINDING_Q[cat][0 if lang == "en" else 1]
    elif variant == "anti" and cat in ANTI_CONFUSION_Q:
        q = RNG.choice(ANTI_CONFUSION_Q[cat][lang])
    elif lang == "en":
        q = RNG.choice(STRICT_EN_Q[cat])
    else:
        q = RNG.choice(STRICT_JA_Q[cat])

    if lang == "en":
        text = f"<|system|>{SYS_EN}<|user|>{q.format(sub=sub_en)}<|assistant|>{a_en_t.format(sub=sub_en, val=val_en)}<|end|>"
    else:
        text = f"<|system|>{SYS_JA}<|user|>{q.format(sub=sub_ja)}<|assistant|>{a_ja_t.format(sub=sub_ja, val=val_ja)}<|end|>"
    return {
        "text": text,
        "lang": lang,
        "task": "fact_qa",
        "category": cat,
        "subject_en": sub_en,
        "subject_ja": sub_ja,
        "value_en": val_en,
        "value_ja": val_ja,
        "subject_id": sub_id,
        "value_id": val_id,
        "aliases_en": fact.get("aliases_en", []),
        "aliases_ja": fact.get("aliases_ja", []),
        "source": "wikidata_strict",
        "fact_id": f"{cat}::{sub_en.casefold()}",
        "template_id": f"{cat}_{lang}_{variant}",
        "variant": variant,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_strict_facts():
    """Return clean, unambiguous fact tuples and source stats.

    Prefers the trilingual TSV (7 cols: cat, sub_en, sub_ja, sub_id, val_en,
    val_ja, val_id) when present; falls back to the bilingual 5-col TSV.
    """
    overrides = load_fact_overrides()
    raw = []
    override_drops = 0
    tri_path = ROOT / "data" / "wikidata_facts_trilingual.tsv"
    wd_paths = []
    if tri_path.exists():
        wd_paths.append(tri_path)
    wd_paths.append(ROOT / "data" / "wikidata_facts_all.tsv")
    seen_tri_keys = set()
    for wd_path in wd_paths:
        if not wd_path.exists():
            continue
        trilingual = wd_path == tri_path
        for line in open(wd_path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if trilingual:
                if len(p) < 7:
                    continue
                cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id = [x.strip() for x in p[:7]]
            else:
                if len(p) < 5:
                    continue
                cat, sub_en, sub_ja, val_en, val_ja = [x.strip() for x in p[:5]]
                sub_id, val_id = "", ""
            # Skip bilingual rows whose fact is already loaded from the
            # trilingual TSV (trilingual is preferred/fresher).
            tri_key = (cat, sub_en.casefold())
            if trilingual:
                seen_tri_keys.add(tri_key)
            elif tri_key in seen_tri_keys:
                continue
            if invalid_strict_fact(cat, sub_en, sub_ja, val_en, val_ja):
                continue
            if cat in {"area", "population", "element_number"}:
                val_en = clean_numeric_value(val_en)
                val_ja = val_en
                val_id = val_en
            fact = apply_fact_overrides(cat, sub_en, sub_ja, val_en, val_ja, overrides,
                                        sub_id=sub_id, val_id=val_id)
            if fact is None:
                override_drops += 1
                continue
            raw.append(fact)

    # Keep only unambiguous single-answer facts per category+subject. This is
    # intentionally strict: multi-value official languages/currencies are valid
    # world facts, but they produce unstable single-answer SFT targets.
    by_key = {}
    conflicts = set()
    for fact in raw:
        cat = fact["category"]
        sub_en = fact["subject_en"]
        sub_ja = fact["subject_ja"]
        val_en = fact["value_en"]
        val_ja = fact["value_ja"]
        key = (cat, sub_en.casefold(), sub_ja)
        val = (val_en.casefold(), val_ja)
        if key in by_key and by_key[key][0] != val:
            conflicts.add(key)
        elif key in by_key:
            existing = by_key[key][1]
            existing["aliases_en"] = sorted(set(existing.get("aliases_en", [])) | set(fact.get("aliases_en", [])))
            existing["aliases_ja"] = sorted(set(existing.get("aliases_ja", [])) | set(fact.get("aliases_ja", [])))
        else:
            by_key[key] = (val, fact)
    facts = [item for key, (_, item) in by_key.items() if key not in conflicts]
    return facts, {"raw_kept": len(raw), "conflict_keys": len(conflicts), "override_drops": override_drops}


def build_strict_dataset():
    """Build a high-precision facts dataset with no generic QA prompts."""
    facts, src_stats = load_strict_facts()

    # Use all unambiguous facts. The raw strict categories are already modest;
    # dropping down to the smallest category wastes clean supervision.
    by_cat = {}
    for fact in facts:
        by_cat.setdefault(fact["category"], []).append(fact)
    for rows in by_cat.values():
        RNG.shuffle(rows)
    balanced_facts = []
    for cat in sorted(by_cat):
        balanced_facts.extend(by_cat[cat])
    RNG.shuffle(balanced_facts)

    split_keys = list({(fact["category"], fact["subject_en"].casefold()) for fact in balanced_facts})
    RNG.shuffle(split_keys)
    n_eval = max(1, int(len(split_keys) * 0.10))
    eval_keys = set(split_keys[:n_eval])

    train, eval_rows = [], []
    prompt_to_answer = {}
    dropped_prompt_conflicts = 0
    for fact in balanced_facts:
        cat = fact["category"]
        sub_en = fact["subject_en"]
        target = eval_rows if (cat, sub_en.casefold()) in eval_keys else train
        for row in render_strict_fact(fact):
            q = row["text"].split("<|user|>")[1].split("<|assistant|>")[0]
            a = row["text"].split("<|assistant|>")[1].split("<|end|>")[0]
            pk = (row["lang"], q)
            if pk in prompt_to_answer and prompt_to_answer[pk] != a:
                dropped_prompt_conflicts += 1
                continue
            prompt_to_answer[pk] = a
            target.append(row)

    RNG.shuffle(train)
    RNG.shuffle(eval_rows)
    write_jsonl(OUT_FILE, train)
    write_jsonl(OUT_DIR / STRICT_EVAL_FILE, eval_rows)

    from collections import Counter
    print(f"  strict raw kept: {src_stats['raw_kept']}")
    print(f"  strict conflicts dropped: {src_stats['conflict_keys']} fact keys")
    print(f"  strict override drops: {src_stats['override_drops']}")
    print(f"  strict categories: { {k: len(v) for k, v in sorted(by_cat.items())} }")
    print(f"  strict prompt conflicts dropped: {dropped_prompt_conflicts}")
    print(f"\n{'='*56}")
    print(f"Final strict facts dataset: {len(train)} train + {len(eval_rows)} eval rows")
    print(f"  train langs: {dict(Counter(r['lang'] for r in train))}")
    print(f"  eval langs: {dict(Counter(r['lang'] for r in eval_rows))}")
    print(f"  train categories: {dict(Counter(r['category'] for r in train))}")
    print(f"  Output: {OUT_FILE}")
    print(f"  Eval:   {OUT_DIR / STRICT_EVAL_FILE}")


def weighted_choice(items, weights):
    total = sum(weights)
    r = RNG.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def subject_fact_index(facts):
    by_subject = {}
    for fact in facts:
        key = fact["subject_en"].casefold()
        by_subject.setdefault(key, {})[fact["category"]] = fact
    return by_subject


def distractor_phrase(category, fact, lang):
    value = fact["value_en"] if lang == "en" else fact["value_ja"]
    if lang == "id":
        value = fact.get("value_id") or fact["value_en"]
    if category in {"area", "population", "element_number"}:
        value = clean_numeric_value(value)
    if lang == "en":
        if category == "area":
            return f"the area {value} square kilometers"
        return f"{REQUEST_LABELS[category][0]} {value}"
    if lang == "id":
        if category == "area":
            return f"luas {value} kilometer persegi"
        if category == "population":
            return f"populasi {value}"
        return f"{REQUEST_LABELS_ID[category]} {value}"
    if category == "area":
        return f"面積の{value}平方キロメートル"
    if category == "population":
        return f"人口の{value}人"
    return f"{REQUEST_LABELS[category][1]}の{value}"


def make_contrast_question(fact, lang, subject_facts):
    if not subject_facts:
        return None
    cat = fact["category"]
    distractors = []
    for distractor_cat in CONTRAST_DISTRACTORS.get(cat, []):
        distractor_fact = subject_facts.get(distractor_cat)
        if distractor_fact and distractor_cat != cat:
            distractors.append(distractor_phrase(distractor_cat, distractor_fact, lang))
    distractors = distractors[:2]
    if not distractors:
        return None
    if lang == "id":
        requested = REQUEST_LABELS_ID[cat]
        subject = fact.get("subject_id") or fact["subject_en"]
        if len(distractors) == 1:
            return f"Untuk {subject}, apa {requested}nya, bukan {distractors[0]}? Jawab dengan {requested} saja."
        return f"Untuk {subject}, apa {requested}nya, bukan {distractors[0]} atau {distractors[1]}? Jawab dengan {requested} saja."
    requested = REQUEST_LABELS[cat][0 if lang == "en" else 1]
    subject = fact["subject_en"] if lang == "en" else fact["subject_ja"]
    if lang == "en":
        if len(distractors) == 1:
            return f"For {subject}, what is {requested}, not {distractors[0]}? Answer with {requested} only."
        return f"For {subject}, what is {requested}, not {distractors[0]} or {distractors[1]}? Answer with {requested} only."
    if len(distractors) == 1:
        return f"{subject}について、{requested}を答えてください。{distractors[0]}ではなく、{requested}だけを答えてください。"
    return f"{subject}について、{requested}を答えてください。{distractors[0]}や{distractors[1]}ではなく、{requested}だけを答えてください。"


def variant_shares(profile, lang, cat):
    if profile == "v6_en_repair":
        if lang == "en":
            if cat in {"official_language", "highest_point"}:
                return 0.60, 0.25, 0.0
            if cat == "currency":
                return 0.45, 0.25, 0.0
            if cat in {"area", "capital", "continent_of"}:
                return 0.10, 0.15, 0.0
            return 0.22, 0.20, 0.0
        if cat in {"population", "capital", "element_symbol", "element_number", "continent_of", "highest_point"}:
            return 0.18, 0.22, 0.0
        return 0.12, 0.18, 0.0
    if profile == "v8_balanced_repair":
        if lang == "en":
            if cat in {"official_language", "highest_point"}:
                return 0.24, 0.16, 0.44
            if cat == "currency":
                return 0.20, 0.15, 0.24
            if cat in {"area", "capital", "continent_of"}:
                return 0.08, 0.12, 0.08
            return 0.14, 0.18, 0.10
        if cat in {"population", "capital", "element_symbol", "element_number", "continent_of", "highest_point"}:
            return 0.10, 0.26, 0.08
        return 0.08, 0.18, 0.04
    if profile == "v9_full":
        # EN: same repair emphasis as V8 (weakest categories get heavy
        # anti+contrast exposure).
        if lang == "en":
            if cat in {"official_language", "highest_point"}:
                return 0.24, 0.16, 0.40
            if cat == "currency":
                return 0.20, 0.15, 0.24
            if cat in {"area", "capital", "continent_of"}:
                return 0.08, 0.14, 0.10
            return 0.14, 0.20, 0.12
        # JA: raise binding everywhere to fix the 10% subject-binding rate;
        # keep modest contrast so attribute discrimination is maintained.
        if cat in {"population", "capital", "element_symbol", "element_number", "continent_of", "highest_point"}:
            return 0.08, 0.40, 0.10
        return 0.08, 0.34, 0.06
    if profile == "v10_trilingual":
        # EN: keep the V9 repair emphasis.
        if lang == "en":
            if cat in {"official_language", "highest_point"}:
                return 0.24, 0.16, 0.40
            if cat == "currency":
                return 0.20, 0.15, 0.24
            if cat in {"area", "capital", "continent_of"}:
                return 0.08, 0.14, 0.10
            return 0.14, 0.20, 0.12
        # JA: same binding-heavy fix as V9.
        if lang == "ja":
            if cat in {"population", "capital", "element_symbol", "element_number", "continent_of", "highest_point"}:
                return 0.08, 0.40, 0.10
            return 0.08, 0.34, 0.06
        # ID: new language — binding-heavy (subject must appear, mirroring the
        # JA subject-binding lesson) + moderate contrast for discrimination.
        if cat in {"official_language", "highest_point"}:
            return 0.15, 0.35, 0.20
        if cat == "currency":
            return 0.15, 0.30, 0.15
        return 0.10, 0.35, 0.10
    if profile == "v5":
        return 0.15, 0.20, 0.0
    if profile == "value_fix":
        return 0.0, 0.30, 0.0
    if profile == "hard_en":
        return (0.55, 0.45, 0.0) if lang == "en" else (0.0, 0.0, 0.0)
    if profile == "v4_en":
        return (0.40, 0.25, 0.0) if lang == "en" else (0.20, 0.15, 0.0)
    if profile == "v3_en":
        return (0.30, 0.25, 0.0) if lang == "en" else (0.20, 0.15, 0.0)
    return 0.20, 0.15, 0.0


def build_oversampled_dataset(target_rows=100000, en_share=0.58, eval_rows=4000, profile="v1", id_share=0.5):
    """Build a large strict fact dataset by weighted oversampling.

    Rows may repeat exact prompts; this is intentional SFT weighting. Held-out
    eval is not oversampled and is split by fact_id so train never sees the same
    category+subject facts.
    """
    facts, src_stats = load_strict_facts()
    if not facts:
        raise RuntimeError("No strict facts available")

    split_keys = list({(fact["category"], fact["subject_en"].casefold()) for fact in facts})
    RNG.shuffle(split_keys)
    # Keep eval bounded and fact-key based. Each held-out fact renders 16 rows
    # with the current template set (8 EN + 8 JA).
    n_eval_keys = max(1, min(len(split_keys) // 5, max(1, eval_rows // 16)))
    eval_keys = set(split_keys[:n_eval_keys])
    train_facts = [fact for fact in facts if (fact["category"], fact["subject_en"].casefold()) not in eval_keys]
    eval_facts = [fact for fact in facts if (fact["category"], fact["subject_en"].casefold()) in eval_keys]
    by_subject = subject_fact_index(train_facts)

    by_cat = {}
    for fact in train_facts:
        by_cat.setdefault(fact["category"], []).append(fact)
    categories = sorted(by_cat)
    if profile == "v10_trilingual":
        base_weights = V9_FULL_WEIGHTS
        lang_mult = V9_FULL_LANG_MULT
    elif profile == "v8_balanced_repair":
        base_weights = V8_BALANCED_REPAIR_WEIGHTS
        lang_mult = V8_BALANCED_REPAIR_LANG_MULT
    elif profile == "v9_full":
        base_weights = V9_FULL_WEIGHTS
        lang_mult = V9_FULL_LANG_MULT
    elif profile == "v6_en_repair":
        base_weights = V6_EN_REPAIR_WEIGHTS
        lang_mult = V6_EN_REPAIR_LANG_MULT
    elif profile == "v3_en":
        base_weights = OVERSAMPLE_WEIGHTS_V3_EN
        lang_mult = OVERSAMPLE_LANG_MULT_V3_EN
    elif profile == "v4_en":
        base_weights = OVERSAMPLE_WEIGHTS_V4_EN
        lang_mult = OVERSAMPLE_LANG_MULT_V4_EN
    elif profile == "hard_en":
        base_weights = HARD_EN_WEIGHTS
        lang_mult = {"en": {}, "ja": {}}
    elif profile == "value_fix":
        base_weights = VALUE_FIX_WEIGHTS
        lang_mult = {"en": {}, "ja": {}}
    elif profile == "v5":
        base_weights = V5_WEIGHTS
        lang_mult = V5_LANG_MULT
    elif profile == "v2":
        base_weights = OVERSAMPLE_WEIGHTS_V2
        lang_mult = OVERSAMPLE_LANG_MULT_V2
    else:
        base_weights = OVERSAMPLE_WEIGHTS
        lang_mult = {"en": {}, "ja": {}}

    trilingual = profile == "v10_trilingual"
    if trilingual:
        # en_share applies to EN; the remainder splits JA/ID via id_share
        # (default 0.5 → half of the non-EN budget).
        target_en = int(target_rows * en_share)
        target_id = int((target_rows - target_en) * id_share)
        target_ja = target_rows - target_en - target_id
        produced = {"en": 0, "ja": 0, "id": 0}
    else:
        target_en = int(target_rows * en_share)
        target_ja = target_rows - target_en
        produced = {"en": 0, "ja": 0}
    train = []

    def pick_lang():
        if trilingual:
            remaining = []
            if produced["en"] < target_en:
                remaining.append(("en", target_en - produced["en"]))
            if produced["ja"] < target_ja:
                remaining.append(("ja", target_ja - produced["ja"]))
            if produced["id"] < target_id:
                remaining.append(("id", target_id - produced["id"]))
            if not remaining:
                return None
            total_left = sum(n for _, n in remaining)
            r = RNG.random() * total_left
            acc = 0
            for lname, n in remaining:
                acc += n
                if r <= acc:
                    return lname
            return remaining[-1][0]
        return "en" if produced["en"] < target_en and (produced["ja"] >= target_ja or RNG.random() < en_share) else "ja"

    while True:
        lang = pick_lang()
        if lang is None:
            break
        cat_weights = [base_weights.get(cat, 1.0) * lang_mult.get(lang, {}).get(cat, 1.0) for cat in categories]
        cat = weighted_choice(categories, cat_weights)
        if trilingual and lang == "id":
            # Only pick facts that have Indonesian labels.
            id_pool = [f for f in by_cat[cat] if f.get("subject_id") and f.get("value_id")]
            if not id_pool:
                id_pool = by_cat[cat]
            fact = RNG.choice(id_pool)
        else:
            fact = RNG.choice(by_cat[cat])
        if profile in {"v2", "v3_en", "v4_en", "hard_en", "value_fix", "v5", "v6_en_repair", "v8_balanced_repair", "v9_full", "v10_trilingual"}:
            p = RNG.random()
            anti_share, binding_share, contrast_share = variant_shares(profile, lang, cat)
            if p < anti_share and cat in ANTI_CONFUSION_Q:
                variant = "anti"
            elif p < anti_share + binding_share:
                variant = "binding"
            elif p < anti_share + binding_share + contrast_share:
                variant = "contrast"
            else:
                variant = "normal"
            row = render_fact_variant(
                fact,
                lang=lang,
                variant=variant,
                subject_facts=by_subject.get(fact["subject_en"].casefold()),
            )
        else:
            rendered = [r for r in render_strict_fact(fact) if r["lang"] == lang]
            row = dict(RNG.choice(rendered))
        row["sample_weighted"] = True
        train.append(row)
        produced[lang] += 1

    eval_out = []
    for fact in eval_facts:
        eval_out.extend(render_strict_fact(fact))

    RNG.shuffle(train)
    RNG.shuffle(eval_out)
    write_jsonl(OUT_FILE, train)
    write_jsonl(OUT_DIR / STRICT_EVAL_FILE, eval_out)

    from collections import Counter
    print(f"  oversample raw kept: {src_stats['raw_kept']}")
    print(f"  oversample conflicts dropped: {src_stats['conflict_keys']} fact keys")
    print(f"  oversample override drops: {src_stats['override_drops']}")
    print(f"  profile: {profile}")
    print(f"  target rows: {target_rows} (EN share {en_share:.2f})")
    print(f"  train fact keys: {len(train_facts)}")
    print(f"  eval fact keys: {len(eval_facts)}")
    print(f"\n{'='*56}")
    print(f"Final oversampled facts dataset: {len(train)} train + {len(eval_out)} eval rows")
    print(f"  train langs: {dict(Counter(r['lang'] for r in train))}")
    print(f"  eval langs: {dict(Counter(r['lang'] for r in eval_out))}")
    print(f"  train categories: {dict(Counter(r['category'] for r in train))}")
    print(f"  Output: {OUT_FILE}")
    print(f"  Eval:   {OUT_DIR / STRICT_EVAL_FILE}")


def main():
    import argparse
    global MODE, OUT_DIR, OUT_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "ja_only", "en_dominant", "strict", "oversample"], default="all",
                    help="all=50/50 combined, ja_only=100% JA, en_dominant=~80% EN, strict/oversample=clean facts")
    ap.add_argument("--target_rows", type=int, default=100000,
                    help="train rows for --mode oversample")
    ap.add_argument("--en_share", type=float, default=0.58,
                    help="EN row share for --mode oversample")
    ap.add_argument("--eval_rows", type=int, default=4000,
                    help="approx eval rows for --mode oversample")
    ap.add_argument("--out_dir", default=None,
                    help="override output dir for strict/oversample modes")
    ap.add_argument("--profile", choices=["v1", "v2", "v3_en", "v4_en", "hard_en", "value_fix", "v5", "v6_en_repair", "v8_balanced_repair", "v9_full", "v10_trilingual"], default="v1",
                    help="oversampling profile")
    ap.add_argument("--id_share", type=float, default=0.5,
                    help="share of the non-EN budget given to ID rows (v10_trilingual only)")
    args = ap.parse_args()
    MODE = args.mode
    if args.out_dir:
        OUT_DIR = ROOT / args.out_dir
    elif MODE == "strict":
        OUT_DIR = ROOT / "sft_data" / "sft_facts_strict"
    elif MODE == "oversample":
        suffix = f"_{args.profile}" if args.profile != "v1" else ""
        if args.profile == "hard_en":
            suffix = "_hard_en"
        elif args.profile == "value_fix":
            suffix = "_value_fix"
        elif args.profile == "v5":
            suffix = "_v5"
        elif args.profile == "v6_en_repair":
            suffix = "_v6_en_repair"
        elif args.profile == "v8_balanced_repair":
            suffix = "_v8_balanced_repair"
        elif args.profile == "v9_full":
            suffix = "_v9_full"
        elif args.profile == "v10_trilingual":
            suffix = "_v10_trilingual"
        OUT_DIR = ROOT / "sft_data" / f"sft_facts_oversampled_{args.target_rows//1000}k{suffix}"
    elif MODE == "ja_only":
        OUT_DIR = ROOT / "sft_data" / "sft_facts_ja"
    elif MODE == "en_dominant":
        OUT_DIR = ROOT / "sft_data" / "sft_facts_en"
    else:
        OUT_DIR = ROOT / "sft_data" / "sft_facts_large"
    OUT_FILE = OUT_DIR / "train.jsonl"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODE == "strict":
        build_strict_dataset()
        return
    if MODE == "oversample":
        build_oversampled_dataset(args.target_rows, args.en_share, args.eval_rows, args.profile, args.id_share)
        return

    rows = []

    # ── 0. Wikidata facts (massal, bilingual, akurat) + template augment ──
    # TSV: CATEGORY \t sub_en \t sub_ja \t val_en \t val_ja
    wd_paths = [ROOT / "data" / "wikidata_facts.tsv",
                ROOT / "data" / "wd_light.tsv"]
    wd_en, wd_ja = 0, 0
    seen_pairs = set()
    for wd_path in wd_paths:
        if not wd_path.exists():
            continue
        for line in open(wd_path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            cat, sub_en, sub_ja, val_en, val_ja = p[:5]
            # Reject QID-noise.
            if re.search(r'\bQ\d{4,}\b', val_en) or re.search(r'\bQ\d{4,}\b', val_ja):
                continue
            if not sub_en.strip() or not val_en.strip() or not sub_ja.strip() or not val_ja.strip():
                continue
            key = (cat, sub_en.strip().lower(), val_en.strip().lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            for t in render_fact_templates(cat, sub_en.strip(), sub_ja.strip(),
                                           val_en.strip(), val_ja.strip()):
                if not t:
                    continue
                rows.append({"text": f"<|system|>{SYS_EN}<|user|>{t['q_en']}<|assistant|>{t['a_en']}<|end|>",
                             "lang": "en", "task": "instruction"})
                rows.append({"text": f"<|system|>{SYS_JA}<|user|>{t['q_ja']}<|assistant|>{t['a_ja']}<|end|>",
                             "lang": "ja", "task": "instruction"})
                wd_en += 1; wd_ja += 1
    print(f"  wikidata (template-augmented) facts: {wd_en} EN + {wd_ja} JA")

    # ── 1. Curated facts (gold) ────────────────────────────────────────────
    curated_en, curated_ja = 0, 0
    if (SFT_CLEAN / "sft_facts_curated.jsonl").exists():
        for line in open(SFT_CLEAN / "sft_facts_curated.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("lang") == "en":
                rows.append(d); curated_en += 1
            elif d.get("lang") == "ja":
                rows.append(d); curated_ja += 1
    print(f"  curated: {curated_en} EN + {curated_ja} JA")

    # ── 2. Strict-filtered objective JA facts ──────────────────────────────
    ja_added = 0
    if (SFT_CLEAN / "sft_ja.jsonl").exists():
        for line in open(SFT_CLEAN / "sft_ja.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("lang") != "ja":
                continue
            t = d.get("text", "")
            if "<|user|>" not in t or "<|assistant|>" not in t:
                continue
            u = t.split("<|user|>")[1].split("<|assistant|>")[0].strip()
            a = t.split("<|assistant|>")[1].split("<|end|>")[0].strip()
            if len(u) > 80 or len(a) > 70 or not has_ja(a):
                continue
            if not any(re.search(p, u) for p in JA_OBJ_PATTERNS):
                continue
            if any(k in u for k in JA_REJECT):
                continue
            if not a.rstrip().endswith(("です", "です。", "ます", "ます。")):
                continue
            rows.append({"text": f"<|system|>{SYS_JA}<|user|>{u}<|assistant|>{a}<|end|>",
                         "lang": "ja", "task": "instruction"})
            ja_added += 1
    print(f"  sft_ja strict-filtered JA facts: {ja_added}")

    # ── 3. Strict-filtered + normalized EN facts ───────────────────────────
    en_added = 0
    if (SFT_CLEAN / "sft_en_facts.jsonl").exists():
        for line in open(SFT_CLEAN / "sft_en_facts.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("lang") != "en":
                continue
            t = d.get("text", "")
            if "<|user|>" not in t or "<|assistant|>" not in t:
                continue
            u = clean_question(t.split("<|user|>")[1].split("<|assistant|>")[0])
            a = t.split("<|assistant|>")[1].split("<|end|>")[0].strip()
            if len(u) > 90 or len(a) > 80 or not has_lat(a):
                continue
            if not any(re.search(p, u, re.I) for p in EN_OBJ_PATTERNS):
                continue
            if any(k in u.lower() for k in EN_REJECT):
                continue
            # Require a clean question marker (What/Which/How/Where/Who/At).
            if not re.match(r'^(What|Which|How|Where|Who|At|Name the)', u, re.I):
                continue
            if not a[0].isupper():  # require proper sentence-like answer
                continue
            rows.append({"text": f"<|system|>{SYS_EN}<|user|>{u}<|assistant|>{a}<|end|>",
                         "lang": "en", "task": "instruction"})
            en_added += 1
    print(f"  sft_en_facts strict-filtered EN facts: {en_added}")

    # ── 4. Balance EN / JA based on MODE ───────────────────────────────────
    en_rows = [r for r in rows if r["lang"] == "en"]
    ja_rows = [r for r in rows if r["lang"] == "ja"]
    RNG.shuffle(en_rows); RNG.shuffle(ja_rows)

    if MODE == "ja_only":
        balanced = ja_rows
        print(f"  [{MODE}] JA-only: {len(ja_rows)} JA rows")
    elif MODE == "en_dominant":
        # ~80% EN, ~20% JA (JA kept so the model doesn't forget what it learned).
        n_en = int(len(en_rows) * EN_SHARE)
        n_ja = int(n_en * 0.25)  # 20% of EN count
        n_ja = min(n_ja, len(ja_rows))
        balanced = en_rows[:n_en] + ja_rows[:n_ja]
        print(f"  [{MODE}] EN-dominant: {n_en} EN + {n_ja} JA")
    else:  # all
        n = min(len(en_rows), len(ja_rows))
        balanced = en_rows[:n] + ja_rows[:n]
        print(f"  [all] balanced: {n} EN + {n} JA")
    RNG.shuffle(balanced)

    # ── 5. Dedup + write ───────────────────────────────────────────────────
    seen = set()
    deduped = []
    for r in balanced:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        deduped.append(r)
    RNG.shuffle(deduped)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"\n{'='*56}")
    print(f"Final facts dataset: {len(deduped)} rows")
    print(f"  {dict(Counter(r['lang'] for r in deduped))}")
    print(f"  Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
