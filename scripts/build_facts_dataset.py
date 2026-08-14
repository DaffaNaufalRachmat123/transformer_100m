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


def render_strict_fact(cat, sub_en, sub_ja, val_en, val_ja):
    ans = CAT_ANSWERS.get(cat)
    if not ans:
        return []
    a_en_t, a_ja_t = ans
    if cat in {"area", "population", "element_number"}:
        val_en = clean_numeric_value(val_en)
        val_ja = clean_numeric_value(val_ja)
    a_en = a_en_t.format(sub=sub_en, val=val_en)
    a_ja = a_ja_t.format(sub=sub_ja, val=val_ja)
    out = []
    fact_id = f"{cat}::{sub_en.casefold()}"
    for i, q in enumerate(STRICT_EN_Q[cat]):
        out.append({
            "text": f"<|system|>{SYS_EN}<|user|>{q.format(sub=sub_en)}<|assistant|>{a_en}<|end|>",
            "lang": "en", "task": "fact_qa", "category": cat,
            "subject_en": sub_en, "subject_ja": sub_ja, "value_en": val_en, "value_ja": val_ja,
            "source": "wikidata_strict", "fact_id": fact_id, "template_id": f"{cat}_en_{i:02d}",
        })
    for i, q in enumerate(STRICT_JA_Q[cat]):
        out.append({
            "text": f"<|system|>{SYS_JA}<|user|>{q.format(sub=sub_ja)}<|assistant|>{a_ja}<|end|>",
            "lang": "ja", "task": "fact_qa", "category": cat,
            "subject_en": sub_en, "subject_ja": sub_ja, "value_en": val_en, "value_ja": val_ja,
            "source": "wikidata_strict", "fact_id": fact_id, "template_id": f"{cat}_ja_{i:02d}",
        })
    return out


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_strict_facts():
    """Return clean, unambiguous fact tuples and source stats."""
    raw = []
    for wd_path in [ROOT / "data" / "wikidata_facts_all.tsv"]:
        if not wd_path.exists():
            continue
        for line in open(wd_path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            cat, sub_en, sub_ja, val_en, val_ja = [x.strip() for x in p[:5]]
            if invalid_strict_fact(cat, sub_en, sub_ja, val_en, val_ja):
                continue
            if cat in {"area", "population", "element_number"}:
                val_en = clean_numeric_value(val_en)
                val_ja = clean_numeric_value(val_ja)
            raw.append((cat, sub_en, sub_ja, val_en, val_ja))

    # Keep only unambiguous single-answer facts per category+subject. This is
    # intentionally strict: multi-value official languages/currencies are valid
    # world facts, but they produce unstable single-answer SFT targets.
    by_key = {}
    conflicts = set()
    for cat, sub_en, sub_ja, val_en, val_ja in raw:
        key = (cat, sub_en.casefold(), sub_ja)
        val = (val_en.casefold(), val_ja)
        if key in by_key and by_key[key][0] != val:
            conflicts.add(key)
        else:
            by_key[key] = (val, (cat, sub_en, sub_ja, val_en, val_ja))
    facts = [item for key, (_, item) in by_key.items() if key not in conflicts]
    return facts, {"raw_kept": len(raw), "conflict_keys": len(conflicts)}


def build_strict_dataset():
    """Build a high-precision facts dataset with no generic QA prompts."""
    facts, src_stats = load_strict_facts()

    # Use all unambiguous facts. The raw strict categories are already modest;
    # dropping down to the smallest category wastes clean supervision.
    by_cat = {}
    for fact in facts:
        by_cat.setdefault(fact[0], []).append(fact)
    for rows in by_cat.values():
        RNG.shuffle(rows)
    balanced_facts = []
    for cat in sorted(by_cat):
        balanced_facts.extend(by_cat[cat])
    RNG.shuffle(balanced_facts)

    split_keys = list({(cat, sub_en.casefold()) for cat, sub_en, _, _, _ in balanced_facts})
    RNG.shuffle(split_keys)
    n_eval = max(1, int(len(split_keys) * 0.10))
    eval_keys = set(split_keys[:n_eval])

    train, eval_rows = [], []
    prompt_to_answer = {}
    dropped_prompt_conflicts = 0
    for fact in balanced_facts:
        cat, sub_en, sub_ja, _, _ = fact
        target = eval_rows if (cat, sub_en.casefold()) in eval_keys else train
        for row in render_strict_fact(*fact):
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


def build_oversampled_dataset(target_rows=100000, en_share=0.58, eval_rows=4000):
    """Build a large strict fact dataset by weighted oversampling.

    Rows may repeat exact prompts; this is intentional SFT weighting. Held-out
    eval is not oversampled and is split by fact_id so train never sees the same
    category+subject facts.
    """
    facts, src_stats = load_strict_facts()
    if not facts:
        raise RuntimeError("No strict facts available")

    split_keys = list({(cat, sub_en.casefold()) for cat, sub_en, _, _, _ in facts})
    RNG.shuffle(split_keys)
    # Keep eval bounded and fact-key based. Each held-out fact renders 16 rows
    # with the current template set (8 EN + 8 JA).
    n_eval_keys = max(1, min(len(split_keys) // 5, max(1, eval_rows // 16)))
    eval_keys = set(split_keys[:n_eval_keys])
    train_facts = [f for f in facts if (f[0], f[1].casefold()) not in eval_keys]
    eval_facts = [f for f in facts if (f[0], f[1].casefold()) in eval_keys]

    by_cat = {}
    for fact in train_facts:
        by_cat.setdefault(fact[0], []).append(fact)
    categories = sorted(by_cat)
    cat_weights = [OVERSAMPLE_WEIGHTS.get(cat, 1.0) for cat in categories]

    target_en = int(target_rows * en_share)
    target_ja = target_rows - target_en
    produced = {"en": 0, "ja": 0}
    train = []
    while produced["en"] < target_en or produced["ja"] < target_ja:
        lang = "en" if produced["en"] < target_en and (produced["ja"] >= target_ja or RNG.random() < en_share) else "ja"
        cat = weighted_choice(categories, cat_weights)
        fact = RNG.choice(by_cat[cat])
        rendered = [r for r in render_strict_fact(*fact) if r["lang"] == lang]
        row = dict(RNG.choice(rendered))
        row["sample_weighted"] = True
        train.append(row)
        produced[lang] += 1

    eval_out = []
    for fact in eval_facts:
        eval_out.extend(render_strict_fact(*fact))

    RNG.shuffle(train)
    RNG.shuffle(eval_out)
    write_jsonl(OUT_FILE, train)
    write_jsonl(OUT_DIR / STRICT_EVAL_FILE, eval_out)

    from collections import Counter
    print(f"  oversample raw kept: {src_stats['raw_kept']}")
    print(f"  oversample conflicts dropped: {src_stats['conflict_keys']} fact keys")
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
    args = ap.parse_args()
    MODE = args.mode
    if args.out_dir:
        OUT_DIR = ROOT / args.out_dir
    elif MODE == "strict":
        OUT_DIR = ROOT / "sft_data" / "sft_facts_strict"
    elif MODE == "oversample":
        OUT_DIR = ROOT / "sft_data" / f"sft_facts_oversampled_{args.target_rows//1000}k"
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
        build_oversampled_dataset(args.target_rows, args.en_share, args.eval_rows)
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
