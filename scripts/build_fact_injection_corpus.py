#!/usr/bin/env python3
"""build_fact_injection_corpus.py — pretraining knowledge-injection corpus.

Renders Wikidata facts as natural declarative text (no chat template) and
repeats them until every fact has been seen `--exposures` times.

WHY THE DESIGN LOOKS LIKE THIS
------------------------------
The previous version emitted one isolated sentence per line from 2-3 templates
per category. Measured result on the 204M model (results/facts_eval.json):

    lang_ok    EN 100%   JA 100%     <- format, already solved
    attr_ok    EN  44%   JA  60%     <- answers the wrong attribute
    value_hit  EN  48%   JA  48%
    subj-bind  EN  92%   JA  28%     <- answer belongs to another subject

and the recorded error causes name the failure modes exactly:
`other_fact:continent_of = 10` (EN answers the continent when asked something
else) and `subject_missing = 8` (JA drops the subject entirely).

Neither is a capacity problem — they are extraction problems. Allen-Zhu & Li,
*Physics of Language Models 3.1*, show a fact seen in only one phrasing is
stored but not retrievable, and that paraphrase diversity matters more than raw
repetition. So this version:

  * renders 2-6 declarative templates plus question/answer forms per category
    per language, instead of 2-3 declaratives only;
  * substitutes subject ALIASES (data/facts_aliases_tier1.tsv) so a subject is
    not one memorized token sequence;
  * assembles multi-fact DOCUMENTS instead of standalone sentences, in shapes
    that each target one measured failure:
      profile   - every attribute of one subject together  -> attr_ok
      contrast  - one attribute across several subjects    -> subj-bind
      qa        - question immediately followed by answer  -> extraction
      mixed     - unrelated facts as a short paragraph     -> generalisation
  * writes documents separated by a "<|endoftext|>" line, so the pretrain
    loader treats each document as one unit.

EXPOSURE BUDGET
---------------
Cost is `facts x langs x exposures x ~22 tokens`. Tier-1 (4,580 facts) at 1000
exposures over EN+JA is ~200M tokens; the 44,973-fact expanded set at 150 is
~300M. Together ~5% of a 10B-token run, which is why the two tiers take
different --exposures values rather than one global number.

Usage:
  python3 scripts/build_fact_injection_corpus.py \
      --tsv data/wikidata_facts_tier1.tsv --exposures 1000
  python3 scripts/build_fact_injection_corpus.py \
      --tsv data/wikidata_facts_expanded.tsv --exposures 150 --append
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_SEP = "<|endoftext|>"

# ── Templates ──────────────────────────────────────────────────────────────
# "d" = declarative, "q" = question immediately answered. {sub} receives an
# alias part of the time so the subject is not a fixed token sequence.
T = {
    "capital": {
        "en": {"d": ["The capital of {sub} is {val}.",
                     "{val} is the capital of {sub}.",
                     "{sub}'s capital city is {val}.",
                     "{sub} has {val} as its capital.",
                     "The seat of government of {sub} is {val}.",
                     "{val} serves as the capital city of {sub}."],
               "q": ["What is the capital of {sub}? The capital of {sub} is {val}.",
                     "Which city is the capital of {sub}? It is {val}."]},
        "ja": {"d": ["{sub}の首都は{val}です。",
                     "{val}は{sub}の首都です。",
                     "{sub}の首都は{val}である。",
                     "{sub}の政治の中心は{val}です。",
                     "{val}が{sub}の首都として機能しています。"],
               "q": ["{sub}の首都はどこですか。{sub}の首都は{val}です。",
                     "{sub}の首都は何ですか。{val}です。"]},
    },
    "currency": {
        "en": {"d": ["The currency of {sub} is the {val}.",
                     "{sub} uses the {val} as its currency.",
                     "In {sub}, people pay with the {val}.",
                     "The official currency of {sub} is the {val}.",
                     "{val} is the money used in {sub}."],
               "q": ["What currency does {sub} use? {sub} uses the {val}.",
                     "Which currency is used in {sub}? The {val}."]},
        "ja": {"d": ["{sub}の通貨は{val}です。",
                     "{sub}では{val}が使われています。",
                     "{val}は{sub}の通貨です。",
                     "{sub}の公式通貨は{val}である。"],
               "q": ["{sub}の通貨は何ですか。{sub}の通貨は{val}です。"]},
    },
    "official_language": {
        "en": {"d": ["An official language of {sub} is {val}.",
                     "{val} is an official language of {sub}.",
                     "{sub} has {val} as an official language.",
                     "People in {sub} officially speak {val}.",
                     "{val} holds official status in {sub}."],
               "q": ["What language is official in {sub}? {val} is an official language of {sub}.",
                     "Which language does {sub} use officially? {val}."]},
        "ja": {"d": ["{sub}の公用語は{val}です。",
                     "{val}は{sub}の公用語です。",
                     "{sub}では{val}が公用語として使われています。",
                     "{sub}の公用語の一つは{val}である。"],
               "q": ["{sub}の公用語は何ですか。{sub}の公用語は{val}です。"]},
    },
    "population": {
        "en": {"d": ["The population of {sub} is about {val}.",
                     "{sub} has a population of roughly {val}.",
                     "About {val} people live in {sub}.",
                     "{sub} is home to approximately {val} inhabitants."],
               "q": ["How many people live in {sub}? About {val}."]},
        "ja": {"d": ["{sub}の人口は約{val}人です。",
                     "{sub}にはおよそ{val}人が住んでいます。",
                     "{sub}の人口はおよそ{val}人である。"],
               "q": ["{sub}の人口はどれくらいですか。約{val}人です。"]},
    },
    "highest_point": {
        "en": {"d": ["The highest point of {sub} is {val}.",
                     "{val} is the highest point of {sub}.",
                     "{sub}'s highest elevation is at {val}.",
                     "The tallest peak in {sub} is {val}."],
               "q": ["What is the highest point of {sub}? It is {val}."]},
        "ja": {"d": ["{sub}の最高地点は{val}です。",
                     "{val}は{sub}の最高峰です。",
                     "{sub}で最も標高が高い場所は{val}である。"],
               "q": ["{sub}の最高地点はどこですか。{val}です。"]},
    },
    "continent_of": {
        "en": {"d": ["{sub} is located in {val}.",
                     "{sub} is a country in {val}.",
                     "{sub} lies in {val}.",
                     "Geographically, {sub} belongs to {val}."],
               "q": ["Which continent is {sub} in? {sub} is in {val}."]},
        "ja": {"d": ["{sub}は{val}にあります。",
                     "{sub}は{val}に位置する国です。",
                     "{sub}は{val}大陸に属しています。"],
               "q": ["{sub}はどの大陸にありますか。{val}です。"]},
    },
    "area": {
        "en": {"d": ["The area of {sub} is about {val} square kilometers.",
                     "{sub} covers roughly {val} square kilometers.",
                     "{sub} has a land area of about {val} square kilometers."],
               "q": ["How large is {sub}? About {val} square kilometers."]},
        "ja": {"d": ["{sub}の面積は約{val}平方キロメートルです。",
                     "{sub}はおよそ{val}平方キロメートルの広さがあります。"],
               "q": ["{sub}の面積はどれくらいですか。約{val}平方キロメートルです。"]},
    },
    "element_symbol": {
        "en": {"d": ["The chemical symbol of {sub} is {val}.",
                     "{sub} has the chemical symbol {val}.",
                     "{val} is the symbol for the element {sub}."],
               "q": ["What is the chemical symbol of {sub}? It is {val}."]},
        "ja": {"d": ["{sub}の元素記号は{val}です。",
                     "{val}は{sub}の元素記号である。"],
               "q": ["{sub}の元素記号は何ですか。{val}です。"]},
    },
    "element_number": {
        "en": {"d": ["The atomic number of {sub} is {val}.",
                     "{sub} has the atomic number {val}.",
                     "Element number {val} is {sub}."],
               "q": ["What is the atomic number of {sub}? It is {val}."]},
        "ja": {"d": ["{sub}の原子番号は{val}です。",
                     "原子番号{val}の元素は{sub}である。"],
               "q": ["{sub}の原子番号は何ですか。{val}です。"]},
    },
    "driving_side": {
        "en": {"d": ["In {sub}, people drive on the {val}.",
                     "Traffic in {sub} drives on the {val}.",
                     "{sub} uses {val}-hand traffic."],
               "q": ["Which side of the road does {sub} drive on? The {val}."]},
        "ja": {"d": ["{sub}では{val}側通行です。",
                     "{sub}の交通は{val}側通行である。"],
               "q": ["{sub}はどちら側通行ですか。{val}側です。"]},
    },
    "calling_code": {
        "en": {"d": ["The international calling code of {sub} is +{val}.",
                     "To call {sub}, dial +{val}.",
                     "{sub} has the country code +{val}."],
               "q": ["What is the calling code of {sub}? It is +{val}."]},
        "ja": {"d": ["{sub}の国際電話番号は+{val}です。",
                     "{sub}に電話をかけるには+{val}を使います。"],
               "q": ["{sub}の国番号は何ですか。+{val}です。"]},
    },
    "founding_date": {
        "en": {"d": ["{sub} was founded on {val}.",
                     "{sub} was established on {val}.",
                     "The founding date of {sub} is {val}."],
               "q": ["When was {sub} founded? On {val}."]},
        "ja": {"d": ["{sub}は{val}に設立されました。",
                     "{sub}の設立日は{val}です。"],
               "q": ["{sub}はいつ設立されましたか。{val}です。"]},
    },
    # ── expanded set ──────────────────────────────────────────────────────
    "person_country": {
        "en": {"d": ["{sub} is from {val}.", "{sub} is a citizen of {val}.",
                     "{sub} comes from {val}.", "The nationality of {sub} is {val}."],
               "q": ["Where is {sub} from? {sub} is from {val}."]},
        "ja": {"d": ["{sub}は{val}出身です。", "{sub}の国籍は{val}です。",
                     "{sub}は{val}の人物である。"],
               "q": ["{sub}はどこの国の人ですか。{val}です。"]},
    },
    "person_birthyear": {
        "en": {"d": ["{sub} was born in {val}.", "The birth year of {sub} is {val}.",
                     "{sub} was born in the year {val}."],
               "q": ["When was {sub} born? In {val}."]},
        "ja": {"d": ["{sub}は{val}年に生まれました。", "{sub}の生年は{val}年です。"],
               "q": ["{sub}はいつ生まれましたか。{val}年です。"]},
    },
    "person_occupation": {
        "en": {"d": ["{sub} works as a {val}.", "{sub} is a {val}.",
                     "The occupation of {sub} is {val}."],
               "q": ["What does {sub} do? {sub} is a {val}."]},
        "ja": {"d": ["{sub}の職業は{val}です。", "{sub}は{val}として知られています。"],
               "q": ["{sub}の職業は何ですか。{val}です。"]},
    },
    "city_country": {
        "en": {"d": ["{sub} is a city in {val}.", "{sub} is located in {val}.",
                     "The city of {sub} belongs to {val}."],
               "q": ["Which country is {sub} in? {sub} is in {val}."]},
        "ja": {"d": ["{sub}は{val}の都市です。", "{sub}は{val}にあります。"],
               "q": ["{sub}はどこの国の都市ですか。{val}です。"]},
    },
    "city_population": {
        "en": {"d": ["The population of {sub} is about {val}.",
                     "About {val} people live in the city of {sub}."],
               "q": ["How many people live in {sub}? About {val}."]},
        "ja": {"d": ["{sub}の人口は約{val}人です。", "{sub}にはおよそ{val}人が住んでいます。"],
               "q": ["{sub}の人口はどれくらいですか。約{val}人です。"]},
    },
    "film_director": {
        "en": {"d": ["{sub} was directed by {val}.", "{val} directed {sub}.",
                     "The director of {sub} is {val}."],
               "q": ["Who directed {sub}? {val} directed it."]},
        "ja": {"d": ["{sub}の監督は{val}です。", "{val}が{sub}を監督しました。"],
               "q": ["{sub}を監督したのは誰ですか。{val}です。"]},
    },
    "film_year": {
        "en": {"d": ["{sub} was released in {val}.", "The film {sub} came out in {val}."],
               "q": ["When was {sub} released? In {val}."]},
        "ja": {"d": ["{sub}は{val}年に公開されました。", "映画{sub}の公開年は{val}年です。"],
               "q": ["{sub}はいつ公開されましたか。{val}年です。"]},
    },
    "book_author": {
        "en": {"d": ["{sub} was written by {val}.", "{val} wrote {sub}.",
                     "The author of {sub} is {val}."],
               "q": ["Who wrote {sub}? {val} wrote it."]},
        "ja": {"d": ["{sub}の著者は{val}です。", "{val}が{sub}を書きました。"],
               "q": ["{sub}を書いたのは誰ですか。{val}です。"]},
    },
    "band_origin": {
        "en": {"d": ["{sub} formed in {val}.", "The band {sub} originates from {val}."],
               "q": ["Where is {sub} from? {sub} is from {val}."]},
        "ja": {"d": ["{sub}は{val}で結成されました。", "{sub}の出身地は{val}です。"],
               "q": ["{sub}はどこで結成されましたか。{val}です。"]},
    },
    "river_country": {
        "en": {"d": ["The {sub} flows through {val}.", "The river {sub} is in {val}."],
               "q": ["Which country does the {sub} flow through? {val}."]},
        "ja": {"d": ["{sub}は{val}を流れる川です。", "{sub}は{val}にあります。"],
               "q": ["{sub}はどこの国の川ですか。{val}です。"]},
    },
    "river_length": {
        "en": {"d": ["The {sub} is about {val} kilometers long.",
                     "The length of the {sub} is roughly {val} kilometers."],
               "q": ["How long is the {sub}? About {val} kilometers."]},
        "ja": {"d": ["{sub}の長さは約{val}キロメートルです。"],
               "q": ["{sub}の全長はどれくらいですか。約{val}キロメートルです。"]},
    },
    "mountain_country": {
        "en": {"d": ["{sub} is located in {val}.", "The mountain {sub} lies in {val}."],
               "q": ["Where is {sub}? {sub} is in {val}."]},
        "ja": {"d": ["{sub}は{val}にある山です。", "{sub}は{val}に位置しています。"],
               "q": ["{sub}はどこにありますか。{val}です。"]},
    },
    "mountain_elevation": {
        "en": {"d": ["{sub} is about {val} meters high.",
                     "The elevation of {sub} is roughly {val} meters."],
               "q": ["How high is {sub}? About {val} meters."]},
        "ja": {"d": ["{sub}の標高は約{val}メートルです。"],
               "q": ["{sub}の標高はどれくらいですか。約{val}メートルです。"]},
    },
    "island_country": {
        "en": {"d": ["{sub} is an island of {val}.", "The island {sub} belongs to {val}."],
               "q": ["Which country does {sub} belong to? {val}."]},
        "ja": {"d": ["{sub}は{val}の島です。", "{sub}は{val}に属しています。"],
               "q": ["{sub}はどこの国の島ですか。{val}です。"]},
    },
    "lake_country": {
        "en": {"d": ["{sub} is a lake in {val}.", "The lake {sub} lies in {val}."],
               "q": ["Which country is {sub} in? {val}."]},
        "ja": {"d": ["{sub}は{val}にある湖です。"],
               "q": ["{sub}はどこの国の湖ですか。{val}です。"]},
    },
    "airport_country": {
        "en": {"d": ["{sub} is an airport in {val}.", "{sub} serves {val}."],
               "q": ["Which country is {sub} in? {val}."]},
        "ja": {"d": ["{sub}は{val}にある空港です。"],
               "q": ["{sub}はどこの国の空港ですか。{val}です。"]},
    },
    "company_country": {
        "en": {"d": ["{sub} is a company based in {val}.",
                     "{sub} has its headquarters in {val}."],
               "q": ["Where is {sub} based? In {val}."]},
        "ja": {"d": ["{sub}は{val}の企業です。", "{sub}の本社は{val}にあります。"],
               "q": ["{sub}はどこの国の企業ですか。{val}です。"]},
    },
    "university_country": {
        "en": {"d": ["{sub} is a university in {val}.", "{sub} is located in {val}."],
               "q": ["Which country is {sub} in? {val}."]},
        "ja": {"d": ["{sub}は{val}にある大学です。"],
               "q": ["{sub}はどこの国の大学ですか。{val}です。"]},
    },
    "species_parent": {
        "en": {"d": ["{sub} belongs to {val}.", "{sub} is classified under {val}."],
               "q": ["What group does {sub} belong to? {val}."]},
        "ja": {"d": ["{sub}は{val}に属します。"],
               "q": ["{sub}はどの分類群に属しますか。{val}です。"]},
    },
    # --- expanded entities (net-new categories; see expand_wikidata_facts.py) ---
    "person_birthplace": {
        "en": {"d": ["{sub} was born in {val}.", "{sub}'s place of birth is {val}."],
               "q": ["Where was {sub} born? In {val}."]},
        "ja": {"d": ["{sub}の出身地は{val}です。", "{sub}は{val}で生まれました。"],
               "q": ["{sub}の出身地はどこですか。{val}です。"]},
    },
    "person_deathyear": {
        "en": {"d": ["{sub} died in {val}.", "The death year of {sub} is {val}."],
               "q": ["When did {sub} die? In {val}."]},
        "ja": {"d": ["{sub}は{val}年に亡くなりました。", "{sub}の没年は{val}年です。"],
               "q": ["{sub}はいつ亡くなりましたか。{val}年です。"]},
    },
    "person_educated": {
        "en": {"d": ["{sub} studied at {val}.", "{sub} was educated at {val}."],
               "q": ["Where did {sub} study? At {val}."]},
        "ja": {"d": ["{sub}は{val}で学びました。", "{sub}は{val}の出身です。"],
               "q": ["{sub}はどこで学びましたか。{val}です。"]},
    },
    "company_founded": {
        "en": {"d": ["{sub} was founded in {val}.", "{sub} was established in {val}."],
               "q": ["When was {sub} founded? In {val}."]},
        "ja": {"d": ["{sub}は{val}年に設立されました。", "{sub}の創業年は{val}年です。"],
               "q": ["{sub}はいつ設立されましたか。{val}年です。"]},
    },
    "company_industry": {
        "en": {"d": ["{sub} operates in the {val} industry.",
                     "{sub} is active in the {val} sector."],
               "q": ["What industry is {sub} in? The {val} industry."]},
        "ja": {"d": ["{sub}は{val}業界で事業を展開しています。", "{sub}の業種は{val}です。"],
               "q": ["{sub}はどこの業界ですか。{val}業界です。"]},
    },
    "company_hq": {
        "en": {"d": ["{sub} is headquartered in {val}.", "The headquarters of {sub} is in {val}."],
               "q": ["Where is {sub} headquartered? In {val}."]},
        "ja": {"d": ["{sub}の本社は{val}にあります。", "{sub}は{val}に本社を置いています。"],
               "q": ["{sub}の本社はどこですか。{val}です。"]},
    },
    "mountain_range": {
        "en": {"d": ["{sub} is part of the {val} range.", "{sub} lies in the {val} mountains."],
               "q": ["Which mountain range does {sub} belong to? The {val}."]},
        "ja": {"d": ["{sub}は{val}山脈の一部です。", "{sub}は{val}山系に属します。"],
               "q": ["{sub}はどの山脈に属しますか。{val}山脈です。"]},
    },
    "city_region": {
        "en": {"d": ["{sub} is located in {val}.", "{sub} is part of {val}."],
               "q": ["Where is {sub} located? In {val}."]},
        "ja": {"d": ["{sub}は{val}に位置します。", "{sub}は{val}の一部です。"],
               "q": ["{sub}はどこに位置しますか。{val}です。"]},
    },
    "species_rank": {
        "en": {"d": ["The taxonomic rank of {sub} is {val}.", "{sub} is ranked as {val}."],
               "q": ["What taxonomic rank is {sub}? {val}."]},
        "ja": {"d": ["{sub}の分類階級は{val}です。"],
               "q": ["{sub}の分類階級は何ですか。{val}です。"]},
    },
    "star_constellation": {
        "en": {"d": ["{sub} is in the constellation {val}.", "{sub} belongs to {val}."],
               "q": ["Which constellation is {sub} in? {val}."]},
        "ja": {"d": ["{sub}は{val}座に属します。", "{sub}は星座{val}にあります。"],
               "q": ["{sub}はどこの星座ですか。{val}座です。"]},
    },
    "planet_star": {
        "en": {"d": ["{sub} orbits {val}.", "{sub} is a planet of the star {val}."],
               "q": ["Which star does {sub} orbit? {val}."]},
        "ja": {"d": ["{sub}は恒星{val}を周回します。", "{sub}は{val}の惑星です。"],
               "q": ["{sub}はどの恒星を周回しますか。{val}です。"]},
    },
}
# capital_admin shares its wording with capital.
T["capital_admin"] = T["capital"]

CONTRAST_LEAD = {
    "en": ["Here are several related facts.",
           "The following entries can be compared directly.",
           "Compare the entries below."],
    "ja": ["以下は関連する事実です。", "次の項目を比較してみましょう。",
           "以下の項目を並べて確認します。"],
}


def load_aliases(path: Path):
    """QID -> lang -> [alias]."""
    out = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        print(f"[warn] no alias table at {path}")
        return out
    for line in path.open(encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        qid, lang, alias = p[0].strip(), p[1].strip(), p[2].strip()
        # "ga" / "GAB" are ambiguous standing alone and would teach a wrong bind.
        if len(alias) < 4 or alias.isupper():
            continue
        out[qid][lang].append(alias)
    return out


def load_facts(path: Path):
    facts, dropped = [], 0
    for line in path.open(encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        cat = p[0].strip()
        if cat not in T:
            dropped += 1
            continue
        sub = {"en": p[1].strip(), "ja": p[2].strip()}
        val = {"en": p[4].strip(), "ja": p[5].strip()}
        qid = p[7].strip() if len(p) > 7 else ""
        # IUPAC placeholder names for undiscovered elements.
        if cat == "element_number" and val["en"].isdigit() and int(val["en"]) > 118:
            dropped += 1
            continue
        if cat == "official_language":
            for lg in ("en", "ja"):
                for pref in ("bahasa ", "Bahasa "):
                    if val[lg].startswith(pref):
                        val[lg] = val[lg][len(pref):]
        facts.append({"cat": cat, "sub": sub, "val": val, "qid": qid})
    return facts, dropped


# Wikidata values already carry a unit symbol ("3,661 m", "758 km", "1949年")
# while these templates spell the unit out, so the two stack into "3,661 m
# meters" and "1949年年". Strip the symbol and let the template supply the unit.
# Categories whose values are bare numbers (area, population, city_population)
# are absent here on purpose.
UNIT_STRIP = {
    "mountain_elevation": ("m",),
    "river_length": ("km",),
    "film_year": ("年",),
    "person_birthyear": ("年",),
    "person_deathyear": ("年",),
    "company_founded": ("年",),
}


def strip_unit(cat, val):
    for u in UNIT_STRIP.get(cat, ()):
        # Full dates ("1912年11月28日") do not end in the unit, so they survive.
        if val.endswith(u):
            return val[: -len(u)].rstrip()
    return val


# "a actor" / "a engineer": the occupation comes from Wikidata so the article
# cannot be baked into the template. Exclude the vowel-letter-but-consonant-sound
# words ("a university", "a European", "a one-time").
_A_AN_RE = re.compile(r"\ba (?=[aeiou])(?!uni|use|uti|ubi|eu|one)", re.I)

# "The mountain Mount Fuji": the template adds a noun the name already starts
# with. Mountains drop the prefix outright ("Mount Fuji lies in..."), rivers keep
# the article because "The River Severn" is correct.
_DUP_MOUNT_RE = re.compile(r"\bThe mountain (?=(?:Mount(?:ains?)?|Mt\.?)\b)", re.I)
_DUP_RIVER_RE = re.compile(r"\bThe river (?=(?:River|Rio)\b)", re.I)


def render(fact, lang, rng, aliases, alias_prob, qa_prob):
    sub, val = fact["sub"][lang], fact["val"][lang]
    if not sub or not val:
        return None
    val = strip_unit(fact["cat"], val)
    pool = aliases.get(fact["qid"], {}).get(lang, []) if fact["qid"] else []
    if pool and rng.random() < alias_prob:
        sub = rng.choice(pool)
    bank = T[fact["cat"]][lang]
    forms = bank["q"] if (bank.get("q") and rng.random() < qa_prob) else bank["d"]
    text = rng.choice(forms).format(sub=sub, val=val)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if lang == "en":
        text = _DUP_MOUNT_RE.sub("", text)
        text = _DUP_RIVER_RE.sub("The ", text)
        text = _A_AN_RE.sub("an ", text)
    # A value that already ends in a period ("Washington, D.C.") doubles it.
    if text.endswith("..") and not text.endswith("..."):
        text = text[:-1]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/wikidata_facts_tier1.tsv")
    ap.add_argument("--aliases", default="data/facts_aliases_tier1.tsv")
    ap.add_argument("--out_dir", default="data/corpus_inject_facts")
    ap.add_argument("--langs", default="en,ja")
    ap.add_argument("--exposures", type=int, default=1000,
                    help="times each fact is rendered")
    ap.add_argument("--append", action="store_true",
                    help="add to existing files instead of overwriting")
    ap.add_argument("--alias_prob", type=float, default=0.25)
    ap.add_argument("--qa_prob", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    facts, dropped = load_facts(ROOT / args.tsv)
    aliases = load_aliases(ROOT / args.aliases)
    print(f"loaded {len(facts)} facts from {args.tsv} "
          f"({dropped} skipped: no template or placeholder element)")
    if not facts:
        return 1

    by_qid, by_cat = defaultdict(list), defaultdict(list)
    for f in facts:
        if f["qid"]:
            by_qid[f["qid"]].append(f)
        by_cat[f["cat"]].append(f)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for lang in langs:
        path = out_dir / f"facts_{lang}.txt"
        counters = {"docs": 0, "bytes": 0}
        with path.open("a" if args.append else "w", encoding="utf-8") as fo:

            def emit(sentences):
                sentences = [s for s in sentences if s]
                if not sentences:
                    return
                body = " ".join(sentences)
                fo.write(body + "\n" + DOC_SEP + "\n")
                counters["docs"] += 1
                counters["bytes"] += len(body.encode("utf-8")) + len(DOC_SEP) + 2

            # Each pass renders every fact exactly once, so exposure counts are
            # exact; the document shape rotates so a fact is seen in four
            # different contexts rather than four copies of one context.
            for p in range(args.exposures):
                shape = ("profile", "contrast", "qa", "mixed")[p % 4]

                if shape == "profile":
                    # Subjects with several attributes become one document each;
                    # subjects with a single attribute are batched, otherwise
                    # half the corpus would be 8-token documents.
                    multi = [g for g in by_qid.values() if len(g) > 1]
                    singles = [g[0] for g in by_qid.values() if len(g) == 1]
                    singles += [f for f in facts if not f["qid"]]
                    rng.shuffle(multi)
                    for g in multi:
                        emit([render(f, lang, rng, aliases, args.alias_prob,
                                     args.qa_prob) for f in g])
                    rng.shuffle(singles)
                    for i in range(0, len(singles), 4):
                        emit([render(f, lang, rng, aliases, args.alias_prob,
                                     args.qa_prob) for f in singles[i:i + 4]])

                elif shape == "contrast":
                    for group in by_cat.values():
                        g = list(group)
                        rng.shuffle(g)
                        for i in range(0, len(g), 5):
                            chunk = g[i:i + 5]
                            lead = (rng.choice(CONTRAST_LEAD[lang])
                                    if len(chunk) > 2 else None)
                            body = [render(f, lang, rng, aliases,
                                           args.alias_prob, 0.0) for f in chunk]
                            emit(([lead] if lead else []) + body)

                else:
                    qp = 0.85 if shape == "qa" else 0.15
                    pool = list(facts)
                    rng.shuffle(pool)
                    for i in range(0, len(pool), 4):
                        emit([render(f, lang, rng, aliases, args.alias_prob, qp)
                              for f in pool[i:i + 4]])

        est_tok = counters["bytes"] / (3.18 if lang == "ja" else 3.53)
        stats[lang] = {"documents": counters["docs"], "bytes": counters["bytes"],
                       "est_tokens": int(est_tok), "path": str(path)}
        print(f"[{lang}] {counters['docs']:,} documents, "
              f"{counters['bytes']/1e6:.0f} MB, ~{est_tok/1e6:.0f}M tokens -> {path}")

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "fact_injection_stats.json").write_text(
        json.dumps({"tsv": args.tsv, "facts": len(facts),
                    "exposures": args.exposures, "by_lang": stats},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
