#!/usr/bin/env python3
"""fetch_wikidata_facts.py — Fetch a large, accurate, bilingual (EN+JA) fact
dataset from Wikidata via SPARQL.

Each fact category is a SPARQL query that returns entity pairs (subject,
value) with BOTH an EN and a JA label in a single query (two label services).
This gives a true bilingual fact (EN sentence + JA sentence) for each pair,
and only facts with both languages are kept — this guarantees accuracy and
helps fix "language confusion" (answer in the language of the question).

Output: data/wikidata_facts.tsv
  one per line: CATEGORY \t question_EN \t answer_EN \t question_JA \t answer_JA

Usage:
    python3 scripts/fetch_wikidata_facts.py [--out data/wikidata_facts.tsv] [--limit N]
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://query.wikidata.org/sparql"

# ── Fact categories ────────────────────────────────────────────────────────
# key: category name; value: (where_clause, q_en, a_en, q_ja, a_ja)
# where_clause binds ?sub (subject) and ?val (value).
CATEGORIES = {
    "capital": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P36 ?val.",
        "What is the capital of {sub}?", "The capital of {sub} is {val}.",
        "{sub}の首都は？", "{sub}の首都は{val}です。"),
    "largest_city": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P149 ?val.",
        "What is the largest city of {sub}?", "The largest city of {sub} is {val}.",
        "{sub}で一番大きい都市は？", "{sub}で一番大きい都市は{val}です。"),
    "currency": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P38 ?val.",
        "What is the currency of {sub}?", "The currency of {sub} is {val}.",
        "{sub}の通貨は？", "{sub}の通貨は{val}です。"),
    "official_language": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P37 ?val.",
        "What is an official language of {sub}?", "An official language of {sub} is {val}.",
        "{sub}の公用語は？", "{sub}の公用語は{val}です。"),
    "population": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P1082 ?val.",
        "What is the population of {sub}?", "The population of {sub} is about {val}.",
        "{sub}の人口は？", "{sub}の人口はおよそ{val}人です。"),
    "highest_point": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P610 ?val.",
        "What is the highest point of {sub}?", "The highest point of {sub} is {val}.",
        "{sub}で一番高い場所は？", "{sub}で一番高い場所は{val}です。"),
    "continent_of": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P30 ?val.",
        "Which continent is {sub} in?", "{sub} is in {val}.",
        "{sub}はどの大陸にありますか？", "{sub}は{val}にあります。"),
    "head_of_state": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P6 ?val.",
        "Who is the head of state of {sub}?", "The head of state of {sub} is {val}.",
        "{sub}の国家元首は？", "{sub}の国家元首は{val}です。"),
    "area": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P2046 ?val.",
        "What is the area of {sub}?", "The area of {sub} is about {val} square kilometers.",
        "{sub}の面積は？", "{sub}の面積はおよそ{val}平方キロメートルです。"),
    "element_symbol": (
        "?sub wdt:P31 wd:Q11344. ?sub wdt:P246 ?val.",
        "What is the chemical symbol of {sub}?", "The chemical symbol of {sub} is {val}.",
        "{sub}の化学記号は？", "{sub}の化学記号は{val}です。"),
    "element_number": (
        "?sub wdt:P31 wd:Q11344. ?sub wdt:P1086 ?val.",
        "What is the atomic number of {sub}?", "The atomic number of {sub} is {val}.",
        "{sub}の原子番号は？", "{sub}の原子番号は{val}です。"),
    "mountain_country": (
        "?sub wdt:P31 wd:Q8502. ?sub wdt:P17 ?val.",
        "Which country is {sub} in?", "{sub} is in {val}.",
        "{sub}はどの国にありますか？", "{sub}は{val}にあります。"),
    "river_country": (
        "?sub wdt:P31 wd:Q4022. ?sub wdt:P17 ?val.",
        "Which country is {sub} in?", "{sub} is in {val}.",
        "{sub}はどの国にありますか？", "{sub}は{val}にあります。"),
    "planet_kind": (
        "?sub wdt:P31 wd:Q634. ?sub wdt:P2044 ?h.",
        "What is {sub}?", "{sub} is a planet.",
        "{sub}は何ですか？", "{sub}は惑星です。"),
    "language_country": (
        "?sub wdt:P31 wd:Q34770. ?sub wdt:P17 ?val.",
        "In which country is {sub} spoken?", "{sub} is spoken in {val}.",
        "{sub}はどの国で話されていますか？", "{sub}は{val}で話されています。"),
    "person_birthplace": (
        "?sub wdt:P31 wd:Q5. ?sub wdt:P19 ?val. ?sub wdt:P569 ?bd.",
        "Where was {sub} born?", "{sub} was born in {val}.",
        "{sub}はどこで生まれましたか？", "{sub}は{val}で生まれました。"),
    "person_occupation": (
        "?sub wdt:P31 wd:Q5. ?sub wdt:P106 ?val. ?sub wdt:P569 ?bd.",
        "What is {sub} known as?", "{sub} is a {val}.",
        "{sub}は何で知られていますか？", "{sub}は{val}です。"),
}


def sparql(query, max_retries=8):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "Mozilla/5.0 (fact-fetcher/2.0; educational)",
            })
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", {}).get("bindings", [])
        except urllib.error.HTTPError as e:
            wait = 3 ** attempt + 2
            print(f"    [retry {attempt+1}] HTTP {e.code} — waiting {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = 3 ** attempt + 2
            print(f"    [retry {attempt+1}] {e} — waiting {wait}s")
            time.sleep(wait)
    return []


def make_query(where, limit):
    return f"""
SELECT DISTINCT ?sub ?subEN ?subJA ?valEN ?valJA WHERE {{
  {where}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?sub rdfs:label ?subEN. ?val rdfs:label ?valEN. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ja". ?sub rdfs:label ?subJA. ?val rdfs:label ?valJA. }}
}} LIMIT {limit}
"""


def fetch_bilingual(where, limit):
    bindings = sparql(make_query(where, limit))
    out = []
    for b in bindings:
        sub_en = b.get("subEN", {}).get("value", "")
        sub_ja = b.get("subJA", {}).get("value", "")
        val_en = b.get("valEN", {}).get("value", "")
        val_ja = b.get("valJA", {}).get("value", "")
        if sub_en and sub_ja and val_en and val_ja:
            out.append((sub_en, sub_ja, val_en, val_ja))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wikidata_facts.tsv")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--categories", default="all")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cats = list(CATEGORIES.items())
    if args.categories != "all":
        cats = [(k, v) for k, v in cats if k in args.categories.split(",")]

    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for name, (where, q_en, a_en, q_ja, a_ja) in cats:
            print(f"[{name}] fetching bilingual...")
            rows = fetch_bilingual(where, args.limit)
            n = 0
            for sub_en, sub_ja, val_en, val_ja in rows:
                # Write RAW pairs (template expansion happens in
                # build_facts_dataset.py for volume).
                f.write(f"{name}\t{sub_en}\t{sub_ja}\t{val_en}\t{val_ja}\n")
                n += 1
            total += n
            print(f"[{name}] kept {n} bilingual facts")
            time.sleep(6)  # be gentle to Wikidata — avoid 429/502 throttling

    print(f"\nDone: {total} bilingual facts -> {out_path}")


if __name__ == "__main__":
    main()