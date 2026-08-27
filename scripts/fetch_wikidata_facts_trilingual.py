#!/usr/bin/env python3
"""fetch_wikidata_facts_trilingual.py — Fetch EN/JA/ID fact dataset from Wikidata.

Extends fetch_wikidata_facts.py with a third label service ("id") so every
fact carries subject/value in English, Japanese, AND Indonesian.

Output TSV (7 columns):
  CATEGORY \t sub_en \t sub_ja \t sub_id \t val_en \t val_ja \t val_id

After fetching, re-apply the known data fixes (Naoero→Nauru, Q4916→euro,
drop truncated values) — see fix_trilingual_tsv() which runs automatically.

Usage:
    python3 scripts/fetch_wikidata_facts_trilingual.py \
        [--out data/wikidata_facts_trilingual.tsv] [--limit 2000]
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://query.wikidata.org/sparql"

# Same categories as fetch_wikidata_facts.py (strict set used by the builder).
CATEGORIES = {
    "capital": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P36 ?val.",
    ),
    "currency": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P38 ?val.",
    ),
    "official_language": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P37 ?val.",
    ),
    "population": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P1082 ?val.",
    ),
    "highest_point": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P610 ?val.",
    ),
    "continent_of": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P30 ?val.",
    ),
    "area": (
        "?sub wdt:P31 wd:Q6256. ?sub wdt:P2046 ?val.",
    ),
    "element_symbol": (
        "?sub wdt:P31 wd:Q11344. ?sub wdt:P246 ?val.",
    ),
    "element_number": (
        "?sub wdt:P31 wd:Q11344. ?sub wdt:P1086 ?val.",
    ),
}


def sparql(query, max_retries=8):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "Mozilla/5.0 (fact-fetcher/3.0; educational)",
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
SELECT DISTINCT ?sub ?subEN ?subJA ?subID ?valEN ?valJA ?valID WHERE {{
  {where}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". ?sub rdfs:label ?subEN. ?val rdfs:label ?valEN. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ja". ?sub rdfs:label ?subJA. ?val rdfs:label ?valJA. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "id". ?sub rdfs:label ?subID. ?val rdfs:label ?valID. }}
}} LIMIT {limit}
"""


def clean_numeric(v):
    v = str(v).strip().replace(",", "")
    try:
        x = float(v)
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:.1f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(v).strip()


def fetch_trilingual(where, limit):
    bindings = sparql(make_query(where, limit))
    out = []
    for b in bindings:
        row = {
            "sub_en": b.get("subEN", {}).get("value", ""),
            "sub_ja": b.get("subJA", {}).get("value", ""),
            "sub_id": b.get("subID", {}).get("value", ""),
            "val_en": b.get("valEN", {}).get("value", ""),
            "val_ja": b.get("valJA", {}).get("value", ""),
            "val_id": b.get("valID", {}).get("value", ""),
        }
        # Keep only rows with ALL three languages present.
        if all(row.values()):
            out.append(row)
    return out


def fix_trilingual_rows(rows):
    """Re-apply the known data-quality fixes from the bilingual pipeline."""
    DROP_EXACT = {
        ("currency", "France", "CFP Franc"),
        ("capital", "Grenada", "St"),
        ("capital", "United States", "Washington, D"),
    }
    FIX_VAL = {
        ("capital", "Antigua and Barbuda"): "Saint John's",
        ("highest_point", "United States"): "Denali",
    }
    fixed, dropped = [], 0
    for r in rows:
        if (r["cat"], r["sub_en"], r["val_en"]) in DROP_EXACT:
            dropped += 1
            continue
        if r["sub_en"] == "Naoero":
            r["sub_en"] = "Nauru"
        if r["val_en"] == "Q4916":
            # Euro in ID is "euro" as well.
            r["val_en"], r["val_id"] = "euro", "euro"
        key = (r["cat"], r["sub_en"])
        if key in FIX_VAL and r["val_en"].startswith("Q"):
            r["val_en"] = FIX_VAL[key]
        fixed.append(r)
    return fixed, dropped


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wikidata_facts_trilingual.tsv")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    rows = []
    for name, (where,) in CATEGORIES.items():
        print(f"[{name}] fetching trilingual...")
        fetched = fetch_trilingual(where, args.limit)
        n = 0
        for f in fetched:
            val_en, val_id = f["val_en"], f["val_id"]
            val_ja = f["val_ja"]
            if name in {"area", "population", "element_number"}:
                val_en = clean_numeric(val_en)
                val_ja = val_en
                val_id = val_en
            rows.append({
                "cat": name,
                "sub_en": f["sub_en"].strip(),
                "sub_ja": f["sub_ja"].strip(),
                "sub_id": f["sub_id"].strip(),
                "val_en": val_en.strip(),
                "val_ja": val_ja.strip(),
                "val_id": val_id.strip(),
            })
            n += 1
        total += n
        print(f"[{name}] kept {n} trilingual facts")
        time.sleep(6)  # be gentle to Wikidata

    rows, dropped = fix_trilingual_rows(rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join([r["cat"], r["sub_en"], r["sub_ja"], r["sub_id"],
                               r["val_en"], r["val_ja"], r["val_id"]]) + "\n")

    print(f"\nDone: {len(rows)} trilingual facts ({dropped} bad rows dropped) -> {out_path}")


if __name__ == "__main__":
    main()
