#!/usr/bin/env python3
"""fetch_capital_admin_v2.py — Re-fetch capital_admin with subclass expansion.

v1 used a direct P31 match on Q10864048 and found only 34 statements because
provinces/states use specific subclasses ("province of Indonesia", "state of
the United States", ...). This script:
  1. enumerates all subclasses of Q10864048 (P279*),
  2. fetches P36 (capital) statements for instances of those classes,
  3. batch-fetches trilingual labels,
  4. REPLACES the capital_admin rows in data/wikidata_facts_tier1.tsv and
     appends new aliases to data/facts_aliases_tier1.tsv.
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
TSV = ROOT / "data" / "wikidata_facts_tier1.tsv"
ALIASES = ROOT / "data" / "facts_aliases_tier1.tsv"
MIN_SITELINKS = 15


def sparql(query, max_retries=6, timeout=180):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "Mozilla/5.0 (trilingual-fact-fetcher/4.2; educational project)",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", {}).get("bindings", [])
        except Exception as e:
            wait = min(3 ** attempt + 2, 60)
            print(f"    [retry {attempt+1}] {e} — waiting {wait}s", flush=True)
            time.sleep(wait)
    return []


def qid_of(iri):
    return str(iri).rsplit("/", 1)[-1]


def norm_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def rank_value(iri):
    return {"PreferredRank": 2, "NormalRank": 1, "DeprecatedRank": 0}.get(
        str(iri).rsplit("#", 1)[-1], 1)


def main():
    # 1. subclasses of Q10864048
    print("[1/4] enumerating subclasses of Q10864048...", flush=True)
    cls_bindings = sparql("SELECT DISTINCT ?cls WHERE { ?cls wdt:P279* wd:Q10864048 }")
    classes = [qid_of(b["cls"]["value"]) for b in cls_bindings]
    print(f"    {len(classes)} subclasses", flush=True)
    time.sleep(3)

    # 2. capital statements for instances of those classes
    print("[2/4] fetching P36 statements...", flush=True)
    statements = {}
    chunk = 60
    for i in range(0, len(classes), chunk):
        values = " ".join(f"wd:{c}" for c in classes[i:i + chunk])
        q = f"""
SELECT DISTINCT ?sub ?val ?rank WHERE {{
  VALUES ?cls {{ {values} }}
  ?sub wdt:P31 ?cls ; p:P36 ?st .
  ?st ps:P36 ?val .
  ?st wikibase:rank ?rank .
  ?sub wikibase:sitelinks ?sl . FILTER(?sl >= {MIN_SITELINKS})
}} LIMIT 40000
"""
        for b in sparql(q, timeout=240):
            sub = qid_of(b["sub"]["value"])
            val = qid_of(b["val"]["value"])
            rank = rank_value(b.get("rank", {}).get("value", "NormalRank"))
            cur = statements.get(sub)
            if cur is None or rank > cur[1]:
                statements[sub] = (val, rank)
        print(f"    subclass batch {i // chunk + 1}/{(len(classes) + chunk - 1) // chunk} "
              f"-> {len(statements)} subjects", flush=True)
        time.sleep(2)

    # keep single-answer per subject (already collapsed to best rank)
    print(f"[2/4] {len(statements)} subjects with capitals", flush=True)

    # 3. labels
    qids = sorted(set(statements.keys()) | {v[0] for v in statements.values()})
    print(f"[3/4] fetching labels for {len(qids)} entities...", flush=True)
    labels = {}
    for i in range(0, len(qids), 180):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 180])
        q = f"""
SELECT ?q ?len ?lja ?lid WHERE {{
  VALUES ?q {{ {values} }}
  OPTIONAL {{ ?q rdfs:label ?len . FILTER(LANG(?len) = "en") }}
  OPTIONAL {{ ?q rdfs:label ?lja . FILTER(LANG(?lja) = "ja") }}
  OPTIONAL {{ ?q rdfs:label ?lid . FILTER(LANG(?lid) = "id") }}
}}
"""
        for b in sparql(q, timeout=120):
            e = labels.setdefault(qid_of(b["q"]["value"]), {})
            for key, lang in (("len", "en"), ("lja", "ja"), ("lid", "id")):
                if key in b:
                    e[lang] = norm_ws(b[key]["value"])
        if (i // 180) % 5 == 4:
            print(f"    labels: {i + 180}/{len(qids)}", flush=True)
        time.sleep(1.5)

    # 4. build rows
    rows = []
    for sub, (val, _rank) in statements.items():
        se, sj, si = (labels.get(sub, {}).get(l, "") for l in ("en", "ja", "id"))
        ve, vj, vi = (labels.get(val, {}).get(l, "") for l in ("en", "ja", "id"))
        if not all([se, sj, si, ve, vj, vi]):
            continue
        if any(len(x) > 60 for x in (se, sj, si, ve, vj, vi)):
            continue
        rows.append((se, sj, si, ve, vj, vi, sub, val))

    # merge: drop old capital_admin, add new
    lines = []
    for line in open(TSV, encoding="utf-8"):
        if not line.startswith("capital_admin\t"):
            lines.append(line.rstrip("\n"))
    for r in sorted(rows, key=lambda x: x[0].casefold()):
        lines.append("\t".join(["capital_admin"] + list(r)))
    with open(TSV, "w", encoding="utf-8") as f:
        for l in sorted(lines):
            f.write(l + "\n")

    # aliases for new entities
    have = set()
    for line in open(ALIASES, encoding="utf-8"):
        have.add(line.split("\t")[0])
    new_qids = [q for q in qids if q not in have]
    n_alias = 0
    if new_qids:
        print(f"[4/4] fetching aliases for {len(new_qids)} new entities...", flush=True)
        with open(ALIASES, "a", encoding="utf-8") as f:
            for i in range(0, len(new_qids), 180):
                values = " ".join(f"wd:{q}" for q in new_qids[i:i + 180])
                q = f"""
SELECT ?q ?alias ?l WHERE {{
  VALUES ?q {{ {values} }}
  ?q skos:altLabel ?alias .
  BIND(LANG(?alias) AS ?l)
  FILTER(LANG(?alias) IN ("en", "ja", "id"))
}} LIMIT 30000
"""
                per = {}
                for b in sparql(q, timeout=120):
                    qid = qid_of(b["q"]["value"])
                    lang = str(b["l"]["value"])
                    alias = norm_ws(b["alias"]["value"])
                    if not (2 <= len(alias) <= 40) or alias.isdigit():
                        continue
                    d = per.setdefault(qid, {"en": [], "ja": [], "id": []})
                    if alias not in d[lang] and len(d[lang]) < 10:
                        d[lang].append(alias)
                for qid, d in per.items():
                    for lang in ("en", "ja", "id"):
                        for a in d[lang]:
                            f.write(f"{qid}\t{lang}\t{a}\n")
                            n_alias += 1
                time.sleep(1.5)

    print(f"\nDone: capital_admin now {len(rows)} rows "
          f"(was 19); +{n_alias} aliases appended")


if __name__ == "__main__":
    main()
