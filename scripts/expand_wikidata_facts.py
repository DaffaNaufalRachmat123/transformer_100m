#!/usr/bin/env python3
"""Expand the Wikidata fact DB beyond countries and chemical elements.

fetch_wikidata_facts_tier1.py queries three entity classes (country, chemical
element, first-level admin division). There are only ~200 countries and ~118
elements, so the DB tops out at 4,580 facts, which the SFT builder inflates to
52,377 rows through templates. That inflation is why fact accuracy sits at 24%:
the model memorises phrasings, not facts.

This widens the *entity* axis instead, which is what actually adds knowledge.
Every row requires both an English and a Japanese label, so nothing enters the
DB that the model cannot be asked about in either language.

Output matches the existing 8-column TSV so build_facts_dataset_v11.py needs no
change: category, subject_en, subject_ja, subject_id, value_en, value_ja,
value_id, qid. The _id columns mirror English; the mixed-SFT builder drops
Indonesian rows anyway.

  python3 scripts/expand_wikidata_facts.py --out data/wikidata_facts_expanded.tsv
  python3 scripts/expand_wikidata_facts.py --only city_country,film_director
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://query.wikidata.org/sparql"

# (entity class constraint, property, value kind, min sitelinks, single_valued)
#   value kind: "item" resolves the object's own EN/JA labels
#               "qty"  SI-normalised amount (metres for length/elevation)
#               "count" unitless integer (population)
#               "date" year only, negative years rendered as BC
#   single_valued: reject entities carrying more than one value for the
#   property. Wikidata lists every co-director, co-author, dual citizenship and
#   secondary occupation, so picking one arbitrarily produced claims like
#   "George Washington is a writer".
SPECS = {
    # --- geography -------------------------------------------------------
    "city_country":      ("wd:Q515",     "P17",   "item",  30, True),
    "city_population":   ("wd:Q515",     "P1082", "count", 30, True),
    "mountain_elevation":("wd:Q8502",    "P2044", "qty",   15, True),
    "mountain_country":  ("wd:Q8502",    "P17",   "item",  15, True),
    "river_length":      ("wd:Q4022",    "P2043", "qty",   15, True),
    "river_country":     ("wd:Q4022",    "P17",   "item",  15, True),
    "lake_country":      ("wd:Q23397",   "P17",   "item",  15, True),
    "island_country":    ("wd:Q23442",   "P17",   "item",  15, True),
    # --- culture ---------------------------------------------------------
    "film_director":     ("wd:Q11424",   "P57",   "item",  25, True),
    "film_year":         ("wd:Q11424",   "P577",  "date",  25, True),
    "book_author":       ("wd:Q7725634", "P50",   "item",  20, True),
    "band_origin":       ("wd:Q215380",  "P495",  "item",  20, True),
    # --- people ----------------------------------------------------------
    "person_birthyear":  ("wd:Q5",       "P569",  "date",  45, True),
    # Occupation and citizenship are legitimately multi-valued: most notable
    # people hold several. Demanding a single value discarded ~90% of them, so
    # these keep every entity and build_facts_expanded.py asks for "one of".
    "person_country":    ("wd:Q5",       "P27",   "item",  45, False),
    "person_occupation": ("wd:Q5",       "P106",  "item",  45, False),
    # --- organisations ---------------------------------------------------
    "university_country":("wd:Q3918",    "P17",   "item",  20, True),
    "company_country":   ("wd:Q4830453", "P17",   "item",  25, True),
    "airport_country":   ("wd:Q1248784", "P17",   "item",  15, True),
    # --- science ---------------------------------------------------------
    # P171 is "parent taxon", which resolves to whatever rank sits directly
    # above the entity (genus, subgenus, family). Calling it "class" made
    # every row false; the category is named for what the property means.
    "species_parent":    ("wd:Q16521",   "P171",  "item",  25, True),
    "planet_star":       ("wd:Q634",     "P397",  "item",  10, True),
}

# Rendering unit for SI-normalised quantities, per property.
QTY_UNITS = {
    "P2043": ("km", 0.001),   # length: metres -> kilometres
    "P2044": ("m", 1.0),      # elevation stays in metres
}


def sparql(query, max_retries=6, timeout=180):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "enja-fact-expander/1.0 (educational project)",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))["results"]["bindings"]
        except urllib.error.HTTPError as e:
            # 429/503 are the endpoint asking us to slow down, not a failure.
            wait = min(3 ** attempt + 2, 90)
            print(f"    [retry {attempt+1}] HTTP {e.code} — waiting {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            wait = min(3 ** attempt + 2, 90)
            print(f"    [retry {attempt+1}] {type(e).__name__}: {e} — waiting {wait}s",
                  flush=True)
            time.sleep(wait)
    return []


def build_query(cls, prop, kind, min_links, limit, offset, single=True):
    """One flat query; labels and value come back with the entity.

    No ORDER BY: ranking ~5M cities by sitelink count exceeds the endpoint's
    60s budget and returns 504. The sitelink FILTER plus the mandatory Japanese
    label already restrict this to notable entities, which is what the ordering
    was for.
    """
    if kind == "item":
        value_bind = f"""
  ?sub wdt:{prop} ?val .
  ?val rdfs:label ?val_en FILTER(lang(?val_en) = "en")
  ?val rdfs:label ?val_ja FILTER(lang(?val_ja) = "ja")"""
        value_sel = "?val_en ?val_ja"
    elif kind == "qty":
        # Normalised value: Wikidata stores river lengths in km and miles
        # interchangeably, so the raw amount is meaningless without its unit.
        # psn: converts to the SI base unit (metres here).
        value_bind = f"""
  ?sub p:{prop}/psn:{prop} [ wikibase:quantityAmount ?val_raw ] ."""
        value_sel = "?val_raw"
    else:
        value_bind = f"\n  ?sub wdt:{prop} ?val_raw ."
        value_sel = "?val_raw"

    # Uniqueness is enforced client-side, not with a SPARQL subquery: grouping
    # over wd:Q5 (~10M humans) exceeds the endpoint's 60s budget and every
    # person_* category came back HTTP 504. The query already emits one row per
    # value, so a QID appearing more than once in a page has multiple values.
    return f"""
SELECT ?sub ?sub_en ?sub_ja {value_sel} WHERE {{
  ?sub wdt:P31 {cls} ;
       wikibase:sitelinks ?links .
  FILTER(?links >= {min_links})
  ?sub rdfs:label ?sub_en FILTER(lang(?sub_en) = "en")
  ?sub rdfs:label ?sub_ja FILTER(lang(?sub_ja) = "ja"){value_bind}
}}
LIMIT {limit} OFFSET {offset}
"""


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def clean_number(s):
    """Wikidata returns '652230' or '+652230' or '652230.0'."""
    t = norm(s).lstrip("+")
    if t.endswith(".0"):
        t = t[:-2]
    return t


def harvest(name, spec, limit, max_rows, pause):
    cls, prop, kind, min_links, single = spec
    rows, offset, seen = [], 0, set()
    ambiguous = set()  # QIDs observed carrying more than one value
    while len(rows) < max_rows:
        q = build_query(cls, prop, kind, min_links, limit, offset, single)
        batch = sparql(q)
        if not batch:
            break

        # Group the page by entity first, so a multi-valued entity is dropped
        # rather than resolved by whichever row happened to arrive first.
        per_qid = {}
        for b in batch:
            per_qid.setdefault(b["sub"]["value"].rsplit("/", 1)[-1], []).append(b)

        for qid, hits in per_qid.items():
            if qid in seen or qid in ambiguous:
                continue
            if single and len(hits) > 1:
                ambiguous.add(qid)
                continue
            b = hits[0]
            sub_en, sub_ja = norm(b["sub_en"]["value"]), norm(b["sub_ja"]["value"])
            if kind == "item":
                val_en, val_ja = norm(b["val_en"]["value"]), norm(b["val_ja"]["value"])
            else:
                raw = b["val_raw"]["value"]
                if kind == "date":
                    m = re.match(r"(-?\d{1,5})-", raw)
                    if not m:
                        continue
                    y = int(m.group(1))
                    # A bare "-2531" reads as a negative year; spell out the era.
                    val_en = f"{-y} BC" if y < 0 else str(y)
                    val_ja = f"紀元前{-y}年" if y < 0 else f"{y}年"
                elif kind == "qty":
                    unit, scale = QTY_UNITS.get(prop, ("", 1.0))
                    try:
                        amt = float(raw) * scale
                    except ValueError:
                        continue
                    if amt <= 0:
                        continue
                    txt = (f"{amt:,.0f}" if amt >= 10
                           else f"{amt:,.2f}".rstrip("0").rstrip("."))
                    val_en = f"{txt} {unit}".strip()
                    val_ja = f"{txt}{unit}".strip()
                else:
                    val_en = val_ja = clean_number(raw)
            if not (sub_en and sub_ja and val_en and val_ja):
                continue
            seen.add(qid)
            rows.append((name, sub_en, sub_ja, sub_en, val_en, val_ja, val_en, qid))
        print(f"    {name}: {len(rows)} rows (offset {offset})", flush=True)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(pause)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wikidata_facts_expanded.tsv")
    ap.add_argument("--only", default="", help="comma-separated subset of categories")
    ap.add_argument("--limit", type=int, default=2000, help="entities per query")
    ap.add_argument("--max_per_cat", type=int, default=6000)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between queries")
    ap.add_argument("--report", default="logs/expand_facts_report.json")
    a = ap.parse_args()

    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(SPECS)
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        sys.exit(f"unknown categories: {unknown}\nknown: {sorted(SPECS)}")

    out_path = ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append-and-resume: a full harvest takes hours and the endpoint throttles.
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 8:
                done.add((parts[0], parts[7]))
        print(f"[RESUME] {len(done)} rows already in {a.out}")

    stats, total = {}, 0
    with out_path.open("a", encoding="utf-8") as f:
        for name in names:
            print(f"\n[CAT] {name}  {SPECS[name]}", flush=True)
            rows = harvest(name, SPECS[name], a.limit, a.max_per_cat, a.pause)
            fresh = [r for r in rows if (r[0], r[7]) not in done]
            for r in fresh:
                f.write("\t".join(r) + "\n")
                done.add((r[0], r[7]))
            f.flush()
            stats[name] = len(fresh)
            total += len(fresh)
            print(f"  [CAT] {name}: +{len(fresh)} new (total {total})", flush=True)

    rep = ROOT / a.report
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({"per_category": stats, "total_new": total,
                               "out": a.out}, indent=2), encoding="utf-8")
    print(f"\n[DONE] +{total} facts -> {a.out}")


if __name__ == "__main__":
    main()
