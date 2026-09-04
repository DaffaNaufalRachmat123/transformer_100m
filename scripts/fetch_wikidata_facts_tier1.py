#!/usr/bin/env python3
"""fetch_wikidata_facts_tier1.py — Tier-1 trilingual fact fetch (EN/JA/ID).

Two-phase SPARQL strategy (label services on full result sets are too slow):
  Phase 1: per category, fetch statements (QIDs, rank, time/precision) — fast.
  Phase 2: batch-fetch rdfs:label + skos:altLabel for all QIDs via VALUES
           chunks, in all 3 languages.

Quality rules:
  - preferred-rank statements preferred, latest point-in-time for pop/area
  - sitelinks >= 5 (>= 15 for admin divisions)
  - all three language labels required (native Wikidata labels, no MT)
  - one answer per (subject, category) unless the category is multi-valued
  - known data fixes re-applied (Naoero->Nauru, euro, truncated values)

13 categories: 9 originals + driving_side, calling_code, founding_date,
capital_admin.

Outputs:
  data/wikidata_facts_tier1.tsv    cat \t sub_en \t sub_ja \t sub_id \t
                                    val_en \t val_ja \t val_id \t sub_qid \t val_qid
  data/facts_aliases_tier1.tsv      qid \t lang \t alias
  logs/tier1_fetch_report.json      per-category stats
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://query.wikidata.org/sparql"

COUNTRY = "?sub wdt:P31 wd:Q6256 ."
ELEMENT = "?sub wdt:P31 wd:Q11344 ."
ADMIN1 = "?sub wdt:P31 wd:Q10864048 ."

CATEGORIES = {
    "capital":           (COUNTRY, "P36",   "item", False),
    "currency":          (COUNTRY, "P38",   "item", False),
    "official_language": (COUNTRY, "P37",   "item", True),
    "population":        (COUNTRY, "P1082", "num",  False),
    "highest_point":     (COUNTRY, "P610",  "item", False),
    "continent_of":      (COUNTRY, "P30",   "item", False),
    "area":              (COUNTRY, "P2046", "num",  False),
    "element_symbol":    (ELEMENT, "P246",  "symbol", False),
    "element_number":    (ELEMENT, "P1086", "num",  False),
    "driving_side":      (COUNTRY, "P1622", "item", False),
    "calling_code":      (COUNTRY, "P2852", "symbol", True),
    "founding_date":     (COUNTRY, "P571",  "date", False),
    "capital_admin":     (ADMIN1,  "P36",   "item", False),
}

MIN_SITELINKS = 5
MIN_SITELINKS_ADMIN = 15

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "ja": ["1月", "2月", "3月", "4月", "5月", "6月",
           "7月", "8月", "9月", "10月", "11月", "12月"],
    "id": ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
           "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
}


def sparql(query, max_retries=6, timeout=120):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "Mozilla/5.0 (trilingual-fact-fetcher/4.1; educational project)",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", {}).get("bindings", [])
        except urllib.error.HTTPError as e:
            wait = min(3 ** attempt + 2, 60)
            print(f"    [retry {attempt+1}] HTTP {e.code} — waiting {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            wait = min(3 ** attempt + 2, 60)
            print(f"    [retry {attempt+1}] {e} — waiting {wait}s", flush=True)
            time.sleep(wait)
    return []


def qid_of(iri):
    s = str(iri)
    return s.rsplit("/", 1)[-1] if "/" in s else s


def norm_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def clean_numeric(v):
    v = str(v).strip().replace(",", "")
    try:
        x = float(v)
        if abs(x - round(x)) < 1e-6:
            return str(int(round(x)))
        return f"{x:.1f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(v).strip()


def parse_wd_time(raw):
    m = re.match(r"^(-?)(\d{4,})-(\d{2})-(\d{2})T", str(raw))
    if not m:
        return None
    return (m.group(1) == "-", int(m.group(2)), int(m.group(3)), int(m.group(4)))


def format_date(raw, prec, lang):
    t = parse_wd_time(raw)
    if t is None:
        return None
    neg, y, mo, d = t
    era = {"en": " BC", "ja": "紀元前", "id": " SM"} if neg else {"en": "", "ja": "", "id": ""}
    if prec is None:
        prec = 11
    if prec <= 7:
        return None
    if prec == 8:
        dec = (y // 10) * 10
        return {"en": f"{dec}s{era['en']}", "ja": f"{era['ja']}{dec}年代",
                "id": f"{dec}-an{era['id']}"}[lang]
    if prec == 9:
        return {"en": f"{y}{era['en']}", "ja": f"{era['ja']}{y}年", "id": f"{y}{era['id']}"}[lang]
    if prec == 10:
        m = MONTHS[lang][mo - 1]
        if lang == "ja":
            return f"{era['ja']}{y}年{m}"
        return f"{m} {y}{era[lang]}"
    if lang == "ja":
        return f"{era['ja']}{y}年{MONTHS['ja'][mo-1]}{d}日"
    return f"{d} {MONTHS[lang][mo-1]} {y}{era[lang]}"


def rank_value(rank_iri):
    s = str(rank_iri).rsplit("#", 1)[-1]
    return {"PreferredRank": 2, "NormalRank": 1, "DeprecatedRank": 0}.get(s, 1)


# ── Phase 1: statements (no labels — fast) ──────────────────────────────────
def fetch_statements(cat):
    subj_filter, prop, val_kind, _ = CATEGORIES[cat]
    sl = MIN_SITELINKS_ADMIN if cat == "capital_admin" else MIN_SITELINKS
    val_select = "?val ?rank"
    time_clause = ""
    prec_clause = ""
    if cat in ("population", "area"):
        val_select = "?val ?rank ?time"
        time_clause = "OPTIONAL { ?st pq:P585 ?time . }"
    if cat == "founding_date":
        val_select = "?val ?rank ?prec"
        prec_clause = "OPTIONAL { ?st psv:P571 ?vn . ?vn wikibase:timePrecision ?prec . }"
    q = f"""
SELECT DISTINCT ?sub {val_select} WHERE {{
  {subj_filter}
  ?sub p:{prop} ?st .
  ?st ps:{prop} ?val .
  ?st wikibase:rank ?rank .
  {time_clause}
  {prec_clause}
  ?sub wikibase:sitelinks ?sl . FILTER(?sl >= {sl})
}} LIMIT 20000
"""
    return sparql(q, timeout=180)


# ── Phase 2: batched label + alias fetch ────────────────────────────────────
def fetch_labels(qids, chunk=180):
    """qid -> {en, ja, id} labels (missing = absent)."""
    labels = {}
    qids = sorted(qids)
    for i in range(0, len(qids), chunk):
        values = " ".join(f"wd:{q}" for q in qids[i:i + chunk])
        q = f"""
SELECT ?q ?len ?lja ?lid WHERE {{
  VALUES ?q {{ {values} }}
  OPTIONAL {{ ?q rdfs:label ?len . FILTER(LANG(?len) = "en") }}
  OPTIONAL {{ ?q rdfs:label ?lja . FILTER(LANG(?lja) = "ja") }}
  OPTIONAL {{ ?q rdfs:label ?lid . FILTER(LANG(?lid) = "id") }}
}}
"""
        for b in sparql(q, timeout=120):
            qid = qid_of(b["q"]["value"])
            entry = labels.setdefault(qid, {})
            for key, lang in (("len", "en"), ("lja", "ja"), ("lid", "id")):
                if key in b:
                    entry[lang] = norm_ws(b[key]["value"])
        if (i // chunk) % 5 == 4:
            print(f"    labels: {i + chunk}/{len(qids)}", flush=True)
        time.sleep(1.5)
    return labels


def fetch_aliases(qids, chunk=180):
    aliases = {}
    qids = sorted(qids)
    for i in range(0, len(qids), chunk):
        values = " ".join(f"wd:{q}" for q in qids[i:i + chunk])
        q = f"""
SELECT ?q ?alias ?l WHERE {{
  VALUES ?q {{ {values} }}
  ?q skos:altLabel ?alias .
  BIND(LANG(?alias) AS ?l)
  FILTER(LANG(?alias) IN ("en", "ja", "id"))
}} LIMIT 30000
"""
        for b in sparql(q, timeout=120):
            qid = qid_of(b["q"]["value"])
            lang = str(b["l"]["value"])
            alias = norm_ws(b["alias"]["value"])
            if not (2 <= len(alias) <= 40) or alias.isdigit():
                continue
            if "\t" in alias or "\n" in alias:
                continue
            per = aliases.setdefault(qid, {"en": [], "ja": [], "id": []})
            if alias not in per[lang] and len(per[lang]) < 10:
                per[lang].append(alias)
        if (i // chunk) % 5 == 4:
            print(f"    aliases: {i + chunk}/{len(qids)}", flush=True)
        time.sleep(1.5)
    return aliases


# ── Known data fixes ────────────────────────────────────────────────────────
DROP_EXACT = {
    ("currency", "France", "CFP Franc"),
    ("capital", "Grenada", "St"),
    ("capital", "United States", "Washington, D"),
}
FIX_VAL = {
    ("capital", "Antigua and Barbuda"): "Saint John's",
    ("highest_point", "United States"): "Denali",
}


def apply_known_fixes(rows):
    fixed, dropped = [], 0
    for r in rows:
        if (r["cat"], r["sub_en"], r["val_en"]) in DROP_EXACT:
            dropped += 1
            continue
        if r["sub_en"] == "Naoero":
            r["sub_en"] = "Nauru"
        if r["val_en"] == "Q4916":
            r["val_en"], r["val_id"] = "euro", "euro"
        key = (r["cat"], r["sub_en"])
        if key in FIX_VAL and r["val_en"].startswith("Q"):
            r["val_en"] = FIX_VAL[key]
        if r["cat"] in ("population", "area"):
            r["val_en"] = clean_numeric(r["val_en"])
            r["val_ja"] = r["val_id"] = r["val_en"]
        fixed.append(r)
    return fixed, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wikidata_facts_tier1.tsv")
    ap.add_argument("--aliases_out", default="data/facts_aliases_tier1.tsv")
    ap.add_argument("--report", default="logs/tier1_fetch_report.json")
    args = ap.parse_args()

    # ── Phase 1: statements per category
    statements = {}   # cat -> [ {sub_qid, val_qid/raw, rank, time, prec}, ... ]
    for cat in CATEGORIES:
        print(f"[{cat}] fetching statements...", flush=True)
        bindings = fetch_statements(cat)
        _, _, val_kind, _ = CATEGORIES[cat]
        rows = []
        for b in bindings:
            entry = {
                "sub_qid": qid_of(b["sub"]["value"]),
                "rank": rank_value(b.get("rank", {}).get("value", "NormalRank")),
                "time": b.get("time", {}).get("value", ""),
                "prec": b.get("prec", {}).get("value"),
            }
            if val_kind == "item":
                entry["val_qid"] = qid_of(b["val"]["value"])
            else:
                entry["raw_val"] = str(b["val"]["value"])
            rows.append(entry)
        statements[cat] = rows
        print(f"[{cat}] {len(rows)} statements", flush=True)
        time.sleep(4)

    # ── Resolve one answer per (cat, subject)
    resolved = []   # (cat, sub_qid, val_qid or raw, multi_flag)
    for cat, rows in statements.items():
        _, _, val_kind, multi_ok = CATEGORIES[cat]
        by_sub = {}
        for r in rows:
            key = (cat, r["sub_qid"])
            by_sub.setdefault(key, []).append(r)
        for key, group in by_sub.items():
            best_rank = max(r["rank"] for r in group)
            group = [r for r in group if r["rank"] == best_rank]
            if all(r["time"] for r in group):
                latest = max(r["time"] for r in group)
                group = [r for r in group if r["time"] == latest]
            if val_kind == "item":
                distinct = {r["val_qid"] for r in group}
                if len(distinct) == 1:
                    resolved.append((cat, key[1], group[0]["val_qid"], multi_ok))
                elif multi_ok:
                    for r in group:
                        resolved.append((cat, key[1], r["val_qid"], True))
                # else: conflicting multi-value single-answer cat -> dropped
            else:
                vals = {(clean_numeric(r["raw_val"]) if val_kind == "num"
                         else norm_ws(r["raw_val"])) for r in group}
                if len(vals) == 1:
                    r = group[0]
                    resolved.append((cat, key[1], {
                        "kind": val_kind, "raw": r["raw_val"], "prec": r["prec"]}, multi_ok))
                elif multi_ok:
                    for r in group:
                        resolved.append((cat, key[1], {
                            "kind": val_kind, "raw": r["raw_val"], "prec": r["prec"]}, True))

    # ── Phase 2: labels for all subject + item-value QIDs
    need_labels = set()
    for cat, sub_qid, val, _ in resolved:
        need_labels.add(sub_qid)
        if isinstance(val, str) and val.startswith("Q"):
            need_labels.add(val)
    print(f"\n[labels] fetching for {len(need_labels)} entities...", flush=True)
    labels = fetch_labels(sorted(need_labels))

    def label3(qid):
        e = labels.get(qid, {})
        return e.get("en", ""), e.get("ja", ""), e.get("id", "")

    facts, dropped_missing_label = [], 0
    for cat, sub_qid, val, _multi in resolved:
        val_kind = CATEGORIES[cat][2]
        sub_en, sub_ja, sub_id = label3(sub_qid)
        # all 3 subject labels required, none may equal the QID
        if not all([sub_en, sub_ja, sub_id]):
            dropped_missing_label += 1
            continue
        if len(sub_en) > 60 or len(sub_ja) > 60 or len(sub_id) > 60:
            dropped_missing_label += 1
            continue
        if val_kind == "item":
            val_en, val_ja, val_id = label3(val)
            if not all([val_en, val_ja, val_id]):
                dropped_missing_label += 1
                continue
            if len(val_en) > 60 or len(val_ja) > 60 or len(val_id) > 60:
                dropped_missing_label += 1
                continue
            val_qid = val
        else:
            if val_kind == "num":
                val_en = clean_numeric(val["raw"])
            elif val_kind == "symbol":
                val_en = norm_ws(val["raw"])
            else:  # date
                prec = int(float(val["prec"])) if val["prec"] else 11
                vals = [format_date(val["raw"], prec, lg) for lg in ("en", "ja", "id")]
                if any(v is None for v in vals):
                    dropped_missing_label += 1
                    continue
                val_en, val_ja, val_id = vals
            val_ja = val_id = val_en
            val_qid = ""
        facts.append({
            "cat": cat, "sub_qid": sub_qid,
            "sub_en": sub_en, "sub_ja": sub_ja, "sub_id": sub_id,
            "val_qid": val_qid, "val_en": val_en, "val_ja": val_ja, "val_id": val_id,
        })

    print(f"[labels] facts with full trilingual labels: {len(facts)} "
          f"(dropped {dropped_missing_label} missing-label)", flush=True)

    facts, dropped_fixes = apply_known_fixes(facts)

    # ID-label leak check (item-valued rows where val_id == val_en)
    leak_flags = [
        {"cat": r["cat"], "sub_en": r["sub_en"], "val_en": r["val_en"]}
        for r in facts
        if r["val_qid"] and r["val_id"] == r["val_en"] and re.search(r"[A-Za-z]", r["val_en"])
    ]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sorted(facts, key=lambda x: (x["cat"], x["sub_en"].casefold())):
            f.write("\t".join([r["cat"], r["sub_en"], r["sub_ja"], r["sub_id"],
                               r["val_en"], r["val_ja"], r["val_id"],
                               r["sub_qid"], r["val_qid"]]) + "\n")

    # ── Phase 2b: aliases
    alias_map = {}
    qids = sorted({r["sub_qid"] for r in facts} | {r["val_qid"] for r in facts if r["val_qid"]})
    print(f"\n[aliases] fetching for {len(qids)} entities...", flush=True)
    alias_map = fetch_aliases(qids)
    al_path = ROOT / args.aliases_out
    n_alias = 0
    with open(al_path, "w", encoding="utf-8") as f:
        for qid in sorted(alias_map):
            for lang in ("en", "ja", "id"):
                for a in alias_map[qid][lang]:
                    f.write(f"{qid}\t{lang}\t{a}\n")
                    n_alias += 1
    print(f"[aliases] {n_alias} aliases -> {al_path}", flush=True)

    # ── Report
    by_cat = {}
    for r in facts:
        by_cat[r["cat"]] = by_cat.get(r["cat"], 0) + 1
    report = {
        "total_facts": len(facts),
        "by_category": by_cat,
        "dropped_missing_label": dropped_missing_label,
        "dropped_known_fixes": dropped_fixes,
        "id_label_leak_candidates": len(leak_flags),
        "alias_entities": len(alias_map),
        "total_aliases": n_alias,
    }
    rep_path = ROOT / args.report
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if leak_flags:
        with open(ROOT / "logs/tier1_id_leak_candidates.json", "w", encoding="utf-8") as f:
            json.dump(leak_flags, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(facts)} Tier-1 facts -> {out_path}")
    print(f"  by category: {by_cat}")
    print(f"  id-label leak candidates: {len(leak_flags)}")


if __name__ == "__main__":
    main()
