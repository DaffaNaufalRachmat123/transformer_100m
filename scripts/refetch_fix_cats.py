#!/usr/bin/env python3
"""refetch_fix_cats.py — Re-fetch founding_date & calling_code (bug fixes).

Two bugs in the first Tier-1 pass:
  1. founding_date: the trilingual formatted dates were overwritten by
     `val_ja = val_id = val_en` (the numeric-category line).
  2. calling_code: P2852 is ITEM-valued (one entity per code like Q1061257),
     but it was treated as a literal — raw IRIs leaked into the TSV.

This script re-fetches both categories correctly and REPLACES their rows in
data/wikidata_facts_tier1.tsv.
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

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    "ja": ["1月", "2月", "3月", "4月", "5月", "6月",
           "7月", "8月", "9月", "10月", "11月", "12月"],
    "id": ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
           "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
}


def sparql(query, max_retries=6, timeout=180):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": query})
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "Mozilla/5.0 (trilingual-fact-fetcher/4.3; educational project)",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")).get("results", {}).get("bindings", [])
        except Exception as e:
            wait = min(3 ** attempt + 2, 60)
            print(f"    [retry {attempt+1}] {e} — waiting {wait}s", flush=True)
            time.sleep(wait)
    return []


def qid_of(iri):
    return str(iri).rsplit("/", 1)[-1] if "/" in str(iri) else str(iri)


def norm_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def rank_value(iri):
    return {"PreferredRank": 2, "NormalRank": 1, "DeprecatedRank": 0}.get(
        str(iri).rsplit("#", 1)[-1], 1)


def parse_wd_time(raw):
    m = re.match(r"^(-?)(\d{4,})-(\d{2})-(\d{2})T", str(raw))
    return (m.group(1) == "-", int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None


def format_date(raw, prec, lang):
    t = parse_wd_time(raw)
    if t is None:
        return None
    neg, y, mo, d = t
    era = {"en": " BC", "ja": "紀元前", "id": " SM"} if neg else {"en": "", "ja": "", "id": ""}
    prec = prec or 11
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
        return f"{era['ja']}{y}年{m}" if lang == "ja" else f"{m} {y}{era[lang]}"
    if lang == "ja":
        return f"{era['ja']}{y}年{MONTHS['ja'][mo-1]}{d}日"
    return f"{d} {MONTHS[lang][mo-1]} {y}{era[lang]}"


def fetch_labels(qids, chunk=180):
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
            e = labels.setdefault(qid_of(b["q"]["value"]), {})
            for key, lang in (("len", "en"), ("lja", "ja"), ("lid", "id")):
                if key in b:
                    e[lang] = norm_ws(b[key]["value"])
        time.sleep(1.5)
    return labels


def main():
    # ── founding_date: statement + precision ───────────────────────────────
    print("[founding_date] fetching...", flush=True)
    bindings = sparql("""
SELECT DISTINCT ?sub ?val ?rank ?prec WHERE {
  ?sub wdt:P31 wd:Q6256 ; p:P571 ?st .
  ?st ps:P571 ?val .
  ?st wikibase:rank ?rank .
  OPTIONAL { ?st psv:P571 ?vn . ?vn wikibase:timePrecision ?prec . }
  ?sub wikibase:sitelinks ?sl . FILTER(?sl >= 5)
} LIMIT 5000
""")
    best = {}
    for b in bindings:
        sub = qid_of(b["sub"]["value"])
        rank = rank_value(b.get("rank", {}).get("value"))
        if sub not in best or rank > best[sub][1]:
            best[sub] = (b["val"]["value"], rank, b.get("prec", {}).get("value"))
    fd_labels = fetch_labels(sorted(best.keys()))

    fd_rows = []
    for sub, (raw, _rank, prec) in best.items():
        se, sj, si = (fd_labels.get(sub, {}).get(l, "") for l in ("en", "ja", "id"))
        if not (se and sj and si):
            continue
        prec_i = int(float(prec)) if prec else 11
        vals = [format_date(raw, prec_i, lg) for lg in ("en", "ja", "id")]
        if any(v is None for v in vals):
            continue
        fd_rows.append((se, sj, si, vals[0], vals[1], vals[2], sub, ""))
    print(f"[founding_date] {len(fd_rows)} trilingual rows", flush=True)

    # ── calling_code: item-valued, label + numeric extraction ─────────────
    # P2852 mixes real ITU calling codes with historic/emergency numbers
    # (000, 911, 110 police-JP, 180 fire-EG...). Validate every candidate
    # against the official ITU-T E.164 country-code whitelist.
    ITU_CODES = {
        1, 7, 20, 27, 30, 31, 32, 33, 34, 36, 39, 40, 41, 43, 44, 45, 46, 47,
        48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66,
        81, 82, 84, 86, 90, 91, 92, 93, 94, 95, 98,
        211, 212, 213, 216, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228,
        229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
        243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
        257, 258, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
        290, 291, 297, 298, 299,
        350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
        370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
        385, 386, 387, 389, 420, 421, 423,
        500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
        590, 591, 592, 593, 594, 595, 596, 597, 598, 599,
        670, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683,
        685, 686, 687, 688, 689, 690, 691, 692,
        850, 852, 853, 855, 856, 880, 886,
        960, 961, 962, 963, 964, 965, 966, 967, 968, 970, 971, 972, 973, 974,
        975, 976, 977, 992, 993, 994, 995, 996, 998,
    }
    print("[calling_code] fetching...", flush=True)
    bindings = sparql("""
SELECT DISTINCT ?sub ?val ?rank WHERE {
  ?sub wdt:P31 wd:Q6256 ; p:P2852 ?st .
  ?st ps:P2852 ?val .
  ?st wikibase:rank ?rank .
  ?sub wikibase:sitelinks ?sl . FILTER(?sl >= 5)
} LIMIT 5000
""")
    cc = {}
    for b in bindings:
        sub = qid_of(b["sub"]["value"])
        val = qid_of(b["val"]["value"])
        rank = rank_value(b.get("rank", {}).get("value"))
        cc.setdefault(sub, []).append((val, rank))
    cc_labels = fetch_labels(sorted(set(list(cc.keys()) + [v for vs in cc.values() for v, _ in vs])))

    cc_rows = []
    for sub, vals in cc.items():
        se, sj, si = (cc_labels.get(sub, {}).get(l, "") for l in ("en", "ja", "id"))
        if not (se and sj and si):
            continue
        best_rank = max(r for _, r in vals)
        codes = {}   # code -> val_qid
        for val, rank in vals:
            if rank != best_rank:
                continue
            label = cc_labels.get(val, {}).get("en", "")
            m = re.match(r"^\+?(\d{1,4})$", label.strip())
            if not m:
                continue
            digits = m.group(1)
            if int(digits) not in ITU_CODES:
                continue
            codes.setdefault(digits, val)
        if not codes:
            continue
        # deterministic pick: shortest code (the umbrella ITU zone code)
        code = min(codes, key=len)
        cc_rows.append((se, sj, si, code, code, code, sub, codes[code]))
    print(f"[calling_code] {len(cc_rows)} trilingual rows", flush=True)

    # ── replace rows in TSV ────────────────────────────────────────────────
    lines = [l.rstrip("\n") for l in open(TSV, encoding="utf-8")
             if not l.startswith("founding_date\t") and not l.startswith("calling_code\t")]
    for r in fd_rows:
        lines.append("\t".join(["founding_date"] + list(r)))
    for r in cc_rows:
        lines.append("\t".join(["calling_code"] + list(r)))
    with open(TSV, "w", encoding="utf-8") as f:
        for l in sorted(lines):
            f.write(l + "\n")
    print(f"\nDone: TSV now {len(lines)} rows "
          f"(founding_date {len(fd_rows)}, calling_code {len(cc_rows)})")


if __name__ == "__main__":
    main()
