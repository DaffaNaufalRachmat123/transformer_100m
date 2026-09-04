#!/usr/bin/env python3
"""audit_facts_tier1.py — Audit the Tier-1 trilingual facts TSV.

Checks:
  1. Schema: 9 columns, no empty fields among the 7 required
  2. Uniqueness: one answer per (category, sub_qid) except multi-ok cats
  3. QID validity: labels must not equal QIDs (label-service fallback bug)
  4. Numeric categories parse as numbers
  5. Dates present for founding_date in all 3 languages
  6. ID-leak: item-valued rows where val_id == val_en (review candidates)
  7. Subject sanity: same QID must not map to wildly different subject names
  8. Category distribution report

Usage:
  python3 scripts/audit_facts_tier1.py [--tsv data/wikidata_facts_tier1.tsv]
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MULTI_OK = {"official_language", "calling_code"}
NUMERIC_CATS = {"population", "area", "element_number"}
SYMBOL_CATS = {"element_symbol", "calling_code"}
DATE_CATS = {"founding_date"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/wikidata_facts_tier1.tsv")
    ap.add_argument("--report", default="logs/tier1_audit.json")
    args = ap.parse_args()

    path = ROOT / args.tsv
    rows = []
    errors = []
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        p = line.rstrip("\n").split("\t")
        if len(p) < 9:
            errors.append(f"line {ln}: expected 9 cols, got {len(p)}")
            continue
        cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id, sub_qid, val_qid = [x.strip() for x in p[:9]]
        if not all([cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id, sub_qid]):
            errors.append(f"line {ln}: empty required field")
            continue
        if not re.match(r"^Q\d+$", sub_qid):
            errors.append(f"line {ln}: bad sub_qid {sub_qid}")
            continue
        if cat not in {"capital", "currency", "official_language", "population",
                       "highest_point", "continent_of", "area", "element_symbol",
                       "element_number", "driving_side", "calling_code",
                       "founding_date", "capital_admin"}:
            errors.append(f"line {ln}: unknown category {cat}")
            continue
        # QID fallback leak in labels
        for label, qid in ((sub_en, sub_qid), (sub_ja, sub_qid), (sub_id, sub_qid)):
            if label == qid:
                errors.append(f"line {ln}: label equals QID ({label})")
        if val_qid:
            for label, qid in ((val_en, val_qid), (val_ja, val_qid), (val_id, val_qid)):
                if label == qid:
                    errors.append(f"line {ln}: value label equals QID ({label})")
        # numeric check
        if cat in NUMERIC_CATS:
            if not re.match(r"^\d+(\.\d+)?$", val_en.replace(",", "")):
                errors.append(f"line {ln}: non-numeric value for {cat}: {val_en!r}")
        if cat in DATE_CATS:
            if val_en == val_ja == val_id and re.match(r"^\d{4}-\d{2}-\d{2}T", val_en):
                errors.append(f"line {ln}: raw unformatted date {val_en!r}")
        rows.append({
            "cat": cat, "sub_en": sub_en, "sub_ja": sub_ja, "sub_id": sub_id,
            "val_en": val_en, "val_ja": val_ja, "val_id": val_id,
            "sub_qid": sub_qid, "val_qid": val_qid,
        })

    # uniqueness per (cat, sub_qid)
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["cat"], r["sub_qid"])].append(r)
    conflicts = {k: [x["val_en"] for x in v] for k, v in by_key.items()
                 if len({x["val_en"].casefold() for x in v}) > 1 and k[0] not in MULTI_OK}

    # same QID with inconsistent subject labels (should be identical since one fetch)
    qid_names = defaultdict(set)
    for r in rows:
        qid_names[r["sub_qid"]].add(r["sub_en"])
    unstable = {q: sorted(v) for q, v in qid_names.items() if len(v) > 1}

    # ID-leak candidates (item-valued)
    id_leak = [r for r in rows if r["val_qid"] and r["val_id"] == r["val_en"]
               and re.search(r"[A-Za-z]", r["val_en"])]

    by_cat = Counter(r["cat"] for r in rows)
    report = {
        "total_rows": len(rows),
        "errors": errors[:200],
        "n_errors": len(errors),
        "multi_value_conflicts": {f"{c}/{q}": v for (c, q), v in conflicts.items()},
        "n_conflicts": len(conflicts),
        "unstable_subject_qids": {k: v for k, v in list(unstable.items())[:50]},
        "n_unstable": len(unstable),
        "id_leak_candidates": [{"cat": r["cat"], "sub_en": r["sub_en"], "val_en": r["val_en"]}
                                for r in id_leak],
        "n_id_leak": len(id_leak),
        "by_category": dict(by_cat),
    }

    rep_path = ROOT / args.report
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    ok = not errors and not conflicts and not unstable
    print(f"Rows: {len(rows)} | errors: {len(errors)} | conflicts: {len(conflicts)} "
          f"| unstable QIDs: {len(unstable)} | id-leak candidates: {len(id_leak)}")
    print(f"By category: {dict(by_cat)}")
    print(f"AUDIT: {'PASS' if ok else 'FAIL'} -> {rep_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
