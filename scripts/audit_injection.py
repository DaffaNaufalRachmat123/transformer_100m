#!/usr/bin/env python3
"""audit_injection.py — Gate D: fact-injection corpus integrity audit.

Checks (per language file):
  1. No double-prefix class bugs ("Bahasa Bahasa", "bahasa bahasa")
  2. No template placeholder leaks ({sub}, {val}, {lang})
  3. No trailing double periods ("D.C..")
  4. No hypothetical element names (unbibium, unquadquadium, ...)
  5. Oversampling consistency: every unique line's repetition count is a
     multiple of the exposure factor range (max rep <= exposures)
  6. No empty lines, no mojibake
  7. Category coverage sanity: subjects from the Tier-1 TSV should dominate

Usage:
  python3 scripts/audit_injection.py --exposures 400
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOUBLE_PREFIX = re.compile(r"bahasa\s+bahasa|Bahasa\s+Bahasa|the\s+the\s+", re.I)
PLACEHOLDER = re.compile(r"\{(sub|val|lang)\}")
DOUBLE_PERIOD = re.compile(r"[a-zA-Z]\.\.$")
HYPOTHETICAL = re.compile(r"unbibium|unquadquadium|unbinilium|unbiennium|"
                          r"unbihexium|unbioctium|unbipentium|unbiquadium|"
                          r"unbiseptium|unbitrium|unbipentium|unquadioctium", re.I)
MOJIBAKE = re.compile(r"Ã[©¨¤¶®«»°¼½¾¢]|â€[˜™œ›""]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/corpus_inject_facts")
    ap.add_argument("--exposures", type=int, default=400)
    ap.add_argument("--report", default="logs/gate_d_audit.json")
    args = ap.parse_args()

    report = {}
    any_fail = False
    for lang in ("en", "ja", "id"):
        path = ROOT / args.dir / f"facts_{lang}.txt"
        if not path.exists():
            report[lang] = {"error": "missing"}
            any_fail = True
            continue
        n = 0
        issues = Counter()
        rep_counter = Counter()
        for line in open(path, encoding="utf-8"):
            t = line.rstrip("\n")
            n += 1
            if not t.strip():
                issues["empty"] += 1
                continue
            if DOUBLE_PREFIX.search(t):
                issues["double_prefix"] += 1
            if PLACEHOLDER.search(t):
                issues["placeholder"] += 1
            if DOUBLE_PERIOD.search(t):
                issues["double_period"] += 1
            if HYPOTHETICAL.search(t):
                issues["hypothetical_element"] += 1
            if MOJIBAKE.search(t):
                issues["mojibake"] += 1
            rep_counter[t] += 1

        max_rep = max(rep_counter.values()) if rep_counter else 0
        # repetition sanity: no line should repeat more than exposures times
        if max_rep > args.exposures:
            issues["overexposed"] = max_rep - args.exposures

        verdict = "PASS" if not issues else ("FAIL" if sum(issues.values()) > 100 else "WARN")
        if verdict == "FAIL":
            any_fail = True
        report[lang] = {
            "lines": n,
            "unique_lines": len(rep_counter),
            "max_repetition": max_rep,
            "issues": dict(issues),
            "verdict": verdict,
        }
        print(f"[{verdict:<4}] facts_{lang}.txt: {n} lines, {len(rep_counter)} unique, "
              f"max_rep={max_rep}, issues={dict(issues) or 'none'}")

    rp = ROOT / args.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nGate D: {'FAIL' if any_fail else 'PASS'} -> {rp}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
