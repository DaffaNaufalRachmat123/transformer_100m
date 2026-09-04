#!/usr/bin/env python3
"""build_fact_injection_corpus.py — Pretrain knowledge-injection corpus.

Renders Tier-1 trilingual facts as NATURAL declarative text (no chat
template) with multiple surface forms per fact per language, then repeats
them to reach a target exposure count. Output is plain text, one document
per line — the exact format the pretrain loader consumes.

Volume math: ~3.4K facts x 3 langs x ~8 variants x ~20 tokens ≈ 160M tokens
when exposures=1 (each unique rendering once). The --exposures flag repeats
shuffled renderings; recommended total budget: 120-180M tokens (pretrain
weight ~0.10-0.12).

Output:
  data/corpus_inject_facts/facts_{en,ja,id}.txt

Usage:
  python3 scripts/build_fact_injection_corpus.py [--exposures 1]
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Declarative surface forms per category per language ────────────────────
# {sub}/{val} substitution; multiple variants per fact.
DECL = {
    "en": {
        "capital": [
            "The capital of {sub} is {val}.",
            "{val} is the capital of {sub}.",
            "{sub}'s capital city is {val}.",
        ],
        "capital_admin": [
            "The capital of {sub} is {val}.",
            "{val} is the capital of {sub}.",
        ],
        "currency": [
            "The currency of {sub} is the {val}.",
            "{sub} uses the {val} as its currency.",
            "In {sub}, people pay with the {val}.",
        ],
        "official_language": [
            "An official language of {sub} is {val}.",
            "{val} is an official language of {sub}.",
            "{sub} has {val} as an official language.",
        ],
        "population": [
            "The population of {sub} is about {val}.",
            "{sub} has a population of roughly {val}.",
            "About {val} people live in {sub}.",
        ],
        "highest_point": [
            "The highest point of {sub} is {val}.",
            "{val} is the highest point of {sub}.",
        ],
        "continent_of": [
            "{sub} is located in {val}.",
            "{sub} is a country in {val}.",
            "{sub} lies in {val}.",
        ],
        "area": [
            "The area of {sub} is about {val} square kilometers.",
            "{sub} covers an area of roughly {val} square kilometers.",
        ],
        "element_symbol": [
            "The chemical symbol of {sub} is {val}.",
            "{sub} has the chemical symbol {val}.",
        ],
        "element_number": [
            "The atomic number of {sub} is {val}.",
            "{sub} has the atomic number {val}.",
        ],
        "driving_side": [
            "In {sub}, people drive on the {val}.",
            "Traffic in {sub} drives on the {val}.",
        ],
        "calling_code": [
            "The international calling code of {sub} is +{val}.",
            "To call {sub}, dial +{val}.",
        ],
        "founding_date": [
            "{sub} was founded on {val}.",
            "{sub} was established on {val}.",
        ],
    },
    "ja": {
        "capital": [
            "{sub}の首都は{val}です。",
            "{val}は{sub}の首都です。",
            "{sub}の首都は{val}である。",
        ],
        "capital_admin": [
            "{sub}の州都は{val}です。",
            "{val}は{sub}の州都です。",
        ],
        "currency": [
            "{sub}の通貨は{val}です。",
            "{sub}では{val}が使われています。",
        ],
        "official_language": [
            "{sub}の公用語は{val}です。",
            "{val}は{sub}の公用語です。",
        ],
        "population": [
            "{sub}の人口はおよそ{val}人です。",
            "{sub}には約{val}人が住んでいます。",
        ],
        "highest_point": [
            "{sub}で一番高い場所は{val}です。",
            "{val}は{sub}の最高地点です。",
        ],
        "continent_of": [
            "{sub}は{val}にあります。",
            "{sub}は{val}の国です。",
        ],
        "area": [
            "{sub}の面積はおよそ{val}平方キロメートルです。",
            "{sub}の面積は約{val}km²です。",
        ],
        "element_symbol": [
            "{sub}の化学記号は{val}です。",
            "{sub}の元素記号は{val}です。",
        ],
        "element_number": [
            "{sub}の原子番号は{val}です。",
        ],
        "driving_side": [
            "{sub}では{val}です。",
            "{sub}の交通は{val}を採用しています。",
        ],
        "calling_code": [
            "{sub}の国際電話番号は+{val}です。",
        ],
        "founding_date": [
            "{sub}は{val}に建国されました。",
            "{sub}の建国日は{val}です。",
        ],
    },
    "id": {
        "capital": [
            "Ibu kota {sub} adalah {val}.",
            "{val} adalah ibu kota {sub}.",
            "{sub} beribu kota di {val}.",
        ],
        "capital_admin": [
            "Ibu kota {sub} adalah {val}.",
            "{val} adalah ibu kota {sub}.",
        ],
        "currency": [
            "Mata uang {sub} adalah {val}.",
            "{sub} menggunakan {val} sebagai mata uangnya.",
        ],
        "official_language": [
            "Bahasa resmi {sub} adalah bahasa {val}.",
            "Bahasa {val} merupakan bahasa resmi {sub}.",
        ],
        "population": [
            "Jumlah penduduk {sub} sekitar {val} jiwa.",
            "{sub} memiliki populasi sekitar {val} jiwa.",
        ],
        "highest_point": [
            "Titik tertinggi di {sub} adalah {val}.",
            "{val} adalah titik tertinggi di {sub}.",
        ],
        "continent_of": [
            "{sub} terletak di {val}.",
            "{sub} adalah negara di {val}.",
        ],
        "area": [
            "Luas wilayah {sub} sekitar {val} kilometer persegi.",
            "{sub} memiliki luas sekitar {val} km persegi.",
        ],
        "element_symbol": [
            "Lambang kimia {sub} adalah {val}.",
            "{sub} memiliki lambang kimia {val}.",
        ],
        "element_number": [
            "Nomor atom {sub} adalah {val}.",
        ],
        "driving_side": [
            "Di {sub}, orang berkendara di {val}.",
            "Lalu lintas di {sub} menggunakan {val}.",
        ],
        "calling_code": [
            "Kode telepon internasional {sub} adalah +{val}.",
            "Untuk menelepon {sub}, gunakan kode +{val}.",
        ],
        "founding_date": [
            "{sub} didirikan pada {val}.",
            "{sub} berdiri pada tanggal {val}.",
        ],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/wikidata_facts_tier1.tsv")
    ap.add_argument("--out_dir", default="data/corpus_inject_facts")
    ap.add_argument("--exposures", type=int, default=1,
                    help="how many times the full rendering pool repeats")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    facts = []
    dropped_hypothetical = 0
    for line in open(ROOT / args.tsv, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id = [x.strip() for x in p[:7]]
        if cat not in DECL["en"]:
            continue
        # drop hypothetical/superheavy placeholder elements (IUPAC systematic
        # names for undiscovered elements: unbibium Z=122, ...)
        if cat == "element_number" and val_en.isdigit() and int(val_en) > 118:
            dropped_hypothetical += 1
            continue
        if cat == "element_symbol" and sub_en.startswith("un") and sub_en.endswith(("ium", "ium")):
            # systematic placeholder names like unbibium/unquadquadium
            if sub_en not in {"ununseptium", "ununium"} and len(sub_en) > 8:
                dropped_hypothetical += 1
                continue
        # normalize ID language labels: Wikidata is inconsistent — some carry
        # a leading "bahasa" prefix, some don't. Strip it; templates add it.
        if cat == "official_language":
            for pref in ("bahasa ", "Bahasa "):
                if val_id.startswith(pref):
                    val_id = val_id[len(pref):]
                    break
        facts.append({
            "cat": cat,
            "sub": {"en": sub_en, "ja": sub_ja, "id": sub_id},
            "val": {"en": val_en, "ja": val_ja, "id": val_id},
        })
    print(f"loaded {len(facts)} facts (dropped {dropped_hypothetical} hypothetical elements)")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for lang in ("en", "ja", "id"):
        lines = []
        for f in facts:
            sub, val = f["sub"][lang], f["val"][lang]
            if not sub or not val:
                continue
            for tmpl in DECL[lang][f["cat"]]:
                text = tmpl.format(sub=sub, val=val)
                text = re.sub(r"\s+", " ", text).strip()
                # guard: value already ends with a period ("Washington, D.C.")
                # — collapse the template's trailing period, keep "..." intact
                if text.endswith("..") and not text.endswith("..."):
                    text = text[:-1]
                if text:
                    lines.append(text)
        # dedup exact renderings — e.g. country "Luxembourg" and its
        # same-named district can produce byte-identical sentences through
        # capital / capital_admin
        seen_renders = set()
        deduped = []
        for l in lines:
            if l not in seen_renders:
                seen_renders.add(l)
                deduped.append(l)
        lines = deduped
        base = len(lines)
        total_lines = lines * 1
        for _ in range(args.exposures - 1):
            shuffled = list(lines)
            rng.shuffle(shuffled)
            total_lines.extend(shuffled)
        rng.shuffle(total_lines)
        out = out_dir / f"facts_{lang}.txt"
        with open(out, "w", encoding="utf-8") as fo:
            for l in total_lines:
                fo.write(l + "\n")
        # token estimate: ~1.3 tok/word heuristic for en/id, ~0.9 tok/char-ish for ja
        if lang == "ja":
            est_tokens = sum(len(l) for l in total_lines) * 0.9
        else:
            est_tokens = sum(len(l.split()) for l in total_lines) * 1.35
        stats[lang] = {
            "unique_renderings": base,
            "total_lines": len(total_lines),
            "est_tokens": int(est_tokens),
            "path": str(out),
        }
        print(f"[{lang}] {base} unique renderings x{args.exposures} = "
              f"{len(total_lines)} lines (~{int(est_tokens)/1e6:.0f}M tokens) -> {out}")

    with open(ROOT / "logs" / "fact_injection_stats.json", "w", encoding="utf-8") as f:
        json.dump({"facts": len(facts), "exposures": args.exposures,
                   "by_lang": stats}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
