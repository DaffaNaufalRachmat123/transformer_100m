#!/usr/bin/env python3
"""Turn the expanded Wikidata TSV into EN/JA fact-QA SFT rows.

The v11 builder reads a 9-column tier1 schema and only covers countries,
elements and administrative regions, which is why the fact set topped out at
4,580 rows and value_hit sat at 0.24. This reads the 8-column expanded dump
(46,925 facts over 19 entity types) produced by scripts/expand_facts.py.

Held-out rows are split by fact_id, never by row: the same fact rendered
through two templates would otherwise land in both train and eval and report
memorisation as generalisation.

  python3 scripts/build_facts_expanded.py --out sft_data/sft_facts_expanded
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SYS_EN = "You are a helpful, accurate multilingual assistant."
SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"

# category -> (english attribute, japanese attribute, sibling attribute used by
# the contrastive template that teaches the model not to answer a neighbour).
ATTRS = {
    "city_country":       ("country", "国", "population"),
    "city_population":    ("population", "人口", "country"),
    "person_occupation":  ("occupation", "職業", "country of citizenship"),
    "person_birthyear":   ("birth year", "生年", "occupation"),
    "person_country":     ("country of citizenship", "国籍", "occupation"),
    "film_director":      ("director", "監督", "release year"),
    "film_year":          ("release year", "公開年", "director"),
    # P171 is "parent taxon", not class: it resolves to whichever rank sits
    # directly above the entity, so the attribute is named for what it is.
    "species_parent":     ("parent taxon", "親分類群", "country"),
    "river_length":       ("length", "長さ", "country"),
    "river_country":      ("country", "国", "length"),
    "island_country":     ("country", "国", "population"),
    "airport_country":    ("country", "国", "city"),
    "book_author":        ("author", "著者", "publication year"),
    "band_origin":        ("origin", "出身地", "genre"),
    "company_country":    ("country", "国", "industry"),
    "mountain_elevation": ("elevation", "標高", "country"),
    "mountain_country":   ("country", "国", "elevation"),
    "lake_country":       ("country", "国", "area"),
    "university_country": ("country", "国", "founding year"),
    "planet_star":        ("host star", "主星", "orbital period"),
}

# Categories whose property is legitimately multi-valued. Their prompts ask for
# "one of" and their answers assert only that the value is *a* valid one, so a
# person with five occupations is not taught that exactly one is correct.
MULTI = {"person_occupation", "person_country", "film_director", "book_author"}

Q_EN = [
    "What is the {attr} of {subj}?",
    "Tell me the {attr} of {subj}.",
    "{subj} — what is its {attr}?",
    "Give the {attr} of {subj}.",
    "Which {attr} does {subj} have?",
]
Q_EN_MULTI = [
    "Name one {attr} of {subj}.",
    "Give one {attr} of {subj}.",
    "What is one {attr} of {subj}?",
    "Tell me one {attr} of {subj}.",
]
Q_EN_CONTRAST = [
    "Give the {attr} of {subj}, not its {other}.",
    "I want the {attr} of {subj}, not the {other}.",
]
A_EN = [
    "The {attr} of {subj} is {val}.",
    "For {subj}, the {attr} is {val}.",
    "{subj} has a {attr} of {val}.",
]
A_EN_MULTI = [
    "One {attr} of {subj} is {val}.",
    "{subj} has {val} as one {attr}.",
]

Q_JA = [
    "{subj}の{attr}は何ですか？",
    "{subj}については、{attr}は？",
    "{subj}の{attr}を教えてください。",
    "{subj}の{attr}は？",
]
Q_JA_MULTI = [
    "{subj}の{attr}を一つ挙げてください。",
    "{subj}の{attr}の一つは何ですか？",
]
Q_JA_CONTRAST = [
    "{subj}の{attr}を教えてください。他の情報は不要です。",
]
A_JA = [
    "{subj}の{attr}は{val}です。",
    "{subj}については、{attr}は{val}です。",
]
A_JA_MULTI = [
    "{subj}の{attr}の一つは{val}です。",
]


def rows_for(rec, rng, per_fact):
    """Render one fact into up to per_fact rows per language."""
    cat = rec["category"]
    attr_en, attr_ja, other = ATTRS[cat]
    multi = cat in MULTI
    out = []

    for lang in ("en", "ja"):
        subj = rec["subject_en"] if lang == "en" else rec["subject_ja"]
        val = rec["value_en"] if lang == "en" else rec["value_ja"]
        if not subj or not val:
            continue
        attr = attr_en if lang == "en" else attr_ja
        if lang == "en":
            qs = Q_EN_MULTI if multi else (Q_EN + Q_EN_CONTRAST)
            ans = A_EN_MULTI if multi else A_EN
        else:
            qs = Q_JA_MULTI if multi else (Q_JA + Q_JA_CONTRAST)
            ans = A_JA_MULTI if multi else A_JA
        sysmsg = SYS_EN if lang == "en" else SYS_JA

        picked = rng.sample(qs, min(per_fact, len(qs)))
        for q in picked:
            question = q.format(subj=subj, attr=attr, other=other)
            answer = rng.choice(ans).format(subj=subj, attr=attr, val=val)
            out.append({
                "text": f"<|system|>{sysmsg}<|user|>{question}"
                        f"<|assistant|>{answer}<|end|>",
                "lang": lang,
                "task": "fact_qa",
                "category": cat,
                "subject_en": rec["subject_en"],
                "subject_ja": rec["subject_ja"],
                "value_en": rec["value_en"],
                "value_ja": rec["value_ja"],
                "fact_id": rec["fact_id"],
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/wikidata_facts_expanded.tsv")
    ap.add_argument("--out", default="sft_data/sft_facts_expanded")
    ap.add_argument("--per_fact", type=int, default=3,
                    help="question templates rendered per fact per language")
    ap.add_argument("--eval_facts", type=int, default=1500,
                    help="facts (not rows) reserved for eval")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    facts, skipped = [], Counter()
    with open(a.tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                skipped["short_row"] += 1
                continue
            cat, s_en, s_ja, _s_id, v_en, v_ja, _v_id, qid = p[:8]
            if cat not in ATTRS:
                skipped[f"unknown_cat:{cat}"] += 1
                continue
            if not (s_en and s_ja and v_en and v_ja):
                skipped["missing_field"] += 1
                continue
            facts.append({
                "category": cat, "subject_en": s_en, "subject_ja": s_ja,
                "value_en": v_en, "value_ja": v_ja,
                "fact_id": f"{cat}::{qid}",
            })

    # Deduplicate on fact_id so a repeated QID cannot straddle the split.
    seen, uniq = set(), []
    for r in facts:
        if r["fact_id"] in seen:
            skipped["dup_fact_id"] += 1
            continue
        seen.add(r["fact_id"])
        uniq.append(r)

    rng.shuffle(uniq)
    n_eval = min(a.eval_facts, len(uniq) // 10)
    eval_facts, train_facts = uniq[:n_eval], uniq[n_eval:]

    tr = [r for fact in train_facts for r in rows_for(fact, rng, a.per_fact)]
    ev = [r for fact in eval_facts for r in rows_for(fact, rng, 1)]
    rng.shuffle(tr)
    rng.shuffle(ev)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_ids = {r["fact_id"] for r in tr}
    leak = sum(1 for r in ev if r["fact_id"] in train_ids)
    stats = {
        "facts_total": len(uniq), "facts_train": len(train_facts),
        "facts_eval": len(eval_facts),
        "rows_train": len(tr), "rows_eval": len(ev),
        "fact_id_leakage": leak,
        "lang": dict(Counter(r["lang"] for r in tr).most_common()),
        "category": dict(Counter(r["category"] for r in tr).most_common()),
        "skipped": dict(skipped.most_common(10)),
    }
    (out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[FACTS] {len(uniq)} unique facts -> "
          f"train {len(tr)} rows / eval {len(ev)} rows")
    print(f"  lang    : {stats['lang']}")
    print(f"  leakage : {leak} (must be 0)")
    if skipped:
        print(f"  skipped : {dict(skipped.most_common(5))}")


if __name__ == "__main__":
    main()
