#!/usr/bin/env python3
"""build_facts_holdout.py — split Wikidata facts into train + two held-out benchmarks.

Two held-out types, so the pretrain evaluation can tell apart two failure modes:

  A. entity-QID holdout (MAIN metric)
     Hold out a fraction of ENTITIES (all their properties go with them).
     Answers "can the model generalise to an entity it has never seen?"

  B. property x entity holdout (DIAGNOSTIC)
     For entities that STAY in training, hold out ONE property, leaving the
     entity's other properties trained. Answers "can the model generalise a
     relation to a seen entity?" — if A is high but B is low, the model is
     memorising entities rather than learning properties.

Only categories that exist in build_fact_injection_corpus.T are considered:
categories with no rendering template are dropped by the injection builder
anyway, so holding them out would be meaningless.

Outputs:
  data/wikidata_facts_expanded_train.tsv   <- feed this to build_fact_injection_corpus
  data/facts_holdout_entity.tsv
  data/facts_holdout_property.tsv
  data/eval/facts_holdout_entity.jsonl     <- feed to eval_facts.py (split A)
  data/eval/facts_holdout_property.jsonl   <- feed to eval_facts.py (split B)

Usage:
  python3 scripts/build_facts_holdout.py --tsv data/wikidata_facts_expanded.tsv
      [--holdout_entity_frac 0.05] [--holdout_property_frac 0.05] [--seed 42]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fact_injection_corpus import T, strip_unit  # noqa: E402

SYS_EN = "You are a helpful, accurate multilingual assistant."
SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"


def parse_tsv(path):
    rows = []
    for line in path.open(encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        cat = p[0].strip()
        if cat not in T:
            continue
        rows.append({
            "cat": cat,
            "sub_en": p[1].strip(), "sub_ja": p[2].strip(), "sub_id": p[3].strip(),
            "val_en": p[4].strip(), "val_ja": p[5].strip(), "val_id": p[6].strip(),
            "qid": p[7].strip() if len(p) > 7 else "",
        })
    return rows


def tsv_line(f):
    return "\t".join([f["cat"], f["sub_en"], f["sub_ja"], f["sub_id"],
                      f["val_en"], f["val_ja"], f["val_id"], f["qid"]]) + "\n"


def split_question(qtext, lang):
    if lang == "en" and "?" in qtext:
        return qtext.split("? ", 1)[0] + "?"
    if "。" in qtext:
        return qtext.split("。", 1)[0] + "。"
    return qtext


def render_eval(fact, lang):
    """Return (question, answer) for one fact in one language, or None."""
    cat = fact["cat"]
    bank = T[cat][lang]
    sub = fact["sub_" + lang]
    val = strip_unit(cat, fact["val_" + lang])
    if not sub or not val:
        return None
    ans = bank["d"][0].format(sub=sub, val=val)
    qaq = bank["q"][0].format(sub=sub, val=val)
    return split_question(qaq, lang), ans


def eval_row(fact, lang, split_name):
    qa = render_eval(fact, lang)
    if qa is None:
        return None
    q, a = qa
    sys_msg = SYS_EN if lang == "en" else SYS_JA
    text = f"<|system|>{sys_msg}<|user|>{q}<|assistant|>{a}<|end|>"
    return {
        "text": text, "lang": lang, "task": "fact_qa", "split": split_name,
        "category": fact["cat"],
        "subject_en": fact["sub_en"], "subject_ja": fact["sub_ja"],
        "subject_id": fact["sub_id"],
        "value_en": fact["val_en"], "value_ja": fact["val_ja"],
        "value_id": fact["val_id"],
        "fact_id": f"{fact['qid']}::{fact['cat']}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/wikidata_facts_expanded.tsv")
    ap.add_argument("--holdout_entity_frac", type=float, default=0.05)
    ap.add_argument("--holdout_property_frac", type=float, default=0.05)
    ap.add_argument("--max_eval_per_split", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = ROOT / args.tsv
    if not src.exists():
        print(f"[ERR] missing {src}")
        return 1

    rows = parse_tsv(src)
    rng = random.Random(args.seed)
    print(f"loaded {len(rows)} facts ({len({r['cat'] for r in rows})} categories, "
          f"{len({r['qid'] for r in rows if r['qid']})} entities)")

    by_qid = defaultdict(list)
    for r in rows:
        if r["qid"]:
            by_qid[r["qid"]].append(r)

    # ---- Split A: entity-QID holdout -----------------------------------
    qids = list(by_qid.keys())
    rng.shuffle(qids)
    n_hold_entity = max(1, int(len(qids) * args.holdout_entity_frac))
    holdout_a = set(qids[:n_hold_entity])

    # ---- Split B: property x entity holdout -----------------------------
    # For entities still in training with >=2 properties, hold out one property.
    holdout_b_pairs = set()
    for qid in qids[n_hold_entity:]:
        cats = sorted({r["cat"] for r in by_qid[qid]})
        if len(cats) >= 2 and rng.random() < args.holdout_property_frac:
            holdout_b_pairs.add((qid, rng.choice(cats)))

    train_rows, a_rows, b_rows = [], [], []
    for r in rows:
        if r["qid"] and r["qid"] in holdout_a:
            a_rows.append(r)
        elif r["qid"] and (r["qid"], r["cat"]) in holdout_b_pairs:
            b_rows.append(r)
        else:
            train_rows.append(r)

    print(f"split A (entity holdout)   : {len(a_rows)} facts "
          f"({len(holdout_a)} entities)")
    print(f"split B (property holdout) : {len(b_rows)} facts "
          f"({len(holdout_b_pairs)} entity-property pairs)")
    print(f"train                      : {len(train_rows)} facts")

    # ---- write TSVs ------------------------------------------------------
    def dump(path, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fo:
            fo.writelines(tsv_line(f) for f in items)

    dump(ROOT / "data" / "wikidata_facts_expanded_train.tsv", train_rows)
    dump(ROOT / "data" / "facts_holdout_entity.tsv", a_rows)
    dump(ROOT / "data" / "facts_holdout_property.tsv", b_rows)

    # ---- render eval JSONL ----------------------------------------------
    def render_jsonl(items, split_name, out_path):
        out = []
        for f in items:
            if len(out) >= args.max_eval_per_split:
                break
            for lang in ("en", "ja"):
                row = eval_row(f, lang, split_name)
                if row is not None:
                    out.append(row)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fo:
            for row in out:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(out)

    n_a = render_jsonl(a_rows, "unseen_entity", ROOT / "data" / "eval" / "facts_holdout_entity.jsonl")
    n_b = render_jsonl(b_rows, "seen_entity_unseen_property",
                       ROOT / "data" / "eval" / "facts_holdout_property.jsonl")
    print(f"eval JSONL: split A={n_a} rows, split B={n_b} rows")

    print("\n[done]")
    print(f"  train       -> data/wikidata_facts_expanded_train.tsv")
    print(f"  holdout A   -> data/facts_holdout_entity.tsv + data/eval/facts_holdout_entity.jsonl")
    print(f"  holdout B   -> data/facts_holdout_property.tsv + data/eval/facts_holdout_property.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())