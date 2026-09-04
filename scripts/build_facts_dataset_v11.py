#!/usr/bin/env python3
"""build_facts_dataset_v11.py — Tier-1 trilingual facts SFT dataset (V11).

Wraps build_facts_dataset.py WITHOUT modifying it (monkeypatch), adding:

  New categories : driving_side, calling_code, founding_date, capital_admin
  New data source: data/wikidata_facts_tier1.tsv (9-col, QID-audited,
                   preferred-rank, point-in-time resolved) preferred over the
                   old trilingual TSV; old rows kept as fallback only.
  Aliases        : data/facts_aliases_tier1.tsv -> value aliases (en/ja/id)
                   + subject aliases for eval subject-binding.
  V11 variants   : JA/ID binding share >= 0.50 (fixes subject-dropping),
                   capital<->continent and population<->X contrast emphasis
                   (the two dominant V9 error modes), EN keeps V9 repair mix.

Output: sft_data/sft_facts_v11/{train,eval}.jsonl  (profile v10_trilingual
rendering path, v11 variant shares)

Usage:
  python3 scripts/build_facts_dataset_v11.py --target_rows 500000 \
      --en_share 0.34 --eval_rows 6000 --id_share 0.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_facts_dataset as B

# ───────────────────────── new-category value normalization ────────────────
DRIVING_SIDE_MAP = {
    "left": ("left", "左", "kiri"),
    "right": ("right", "右", "kanan"),
    "left-hand traffic": ("left", "左", "kiri"),
    "left hand traffic": ("left", "左", "kiri"),
    "left-hand": ("left", "左", "kiri"),
    "left-hand side": ("left", "左", "kiri"),
    "right-hand traffic": ("right", "右", "kanan"),
    "right hand traffic": ("right", "右", "kanan"),
    "right-hand": ("right", "右", "kanan"),
    "right-hand side": ("right", "右", "kanan"),
}

NEW_CATS = {"driving_side", "founding_date", "capital_admin"}
# NOTE: calling_code was dropped — Wikidata P2852 does not reliably store the
# main ITU code (most countries only list emergency numbers there).

# ───────────────────────── EN/JA answer templates ──────────────────────────
B.CAT_ANSWERS.update({
    "driving_side": ("In {sub}, people drive on the {val}.", "{sub}では{val}通行です。"),
    "calling_code": ("The international calling code of {sub} is +{val}.", "{sub}の国際電話番号は+{val}です。"),
    "founding_date": ("{sub} was founded on {val}.", "{sub}は{val}に建国されました。"),
    "capital_admin": ("The capital of {sub} is {val}.", "{sub}の州都は{val}です。"),
})
B.CAT_ANSWERS_BINDING.update({
    "driving_side": ("For {sub}, the driving side is the {val}.", "{sub}については、{val}通行です。"),
    "calling_code": ("For {sub}, the calling code is +{val}.", "{sub}については、国際電話番号は+{val}です。"),
    "founding_date": ("For {sub}, the founding date is {val}.", "{sub}については、建国日は{val}です。"),
    "capital_admin": ("For {sub}, the capital is {val}.", "{sub}については、州都は{val}です。"),
})
B.BINDING_Q.update({
    "driving_side": ("For {sub}, what is the driving side?", "{sub}については、通行側はどちらですか？"),
    "calling_code": ("For {sub}, what is the calling code?", "{sub}については、国際電話番号は何ですか？"),
    "founding_date": ("For {sub}, what is the founding date?", "{sub}については、建国日はいつですか？"),
    "capital_admin": ("For {sub}, what is the capital?", "{sub}については、州都はどこですか？"),
})
B.REQUEST_LABELS.update({
    "driving_side": ("the driving side", "通行側"),
    "calling_code": ("the calling code", "国際電話番号"),
    "founding_date": ("the founding date", "建国日"),
    "capital_admin": ("the capital", "州都"),
})
B.CONTRAST_DISTRACTORS.update({
    "capital": ["continent_of", "population", "currency"],   # EN continent-fallback fix
    "population": ["currency", "capital", "official_language", "area"],  # JA confusion fix
    "driving_side": ["continent_of", "calling_code"],
    "calling_code": ["currency", "driving_side"],
    "founding_date": ["capital", "continent_of"],
    "capital_admin": ["continent_of", "population"],
})

# ───────────────────────── EN/JA question templates ─────────────────────────
B.STRICT_EN_Q.update({
    "driving_side": [
        "Which side of the road do people drive on in {sub}?",
        "What is the driving side in {sub}?",
        "Do people in {sub} drive on the left or the right?",
        "Tell me the driving side of {sub}.",
        "In {sub}, is traffic on the left or the right?",
    ],
    "calling_code": [
        "What is the international calling code of {sub}?",
        "Which calling code is used for {sub}?",
        "Tell me the country calling code of {sub}.",
        "What code do I dial to call {sub}?",
        "Do you know the international dialing code of {sub}?",
    ],
    "founding_date": [
        "When was {sub} founded?",
        "What is the founding date of {sub}?",
        "Tell me when {sub} was established.",
        "Do you know the founding date of {sub}?",
        "On what date was {sub} founded?",
    ],
    "capital_admin": [
        "What is the capital of {sub}?",
        "Which city is the capital of {sub}?",
        "Tell me the capital of {sub}.",
        "Do you know the capital of {sub}?",
        "What city serves as the capital of {sub}?",
    ],
})
B.STRICT_JA_Q.update({
    "driving_side": [
        "{sub}では車はどちら側を走りますか？",
        "{sub}の通行側はどちらですか？",
        "{sub}は左通行ですか、右通行ですか？",
        "{sub}はどちら側通行ですか？",
    ],
    "calling_code": [
        "{sub}の国際電話番号は何番ですか？",
        "{sub}にかける際の国番号を教えてください。",
        "{sub}の電話の国コードは何ですか？",
        "{sub}のダイヤルコードは？",
    ],
    "founding_date": [
        "{sub}はいつ建国されましたか？",
        "{sub}の建国日はいつですか？",
        "{sub}が設立された日を教えてください。",
        "{sub}の独立日はいつですか？",
    ],
    "capital_admin": [
        "{sub}の州都はどこですか？",
        "{sub}の首都はどこですか？",
        "{sub}の中心都市を教えてください。",
        "{sub}の州都について知っていますか？",
    ],
})

# ───────────────────────── Indonesian templates ─────────────────────────────
B.CAT_ANSWERS_ID.update({
    "driving_side": "Di {sub}, orang berkendara di sebelah {val}.",
    "calling_code": "Kode telepon internasional {sub} adalah +{val}.",
    "founding_date": "{sub} didirikan pada {val}.",
    "capital_admin": "Ibu kota {sub} adalah {val}.",
})
B.CAT_ANSWERS_ID_BINDING.update({
    "driving_side": "Untuk {sub}, orang berkendara di sebelah {val}.",
    "calling_code": "Untuk {sub}, kode teleponnya adalah +{val}.",
    "founding_date": "Untuk {sub}, tanggal pendiriannya adalah {val}.",
    "capital_admin": "Untuk {sub}, ibu kotanya adalah {val}.",
})
B.BINDING_Q_ID.update({
    "driving_side": "Untuk {sub}, di sebelah mana orang berkendara?",
    "calling_code": "Untuk {sub}, berapa kode teleponnya?",
    "founding_date": "Untuk {sub}, kapan tanggal pendiriannya?",
    "capital_admin": "Untuk {sub}, apa ibu kotanya?",
})
B.REQUEST_LABELS_ID.update({
    "driving_side": "sisi berkendara",
    "calling_code": "kode telepon",
    "founding_date": "tanggal pendirian",
    "capital_admin": "ibu kota",
})
B.STRICT_ID_Q.update({
    "driving_side": [
        "Di sebelah mana orang berkendara di {sub}?",
        "Apa sisi berkendara di {sub}?",
        "Di {sub}, mobil berjalan di kiri atau kanan?",
        "Sebutkan sisi berkendara {sub}.",
    ],
    "calling_code": [
        "Berapa kode telepon internasional {sub}?",
        "Apa kode telepon {sub}?",
        "Kode negara mana yang dipakai untuk menelepon {sub}?",
        "Sebutkan kode telepon {sub}.",
    ],
    "founding_date": [
        "Kapan {sub} didirikan?",
        "Apa tanggal pendirian {sub}?",
        "Sebutkan tanggal berdirinya {sub}.",
        "Kapan {sub} berdiri?",
    ],
    "capital_admin": [
        "Apa ibu kota {sub}?",
        "Kota apa yang menjadi ibu kota {sub}?",
        "Sebutkan ibu kota {sub}.",
        "Apakah kamu tahu ibu kota {sub}?",
    ],
})
if hasattr(B, "ANTI_CONFUSION_Q_ID"):
    B.ANTI_CONFUSION_Q_ID.update({
        "driving_side": ["{sub} berada di benua apa, bukan sisi berkendaranya?"],
        "calling_code": ["Apa mata uang {sub}, bukan kode teleponnya?"],
        "founding_date": ["Apa ibu kota {sub}, bukan tanggal pendiriannya?"],
        "capital_admin": ["Di benua apa {sub} berada, bukan ibu kotanya?"],
    })
if hasattr(B, "ANTI_CONFUSION_Q"):
    B.ANTI_CONFUSION_Q.update({
        "driving_side": {"en": ["Which continent is {sub} in, not its driving side?"],
                         "ja": ["{sub}の通行側ではなく、大陸を教えてください。"]},
        "calling_code": {"en": ["What is the currency of {sub}, not its calling code?"],
                         "ja": ["{sub}の国際電話番号ではなく、通貨を教えてください。"]},
        "founding_date": {"en": ["What is the capital of {sub}, not its founding date?"],
                          "ja": ["{sub}の建国日ではなく、首都を教えてください。"]},
        "capital_admin": {"en": ["Which continent is {sub} in, not its capital?"],
                          "ja": ["{sub}の州都ではなく、大陸を教えてください。"]},
    })

# allow the new categories through the strict-category gate
B.STRICT_CATEGORIES.update(NEW_CATS)

# V11 category weights: V9 base + exposure for the new categories.
V11_WEIGHTS = dict(B.V9_FULL_WEIGHTS)
V11_WEIGHTS.update({
    "driving_side": 1.6,
    "calling_code": 1.2,       # 567 facts but easy — moderate exposure
    "founding_date": 2.2,
    "capital_admin": 4.5,      # large new category, high-value knowledge
})
B.V9_FULL_WEIGHTS = V11_WEIGHTS


# ───────────────────────── V11 variant shares ──────────────────────────────
def variant_shares_v11(profile, lang, cat):
    if lang == "en":
        # keep the V9 repair emphasis for the known-weak categories
        if cat in {"official_language", "highest_point"}:
            return 0.24, 0.16, 0.40
        if cat == "currency":
            return 0.20, 0.15, 0.24
        if cat in {"area", "capital", "continent_of"}:
            # capital contrast ↑: continent-fallback was 10/50 of EN errors
            return 0.08, 0.14, 0.22 if cat == "capital" else 0.10
        if cat == "population":
            return 0.10, 0.18, 0.18
        if cat in NEW_CATS:
            return 0.10, 0.35, 0.12
        return 0.14, 0.20, 0.12
    if lang == "ja":
        # binding >= 0.50 everywhere: JA subject-dropping was 8/50 errors
        if cat in {"population", "capital", "element_symbol", "element_number",
                   "continent_of", "highest_point"}:
            return 0.08, 0.55, 0.10
        if cat in NEW_CATS:
            return 0.08, 0.52, 0.10
        return 0.08, 0.50, 0.08
    # id
    if cat in {"official_language", "highest_point"}:
        return 0.15, 0.40, 0.18
    if cat in NEW_CATS:
        return 0.10, 0.52, 0.12
    return 0.10, 0.52, 0.10


B.variant_shares = variant_shares_v11


# ───────────────────────── V11 fact loader ─────────────────────────────────
def load_alias_map():
    alias_map = {}
    p = B.ROOT / "data" / "facts_aliases_tier1.tsv"
    if not p.exists():
        return alias_map
    for line in open(p, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        qid, lang, alias = parts[0], parts[1], parts[2]
        alias_map.setdefault(qid, {"en": [], "ja": [], "id": []}).setdefault(lang, []).append(alias)
    return alias_map


def normalize_new_cat_values(cat, val_en, val_ja, val_id):
    if cat == "driving_side":
        mapped = DRIVING_SIDE_MAP.get(val_en.strip().casefold())
        if not mapped:
            return None
        return mapped
    if cat == "calling_code":
        digits = val_en.strip().lstrip("+")
        if not digits.isdigit():
            return None
        return (digits, digits, digits)
    return (val_en, val_ja, val_id)


def invalid_fact_v11(cat, sub_en, sub_ja, val_en, val_ja):
    if not sub_en.strip() or not sub_ja.strip() or not val_en.strip() or not val_ja.strip():
        return True
    if B.has_qid(sub_en, sub_ja, val_en, val_ja):
        return True
    if "�" in (sub_en + sub_ja + val_en + val_ja):
        return True
    if cat in {"area", "population", "element_number", "calling_code"}:
        return False
    if cat != "element_symbol" and len(val_en.strip()) <= 2:
        return True
    return False


def load_strict_facts_v11():
    overrides = B.load_fact_overrides()
    alias_map = load_alias_map()

    def build_fact(cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id,
                   sub_qid="", val_qid=""):
        if cat == "driving_side":
            norm = normalize_new_cat_values(cat, val_en, val_ja, val_id)
            if norm is None:
                return None
            val_en, val_ja, val_id = norm
        elif cat == "calling_code":
            norm = normalize_new_cat_values(cat, val_en, val_ja, val_id)
            if norm is None:
                return None
            val_en, val_ja, val_id = norm
        if cat in {"area", "population", "element_number"}:
            val_en = B.clean_numeric_value(val_en)
            val_ja = val_id = val_en
        if invalid_fact_v11(cat, sub_en, sub_ja, val_en, val_ja):
            return None
        fact = B.apply_fact_overrides(cat, sub_en, sub_ja, val_en, val_ja, overrides,
                                      sub_id=sub_id, val_id=val_id)
        if fact is None:
            return None
        # enrich with Wikidata aliases (value aliases for eval matching;
        # subject aliases for subject-binding checks)
        val_a = alias_map.get(val_qid, {})
        sub_a = alias_map.get(sub_qid, {})
        fact["aliases_en"] = sorted(set(fact.get("aliases_en", []) + val_a.get("en", [])))
        fact["aliases_ja"] = sorted(set(fact.get("aliases_ja", []) + val_a.get("ja", [])))
        fact["aliases_id"] = sorted(set(val_a.get("id", [])))
        fact["subject_aliases_en"] = sub_a.get("en", [])[:6]
        fact["subject_aliases_ja"] = sub_a.get("ja", [])[:6]
        fact["subject_aliases_id"] = sub_a.get("id", [])[:6]
        return fact

    raw = []
    override_drops = 0
    seen_keys = set()

    # ── primary source: Tier-1 TSV (9-col)
    tier1 = B.ROOT / "data" / "wikidata_facts_tier1.tsv"
    if tier1.exists():
        for line in open(tier1, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id = [x.strip() for x in p[:7]]
            sub_qid = p[7].strip() if len(p) > 7 else ""
            val_qid = p[8].strip() if len(p) > 8 else ""
            fact = build_fact(cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id,
                              sub_qid, val_qid)
            if fact is None:
                override_drops += 1
                continue
            raw.append(fact)
            seen_keys.add((cat, sub_en.casefold()))

    # ── fallback: old trilingual TSV rows not already covered
    old = B.ROOT / "data" / "wikidata_facts_trilingual.tsv"
    if old.exists():
        for line in open(old, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id = [x.strip() for x in p[:7]]
            if (cat, sub_en.casefold()) in seen_keys:
                continue
            fact = build_fact(cat, sub_en, sub_ja, sub_id, val_en, val_ja, val_id)
            if fact is None:
                override_drops += 1
                continue
            raw.append(fact)
            seen_keys.add((cat, sub_en.casefold()))

    # ── single-answer conflict resolution (same as the original loader)
    by_key = {}
    conflicts = set()
    for fact in raw:
        key = (fact["category"], fact["subject_en"].casefold(), fact["subject_ja"])
        val = (fact["value_en"].casefold(), fact["value_ja"])
        if key in by_key and by_key[key][0] != val:
            conflicts.add(key)
        elif key in by_key:
            existing = by_key[key][1]
            for al in ("aliases_en", "aliases_ja", "aliases_id",
                       "subject_aliases_en", "subject_aliases_ja", "subject_aliases_id"):
                existing[al] = sorted(set(existing.get(al, [])) | set(fact.get(al, [])))
        else:
            by_key[key] = (val, fact)
    facts = [item for key, (_, item) in by_key.items() if key not in conflicts]

    # ── global subject-name disambiguation ─────────────────────────────────
    # The same subject NAME can belong to different entities IN ANY LANGUAGE:
    #   EN "Niger" (country) vs "Niger State" whose ID label is also "Niger",
    #   JA "北部州" used by both Sudan's and Cameroon's Northern State,
    #   "Eastern Region" shared by Ghana / Uganda / Iceland.
    # Prompts render in all 3 languages, so a colliding name in ANY language
    # with a different answer poisons single-answer QA. Country facts win
    # over admin-division facts.
    def norm(x):
        return (x or "").strip().casefold()

    def prompt_collides(f1, f2):
        same_group = (lambda c: "capital" if c == "capital_admin" else c)
        return same_group(f1["category"]) == same_group(f2["category"])

    drop_set = set()
    for name_field in ("subject_en", "subject_ja", "subject_id"):
        by_name = {}
        for f in facts:
            key = norm(f.get(name_field, ""))
            if key:
                by_name.setdefault(key, []).append(f)
        for _name, group in by_name.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    f1, f2 = group[i], group[j]
                    if not prompt_collides(f1, f2):
                        continue
                    a1, a2 = f1["category"] == "capital_admin", f2["category"] == "capital_admin"
                    if a1 and not a2:
                        # admin-division name colliding with a country fact:
                        # always drop the division (even when values match,
                        # the answer templates differ: 州都 vs 首都)
                        drop_set.add(id(f1))
                    elif a2 and not a1:
                        drop_set.add(id(f2))
                    elif norm(f1["value_en"]) == norm(f2["value_en"]):
                        continue
                    else:
                        drop_set.add(id(f1))
                        drop_set.add(id(f2))
    if drop_set:
        facts = [f for f in facts if id(f) not in drop_set]

    return facts, {"raw_kept": len(raw), "conflict_keys": len(conflicts),
                   "override_drops": override_drops, "name_disambig_drops": len(drop_set)}


B.load_strict_facts = load_strict_facts_v11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_rows", type=int, default=500000)
    ap.add_argument("--en_share", type=float, default=0.34)
    ap.add_argument("--eval_rows", type=int, default=6000)
    ap.add_argument("--id_share", type=float, default=0.5)
    ap.add_argument("--out_dir", default="sft_data/sft_facts_v11")
    args = ap.parse_args()

    B.OUT_DIR = B.ROOT / args.out_dir
    B.OUT_FILE = B.OUT_DIR / "train.jsonl"

    # quick sanity on the loader before the long build
    facts, stats = load_strict_facts_v11()
    from collections import Counter
    by_cat = Counter(f["category"] for f in facts)
    n_alias = sum(1 for f in facts if f.get("aliases_en"))
    print(f"V11 loader sanity: {len(facts)} facts, {len(by_cat)} categories")
    print(f"  by category: {dict(by_cat)}")
    print(f"  facts with value aliases: {n_alias}")
    print(f"  stats: {stats}")

    B.build_oversampled_dataset(
        target_rows=args.target_rows,
        en_share=args.en_share,
        eval_rows=args.eval_rows,
        profile="v10_trilingual",   # activates the trilingual path
        id_share=args.id_share,
    )


if __name__ == "__main__":
    main()
