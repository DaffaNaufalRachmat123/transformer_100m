#!/usr/bin/env python3
"""eval_facts.py — Evaluate EN/JA fact answering on a held-out set.

Unlike BLEU (which needs exact-translation references), fact evaluation checks:
  1. Language correctness — an EN question must be answered in English, a JA
     question in Japanese (this is the "language confusion" fix).
  2. Fact accuracy — the answer must contain key content words from the
     reference answer (e.g. "Tokyo" for "capital of Japan").

Usage:
    python3 scripts/eval_facts.py \
        --model translation_resources/ckpt_sft_final.safetensors \
        --eval sft_data/sft_facts_large/train.jsonl \
        --max_test 200
"""

import argparse
from collections import Counter, defaultdict
import json
import re
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_EN = "You are a helpful, accurate multilingual assistant."
SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"

ATTR_MARKERS = {
    "capital": {"en": ["capital"], "ja": ["首都"], "id": ["ibu kota", "ibukota"]},
    "currency": {"en": ["currency"], "ja": ["通貨"], "id": ["mata uang"]},
    "official_language": {"en": ["official language", "language"], "ja": ["公用語"], "id": ["bahasa resmi"]},
    "population": {"en": ["population", "people"], "ja": ["人口", "人です", "住んでいます"], "id": ["populasi", "penduduk"]},
    "highest_point": {"en": ["highest point"], "ja": ["一番高い", "最高地点"], "id": ["titik tertinggi"]},
    "continent_of": {"en": ["in Africa", "in Europe", "in Asia", "in North America", "in South America", "in Oceania"],
                     "ja": ["アフリカ", "ヨーロッパ", "アジア", "北アメリカ", "南アメリカ", "オセアニア"],
                     "id": ["Afrika", "Eropa", "Asia", "Amerika Utara", "Amerika Selatan", "Oseania", "benua"]},
    "area": {"en": ["area", "square kilometer"], "ja": ["面積", "平方キロメートル"], "id": ["luas", "kilometer persegi"]},
    "element_symbol": {"en": ["chemical symbol"], "ja": ["化学記号"], "id": ["simbol kimia"]},
    "element_number": {"en": ["atomic number"], "ja": ["原子番号"], "id": ["nomor atom"]},
}

NUMERIC_CATEGORIES = {"population", "area", "element_number"}
SHORT_TOKEN_CATEGORIES = {"element_symbol"}


def has_ja(s):
    return any('\u3040' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' for c in s)


def has_lat(s):
    return bool(re.search(r'[A-Za-z]{2,}', s))


def tokenize(s, lang):
    if lang == "ja":
        return re.findall(r"[\u3040-\u30ff]|[\u4e00-\u9fff]|[A-Za-z0-9]+", s)
    return re.findall(r"[A-Za-z0-9]+", s)


def content_hit(ref, pred, lang):
    """Fraction of reference content tokens that appear in the prediction."""
    rt = tokenize(ref, lang)
    pt = set(tokenize(pred, lang))
    if not rt:
        return 0.0
    return sum(1 for t in rt if t.lower() in {x.lower() for x in pt}) / len(rt)


def nfkc_text(s):
    return unicodedata.normalize("NFKC", s or "")


def strip_diacritics(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def canonical_number(token):
    token = (token or "").replace(",", "").strip()
    if not token:
        return ""
    try:
        value = float(token)
    except ValueError:
        return token
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def normalize_text(s, lang):
    s = nfkc_text(s).strip()
    latin = lang in {"en", "id"}
    s = s.casefold() if latin else s
    if latin:
        s = strip_diacritics(s)
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[‐‑‒–—―]", "-", s)
    s = re.sub(r'[\(\)\[\]\{\}",.。:：;；!?！？/\\]', " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_lookup_forms(s, lang):
    raw = nfkc_text(s)
    norm = normalize_text(raw, lang)
    compact = re.sub(r"\s+", "", norm)
    numbers = {
        canonical_number(token)
        for token in re.findall(r"\d+(?:\.\d+)?", raw.replace(",", ""))
    }
    numbers.discard("")
    tokens = {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", raw)}
    return {
        "norm": norm,
        "compact": compact,
        "numbers": numbers,
        "tokens": tokens,
    }


def alias_forms(value, lang, category):
    forms = set()
    if not value:
        return forms
    lookup = make_lookup_forms(value, lang)
    if category in NUMERIC_CATEGORIES:
        return set(lookup["numbers"])
    if category in SHORT_TOKEN_CATEGORIES:
        raw = nfkc_text(value).strip()
        if raw:
            forms.add(raw.casefold())
        forms.update(lookup["tokens"])
        return {item for item in forms if item}
    if lookup["norm"]:
        forms.add(lookup["norm"])
    if lookup["compact"]:
        forms.add(lookup["compact"])
    if lang == "ja" and lookup["norm"]:
        plain = lookup["norm"].replace("・", "")
        if plain:
            forms.add(plain)
    return {item for item in forms if item}


def value_aliases(row, lang):
    category = row.get("category")
    if lang == "en":
        value = row.get("value_en")
        extra_key = "aliases_en"
    elif lang == "id":
        value = row.get("value_id") or row.get("value_en")
        extra_key = "aliases_en"  # no ID-specific aliases; EN aliases apply (Latin script)
    else:
        value = row.get("value_ja")
        extra_key = "aliases_ja"
    extras = row.get(extra_key)
    candidates = [value] if value else []
    if isinstance(extras, str):
        candidates.extend(part.strip() for part in re.split(r"[|;,]", extras) if part.strip())
    elif isinstance(extras, list):
        candidates.extend(str(part).strip() for part in extras if str(part).strip())
    forms = set()
    for candidate in candidates:
        forms.update(alias_forms(candidate, lang, category))
    return forms


def normalize_value(s):
    s = s.lower().strip()
    s = re.sub(r"[,。.]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def attr_ok(category, pred, lang):
    if not category:
        return None
    markers = ATTR_MARKERS.get(category, {}).get(lang, [])
    if not markers:
        return None
    p = pred.lower() if lang in {"en", "id"} else pred
    return any((m.lower() if lang in {"en", "id"} else m) in p for m in markers)


def value_hit(row, pred, lang):
    aliases = value_aliases(row, lang)
    if not aliases:
        return None
    forms = make_lookup_forms(pred, lang)
    category = row.get("category")
    for alias in aliases:
        if alias_matches_prediction(alias, forms, category, pred):
            return True
    return False


def detect_marker_categories(pred, lang):
    matches = []
    probe = pred.lower() if lang in {"en", "id"} else pred
    for category, langs in ATTR_MARKERS.items():
        markers = langs.get(lang, [])
        if any((marker.lower() if lang in {"en", "id"} else marker) in probe for marker in markers):
            matches.append(category)
    return matches


def _subject_field(row, lang):
    if lang == "en":
        return row.get("subject_en")
    if lang == "id":
        return row.get("subject_id") or row.get("subject_en")
    return row.get("subject_ja")


def build_indexes(rows):
    langs = {row["target_lang"] for row in rows}
    subject_index = {l: defaultdict(lambda: defaultdict(set)) for l in langs}
    global_entries = {l: [] for l in langs}
    seen_global = {l: set() for l in langs}
    for row in rows:
        lang = row["target_lang"]
        subject = _subject_field(row, lang)
        subject_key = normalize_text(subject, lang)
        if not subject_key or not row.get("category"):
            continue
        for alias in value_aliases(row, lang):
            subject_index[lang][subject_key][row["category"]].add(alias)
            dedupe_key = (subject_key, row["category"], alias)
            if dedupe_key in seen_global[lang]:
                continue
            seen_global[lang].add(dedupe_key)
            global_entries[lang].append({
                "subject": subject_key,
                "category": row["category"],
                "alias": alias,
                "fact_id": row["fact_id"],
            })
    return subject_index, global_entries


def alias_matches_prediction(alias, pred_forms, category, pred_raw="", lang=None):
    if not alias:
        return False
    if category in NUMERIC_CATEGORIES:
        return alias in pred_forms["numbers"]
    if category in SHORT_TOKEN_CATEGORIES or (len(alias) <= 3 and alias.isalpha()):
        return alias.casefold() in pred_forms["tokens"]
    # Long alphabetic aliases must match on word boundaries to avoid
    # "kyrgyz" matching "kyrgyzstan" (substring false positive).
    alias_norm = alias.casefold()
    if re.fullmatch(r"[a-z0-9 ]+", alias_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(alias_norm) + r"(?![a-z0-9])"
        if re.search(pattern, pred_forms["norm"]):
            return True
        # Compact (whitespace-stripped) matching is only meaningful for
        # non-spaced scripts (Japanese); skip it for Latin aliases.
        return False
    return alias in pred_forms["norm"] or alias in pred_forms["compact"]


def detect_value_matches(row, pred, lang, subject_index, global_entries):
    subject = _subject_field(row, lang)
    subject_key = normalize_text(subject, lang)
    pred_forms = make_lookup_forms(pred, lang)
    same_subject = set()
    for category, aliases in subject_index.get(lang, {}).get(subject_key, {}).items():
        if any(alias_matches_prediction(alias, pred_forms, category, pred) for alias in aliases):
            same_subject.add(category)
    global_matches = []
    for entry in global_entries.get(lang, []):
        if alias_matches_prediction(entry["alias"], pred_forms, entry["category"], pred):
            global_matches.append(entry)
    return same_subject, global_matches


def choose_predicted_category(expected_category, same_subject, marker_categories, global_matches):
    if len(same_subject) == 1:
        return next(iter(same_subject))
    if len(same_subject) > 1 and expected_category in same_subject:
        return expected_category
    if len(marker_categories) == 1:
        return marker_categories[0]
    unique_global = sorted({entry["category"] for entry in global_matches})
    if len(unique_global) == 1:
        return unique_global[0]
    if len(marker_categories) > 1 and expected_category in marker_categories:
        return expected_category
    return None


def subject_binding_ok(row, pred, lang, subject_index):
    """Check the prediction mentions the asked subject (or an alias of it).

    Prevents "right value, wrong country" (e.g. asked Poland, answered
    "アフリカはヨーロッパにあります") from being scored as correct.
    """
    subject = _subject_field(row, lang)
    subject_key = normalize_text(subject, lang)
    if not subject_key:
        return True
    pred_forms = make_lookup_forms(pred, lang)
    subject_norm = subject_key.casefold()
    if re.fullmatch(r"[a-z0-9 ]+", subject_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(subject_norm) + r"(?![a-z0-9])"
        if re.search(pattern, pred_forms["norm"]):
            return True
    else:
        if subject_key in pred_forms["norm"] or subject_key.replace("・", "") in pred_forms["norm"]:
            return True
    # Fallback: the subject may be echoed from a DIFFERENT category's value
    # (e.g. the model answers "ワルシャワはポーランドの首都です" where Poland
    # appears as part of another fact's answer). The asked row's own value
    # must NOT count as subject evidence.
    if lang == "en":
        asked_value = row.get("value_en")
    elif lang == "id":
        asked_value = row.get("value_id") or row.get("value_en")
    else:
        asked_value = row.get("value_ja")
    asked_value_forms = make_lookup_forms(asked_value, lang) if asked_value else None
    for category, aliases in subject_index.get(lang, {}).get(subject_key, {}).items():
        for alias in aliases:
            if len(alias) < 4:
                continue
            if alias in pred_forms["norm"]:
                if asked_value_forms is None:
                    return True
                if alias not in asked_value_forms["norm"] and alias not in asked_value_forms["compact"]:
                    return True
    # Japanese subject may be partially echoed (first 2+ chars) due to BPE splits.
    if lang == "ja" and len(subject_key) >= 3:
        head = subject_key[:2]
        if head and head in pred:
            return True
    return False


def classify_error(row, pred, lang, lang_ok, value_ok, same_subject, global_matches, predicted_category, subject_bound):
    if not pred.strip():
        return "empty"
    if value_ok and not subject_bound:
        return "subject_missing"
    if value_ok:
        return "correct"
    expected = row.get("category")
    if not lang_ok:
        return "language_leak"
    wrong_same_subject = sorted(cat for cat in same_subject if cat != expected)
    if wrong_same_subject:
        return f"same_subject:{wrong_same_subject[0]}"
    subject_key = normalize_text(_subject_field(row, lang), lang)
    same_category_other_subject = [
        entry for entry in global_matches
        if entry["category"] == expected and entry["subject"] != subject_key
    ]
    if same_category_other_subject:
        return "other_subject_same_category"
    wrong_category_global = sorted({entry["category"] for entry in global_matches if entry["category"] != expected})
    if wrong_category_global:
        return f"other_fact:{wrong_category_global[0]}"
    if predicted_category and predicted_category != expected:
        return f"marker:{predicted_category}"
    return "unknown"


def blank_stats():
    return {
        "n": 0,
        "lang_ok": 0,
        "hit": 0.0,
        "empty": 0,
        "attr_ok": 0,
        "attr_n": 0,
        "value_ok": 0,
        "value_n": 0,
        "ja_leak": 0,
        "lat_only": 0,
        "subject_ok": 0,
    }


def pct(num, den):
    return round(num / max(den, 1) * 100, 1)


def sample_rows(rows, max_test, seed, sample_unit):
    langs = sorted({row["lang"] for row in rows})
    groups = {l: [row for row in rows if row["lang"] == l] for l in langs}
    rng = random.Random(seed)
    per_lang = max(1, max_test // len(langs))
    if sample_unit == "fact":
        def choose_unique(group):
            fact_rows = defaultdict(list)
            for row in group:
                fact_rows[row["fact_id"]].append(row)
            fact_ids = list(fact_rows)
            rng.shuffle(fact_ids)
            picked = []
            for fact_id in fact_ids[:per_lang]:
                picked.append(rng.choice(fact_rows[fact_id]))
            return picked
        sample = []
        for l in langs:
            sample.extend(choose_unique(groups[l]))
    else:
        sample = []
        for l in langs:
            rng.shuffle(groups[l])
            sample.extend(groups[l][:per_lang])
    rng.shuffle(sample)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="translation_resources/ckpt_sft_final.safetensors")
    ap.add_argument("--config", default="config_pretrain.json")
    ap.add_argument("--vocab", default="vocab_7")
    ap.add_argument("--eval", default="sft_data/sft_facts_large/train.jsonl")
    ap.add_argument("--max_test", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=30)
    ap.add_argument("--infer", default="build_infer/infer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_unit", choices=["row", "fact"], default="row",
                    help="row = random prompt rows, fact = one prompt per unique fact_id per language")
    args = ap.parse_args()

    infer = ROOT / args.infer
    assert infer.exists(), f"infer not found: {infer}"

    # 1. Load eval rows (balanced EN/JA).
    rows = []
    for line in open(ROOT / args.eval, encoding="utf-8"):
        r = json.loads(line)
        lang = r["lang"]
        text = r["text"]
        user = text.split("<|user|>")[1].split("<|assistant|>")[0]
        ref = text.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        prompt = text.split("<|assistant|>")[0] + "<|assistant|>"
        fact_id = r.get("fact_id") or f"{r.get('category', 'unknown')}::{(r.get('subject_en') or user).casefold()}"
        rows.append({"lang": lang, "prompt": prompt, "user": user, "ref": ref,
                     "target_lang": lang,
                     "category": r.get("category"), "value_en": r.get("value_en"),
                     "value_ja": r.get("value_ja"), "value_id": r.get("value_id"),
                     "subject_en": r.get("subject_en"),
                     "subject_ja": r.get("subject_ja"), "subject_id": r.get("subject_id"),
                     "fact_id": fact_id,
                     "template_id": r.get("template_id"), "aliases_en": r.get("aliases_en"),
                     "aliases_ja": r.get("aliases_ja")})

    sample = sample_rows(rows, args.max_test, args.seed, args.sample_unit)
    sample_en = sum(1 for row in sample if row["lang"] == "en")
    sample_ja = sum(1 for row in sample if row["lang"] == "ja")
    sample_id = sum(1 for row in sample if row["lang"] == "id")
    print(f"  testing {len(sample)} ({sample_en} EN, {sample_ja} JA, {sample_id} ID, unit={args.sample_unit})")

    subject_index, global_entries = build_indexes(rows)

    # 2. Run infer in batch.
    prompts = [r["prompt"] for r in sample]
    proc = subprocess.run(
        [str(infer), "--model", str(ROOT/args.model), "--config", str(ROOT/args.config),
         "--vocab", str(ROOT/args.vocab), "--greedy", "--max_tokens", str(args.max_tokens),
         "--stdin"],
        input=("\n".join(prompts) + "\n").encode("utf-8"),
        capture_output=True, timeout=3600,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    raw_lines = stdout.split("\n")
    gens = []
    for i in range(len(sample)):
        gens.append(raw_lines[i].rstrip("\n").strip() if i < len(raw_lines) else "")

    # 3. Score.
    eval_langs = ("en", "ja", "id") if any(r["lang"] == "id" for r in rows) else ("en", "ja")
    stats = {l: blank_stats() for l in eval_langs}
    per_category = {l: defaultdict(blank_stats) for l in eval_langs}
    confusion = {l: defaultdict(Counter) for l in eval_langs}
    error_causes = {l: Counter() for l in eval_langs}
    fact_rollup = {l: defaultdict(list) for l in eval_langs}
    detail = []
    for r, pred in zip(sample, gens):
        lang = r["lang"]
        tgt = r["target_lang"]
        s = stats[lang]
        category_stats = per_category[lang][r.get("category") or "unknown"]
        s["n"] += 1
        category_stats["n"] += 1
        if not pred.strip():
            s["empty"] += 1
            category_stats["empty"] += 1
            predicted_category = None
            same_subject = set()
            global_matches = []
            lang_is_ok = False
            hit = 0.0
            aok = attr_ok(r.get("category"), pred, tgt)
            vok = value_hit(r, pred, tgt)
            subject_bound = False
            error_cause = "empty"
        else:
            if tgt in {"en", "id"} and has_ja(pred):
                s["ja_leak"] += 1
                category_stats["ja_leak"] += 1
            if tgt == "ja" and has_lat(pred) and not has_ja(pred):
                s["lat_only"] += 1
                category_stats["lat_only"] += 1
            if tgt in {"en", "id"} and has_lat(pred) and not has_ja(pred):
                s["lang_ok"] += 1
                category_stats["lang_ok"] += 1
                lang_is_ok = True
            elif tgt == "ja" and has_ja(pred):
                s["lang_ok"] += 1
                category_stats["lang_ok"] += 1
                lang_is_ok = True
            else:
                lang_is_ok = False
            hit = content_hit(r["ref"], pred, tgt)
            s["hit"] += hit
            category_stats["hit"] += hit
            aok = attr_ok(r.get("category"), pred, tgt)
            if aok is not None:
                s["attr_n"] += 1
                s["attr_ok"] += int(aok)
                category_stats["attr_n"] += 1
                category_stats["attr_ok"] += int(aok)
            vok = value_hit(r, pred, tgt)
            if vok is not None:
                s["value_n"] += 1
                s["value_ok"] += int(vok)
                category_stats["value_n"] += 1
                category_stats["value_ok"] += int(vok)
            marker_categories = detect_marker_categories(pred, tgt)
            same_subject, global_matches = detect_value_matches(r, pred, tgt, subject_index, global_entries)
            predicted_category = choose_predicted_category(r.get("category"), same_subject, marker_categories, global_matches)
            subject_bound = subject_binding_ok(r, pred, tgt, subject_index)
            if subject_bound:
                s["subject_ok"] += 1
                category_stats["subject_ok"] += 1
            error_cause = classify_error(r, pred, tgt, lang_is_ok, bool(vok), same_subject, global_matches, predicted_category, subject_bound)
        confusion[lang][r.get("category") or "unknown"][predicted_category or "unknown"] += 1
        if error_cause != "correct":
            error_causes[lang][error_cause] += 1
        fact_rollup[lang][r["fact_id"]].append(bool(vok))
        detail.append({
            "lang": lang,
            "fact_id": r["fact_id"],
            "category": r.get("category"),
            "predicted_category": predicted_category,
            "error_cause": error_cause,
            "subject_bound": bool(subject_bound) if pred.strip() else False,
            "q": r["user"][:120],
            "reference": r["ref"],
            "prediction": pred[:120],
            "same_subject_matches": sorted(same_subject),
        })

    print(f"\n{'='*60}")
    print(f" Model: {args.model}")
    print(f" Tested: {len(sample)} (novel fact Q&A, sample_unit={args.sample_unit})")
    print(f"  {'lang':<6} {'n':>4} {'lang-ok':>8} {'attr-ok':>8} {'value-hit':>10} {'subj-bind':>10} {'content-hit':>12} {'empty':>6}")
    for lang in eval_langs:
        s = stats[lang]
        lang_pct = s["lang_ok"] / s["n"] * 100 if s["n"] else 0
        hit = s["hit"] / max(s["n"] - s["empty"], 1)
        attr_pct = s["attr_ok"] / max(s["attr_n"], 1) * 100
        value_pct = s["value_ok"] / max(s["value_n"], 1) * 100
        subj_pct = s["subject_ok"] / max(s["n"] - s["empty"], 1) * 100
        print(f"  {lang:<6} {s['n']:>4} {lang_pct:>7.1f}% {attr_pct:>7.1f}% {value_pct:>9.1f}% {subj_pct:>9.1f}% {hit:>11.1f}% {s['empty']:>6}")
    # combined
    tot_n = sum(stats[l]["n"] for l in eval_langs)
    tot_ok = sum(stats[l]["lang_ok"] for l in eval_langs)
    overall_hit = sum(stats[l]["hit"] for l in eval_langs) / max(tot_n, 1)
    tot_attr_ok = sum(stats[l]["attr_ok"] for l in eval_langs)
    tot_attr_n = sum(stats[l]["attr_n"] for l in eval_langs)
    tot_value_ok = sum(stats[l]["value_ok"] for l in eval_langs)
    tot_value_n = sum(stats[l]["value_n"] for l in eval_langs)
    print(f"  {'both':<6} {tot_n:>4} {tot_ok/max(tot_n,1)*100:>7.1f}% {tot_attr_ok/max(tot_attr_n,1)*100:>7.1f}% {tot_value_ok/max(tot_value_n,1)*100:>9.1f}% {overall_hit*100:>11.1f}%")

    print("\n  leakage:")
    print(f"    EN prompts with JA script in output : {stats['en']['ja_leak']}/{max(stats['en']['n'], 1)}")
    print(f"    JA prompts with Latin-only output   : {stats['ja']['lat_only']}/{max(stats['ja']['n'], 1)}")
    if "id" in eval_langs:
        print(f"    ID prompts with JA script in output : {stats['id']['ja_leak']}/{max(stats['id']['n'], 1)}")

    print("\n  fact-level sampled summary:")
    for lang in eval_langs:
        values = list(fact_rollup[lang].values())
        any_ok = sum(1 for outcome_list in values if any(outcome_list))
        all_ok = sum(1 for outcome_list in values if outcome_list and all(outcome_list))
        print(f"    {lang}: {len(values)} unique fact_ids, any-hit={pct(any_ok, len(values)):.1f}%, all-hit={pct(all_ok, len(values)):.1f}%")

    print("\n  per-category breakdown:")
    for lang in eval_langs:
        for category in sorted(per_category[lang]):
            s = per_category[lang][category]
            print(
                f"    {lang}/{category:<18} n={s['n']:>3} "
                f"lang={pct(s['lang_ok'], s['n']):>5.1f}% "
                f"attr={pct(s['attr_ok'], s['attr_n']):>5.1f}% "
                f"value={pct(s['value_ok'], s['value_n']):>5.1f}%"
            )

    print("\n  top confusions:")
    for lang in eval_langs:
        printed = 0
        for asked_category in sorted(confusion[lang]):
            for predicted_category, count in confusion[lang][asked_category].most_common():
                if predicted_category == asked_category or predicted_category == "unknown":
                    continue
                print(f"    {lang}: {asked_category} -> {predicted_category} = {count}")
                printed += 1
                if printed >= 8:
                    break
            if printed >= 8:
                break
        if not printed:
            print(f"    {lang}: no cross-category confusion captured in this sample")

    print("\n  top error causes:")
    for lang in eval_langs:
        if error_causes[lang]:
            for cause, count in error_causes[lang].most_common(6):
                print(f"    {lang}: {cause} = {count}")
        else:
            print(f"    {lang}: none")
    print(f"{'='*60}")

    fact_level_json = {}
    for lang in eval_langs:
        values = list(fact_rollup[lang].values())
        fact_level_json[lang] = {
            "fact_ids": len(values),
            "any_hit_pct": pct(sum(1 for outcome_list in values if any(outcome_list)), len(values)),
            "all_hit_pct": pct(sum(1 for outcome_list in values if outcome_list and all(outcome_list)), len(values)),
        }

    per_category_json = {}
    for lang in eval_langs:
        per_category_json[lang] = {}
        for category, s in per_category[lang].items():
            per_category_json[lang][category] = {
                "n": s["n"],
                "lang_ok_pct": pct(s["lang_ok"], s["n"]),
                "attr_ok_pct": pct(s["attr_ok"], s["attr_n"]),
                "value_hit_pct": pct(s["value_ok"], s["value_n"]),
                "content_hit_pct": round(s["hit"] / max(s["n"] - s["empty"], 1) * 100, 1),
            }

    confusion_json = {
        lang: {asked: dict(counter) for asked, counter in confusion[lang].items()}
        for lang in eval_langs
    }

    out = {"model": args.model, "num": len(sample),
           "sample_unit": args.sample_unit,
           "lang_ok_pct": {k: round(v["lang_ok"]/max(v["n"],1)*100,1) for k, v in stats.items()},
           "attr_ok_pct": {k: round(v["attr_ok"]/max(v["attr_n"],1)*100,1) for k, v in stats.items()},
           "value_hit_pct": {k: round(v["value_ok"]/max(v["value_n"],1)*100,1) for k, v in stats.items()},
           "content_hit": {k: round(v["hit"]/max(v["n"]-v["empty"],1)*100,1) for k, v in stats.items()},
           "leakage": {
               "en_ja_script": stats["en"]["ja_leak"],
               "ja_latin_only": stats["ja"]["lat_only"],
           },
           "fact_level": fact_level_json,
           "per_category": per_category_json,
           "confusion": confusion_json,
           "error_causes": {lang: dict(counter) for lang, counter in error_causes.items()},
           "samples": detail[:50]}
    outp = ROOT / "results" / "facts_eval.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  detail -> {outp}")


if __name__ == "__main__":
    main()
