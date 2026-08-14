#!/usr/bin/env python3
"""Audit EN/JA fact SFT data before training.

Fails non-zero on issues that poison fact QA: generic single-fact prompts,
same-prompt conflicting answers, language mismatch, QID noise, or category
imbalance. Designed for sft_data/sft_facts_strict/{train,eval}.jsonl.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

GENERIC_EN = re.compile(
    r"^(what is|can you tell me|do you know|what do you know about|tell me about|"
    r"what can you say about|i have a question about|could you tell me about|"
    r"please tell me about|i would like to know|what can you tell me about|"
    r"i am curious about|would you tell me about|can you explain|"
    r"what is known about|tell me what you know about|please explain|"
    r"what do you think)\b(?!.*\b(capital|currency|official language|population|"
    r"highest point|highest place|continent|area|chemical symbol|atomic number|"
    r"people live|money|element)\b)",
    re.I,
)
GENERIC_JA_MARKERS = [
    "について教えてください", "について説明してください", "について話してください",
    "についてどう思いますか", "についてどうですか", "について詳しく教えてください",
    "について知りたいです", "について共有できますか", "について何か言えますか",
    "について興味があります", "について何が知られていますか", "は何ですか？",
    "は何でしょうか", "はどういうものですか", "は何だと思いますか",
]
JA_ATTR_MARKERS = [
    "首都", "通貨", "公用語", "人口", "一番高い", "最高地点", "大陸", "面積",
    "化学記号", "原子番号", "公式に使われる言語", "何人", "住んでいます", "お金",
    "元素", "割り当てられた番号", "表す記号",
]
EN_ATTR_MARKERS = [
    "capital", "currency", "official language", "population", "highest point",
    "continent", "area", "chemical symbol", "atomic number", "language is official",
    "language", "official in", "capital city", "money", "officially used",
    "highest place", "square kilometers", "symbol represents", "element", "people live",
]


def has_ja(s):
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", s))


def has_lat(s):
    return bool(re.search(r"[A-Za-z]{2,}", s))


def parse_text(text):
    try:
        user = text.split("<|user|>")[1].split("<|assistant|>")[0]
        ans = text.split("<|assistant|>")[1].split("<|end|>")[0]
    except Exception:
        return None, None
    return user.strip(), ans.strip()


def is_generic(lang, q):
    if lang == "en":
        return bool(GENERIC_EN.search(q)) or not any(m in q.lower() for m in EN_ATTR_MARKERS)
    if lang == "ja":
        return any(m in q for m in GENERIC_JA_MARKERS) and not any(m in q for m in JA_ATTR_MARKERS)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["sft_data/sft_facts_strict/train.jsonl", "sft_data/sft_facts_strict/eval.jsonl"])
    ap.add_argument("--max_category_share", type=float, default=0.30)
    args = ap.parse_args()

    rows = []
    for raw in args.paths:
        path = ROOT / raw
        if not path.exists():
            print(f"[fail] missing file: {path}")
            return 1
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                r = json.loads(line)
                q, a = parse_text(r.get("text", ""))
                rows.append((raw, i, r, q, a))

    errors = []
    langs = Counter()
    cats = Counter()
    prompt_answers = defaultdict(set)
    fact_values = defaultdict(set)
    generic = []
    qid = []
    lang_mismatch = []
    bad_format = []

    for path, line_no, r, q, a in rows:
        lang = r.get("lang")
        cat = r.get("category", "")
        langs[lang] += 1
        cats[cat] += 1
        if q is None or a is None:
            bad_format.append((path, line_no))
            continue
        if re.search(r"\bQ\d{4,}\b", r.get("text", "")):
            qid.append((path, line_no, q))
        if is_generic(lang, q):
            generic.append((path, line_no, lang, q))
        if lang == "en" and (has_ja(q) or has_ja(a) or not has_lat(a)):
            lang_mismatch.append((path, line_no, lang, q, a))
        if lang == "ja" and (not has_ja(q) or not has_ja(a)):
            lang_mismatch.append((path, line_no, lang, q, a))
        prompt_answers[(lang, q)].add(a)
        if cat and r.get("subject_en"):
            fact_values[(cat, r.get("subject_en", "").casefold(), lang)].add(a)

    prompt_conflicts = [(k, v) for k, v in prompt_answers.items() if len(v) > 1]
    fact_conflicts = [(k, v) for k, v in fact_values.items() if len(v) > 1]
    if bad_format:
        errors.append(f"bad format rows: {len(bad_format)}")
    if qid:
        errors.append(f"QID/noise rows: {len(qid)}")
    if generic:
        errors.append(f"generic single-fact prompts: {len(generic)}")
    if lang_mismatch:
        errors.append(f"language mismatch rows: {len(lang_mismatch)}")
    if prompt_conflicts:
        errors.append(f"same prompt with multiple answers: {len(prompt_conflicts)}")
    if fact_conflicts:
        errors.append(f"same fact key with multiple rendered answers: {len(fact_conflicts)}")
    if rows:
        max_share = max(cats.values()) / len(rows)
        if max_share > args.max_category_share:
            errors.append(f"category imbalance: max share {max_share:.1%} > {args.max_category_share:.1%}")

    print("=" * 60)
    print(f"rows: {len(rows)}")
    print(f"langs: {dict(langs)}")
    print(f"categories: {dict(cats)}")
    print(f"generic prompts: {len(generic)}")
    print(f"prompt conflicts: {len(prompt_conflicts)}")
    print(f"fact conflicts: {len(fact_conflicts)}")
    print(f"language mismatch: {len(lang_mismatch)}")
    print(f"QID/noise rows: {len(qid)}")
    print("=" * 60)

    for label, sample in [
        ("generic", generic[:5]),
        ("language mismatch", lang_mismatch[:5]),
        ("prompt conflict", prompt_conflicts[:3]),
    ]:
        if sample:
            print(f"\n{label} examples:")
            for x in sample:
                print("  ", x)

    if errors:
        print("\nFAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
