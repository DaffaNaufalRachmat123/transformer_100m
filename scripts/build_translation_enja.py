#!/usr/bin/env python3
"""Build EN<->JA translation SFT from the full parallel corpus.

sft_translation_tri spends 1.2M of its 2M rows on Indonesian and keeps only
400K pairs per direction, while data/parallel_en_ja.tsv holds 1.96M pairs. This
reads the source directly so the EN/JA model gets all of it.

Pairs are filtered before use. Very short sources ("Why?", "どうも") have many
equally valid translations, and the mixer rejects them later as contradictory
targets anyway; screening here keeps the corpus honest instead of discovering
the conflict downstream.

  python3 scripts/build_translation_enja.py --out sft_data/sft_translation_enja
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")

SYS_TO_JA = "You are a helpful translation engine. Output only the translation."
SYS_TO_EN = "You are a multilingual translation assistant."

# Prompt wording must stay in the SFT/inference templates already used by
# scripts/ask.py and eval_multiskill.py, or the model sees a different task.
Q_TO_JA = ["日本語に翻訳: {src}", "次の文を日本語に訳してください: {src}"]
Q_TO_EN = ["Translate to English: {src}", "Render the following in English: {src}"]


def is_en(s):
    return bool(LATIN.search(s)) and not KANA.search(s)


def is_ja(s):
    return bool(KANA.search(s) or HAN.search(s))


def clean(s):
    s = unicodedata.normalize("NFKC", s).replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()


def acceptable(en, ja, min_en_chars, max_chars):
    """Screen a pair; returns a reject reason or None."""
    if not en or not ja:
        return "empty"
    if len(en) > max_chars or len(ja) > max_chars:
        return "too_long"
    if len(en) < min_en_chars:
        return "too_short"
    if not is_en(en):
        return "en_side_not_latin"
    if not is_ja(ja):
        return "ja_side_not_japanese"
    # A pair whose sides differ wildly in length is usually a misalignment.
    ratio = len(en) / max(1, len(ja))
    if not 0.4 <= ratio <= 8.0:
        return "length_ratio"
    if en == ja:
        return "identical"
    # Untranslated copies: the Japanese side is mostly the English source.
    if is_en(ja) and not KANA.search(ja):
        return "ja_side_untranslated"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/parallel_en_ja.tsv")
    ap.add_argument("--out", default="sft_data/sft_translation_enja")
    ap.add_argument("--max_pairs", type=int, default=900_000)
    ap.add_argument("--min_en_chars", type=int, default=25,
                    help="short sources have many valid translations")
    ap.add_argument("--max_chars", type=int, default=400)
    ap.add_argument("--eval_pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    stats = Counter()
    pairs, seen_en, seen_ja = [], set(), set()

    with open(a.tsv, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                stats["malformed"] += 1
                continue
            en, ja = clean(parts[0]), clean(parts[1])
            why = acceptable(en, ja, a.min_en_chars, a.max_chars)
            if why:
                stats[why] += 1
                continue
            # Drop repeats of either side so no source maps to two targets.
            he = hashlib.sha1(en.encode()).digest()
            hj = hashlib.sha1(ja.encode()).digest()
            if he in seen_en or hj in seen_ja:
                stats["dup_side"] += 1
                continue
            seen_en.add(he)
            seen_ja.add(hj)
            pairs.append((en, ja))
            stats["kept"] += 1

    rng.shuffle(pairs)
    pairs = pairs[:a.max_pairs]
    n_eval = min(a.eval_pairs, len(pairs) // 20)
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]

    def render(chunk):
        rows = []
        for en, ja in chunk:
            rows.append({
                "text": f"<|system|>{SYS_TO_JA}<|user|>"
                        f"{rng.choice(Q_TO_JA).format(src=en)}"
                        f"<|assistant|>{ja}<|end|>",
                "lang": "en-ja", "task": "translation", "pair": "en-ja",
            })
            rows.append({
                "text": f"<|system|>{SYS_TO_EN}<|user|>"
                        f"{rng.choice(Q_TO_EN).format(src=ja)}"
                        f"<|assistant|>{en}<|end|>",
                "lang": "ja-en", "task": "translation", "pair": "ja-en",
            })
        rng.shuffle(rows)
        return rows

    tr, ev = render(train_pairs), render(eval_pairs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(
        {"pairs_kept": len(pairs), "train_rows": len(tr), "eval_rows": len(ev),
         "filters": dict(stats.most_common())}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"[TRANS] pairs kept {len(pairs)} -> train {len(tr)} / eval {len(ev)}")
    for k, v in stats.most_common(10):
        print(f"    {k:22s} {v}")


if __name__ == "__main__":
    main()
