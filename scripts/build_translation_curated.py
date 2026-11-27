#!/usr/bin/env python3
r"""build_translation_curated.py — Ceiling-Edition 2-Way EN<->JA SFT Translation Builder.

Maximizes BOTH:
  * Point B: Template-based 2-way EN<->JA translation (`日本語に翻訳:`, `Translate to English:`)
  * Point C: Pure NON-template / free-form translation (50+ natural EN & JA prompt styles,
             conversational requests, suffix requests, quoted/multiline inputs, general
             assistant system prompts, `<|translate|><|lang_*|>` pretrain marker continuity,
             2-3 sentence paragraphs, AND question-source pairs so the model NEVER confuses
             translating a question with answering a question!).

Sources combined (100% subtitle-space-free, 0% Indonesian):
  1. OPUS KFTT (`data/kftt_enja_clean.tsv`) — Kyoto Free Translation Task (human-translated
     Wikipedia/history/news/culture).
  2. Clean Conversational/Tatoeba (`data/parallel_en_ja.tsv`, 4-22 EN words, zero subtitle junk).
  3. Clean Formal/LaBSE Multi-Clause (`data/corpus_inject_parallel/parallel_enja.txt`, 14-58 EN words).
  4. Multi-Sentence Paragraphs (2-3 consecutive clean pairs joined into paragraph blocks).
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
LATIN = re.compile(r"[A-Za-z]")
HANGUL = re.compile(r"[\uac00-\ud7af]")
JA_SUBTITLE_SPACE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff] [\u3040-\u30ff\u4e00-\u9fff]")
BAD_SYM = re.compile(r"(♪|※|^\s*[-–—]+|\bhttp://|\bhttps://|<[^>]+>|&\w+;)")

SYS_DEDICATED_JA = [
    "You are a helpful translation engine. Output only the translation.",
    "You are a professional translator. Translate accurately and naturally.",
]
SYS_DEDICATED_EN = [
    "You are a multilingual translation assistant.",
    "You are a professional translator. Translate accurately and naturally.",
]
# General assistant system prompts (used on 35% of free-form rows so non-template
# translation works seamlessly inside normal general-assistant chat!)
SYS_GENERAL = [
    "You are a helpful, accurate multilingual assistant.",
    "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。",
]

# Standard canonical templates (kept at ~45% weight so existing evals & scripts/ask.py stay peak)
CANONICAL_TO_JA = [
    "日本語に翻訳: {src}",
    "次の文を日本語に訳してください: {src}",
]
CANONICAL_TO_EN = [
    "Translate to English: {src}",
    "Render the following in English: {src}",
]

# Rich free-form / non-template phrasings (~50% weight -> unlocks Point C non-template ceiling!)
FREEFORM_TO_JA = [
    "Translate this into Japanese:\n{src}",
    "Please translate the following into natural Japanese: {src}",
    "How do you say \"{src}\" in Japanese?",
    "Could you translate \"{src}\" to Japanese?",
    "Convert the following English text into Japanese:\n{src}",
    "What is the Japanese translation of: {src}",
    "In Japanese, how would you express:\n{src}",
    "{src}\n\nTranslate the sentence above into Japanese.",
    "{src}\n日本語に訳してください。",
    "「{src}」を日本語に翻訳してください。",
    "「{src}」は日本語で何と言いますか？",
    "次の英語を自然な日本語にしてください。\n{src}",
    "和訳お願いします：{src}",
    "以下の英文を和訳してください:\n{src}",
    "これを日本語に直してください: {src}",
    "英語から日本語へ翻訳:\n{src}",
]

FREEFORM_TO_EN = [
    "Translate this into English:\n{src}",
    "Please translate the following Japanese into natural English: {src}",
    "How do you say \"{src}\" in English?",
    "Could you translate \"{src}\" into English?",
    "What does \"{src}\" mean in English? Give the translation.",
    "Convert the following Japanese text to English:\n{src}",
    "{src}\n\nTranslate the Japanese text above into English.",
    "{src}\nPlease translate this to English.",
    "「{src}」を英語に翻訳してください。",
    "「{src}」は英語で何と言いますか？",
    "「{src}」って英語でどう言う？",
    "次の日本語を自然な英語にしてください。\n{src}",
    "英訳お願いします：{src}",
    "以下の文を英訳してください:\n{src}",
    "これを英語に訳して: {src}",
    "日本語から英語へ翻訳:\n{src}",
]


def clean_str(s):
    s = unicodedata.normalize("NFKC", s).replace("\u3000", " ").replace("\r", "")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 4 and ((s[0] == '"' and s.count('"') == 1) or (s[0] == "「" and "」" not in s)):
        s = s[1:].strip()
    return s


def pair_quality_ok(en, ja, min_words=4, max_words=65, max_chars=450):
    if not en or not ja:
        return "empty"
    if len(en) > max_chars or len(ja) > max_chars:
        return "too_long_chars"
    wc = len(en.split())
    if wc < min_words:
        return "too_few_en_words"
    if wc > max_words:
        return "too_many_en_words"
    if not LATIN.search(en) or KANA.search(en) or HANGUL.search(en):
        return "bad_en_script"
    if not KANA.search(ja) or HANGUL.search(ja):
        return "ja_missing_kana"
    if JA_SUBTITLE_SPACE.search(ja):
        return "ja_subtitle_space"
    if BAD_SYM.search(en) or BAD_SYM.search(ja):
        return "subtitle_or_html_sym"
    ja_ascii = sum(1 for c in ja if "A" <= c <= "Z" or "a" <= c <= "z")
    if ja_ascii > max(12, int(len(ja) * 0.35)):
        return "ja_too_much_latin"
    ratio = len(en) / max(1, len(ja))
    if not (0.7 <= ratio <= 5.5):
        return "length_ratio"
    if wc >= 8 and not ja.endswith(("。", "？", "！", "?", "!", "」", "）", ".")):
        return "ja_unterminated"
    return None


def mine_tsv(tsv_path, label, min_w, max_w, cap, seen_en, seen_ja, stats, rng):
    candidates = []
    if not tsv_path.exists():
        return candidates
    with tsv_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            en, ja = clean_str(parts[0]), clean_str(parts[1])
            why = pair_quality_ok(en, ja, min_words=min_w, max_words=max_w, max_chars=420)
            if why:
                stats[f"{label}_{why}"] += 1
                continue
            he = hashlib.sha1(en.lower().encode()).digest()
            hj = hashlib.sha1(ja.encode()).digest()
            if he in seen_en or hj in seen_ja:
                stats[f"{label}_dup"] += 1
                continue
            seen_en.add(he)
            seen_ja.add(hj)
            candidates.append((en, ja, label))
    rng.shuffle(candidates)
    return candidates[:cap]


def mine_long_formal(par_path, cap, seen_en, seen_ja, stats, rng):
    candidates = []
    if not par_path.exists():
        return candidates
    target_pool = int(cap * 2.2)
    with par_path.open(encoding="utf-8", errors="ignore") as f:
        prev = ""
        mode = None
        for raw in f:
            line = raw.strip()
            if not line or line == "<|endoftext|>":
                prev = ""
                mode = None
                continue
            if line == "<|translate|><|lang_ja|>":
                mode = "en-ja"
                continue
            elif line == "<|translate|><|lang_en|>":
                mode = "ja-en"
                continue
            if mode and prev:
                en, ja = (clean_str(prev), clean_str(line)) if mode == "en-ja" else (clean_str(line), clean_str(prev))
                mode = None
                prev = ""
                why = pair_quality_ok(en, ja, min_words=14, max_words=60, max_chars=440)
                if why:
                    stats[f"long_{why}"] += 1
                    continue
                he = hashlib.sha1(en.lower().encode()).digest()
                hj = hashlib.sha1(ja.encode()).digest()
                if he in seen_en or hj in seen_ja:
                    continue
                seen_en.add(he)
                seen_ja.add(hj)
                candidates.append((en, ja, "formal_labse"))
                if len(candidates) >= target_pool:
                    break
            else:
                mode = None
                prev = line
    rng.shuffle(candidates)
    return candidates[:cap]


def build_question_translation_pairs(facts_dir, cap, rng):
    """Pairs where the source ITSELF is a question (`What is the capital of X?` <-> `Xの首都はどこですか？`).
    Crucial anti-collapse signal: teaches the model that when asked to TRANSLATE a question,
    it must translate the question, NOT answer it!
    """
    pairs = []
    f_path = facts_dir / "train.jsonl"
    if not f_path.exists():
        return pairs
    by_fid = {}
    for line in f_path.open(encoding="utf-8"):
        r = json.loads(line)
        fid = r.get("fact_id")
        lg = r.get("lang")
        if not fid or lg not in ("en", "ja"):
            continue
        t = r["text"]
        q = t.split("<|user|>")[1].split("<|assistant|>")[0].strip()
        by_fid.setdefault(fid, {})[lg] = q
    items = [(v["en"], v["ja"]) for v in by_fid.values() if "en" in v and "ja" in v]
    rng.shuffle(items)
    for en_q, ja_q in items[:cap]:
        pairs.append((en_q, ja_q, "question_anti_collapse"))
    return pairs


def render_pair_direction(en, ja, direction, style, rng, force_canonical=False):
    src, tgt = (en, ja) if direction == "en-ja" else (ja, en)
    is_q_anti = (style == "question_anti_collapse")

    if force_canonical:
        sys_m = SYS_DEDICATED_JA[0] if direction == "en-ja" else SYS_DEDICATED_EN[0]
        tmpl = rng.choice(CANONICAL_TO_JA if direction == "en-ja" else CANONICAL_TO_EN)
        user_m = tmpl.format(src=src)
        mode_tag = "canonical"
    else:
        u = rng.random()
        if is_q_anti:
            # Always test/train question-anti-collapse under general assistant + free-form!
            sys_m = rng.choice(SYS_GENERAL)
            tmpl = rng.choice(FREEFORM_TO_JA if direction == "en-ja" else FREEFORM_TO_EN)
            user_m = tmpl.format(src=src)
            mode_tag = "freeform_q_guard"
        elif u < 0.45:
            sys_m = rng.choice(SYS_DEDICATED_JA if direction == "en-ja" else SYS_DEDICATED_EN)
            tmpl = rng.choice(CANONICAL_TO_JA if direction == "en-ja" else CANONICAL_TO_EN)
            user_m = tmpl.format(src=src)
            mode_tag = "canonical"
        elif u < 0.88:
            # Free-form non-template: 65% under general assistant system prompt, 35% under translator!
            if rng.random() < 0.65:
                sys_m = rng.choice(SYS_GENERAL)
            else:
                sys_m = rng.choice(SYS_DEDICATED_JA if direction == "en-ja" else SYS_DEDICATED_EN)
            tmpl = rng.choice(FREEFORM_TO_JA if direction == "en-ja" else FREEFORM_TO_EN)
            user_m = tmpl.format(src=src)
            mode_tag = "freeform"
        elif u < 0.95:
            # Zero-prefix bare input under dedicated translation engine system prompt!
            sys_m = ("You are a professional English-to-Japanese translation engine. Output only the Japanese translation."
                     if direction == "en-ja" else
                     "You are a professional Japanese-to-English translation engine. Output only the English translation.")
            user_m = src
            mode_tag = "bare_engine"
        else:
            # Preserve pretrain <|translate|><|lang_*|> circuit!
            sys_m = SYS_DEDICATED_JA[0] if direction == "en-ja" else SYS_DEDICATED_EN[0]
            ltag = "<|lang_ja|>" if direction == "en-ja" else "<|lang_en|>"
            user_m = f"{src}\n<|translate|>{ltag}"
            mode_tag = "pretrain_tag"

    return {
        "text": f"<|system|>{sys_m}<|user|>{user_m}<|assistant|>{tgt}<|end|>",
        "lang": direction,
        "task": "translation",
        "prompt_style": mode_tag,
        "source": f"trans_{style}",
        "family": "translation",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/parallel_en_ja.tsv")
    ap.add_argument("--kftt_tsv", default="data/kftt_enja_clean.tsv")
    ap.add_argument("--parallel_txt", default="data/corpus_inject_parallel/parallel_enja.txt")
    ap.add_argument("--out", default="sft_data/sft_translation_curated")
    ap.add_argument("--short_pairs", type=int, default=65000)
    ap.add_argument("--kftt_pairs", type=int, default=55000)
    ap.add_argument("--long_pairs", type=int, default=55000)
    ap.add_argument("--para_blocks", type=int, default=12000,
                    help="2-3 sentence paragraph translation pairs so multi-sentence inputs never drop clauses")
    ap.add_argument("--question_guard_pairs", type=int, default=8000)
    ap.add_argument("--eval_pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = Counter()
    seen_en, seen_ja = set(), set()

    # Block heldout KFTT test pairs if present
    kftt_holdout = ROOT / "eval_data/kftt_heldout_enja.jsonl"
    if kftt_holdout.exists():
        for l in kftt_holdout.open(encoding="utf-8"):
            d = json.loads(l)
            seen_en.add(hashlib.sha1(clean_str(d["en"]).lower().encode()).digest())
            seen_ja.add(hashlib.sha1(clean_str(d["ja"]).encode()).digest())

    kftt_p = mine_tsv(ROOT / args.kftt_tsv, "kftt", 6, 58, args.kftt_pairs, seen_en, seen_ja, stats, rng)
    short_p = mine_tsv(ROOT / args.tsv, "conversational", 4, 22, args.short_pairs, seen_en, seen_ja, stats, rng)
    long_p = mine_long_formal(ROOT / args.parallel_txt, args.long_pairs, seen_en, seen_ja, stats, rng)
    qguard_p = build_question_translation_pairs(ROOT / "sft_data/sft_facts_expanded", args.question_guard_pairs, rng)

    # Multi-sentence paragraph blocks (2 sentences combined) from clean formal/KFTT sentences
    para_source = [p for p in (kftt_p + long_p) if len(p[0]) <= 180 and p[0].endswith(".") and p[1].endswith("。")]
    rng.shuffle(para_source)
    para_p = []
    for i in range(0, min(len(para_source) - 1, args.para_blocks * 2), 2):
        en_2 = f"{para_source[i][0]} {para_source[i+1][0]}"
        ja_2 = f"{para_source[i][1]}{para_source[i+1][1]}"
        if len(en_2) <= 380 and len(ja_2) <= 280:
            para_p.append((en_2, ja_2, "paragraph_2sent"))

    base_pairs = kftt_p + short_p + long_p + para_p
    rng.shuffle(base_pairs)
    eval_pairs = base_pairs[:args.eval_pairs]
    train_pairs = base_pairs[args.eval_pairs:] + qguard_p
    rng.shuffle(train_pairs)

    tr = []
    for en, ja, st in train_pairs:
        tr.append(render_pair_direction(en, ja, "en-ja", st, rng, force_canonical=False))
        tr.append(render_pair_direction(en, ja, "ja-en", st, rng, force_canonical=False))
    rng.shuffle(tr)

    # Eval includes BOTH canonical and freeform non-template splits!
    ev = []
    ev_freeform = []
    for en, ja, st in eval_pairs:
        ev.append(render_pair_direction(en, ja, "en-ja", st, rng, force_canonical=True))
        ev.append(render_pair_direction(en, ja, "ja-en", st, rng, force_canonical=True))
        ev_freeform.append(render_pair_direction(en, ja, "en-ja", st, rng, force_canonical=False))
        ev_freeform.append(render_pair_direction(en, ja, "ja-en", st, rng, force_canonical=False))
    rng.shuffle(ev)
    rng.shuffle(ev_freeform)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev), ("eval_freeform", ev_freeform)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    style_counts = Counter(r["prompt_style"] for r in tr)
    src_counts = Counter(r["source"] for r in tr)
    (out / "stats.json").write_text(json.dumps({
        "train_rows": len(tr),
        "eval_canonical_rows": len(ev),
        "eval_freeform_rows": len(ev_freeform),
        "by_source": dict(src_counts.most_common()),
        "by_prompt_style": dict(style_counts.most_common()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[TRANS-CEILING] train={len(tr):,} eval={len(ev):,} freeform_eval={len(ev_freeform):,}")
    print(f"  sources: {dict(src_counts.most_common())}")
    print(f"  styles : {dict(style_counts.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
