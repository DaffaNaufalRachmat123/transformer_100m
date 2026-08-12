#!/usr/bin/env python3
"""eval_translation.py — Measure EN<->JA translation quality on NOVEL sentences.

Runs the built infer binary over a held-out eval set (sft_translation_eval.jsonl,
sentences excluded from training) and scores translations with BLEU + chrF.
This measures real generalization, not memorization.

Usage:
    python3 scripts/eval_translation.py \
        [--model translation_resources/ckpt_sft_final.safetensors] \
        [--config config_pretrain.json] [--vocab vocab_6] \
        [--eval sft_data/sft_clean/sft_translation_eval.jsonl] \
        [--max_test 200] [--max_tokens 40] \
        [--infer build_infer/infer] [--out results/translation_eval.json]

Output: JSON with per-direction BLEU/chrF + a sample of predictions.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── BLEU (corpus-level, up to 4-gram, closest-length brevity penalty) ──────
def _tokenize(s, lang):
    if lang == "ja":
        # Japanese: split into kanji/kana/katakana runs + punctuation.
        return re.findall(r"[\u3040-\u30ff]|[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\u3040-\u30ff\u4e00-\u9fff]", s)
    return re.findall(r"[A-Za-z0-9]+('[a-z]+)?", s.replace("'", " ' "))


def _n_grams(toks, n):
    return [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]


def corpus_bleu(references, hypotheses, lang, max_order=4):
    """references: list of token lists; hypotheses: list of token lists."""
    n_matches = Counter()
    n_totals = Counter()
    ref_lens = []
    hyp_lens = []
    for ref, hyp in zip(references, hypotheses):
        ref_lens.append(len(ref))
        hyp_lens.append(len(hyp))
        # Build per-order reference n-gram counts.
        ref_counts = {}
        for n in range(1, max_order+1):
            ref_counts[n] = Counter(_n_grams(ref, n))
        for n in range(1, max_order+1):
            hyp_counts = Counter(_n_grams(hyp, n))
            for gram, c in hyp_counts.items():
                n_matches[n] += min(c, ref_counts[n].get(gram, 0))
            n_totals[n] += sum(hyp_counts.values())
    # Brevity penalty on total lengths.
    ref_total = sum(ref_lens)
    hyp_total = sum(hyp_lens)
    math = __import__("math")
    if hyp_total == 0:
        return 0.0
    bp = 1.0 if hyp_total > ref_total else math.exp(1 - ref_total / hyp_total)
    precisions = []
    for n in range(1, max_order+1):
        if n_totals[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(n_matches[n] / n_totals[n])
    if min(precisions) == 0:
        return 0.0
    geo = sum(p*math.log(p) for p in precisions) / max_order
    return bp * math.exp(geo)


def chrF(reference, hypothesis, lang, n=6):
    """Character n-gram F-score (chrF2-style, beta favouring recall slightly)."""
    r = list(reference)
    h = list(hypothesis)
    if not r or not h:
        return 0.0
    ref_ng = Counter("".join(r[i:i+n]) for i in range(len(r)-n+1)) if len(r) >= n else Counter(r)
    hyp_ng = Counter("".join(h[i:i+n]) for i in range(len(h)-n+1)) if len(h) >= n else Counter(h)
    if not hyp_ng:
        return 0.0
    matches = sum(min(ref_ng.get(g, 0), c) for g, c in hyp_ng.items())
    prec = matches / sum(hyp_ng.values())
    rec = matches / sum(ref_ng.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def gen_lang(lang):
    return "ja" if lang == "en-ja" else "en"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="translation_resources/ckpt_sft_final.safetensors")
    ap.add_argument("--config", default="config_pretrain.json")
    ap.add_argument("--vocab", default="vocab_6")
    ap.add_argument("--eval", default="sft_data/sft_clean/sft_translation_eval.jsonl")
    ap.add_argument("--max_test", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=40)
    ap.add_argument("--infer", default="build_infer/infer")
    ap.add_argument("--out", default="results/translation_eval.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    infer = ROOT / args.infer
    assert infer.exists(), f"infer binary not found: {infer}"

    # 1. Load eval rows.
    rows = []
    for line in open(ROOT / args.eval, encoding="utf-8"):
        r = json.loads(line)
        t = r["text"]
        lang = r["lang"]
        user = t.split("<|user|>")[1].split("<|assistant|>")[0]
        ref = t.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        prompt = t.split("<|assistant|>")[0] + "<|assistant|>"  # include assistant marker so model starts generating
        rows.append({"lang": lang, "prompt": prompt, "user": user, "ref": ref})
    print(f"  eval rows loaded: {len(rows)}")

    # 2. Sample balanced subset.
    en_ja = [r for r in rows if r["lang"] == "en-ja"]
    ja_en = [r for r in rows if r["lang"] == "ja-en"]
    import random
    rng = random.Random(args.seed)
    half = max(1, args.max_test // 2)
    rng.shuffle(en_ja); rng.shuffle(ja_en)
    sample = en_ja[:half] + ja_en[:half]
    rng.shuffle(sample)
    print(f"  testing {len(sample)} prompts ({half} en-ja, {half} ja-en)")

    # 3. Run infer in batch (--stdin): model loaded once.
    prompts = [r["prompt"] for r in sample]
    proc = subprocess.run(
        [str(infer), "--model", str(ROOT/args.model), "--config", str(ROOT/args.config),
         "--vocab", str(ROOT/args.vocab), "--greedy", "--max_tokens", str(args.max_tokens),
         "--stdin"],
        input=("\n".join(prompts) + "\n").encode("utf-8"),
        capture_output=True, timeout=3600,
    )
    # stderr may contain non-UTF8 bytes (mojibake tokens); decode leniently.
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"  [warn] infer exited {proc.returncode}: "
              f"{proc.stderr.decode('utf-8', errors='replace')[-500:]}")
    sys.stdout.flush()
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"  [warn] infer exited {proc.returncode}: "
              f"{proc.stderr.decode('utf-8', errors='replace')[-500:]}")
    sys.stdout.flush()
    # stdout = one generation per line (final "\n" per prompt). Align by index;
    # empty generations (model fails to produce output) are kept as "".
    raw_lines = stdout.split("\n")
    gens = []
    for i in range(len(sample)):
        gens.append(raw_lines[i].rstrip("\n").strip() if i < len(raw_lines) else "")
    n_nonempty = sum(1 for g in gens if g.strip())
    print(f"  generations parsed: {len(gens)} (non-empty: {n_nonempty})")

    # 4. Score.
    results = {"en-ja": {"refs": [], "hyps": [], "bleu": 0, "chrf": 0},
               "ja-en": {"refs": [], "hyps": [], "bleu": 0, "chrf": 0}}
    detail = []
    for i, r in enumerate(sample):
        hyp = gens[i] if i < len(gens) else ""
        lang = r["lang"]; glang = gen_lang(lang)
        ref_toks = _tokenize(r["ref"], glang)
        hyp_toks = _tokenize(hyp, glang)
        results[lang]["refs"].append(ref_toks)
        results[lang]["hyps"].append(hyp_toks)
        detail.append({"lang": lang, "user": r["user"], "reference": r["ref"],
                       "prediction": hyp, "chrf": round(chrF(r["ref"], hyp, glang), 4)})

    bleu = {}
    chrf = {}
    for lang in ("en-ja", "ja-en"):
        res = results[lang]
        res["bleu"] = corpus_bleu(res["refs"], res["hyps"], gen_lang(lang))
        res["chrf"] = sum(chrF(r_, h_, gen_lang(lang))
                          for r_, h_ in zip(res["refs"], res["hyps"])) / max(len(res["refs"]), 1)
        bleu[lang] = res["bleu"]
        chrf[lang] = res["chrf"]

    # Combined: average of the two directions (BLEU/chrF are not comparable
    # across different target tokenizations, so we average per-direction scores).
    combined_bleu = (bleu["en-ja"] + bleu["ja-en"]) / 2
    combined_chrf = (chrf["en-ja"] + chrf["ja-en"]) / 2

    out = {
        "model": args.model,
        "num_tested": len(sample),
        "bleu": {"en-ja": bleu["en-ja"], "ja-en": bleu["ja-en"], "combined": combined_bleu},
        "chrf": {"en-ja": chrf["en-ja"], "ja-en": chrf["ja-en"], "combined": combined_chrf},
        "samples": detail[:20],
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f" Model: {args.model}")
    print(f" Tested: {len(sample)} novel prompts")
    print(f"  {'direction':<8} {'BLEU':>8} {'chrF':>8}")
    print(f"  {'en-ja':<8} {bleu['en-ja']:>8.3f} {chrf['en-ja']:>8.3f}")
    print(f"  {'ja-en':<8} {bleu['ja-en']:>8.3f} {chrf['ja-en']:>8.3f}")
    print(f"  {'combined':<8} {combined_bleu:>8.3f} {combined_chrf:>8.3f}")
    print(f"{'='*60}")
    print(f" Detail saved to {out_path}")


if __name__ == "__main__":
    main()