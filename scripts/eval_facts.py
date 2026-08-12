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
import json
import re
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYS_EN = "You are a helpful, accurate multilingual assistant."
SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="translation_resources/ckpt_sft_final.safetensors")
    ap.add_argument("--config", default="config_pretrain.json")
    ap.add_argument("--vocab", default="vocab_6")
    ap.add_argument("--eval", default="sft_data/sft_facts_large/train.jsonl")
    ap.add_argument("--max_test", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=30)
    ap.add_argument("--infer", default="build_infer/infer")
    ap.add_argument("--seed", type=int, default=42)
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
        rows.append({"lang": lang, "prompt": prompt, "user": user, "ref": ref,
                     "target_lang": "en" if lang == "en" else "ja"})

    en = [r for r in rows if r["lang"] == "en"]
    ja = [r for r in rows if r["lang"] == "ja"]
    rng = random.Random(args.seed)
    half = max(1, args.max_test // 2)
    rng.shuffle(en); rng.shuffle(ja)
    sample = en[:half] + ja[:half]
    rng.shuffle(sample)
    print(f"  testing {len(sample)} ({half} EN, {half} JA)")

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
    stats = {"en": {"n": 0, "lang_ok": 0, "hit": 0.0, "empty": 0},
             "ja": {"n": 0, "lang_ok": 0, "hit": 0.0, "empty": 0}}
    detail = []
    for r, pred in zip(sample, gens):
        lang = r["lang"]
        tgt = r["target_lang"]
        s = stats[lang]
        s["n"] += 1
        if not pred.strip():
            s["empty"] += 1
        else:
            if tgt == "en" and has_lat(pred) and not has_ja(pred):
                s["lang_ok"] += 1
            elif tgt == "ja" and has_ja(pred):
                s["lang_ok"] += 1
            s["hit"] += content_hit(r["ref"], pred, tgt)
        detail.append({"lang": lang, "q": r["user"][:60], "reference": r["ref"],
                       "prediction": pred[:60]})

    print(f"\n{'='*60}")
    print(f" Model: {args.model}")
    print(f" Tested: {len(sample)} (novel fact Q&A)")
    print(f"  {'lang':<6} {'n':>4} {'lang-ok':>8} {'content-hit':>12} {'empty':>6}")
    for lang in ("en", "ja"):
        s = stats[lang]
        lang_pct = s["lang_ok"] / s["n"] * 100 if s["n"] else 0
        hit = s["hit"] / max(s["n"] - s["empty"], 1)
        print(f"  {lang:<6} {s['n']:>4} {lang_pct:>7.1f}% {hit:>11.1f}% {s['empty']:>6}")
    # combined
    tot_n = stats["en"]["n"] + stats["ja"]["n"]
    tot_ok = stats["en"]["lang_ok"] + stats["ja"]["lang_ok"]
    overall_hit = (stats["en"]["hit"] + stats["ja"]["hit"]) / max(tot_n, 1)
    print(f"  {'both':<6} {tot_n:>4} {tot_ok/max(tot_n,1)*100:>7.1f}% {overall_hit*100:>11.1f}%")
    print(f"{'='*60}")

    out = {"model": args.model, "num": len(sample),
           "lang_ok_pct": {k: round(v["lang_ok"]/max(v["n"],1)*100,1) for k, v in stats.items()},
           "content_hit": {k: round(v["hit"]/max(v["n"]-v["empty"],1)*100,1) for k, v in stats.items()},
           "samples": detail[:20]}
    outp = ROOT / "results" / "facts_eval.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  detail -> {outp}")


if __name__ == "__main__":
    main()