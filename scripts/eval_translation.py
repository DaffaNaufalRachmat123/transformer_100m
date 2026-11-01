#!/usr/bin/env python3
"""eval_translation.py v2 — Trilingual translation eval with tier metrics.

Fixes vs v1:
  - ALIGNMENT: uses infer's --eot_marker (<<<EOT>>>) so multi-line
    generations can no longer shift the output index (v1's chrF en-ja
    0.0126 vs BLEU 0.63 anomaly).
  - 3 pairs x 2 directions (en<->id, en<->ja, ja<->id), auto-detected from
    the eval file's lang field.
  - Fuzzy contamination screen: eval rows whose 5-gram signature overlaps a
    training row are reported (and excluded from T3 with --strict_dedup).
  - Tiered metrics:
      T1 compliance  — target script purity + number copying (approaching 100%)
      T2 bank        — exact match on task=translation_bank rows
      T3 novel       — corpus BLEU + chrF on non-bank rows

Usage:
  python3 scripts/eval_translation.py \
      --eval sft_data/sft_translation_tri/eval.jsonl \
      --bank_eval sft_data/sft_translation_bank/eval.jsonl \
      [--dedup_train sft_data/sft_translation_tri/train.jsonl,...] \
      [--max_test 300] [--max_tokens 60]
"""

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRS = ("en-ja", "ja-en", "en-id", "id-en", "ja-id", "id-ja")
EOT = "<<<EOT>>>"


def has_ja(s):
    return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in s)


def gen_lang(lang):
    return "ja" if lang.endswith("ja") else ("en" if lang.endswith("en") else "id")


def _tokenize(s, lang):
    if lang == "ja":
        return re.findall(r"[\u3040-\u30ff]|[\u4e00-\u9fff]|[A-Za-z0-9]+|[^\s\u3040-\u30ff\u4e00-\u9fff]", s)
    return re.findall(r"[A-Za-z0-9]+('[a-z]+)?", s.replace("'", " ' "))


def _n_grams(toks, n):
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def corpus_bleu(refs, hyps, max_order=4):
    n_matches, n_totals = Counter(), Counter()
    ref_total = hyp_total = 0
    for ref, hyp in zip(refs, hyps):
        ref_total += len(ref)
        hyp_total += len(hyp)
        ref_counts = {n: Counter(_n_grams(ref, n)) for n in range(1, max_order + 1)}
        for n in range(1, max_order + 1):
            for gram, c in Counter(_n_grams(hyp, n)).items():
                n_matches[n] += min(c, ref_counts[n].get(gram, 0))
            n_totals[n] += sum(Counter(_n_grams(hyp, n)).values())
    if hyp_total == 0:
        return 0.0
    bp = 1.0 if hyp_total > ref_total else math.exp(1 - ref_total / hyp_total)
    precs = []
    for n in range(1, max_order + 1):
        precs.append(n_matches[n] / n_totals[n] if n_totals[n] else 0.0)
    if min(precs) == 0:
        return 0.0
    geo = sum(p * math.log(p) for p in precs) / max_order
    return bp * math.exp(geo)


def chrF(ref, hyp, n=6):
    r, h = list(ref), list(hyp)
    if not r or not h:
        return 0.0
    ref_ng = Counter("".join(r[i:i + n]) for i in range(len(r) - n + 1)) if len(r) >= n else Counter(r)
    hyp_ng = Counter("".join(h[i:i + n]) for i in range(len(h) - n + 1)) if len(h) >= n else Counter(h)
    if not hyp_ng:
        return 0.0
    matches = sum(min(ref_ng.get(g, 0), c) for g, c in hyp_ng.items())
    prec = matches / sum(hyp_ng.values())
    rec = matches / sum(ref_ng.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def numbers_of(s):
    return {m.group(0).replace(",", "") for m in re.finditer(r"\d[\d,\.]*", s) if m.group(0).rstrip(".")}


def sig(s):
    t = re.sub(r"\s+", " ", s.casefold())
    return {t[i:i + 5] for i in range(max(len(t) - 4, 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="translation_resources/ckpt_sft_final.safetensors")
    ap.add_argument("--config", default="config_pretrain.json")
    ap.add_argument("--vocab", default="vocab_9")
    ap.add_argument("--eval", default="sft_data/sft_translation_tri/eval.jsonl")
    ap.add_argument("--bank_eval", default="sft_data/sft_translation_bank/eval.jsonl")
    ap.add_argument("--dedup_train", default="")
    ap.add_argument("--strict_dedup", action="store_true",
                    help="exclude contaminated rows from T3 instead of just reporting")
    ap.add_argument("--max_test", type=int, default=300, help="per direction (novel)")
    ap.add_argument("--max_bank", type=int, default=200, help="per direction (bank)")
    ap.add_argument("--max_tokens", type=int, default=60)
    ap.add_argument("--infer", default="build_infer/infer")
    ap.add_argument("--infer_flags", default="--rep_penalty 1.0",
                    help="extra flags passed to infer. Default turns the repetition "
                         "penalty off so scores measure the model, not the sampler "
                         "(e.g. \"--rep_penalty 1.3 --hard_block 3\" to reproduce old runs, "
                         "\"--no_bos\" to test without BOS)")
    ap.add_argument("--out", default="results/translation_eval_v2.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    # ── 1. Load eval rows (novel + bank)
    def load_rows(path, cap_per_dir):
        out = []
        per = Counter()
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rng.shuffle(rows)
        for r in rows:
            lang = r.get("lang")
            if lang not in DIRS:
                continue
            if per[lang] >= cap_per_dir:
                continue
            t = r["text"]
            try:
                user = t.split("<|user|>")[1].split("<|assistant|>")[0]
                ref = t.split("<|assistant|>")[1].split("<|end|>")[0].strip()
            except IndexError:
                continue
            prompt = t.split("<|assistant|>")[0] + "<|assistant|>"
            out.append({"lang": lang, "prompt": prompt, "ref": ref,
                        "task": r.get("task", "")})
            per[lang] += 1
        return out

    novel_rows = load_rows(ROOT / args.eval, args.max_test)
    bank_rows = load_rows(ROOT / args.bank_eval, args.max_bank)
    all_rows = novel_rows + bank_rows
    print(f"eval rows: {len(novel_rows)} novel + {len(bank_rows)} bank "
          f"({dict(Counter(r['lang'] for r in all_rows))})")

    # ── 2. Contamination screen vs training data
    train_sigs = set()
    for path in filter(None, args.dedup_train.split(",")):
        p = ROOT / path
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            try:
                t = json.loads(line)["text"]
            except json.JSONDecodeError:
                continue
            src = t.split("<|user|>")[1].split("<|assistant|>")[0] if "<|user|>" in t else ""
            train_sigs |= sig(src.split(":", 1)[-1]) if ":" in src else sig(src)
            ans = t.split("<|assistant|>")[1].split("<|end|>")[0] if "<|assistant|>" in t else ""
            train_sigs |= sig(ans)

    contaminated = set()
    for i, r in enumerate(all_rows):
        if sig(r["ref"]) & train_sigs:
            contaminated.add(i)
    print(f"contaminated (5-gram overlap w/ train): {len(contamined)}/{len(all_rows)}")
    if args.strict_dedup:
        kept = [r for i, r in enumerate(all_rows) if i not in contaminated]
        all_rows = kept

    # ── 3. Run infer with EOT markers
    infer = ROOT / args.infer
    assert infer.exists(), f"missing {infer}"
    prompts = [r["prompt"] for r in all_rows]
    proc = subprocess.run(
        [str(infer), "--model", str(ROOT / args.model), "--config", str(ROOT / args.config),
         "--vocab", str(ROOT / args.vocab), "--greedy", "--max_tokens", str(args.max_tokens),
         "--stdin", "--eot_marker", *shlex.split(args.infer_flags)],
        input=("\n".join(prompts) + "\n").encode("utf-8"),
        capture_output=True, timeout=7200,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"[warn] infer rc={proc.returncode}: "
              f"{proc.stderr.decode('utf-8', errors='replace')[-300:]}")

    # split on EOT; strip the trailing empty piece
    pieces = stdout.split(EOT)
    if len(pieces) >= len(all_rows) + 1 or (len(pieces) > 1 and pieces[-1].strip() == ""):
        pieces = pieces[:-1] if pieces and pieces[-1].strip() == "" else pieces
    gens = []
    for i in range(len(all_rows)):
        g = pieces[i].strip() if i < len(pieces) else ""
        gens.append(g)
    n_nonempty = sum(1 for g in gens if g)
    print(f"generations parsed via EOT: {len(gens)} (non-empty {n_nonempty})")

    # ── 4. Score tiers
    res = {d: {"t1_ok": 0, "n": 0, "bank_exact": 0, "bank_n": 0,
               "novel_refs": [], "novel_hyps": []} for d in DIRS}
    for r, hyp in zip(all_rows, gens):
        d = r["lang"]
        tl = gen_lang(d)
        R = res[d]
        # T1: script purity + number copying
        ok_script = has_ja(hyp) if tl == "ja" else (not has_ja(hyp) or tl == "ja")
        if tl == "en" or tl == "id":
            ok_script = not has_ja(hyp)
        nums_ok = numbers_of(r["ref"]) <= numbers_of(hyp) | numbers_of(r["ref"])
        R["n"] += 1
        if ok_script and nums_ok and hyp:
            R["t1_ok"] += 1
        # T2 / T3
        if r["task"] == "translation_bank":
            R["bank_n"] += 1
            if hyp.strip().casefold() == r["ref"].strip().casefold():
                R["bank_exact"] += 1
        else:
            R["novel_refs"].append(_tokenize(r["ref"], tl))
            R["novel_hyps"].append(_tokenize(hyp, tl))

    out = {"model": args.model, "tiers": {}, "chrf_detail": []}
    for d in DIRS:
        R = res[d]
        if not R["n"]:
            continue
        tl = gen_lang(d)
        bleu = corpus_bleu(R["novel_refs"], R["novel_hyps"]) if R["novel_refs"] else 0.0
        chrf = (sum(chrF(" ".join(a), " ".join(b)) for a, b in zip(R["novel_refs"], R["novel_hyps"]))
                / max(len(R["novel_refs"]), 1))
        out["tiers"][d] = {
            "t1_compliance": round(R["t1_ok"] / R["n"], 4),
            "t2_bank_exact": round(R["bank_exact"] / R["bank_n"], 4) if R["bank_n"] else None,
            "bank_n": R["bank_n"],
            "t3_novel_bleu": round(bleu, 4),
            "t3_novel_chrf": round(chrf, 4),
            "novel_n": len(R["novel_refs"]),
        }

    out["contaminated"] = len(contaminated)
    out["total_tested"] = len(all_rows)
    rp = ROOT / args.out
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n{'dir':<6} {'T1':>6} {'T2 bank':>8} {'BLEU':>7} {'chrF':>7}  (n novel/bank)")
    for d, v in out["tiers"].items():
        t2 = f"{v['t2_bank_exact']:.3f}" if v["t2_bank_exact"] is not None else "-"
        print(f"{d:<6} {v['t1_compliance']:>6.3f} {t2:>8} "
              f"{v['t3_novel_bleu']:>7.3f} {v['t3_novel_chrf']:>7.3f}  "
              f"({v['novel_n']}/{v['bank_n']})")
    print(f"\nSaved -> {rp}")


if __name__ == "__main__":
    main()
