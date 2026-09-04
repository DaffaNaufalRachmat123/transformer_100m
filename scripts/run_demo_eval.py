#!/usr/bin/env python3
"""run_demo_eval.py — Fixed-size demo evaluation for the 204M trilingual model.

Runs, against a held-out eval split:
  * 10 EN fact questions
  * 10 JA fact questions
  * 20 EN->JA translations
  * 20 JA->EN translations

Scores language purity (did it answer in the right script?), fact-value recall,
and chrF for translation, then prints a per-item table plus a summary.

Usage:
  python3 scripts/run_demo_eval.py \
      --model translation_resources/facts/ckpt_sft_final.safetensors \
      --config config_204m.json
"""

import argparse
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EOT = "<<<EOT>>>"

SYS_FACT_EN = "You are a helpful, accurate multilingual assistant."
SYS_FACT_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
SYS_TR_JA = "You are a helpful translation engine. Output only the translation."
SYS_TR_EN = "You are a multilingual translation assistant."


def has_ja(s: str) -> bool:
    return any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in s)


def has_latin(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", s))


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u3040-\u30ff\u4e00-\u9fff]", " ", s)).strip()


def chrf(ref: str, hyp: str, n: int = 6) -> float:
    """Character n-gram F-score (chrF), order n."""
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    if not r or not h:
        return 0.0
    tot = 0.0
    orders = 0
    for k in range(1, n + 1):
        rg = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        hg = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        if not rg or not hg:
            continue
        m = sum(min(c, rg.get(g, 0)) for g, c in hg.items())
        p = m / sum(hg.values())
        rc = m / sum(rg.values())
        tot += 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)
        orders += 1
    return tot / orders if orders else 0.0


def value_hit(value: str, pred: str, lang: str) -> bool:
    """True if the gold fact value appears in the prediction."""
    if not value:
        return False
    v, p = norm(value), norm(pred)
    if not v:
        return False
    if v in p:
        return True
    toks = [t for t in v.split() if len(t) > 2] if lang != "ja" else list(v)
    if not toks:
        return False
    return sum(1 for t in toks if t in p) / len(toks) >= 0.7


def build_prompt(system: str, user: str) -> str:
    return f"<|system|>{system}<|user|>{user}<|assistant|>"


def run_infer(binary: Path, model: Path, config: Path, vocab: Path,
              prompts, max_tokens: int):
    """Feed all prompts through one infer process; return list of generations."""
    proc = subprocess.run(
        [str(binary), "--model", str(model), "--config", str(config),
         "--vocab", str(vocab), "--greedy", "--max_tokens", str(max_tokens),
         "--stdin", "--eot_marker"],
        input=("\n".join(prompts) + "\n").encode("utf-8"),
        capture_output=True, timeout=7200,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"[warn] infer rc={proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace')[-400:]}\n")
    pieces = proc.stdout.decode("utf-8", "replace").split(EOT)
    outs = [p.strip() for p in pieces][:len(prompts)]
    while len(outs) < len(prompts):
        outs.append("")
    return outs


def run_gguf(runner: Path, model: Path, vocab: Path, prompts,
             max_tokens: int, n_ctx: int, threads: int):
    """Same prompts through llama.cpp, tokenized by our own BPE port.

    The model's BPE is not GPT-2 compatible, so llama.cpp cannot encode these
    prompts itself; we send token ids instead.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from bpe_reference import BPEReference

    enc = BPEReference(vocab)
    lines = [" ".join(str(i) for i in enc.encode(p)) for p in prompts]
    proc = subprocess.run(
        [str(runner), "--model", str(model), "--n-predict", str(max_tokens),
         "--ctx", str(n_ctx), "--threads", str(threads), "--eot-marker"],
        input=("\n".join(lines) + "\n").encode("utf-8"),
        capture_output=True, timeout=7200,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"[warn] runner rc={proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace')[-400:]}\n")
    pieces = proc.stdout.decode("utf-8", "replace").split(EOT)
    outs = [p.strip() for p in pieces][:len(prompts)]
    while len(outs) < len(prompts):
        outs.append("")
    return outs


def pick_facts(path: Path, lang: str, n: int, rng):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("lang") != lang or r.get("task") != "fact_qa":
            continue
        t = r["text"]
        if "<|user|>" not in t or "<|assistant|>" not in t:
            continue
        q = t.split("<|user|>")[1].split("<|assistant|>")[0]
        ref = t.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        if "\n" in q:
            continue
        rows.append({
            "q": q, "ref": ref, "lang": lang,
            "value": r.get(f"value_{lang}", ""),
            "category": r.get("category", ""),
            "subject": r.get(f"subject_{lang}", ""),
        })
    rng.shuffle(rows)
    # one item per category where possible, for coverage
    seen, out = set(), []
    for r in rows:
        if r["category"] in seen:
            continue
        seen.add(r["category"])
        out.append(r)
        if len(out) == n:
            return out
    for r in rows:
        if len(out) == n:
            break
        if r not in out:
            out.append(r)
    return out[:n]


def pick_trans(path: Path, direction: str, n: int, rng):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("lang") != direction:
            continue
        t = r["text"]
        if "<|user|>" not in t or "<|assistant|>" not in t:
            continue
        user = t.split("<|user|>")[1].split("<|assistant|>")[0]
        ref = t.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        if "\n" in user or not ref:
            continue
        src = user.split(":", 1)[1].strip() if ":" in user else user
        if not (5 <= len(src) <= 160):
            continue
        rows.append({"user": user, "src": src, "ref": ref, "dir": direction})
    rng.shuffle(rows)
    return rows[:n]


def clip(s: str, n: int) -> str:
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="infer", choices=["infer", "gguf"],
                    help="'infer' = original C++ engine, 'gguf' = llama.cpp")
    ap.add_argument("--model", default="translation_resources/facts/ckpt_sft_final.safetensors")
    ap.add_argument("--config", default="config_204m.json")
    ap.add_argument("--vocab", default="vocab_6")
    ap.add_argument("--infer", default="build_infer/infer")
    ap.add_argument("--runner", default="build_infer/llama_runner")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--facts_eval", default="sft_data/sft_facts_v11/eval.jsonl")
    ap.add_argument("--trans_eval", default="sft_data/sft_translation_tri/eval.jsonl")
    ap.add_argument("--n_facts", type=int, default=10)
    ap.add_argument("--n_trans", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_tokens", type=int, default=48)
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    binary = ROOT / (args.runner if args.backend == "gguf" else args.infer)
    model = ROOT / args.model
    config = ROOT / args.config
    vocab = ROOT / args.vocab
    needed = [binary, model, vocab] + ([] if args.backend == "gguf" else [config])
    for p in needed:
        if not p.exists():
            sys.exit(f"missing: {p}")

    print("=" * 78)
    print(f"BACKEND: {args.backend}")
    print(f"MODEL : {args.model}")
    if args.backend == "infer":
        print(f"CONFIG: {args.config}")
    print("=" * 78)

    facts_en = pick_facts(ROOT / args.facts_eval, "en", args.n_facts, rng)
    facts_ja = pick_facts(ROOT / args.facts_eval, "ja", args.n_facts, rng)
    tr_enja = pick_trans(ROOT / args.trans_eval, "en-ja", args.n_trans, rng)
    tr_jaen = pick_trans(ROOT / args.trans_eval, "ja-en", args.n_trans, rng)

    prompts = (
        [build_prompt(SYS_FACT_EN, r["q"]) for r in facts_en]
        + [build_prompt(SYS_FACT_JA, r["q"]) for r in facts_ja]
        + [build_prompt(SYS_TR_JA, r["user"]) for r in tr_enja]
        + [build_prompt(SYS_TR_EN, r["user"]) for r in tr_jaen]
    )
    print(f"running {len(prompts)} prompts through {args.backend} …\n")
    if args.backend == "gguf":
        outs = run_gguf(binary, model, vocab, prompts, args.max_tokens,
                        args.ctx, args.threads)
    else:
        outs = run_infer(binary, model, config, vocab, prompts, args.max_tokens)

    i = 0
    for r in facts_en + facts_ja + tr_enja + tr_jaen:
        r["pred"] = outs[i]
        i += 1

    report = {"model": args.model, "backend": args.backend, "sections": {}}

    # ── facts ────────────────────────────────────────────────────────────
    for tag, rows, lang in (("FACTS EN", facts_en, "en"), ("FACTS JA", facts_ja, "ja")):
        print("─" * 78)
        print(f"{tag}  ({len(rows)} items)")
        print("─" * 78)
        ok_lang = ok_val = 0
        for k, r in enumerate(rows, 1):
            pred = r["pred"]
            lang_ok = has_ja(pred) if lang == "ja" else (has_latin(pred) and not has_ja(pred))
            val_ok = value_hit(r["value"], pred, lang)
            ok_lang += lang_ok
            ok_val += val_ok
            print(f"{k:2d}. [{'L✓' if lang_ok else 'L✗'}|{'V✓' if val_ok else 'V✗'}] {clip(r['q'], 68)}")
            print(f"     gold: {clip(r['ref'], 66)}")
            print(f"     pred: {clip(pred, 66)}")
        n = max(len(rows), 1)
        print(f"\n  language purity : {ok_lang}/{len(rows)}  ({100*ok_lang/n:.0f}%)")
        print(f"  fact value hit  : {ok_val}/{len(rows)}  ({100*ok_val/n:.0f}%)\n")
        report["sections"][tag] = {
            "n": len(rows), "lang_ok": ok_lang, "value_ok": ok_val,
            "items": [{"q": r["q"], "gold": r["ref"], "pred": r["pred"]} for r in rows],
        }

    # ── translation ──────────────────────────────────────────────────────
    for tag, rows, tgt in (("TRANSLATION EN→JA", tr_enja, "ja"), ("TRANSLATION JA→EN", tr_jaen, "en")):
        print("─" * 78)
        print(f"{tag}  ({len(rows)} items)")
        print("─" * 78)
        ok_lang = 0
        scores = []
        for k, r in enumerate(rows, 1):
            pred = r["pred"]
            lang_ok = has_ja(pred) if tgt == "ja" else (has_latin(pred) and not has_ja(pred))
            sc = chrf(r["ref"], pred)
            ok_lang += lang_ok
            scores.append(sc)
            print(f"{k:2d}. [{'L✓' if lang_ok else 'L✗'}] chrF={sc:.3f}  src: {clip(r['src'], 52)}")
            print(f"     gold: {clip(r['ref'], 66)}")
            print(f"     pred: {clip(pred, 66)}")
        n = max(len(rows), 1)
        avg = sum(scores) / n if scores else 0.0
        print(f"\n  language purity : {ok_lang}/{len(rows)}  ({100*ok_lang/n:.0f}%)")
        print(f"  avg chrF        : {avg:.4f}\n")
        report["sections"][tag] = {
            "n": len(rows), "lang_ok": ok_lang, "avg_chrf": avg,
            "items": [{"src": r["src"], "gold": r["ref"], "pred": r["pred"]} for r in rows],
        }

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for tag, s in report["sections"].items():
        extra = (f"value {s['value_ok']}/{s['n']}" if "value_ok" in s
                 else f"chrF {s['avg_chrf']:.4f}")
        print(f"  {tag:<20} lang {s['lang_ok']}/{s['n']:<4} {extra}")

    if args.json_out:
        out = ROOT / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
