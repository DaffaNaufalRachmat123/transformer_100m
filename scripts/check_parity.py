#!/usr/bin/env python3
"""check_parity.py — Numeric parity gate: original C++ engine vs GGUF/llama.cpp.

Why only the FIRST generated token: `infer --greedy` is not pure argmax -- it
still applies repetition penalty (default 1.3) and a hard-block on tokens seen
>= 3 times. Both of those depend on generation history, which is empty at step
0. So the first generated token IS pure argmax in both engines, making it the
only clean apples-to-apples comparison.

A mismatch here almost always means a wrong tensor transpose or the wrong RoPE
convention in the GGUF conversion.

Usage:
  python3 scripts/check_parity.py \
      --safetensors translation_resources/facts/ckpt_sft_final.safetensors \
      --gguf gguf/trilingual-204m-facts-f16.gguf --n 200
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
EOT = "<<<EOT>>>"

SYS_FACT_EN = "You are a helpful, accurate multilingual assistant."
SYS_FACT_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
SYS_TR_JA = "You are a helpful translation engine. Output only the translation."
SYS_TR_EN = "You are a multilingual translation assistant."


def collect_prompts(n, seed):
    """Build a mixed prompt set from the held-out eval splits."""
    rng = random.Random(seed)
    out = []

    facts = ROOT / "sft_data/sft_facts_v11/eval.jsonl"
    trans = ROOT / "sft_data/sft_translation_tri/eval.jsonl"

    def rows(path, limit=6000):
        acc = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                acc.append(json.loads(line))
        return acc

    for r in rows(facts):
        t = r.get("text", "")
        if "<|user|>" not in t or "<|assistant|>" not in t:
            continue
        q = t.split("<|user|>")[1].split("<|assistant|>")[0]
        if "\n" in q:
            continue
        sysmsg = SYS_FACT_JA if r.get("lang") == "ja" else SYS_FACT_EN
        out.append(f"<|system|>{sysmsg}<|user|>{q}<|assistant|>")

    for r in rows(trans):
        t = r.get("text", "")
        if "<|user|>" not in t or "<|assistant|>" not in t:
            continue
        user = t.split("<|user|>")[1].split("<|assistant|>")[0]
        if "\n" in user:
            continue
        sysmsg = SYS_TR_JA if str(r.get("lang", "")).endswith("ja") else SYS_TR_EN
        out.append(f"<|system|>{sysmsg}<|user|>{user}<|assistant|>")

    rng.shuffle(out)
    return out[:n]


def run(cmd, payload):
    proc = subprocess.run(cmd, input=payload.encode("utf-8"),
                          capture_output=True, timeout=7200)
    if proc.returncode != 0:
        sys.stderr.write(f"[warn] rc={proc.returncode} "
                         f"{proc.stderr.decode('utf-8','replace')[-300:]}\n")
    pieces = proc.stdout.decode("utf-8", "replace").split(EOT)
    return [p.strip("\n") for p in pieces]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default="translation_resources/facts/ckpt_sft_final.safetensors")
    ap.add_argument("--gguf", default="gguf/trilingual-204m-facts-f16.gguf")
    ap.add_argument("--config", default="config_204m.json")
    ap.add_argument("--vocab", default="vocab_6")
    ap.add_argument("--infer", default="build_infer/infer")
    ap.add_argument("--runner", default="build_infer/llama_runner")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    from bpe_reference import BPEReference
    enc = BPEReference(ROOT / args.vocab)

    prompts = collect_prompts(args.n, args.seed)
    print(f"parity set: {len(prompts)} prompts, first generated token only\n")

    ref = run([str(ROOT / args.infer),
               "--model", str(ROOT / args.safetensors),
               "--config", str(ROOT / args.config),
               "--vocab", str(ROOT / args.vocab),
               "--greedy", "--max_tokens", "1", "--stdin", "--eot_marker"],
              "\n".join(prompts) + "\n")

    ids = [" ".join(str(i) for i in enc.encode(p)) for p in prompts]
    got = run([str(ROOT / args.runner),
               "--model", str(ROOT / args.gguf),
               "--n-predict", "1", "--ctx", "1024", "--threads", "4", "-ngl", "0",
               "--eot-marker"],
              "\n".join(ids) + "\n")

    n = min(len(prompts), len(ref), len(got))
    match = 0
    diffs = []
    for i in range(n):
        a, b = ref[i].strip(), got[i].strip()
        if a == b:
            match += 1
        elif len(diffs) < 10:
            diffs.append((prompts[i], a, b))

    rate = match / n if n else 0.0
    print(f"compared      : {n}")
    print(f"first-token OK: {match}  ({100*rate:.1f}%)")
    if diffs:
        print("\nfirst mismatches:")
        for p, a, b in diffs:
            print(f"  prompt : {p[:70]}")
            print(f"    infer: {a!r}")
            print(f"    gguf : {b!r}")

    ok = rate >= args.threshold
    print(f"\nGATE {'PASS' if ok else 'FAIL'} (threshold {100*args.threshold:.0f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
