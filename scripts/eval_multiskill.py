#!/usr/bin/env python3
"""eval_multiskill.py — does ONE checkpoint hold every skill at once?

The staged SFT pipeline produces specialists that destroy each other: the facts
checkpoint scores 0.003 chrF on translation, the translation checkpoint answers
0/10 EN facts. Per-skill evals hide this because each is run against the
checkpoint that happens to be good at it. This harness scores facts,
translation, math and coding on a single model and fails if any family
collapses, which is the actual gate for the mixed-SFT experiment.

  python3 scripts/eval_multiskill.py \
      --model checkpoints_mixed/ckpt_sft_final.safetensors \
      --config config_204m.json
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_demo_eval import chrf, has_ja, has_latin, norm, run_infer, value_hit  # noqa: E402

# Below these a family is considered collapsed, i.e. overwritten by another skill.
THRESHOLDS = {
    "facts": ("value_hit", 0.60),
    "translation": ("chrf", 0.45),
    "math": ("answer_hit", 0.30),
    "coding": ("chrf", 0.30),
}


def split_turn(text):
    """Return (prompt_including_assistant_tag, gold_answer)."""
    tag = "<|assistant|>"
    i = text.find(tag)
    if i < 0:
        return None, None
    gold = text[i + len(tag):]
    for stop in ("<|end|>", "<|endoftext|>"):
        j = gold.find(stop)
        if j >= 0:
            gold = gold[:j]
    return text[:i + len(tag)], gold.strip()


def score(rec, pred, gold):
    """Family-specific metrics plus the language checks that catch script drift."""
    fam = rec.get("family", "?")
    lang = str(rec.get("lang", "")).lower()
    out = {}

    if fam == "facts":
        want_ja = lang == "ja"
        out["lang_ok"] = has_ja(pred) if want_ja else (has_latin(pred) and not has_ja(pred))
        val = rec.get("value_ja") if want_ja else rec.get("value_en")
        # Fall back to the gold turn when the row carries no structured value.
        out["value_hit"] = value_hit(val or gold, pred, "ja" if want_ja else "en")

    elif fam == "translation":
        tgt = lang.partition("-")[2] or lang
        out["lang_ok"] = has_ja(pred) if tgt == "ja" else (has_latin(pred) and not has_ja(pred))
        out["chrf"] = chrf(gold, pred)

    elif fam == "math":
        exp = str(rec.get("expected_output", "")).strip()
        # Compare on digits only: the model may format 30500 as "30,500".
        digits = re.sub(r"[^\d.-]", "", exp)
        out["answer_hit"] = bool(digits) and digits in re.sub(r"[^\d.-]", "", pred)
        out["chrf"] = chrf(gold, pred)

    else:  # coding — textual similarity only; real pass/fail needs rl_verifier.py
        out["chrf"] = chrf(gold, pred)
        out["has_code"] = bool(re.search(r"(def |print\(|input\(|=)", pred))

    out["empty"] = not pred.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config_204m.json")
    ap.add_argument("--vocab", default="vocab_6")
    ap.add_argument("--infer_bin", default="build_infer/infer")
    ap.add_argument("--mix_dir", default="sft_data/sft_mixed_enja")
    ap.add_argument("--per_family", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    by_fam = defaultdict(list)
    with (Path(a.mix_dir) / "eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt, gold = split_turn(rec.get("text", ""))
            if prompt and gold:
                rec["_prompt"], rec["_gold"] = prompt, gold
                by_fam[rec.get("family", "?")].append(rec)

    picked = []
    for fam in sorted(by_fam):
        rows = by_fam[fam]
        rng.shuffle(rows)
        picked.extend(rows[: a.per_family])
    if not picked:
        sys.exit(f"[ERR] no usable eval rows in {a.mix_dir}/eval.jsonl")

    # Single-line prompts only: infer's --stdin splits input on newlines.
    for r in picked:
        r["_prompt"] = r["_prompt"].replace("\n", " ")

    print(f"[EVAL] {len(picked)} prompts across {len(by_fam)} families "
          f"-> {a.model}")
    preds = run_infer(Path(a.infer_bin), Path(a.model), Path(a.config),
                      Path(a.vocab), [r["_prompt"] for r in picked],
                      a.max_tokens)

    agg = defaultdict(lambda: defaultdict(list))
    samples = defaultdict(list)
    for rec, pred in zip(picked, preds):
        fam = rec.get("family", "?")
        s = score(rec, pred, rec["_gold"])
        for k, v in s.items():
            agg[fam][k].append(float(v))
        if len(samples[fam]) < 3:
            samples[fam].append({"lang": rec.get("lang"),
                                 "gold": rec["_gold"][:120],
                                 "pred": pred[:120]})

    print(f"\n{'family':<13}{'n':>5}  metrics")
    print("-" * 72)
    report, verdict = {}, {}
    for fam in sorted(agg):
        m = {k: sum(v) / len(v) for k, v in agg[fam].items()}
        n = len(agg[fam]["empty"])
        report[fam] = {"n": n, **{k: round(v, 4) for k, v in m.items()}}
        cells = "  ".join(f"{k}={v:.3f}" for k, v in sorted(m.items()))
        print(f"{fam:<13}{n:>5}  {cells}")

        key, thr = THRESHOLDS.get(fam, (None, 0))
        if key and key in m:
            ok = m[key] >= thr
            verdict[fam] = {"metric": key, "value": round(m[key], 4),
                            "threshold": thr, "pass": ok}

    print(f"\n{'family':<13}{'metric':<12}{'value':>8}{'thr':>8}   result")
    print("-" * 72)
    for fam, v in sorted(verdict.items()):
        print(f"{fam:<13}{v['metric']:<12}{v['value']:>8.3f}{v['threshold']:>8.2f}"
              f"   {'PASS' if v['pass'] else 'FAIL'}")

    n_pass = sum(v["pass"] for v in verdict.values())
    all_pass = n_pass == len(verdict)
    print(f"\n[VERDICT] {n_pass}/{len(verdict)} families pass on one checkpoint"
          f" — {'NO catastrophic forgetting' if all_pass else 'STILL FORGETTING'}")

    for fam in sorted(samples):
        print(f"\n--- {fam} samples ---")
        for s in samples[fam]:
            print(f"  [{s['lang']}] gold: {s['gold']}")
            print(f"        pred: {s['pred']}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"model": a.model, "per_family": report, "verdict": verdict,
             "all_pass": all_pass, "samples": dict(samples)},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[EVAL] wrote {a.out}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
