#!/usr/bin/env python3
"""Expand the Python coding data with MBPP, and reserve held-out eval sets.

The mixed-SFT builder measured only 6,007 unique EN/JA coding rows across
sft_coding_v3 + code_explain + code_bugfix, which caps any exactly-proportioned
dataset at ~30K rows. Those rows come from 46 hand-written task generators, so
they are also narrow: every problem is "read ints from stdin, print a number".
MBPP adds 974 genuine function-writing problems with real assert tests.

Split discipline (this is why HumanEval is not in the training output):
  * MBPP 601-974  -> train      (the official train split)
  * MBPP 511-600  -> validation
  * MBPP  11-510  -> HELD OUT   (the official test split)
  * HumanEval all -> HELD OUT   (benchmark only)
Both held-out sets are written under eval/, never into train.jsonl.

  python3 scripts/build_coding_expanded.py --download
  python3 scripts/build_coding_expanded.py --mbpp /tmp/mbpp.jsonl --humaneval /tmp/he.jsonl
"""
import argparse
import gzip
import json
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MBPP_URL = ("https://raw.githubusercontent.com/google-research/"
            "google-research/master/mbpp/mbpp.jsonl")
HUMANEVAL_URL = ("https://raw.githubusercontent.com/openai/"
                 "human-eval/master/data/HumanEval.jsonl.gz")

# Official MBPP splits (task_id ranges, inclusive).
MBPP_TRAIN = range(601, 975)
MBPP_VALID = range(511, 601)
MBPP_TEST = range(11, 511)

SYS_EN = "You are a coding assistant. Write short, correct Python programs."
SYS_JA = "あなたはコーディングアシスタントです。短く正確なPythonプログラムを書いてください。"

# Japanese framing for the instruction; the problem text itself stays English
# because MBPP has no translation and machine-translating it would inject noise.
JA_PREFIX = "次の問題をPythonで解いてください。\n"


def fetch(url, dest, gz=False):
    print(f"[GET] {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read()
    if gz:
        raw = gzip.decompress(raw)
    dest.write_bytes(raw)
    return dest


def chat(system, user, assistant):
    return f"<|system|>{system}<|user|>{user}<|assistant|>{assistant}<|end|>"


def mbpp_prompt(rec):
    """Problem text plus one assert, so the expected signature is unambiguous."""
    text = rec["text"].strip()
    tests = rec.get("test_list") or []
    sig = tests[0] if tests else ""
    return f"{text}\nYour code should satisfy: {sig}"


def verify(code, tests, setup="", timeout=10):
    """Run solution+tests exactly the way rl_verifier.py will."""
    program = "\n\n".join(p for p in (setup, code, tests) if p and p.strip())
    try:
        p = subprocess.run([sys.executable, "-c", program],
                           capture_output=True, timeout=timeout)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def build_mbpp(path, rng, verify_solutions=True):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    buckets = {"train": [], "valid": [], "test": []}
    dropped = 0
    for rec in rows:
        tid = rec["task_id"]
        if tid in MBPP_TRAIN:
            split = "train"
        elif tid in MBPP_VALID:
            split = "valid"
        elif tid in MBPP_TEST:
            split = "test"
        else:
            continue  # 1-10 are the few-shot prompt examples
        tests = "\n".join(rec.get("test_list") or [])
        setup = rec.get("test_setup_code", "") or ""
        code = rec["code"].strip()
        # A reference solution that fails its own tests would teach the model
        # wrong answers and make the RL reward unreachable.
        if verify_solutions and not verify(code, tests, setup):
            dropped += 1
            continue
        buckets[split].append({
            "task": f"mbpp_{tid}", "source": "mbpp",
            "prompt": mbpp_prompt(rec), "code": code,
            "tests": tests, "setup": setup,
        })
    print(f"[MBPP] train={len(buckets['train'])} valid={len(buckets['valid'])} "
          f"test={len(buckets['test'])} (dropped {dropped} failing reference solutions)")
    return buckets


def build_humaneval(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for rec in rows:
        out.append({
            "task": rec["task_id"].replace("/", "_"), "source": "humaneval",
            "prompt": rec["prompt"].rstrip(),
            "code": rec["prompt"] + rec["canonical_solution"],
            "tests": rec["test"] + f"\n\ncheck({rec['entry_point']})\n",
            "setup": "",
        })
    print(f"[HumanEval] {len(out)} problems (held out, never trained on)")
    return out


def to_sft(rec, lang):
    system = SYS_JA if lang == "ja" else SYS_EN
    user = (JA_PREFIX + rec["prompt"]) if lang == "ja" else rec["prompt"]
    body = f"```python\n{rec['code']}\n```"
    return {"task": rec["task"], "lang": lang, "source": rec["source"],
            "text": chat(system, user, body)}


def to_rl(rec, lang):
    system = SYS_JA if lang == "ja" else SYS_EN
    user = (JA_PREFIX + rec["prompt"]) if lang == "ja" else rec["prompt"]
    # Empty mock_input/expected_output routes rl_verifier.py to assert mode.
    return {"task": rec["task"], "lang": lang, "source": rec["source"],
            "system": system, "prompt": user,
            "mock_input": "", "expected_output": "",
            "tests": rec["tests"], "setup": rec["setup"]}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):5d} -> {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--mbpp", default="/tmp/mbpp.jsonl")
    ap.add_argument("--humaneval", default="/tmp/he.jsonl")
    ap.add_argument("--sft_out", default="sft_data/sft_coding_mbpp")
    ap.add_argument("--rl_out", default="rl_data/coding_mbpp.jsonl")
    ap.add_argument("--eval_out", default="eval_data")
    ap.add_argument("--langs", default="en,ja")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip executing reference solutions (much faster)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    mbpp_path, he_path = Path(a.mbpp), Path(a.humaneval)
    if a.download:
        fetch(MBPP_URL, mbpp_path)
        fetch(HUMANEVAL_URL, he_path, gz=True)

    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    mbpp = build_mbpp(mbpp_path, rng, verify_solutions=not a.no_verify)
    he = build_humaneval(he_path)

    sft, rl = [], []
    for rec in mbpp["train"]:
        for lang in langs:
            sft.append(to_sft(rec, lang))
            rl.append(to_rl(rec, lang))
    rng.shuffle(sft)
    rng.shuffle(rl)

    sft_dir = ROOT / a.sft_out
    write(sft_dir / "train.jsonl", sft)
    write(sft_dir / "eval.jsonl", [to_sft(r, langs[0]) for r in mbpp["valid"]])
    write(ROOT / a.rl_out, rl)

    ev = ROOT / a.eval_out
    write(ev / "mbpp_heldout.jsonl", [to_rl(r, "en") for r in mbpp["test"]])
    write(ev / "humaneval.jsonl", [to_rl(r, "en") for r in he])

    (sft_dir / "stats.json").write_text(json.dumps({
        "sft_train": len(sft), "sft_eval": len(mbpp["valid"]),
        "rl_train": len(rl),
        "heldout_mbpp": len(mbpp["test"]), "heldout_humaneval": len(he),
        "langs": langs, "seed": a.seed,
        "note": "MBPP test split and all of HumanEval are held out for evaluation",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
