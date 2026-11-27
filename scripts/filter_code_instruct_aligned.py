#!/usr/bin/env python3
r"""filter_code_instruct_aligned.py — Filter Magicoder SFT to align with corpus_pretrain_v8 & max_seq=1024 (Option A2).

Filters sft_data/sft_code_instruct/train.jsonl to:
  1. ONLY programming languages present in corpus_pretrain_v8/code:
     Python (~70%), JavaScript (~15%), Java (~6%), PHP (~3%), Go (~3%), Kotlin/C++/Ruby (~3%).
     Drops Swift, C#, Rust, SQL, Bash, Metal/Objective-C, HTML-only, and unfenced prose.
  2. Consistent <|think|> brief plan before the response so 100% of coding SFT rows
     share the same <|think|>...<|/think|> convention as sft_think_code.
  3. CRLF -> LF normalization and zero raw \uXXXX escape hazards.
  4. Exact BPE v9 token length <= 1000 (safe margin under max_seq=1024, Option A2 -> 0% missing <|end|>).
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bpe_reference import BPEReference

FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#-]+)\s*\n(.*?)```", re.DOTALL)
FORBIDDEN_KEYWORDS = re.compile(
    r"\b(SwiftUI|UIKit|Objective-C|MTLDevice|Metal\s+Render|using\s+System;|namespace\s+\w+\s*\{|"
    r"impl\s+\w+\s+for|fn\s+main\s*\(\)|cargo\s+build|SELECT\s+.*\s+FROM\s+|CREATE\s+TABLE)\b"
)

LANG_MAP = {
    "python": "python", "py": "python", "python3": "python",
    "javascript": "javascript", "js": "javascript",
    "java": "java",
    "php": "php",
    "kotlin": "kotlin", "kt": "kotlin",
    "go": "go", "golang": "go",
    "cpp": "cpp", "c++": "cpp", "c": "cpp",
    "ruby": "ruby", "rb": "ruby",
}

# Target language caps matching corpus_pretrain_v8/code distribution (Python-dominant + JS/Java/PHP/Kotlin/Go/C++)
LANG_CAPS = {
    "python": 14000,
    "javascript": 3000,
    "java": 1200,
    "php": 700,
    "kotlin": 500,
    "go": 600,
    "cpp": 600,
    "ruby": 200,
}

PLAN_TEMPLATES = {
    "python": "Plan: implement the Python solution cleanly and handle edge cases.",
    "javascript": "Plan: implement the JavaScript function cleanly and return the expected result.",
    "java": "Plan: define the Java class/method with clear types and logic.",
    "php": "Plan: write a concise PHP implementation for the requested logic.",
    "kotlin": "Plan: implement the idiomatic Kotlin function for the task.",
    "go": "Plan: write a clean Go implementation with proper types.",
    "cpp": "Plan: implement an efficient C/C++ solution.",
    "ruby": "Plan: write a clean, idiomatic Ruby method.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="sft_data/sft_code_instruct/train.jsonl")
    ap.add_argument("--out", default="sft_data/sft_code_instruct_aligned")
    ap.add_argument("--vocab", default="vocab_9")
    ap.add_argument("--max_tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    enc = BPEReference(ROOT / args.vocab)
    inp = ROOT / args.inp
    with inp.open(encoding="utf-8") as fh:
        raw_lines = [l for l in fh if l.strip()]
    rng.shuffle(raw_lines)

    kept_by_lang = Counter()
    stats = Counter()
    rows = []

    for line in raw_lines:
        if sum(kept_by_lang.values()) >= sum(LANG_CAPS.values()):
            break
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        t = rec.get("text", "").replace("\r\n", "\n").replace("\r", "\n")
        m_turn = re.match(r"<\|system\|>(.*?)<\|user\|>(.*?)<\|assistant\|>(.*?)<\|end\|>\s*$", t, re.DOTALL)
        if not m_turn:
            stats["bad_turn"] += 1
            continue
        sys_, user, ans = m_turn.groups()
        if FORBIDDEN_KEYWORDS.search(user + "\n" + ans):
            stats["forbidden_lang_kw"] += 1
            continue

        fences = FENCE_RE.findall(ans)
        if not fences:
            stats["unfenced_prose"] += 1
            continue
        langs_in_ans = {LANG_MAP.get(fl.lower()) for fl, _ in fences}
        if None in langs_in_ans or len(langs_in_ans) != 1:
            stats["other_or_mixed_lang"] += 1
            continue
        plang = next(iter(langs_in_ans))
        if kept_by_lang[plang] >= LANG_CAPS[plang]:
            continue

        # Fast char pre-filter (~3.1 bytes/tok -> 1000 tokens is at most ~3800 chars)
        if len(t) > 3800:
            stats["char_pre_drop"] += 1
            continue

        plan = PLAN_TEMPLATES[plang]
        new_text = f"<|system|>{sys_.strip()}<|user|>{user.strip()}<|assistant|><|think|>{plan}<|/think|>{ans.strip()}<|end|>"

        # Exact BPE v9 token check (with BOS=True)
        tok_ids = enc.encode(new_text, add_bos=True)
        if len(tok_ids) > args.max_tokens:
            stats["over_max_tokens"] += 1
            continue

        out_rec = {
            "task": rec.get("task", "code_instruct"),
            "lang": "en",
            "code_lang": plang,
            "text": new_text,
            "has_think": True,
            "source": f"code_instruct_{plang}",
            "family": "coding",
        }
        # Ensure JSON serialization carries zero \uXXXX escapes
        dumped = json.dumps(out_rec, ensure_ascii=False)
        if "\\u" in dumped:
            stats["u_escape_drop"] += 1
            continue

        rows.append(out_rec)
        kept_by_lang[plang] += 1
        stats["kept"] += 1

    rng.shuffle(rows)
    n_eval = min(300, len(rows) // 40)
    ev, tr = rows[:n_eval], rows[n_eval:]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (out_dir / "stats.json").write_text(json.dumps({
        "train": len(tr),
        "eval": len(ev),
        "max_tokens": args.max_tokens,
        "by_code_lang": dict(kept_by_lang.most_common()),
        "filter_stats": dict(stats.most_common()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[CODE-A2] train={len(tr):,} eval={len(ev):,} by_lang={dict(kept_by_lang.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
