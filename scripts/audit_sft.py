#!/usr/bin/env python3
"""Audit SFT rows for language mismatch, duplication and malformed turns.

Answering an English question in Japanese (or the reverse) teaches the model to
ignore the prompt language, and the multi-skill baseline showed exactly that
failure mode surviving because language purity was only ever checked on the
model's output, never on the training data itself.

Checks per row:
  lang_mismatch  answer script disagrees with the row's declared language
  q_lang_wrong   question script disagrees with the declared language
  dup_exact      identical rendered text seen earlier
  dup_qa         same question with a different answer (contradictory target)
  empty_answer   assistant turn missing or blank
  copy_answer    answer merely repeats the question
  no_target_lang answer for a translation row is not in the target script

  python3 scripts/audit_sft.py sft_data/sft_mixed_enja
  python3 scripts/audit_sft.py sft_data/*/ --quiet
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
CODE_HINT = re.compile(r"(^|\n)\s*(def |class |import |from |print\(|return |"
                       r"for |while |if |#|```)")


def has_ja(s):
    return bool(KANA.search(s) or HAN.search(s))


def has_kana(s):
    """Kana is the reliable Japanese signal; Han alone also occurs in names."""
    return bool(KANA.search(s))


def has_latin(s):
    return bool(LATIN_WORD.search(s))


def split_turn(text):
    tag = "<|assistant|>"
    i = text.find(tag)
    if i < 0:
        return None, None
    q = text[:i]
    a = text[i + len(tag):]
    for stop in ("<|end|>", "<|endoftext|>"):
        j = a.find(stop)
        if j >= 0:
            a = a[:j]
    u = q.find("<|user|>")
    if u >= 0:
        q = q[u + len("<|user|>"):]
    return q.strip(), a.strip()


def audit_row(rec, seen_text, seen_qa):
    """Return a list of problem codes for one row."""
    problems = []
    text = rec.get("text", "")
    q, a = split_turn(text)
    if q is None or not a:
        return ["empty_answer"]

    h = hashlib.sha1(text.encode()).digest()
    if h in seen_text:
        problems.append("dup_exact")
    seen_text.add(h)

    qh = hashlib.sha1(q.encode()).digest()
    ah = hashlib.sha1(a.encode()).digest()
    if qh in seen_qa and seen_qa[qh] != ah:
        problems.append("dup_qa")
    seen_qa.setdefault(qh, ah)

    if a.strip() == q.strip():
        problems.append("copy_answer")

    lang = str(rec.get("lang", "")).lower()
    is_code = bool(CODE_HINT.search(a))

    if "-" in lang:
        # Translation: the answer must be in the TARGET script.
        tgt = lang.partition("-")[2]
        if tgt == "ja" and not has_ja(a):
            problems.append("no_target_lang")
        elif tgt == "en" and (not has_latin(a) or has_kana(a)):
            problems.append("no_target_lang")
    else:
        # Monolingual: code answers are legitimately Latin in any language.
        if lang == "ja":
            if not has_ja(q):
                problems.append("q_lang_wrong")
            if not is_code and not has_ja(a):
                problems.append("lang_mismatch")
        elif lang == "en":
            if has_kana(q):
                problems.append("q_lang_wrong")
            if not is_code and has_kana(a):
                problems.append("lang_mismatch")
    return problems


def audit_file(path, limit=0, show=0):
    seen_text, seen_qa = set(), {}
    counts = Counter()
    samples = defaultdict(list)
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            n += 1
            if limit and n > limit:
                break
            for p in audit_row(rec, seen_text, seen_qa):
                counts[p] += 1
                if len(samples[p]) < show:
                    q, a = split_turn(rec.get("text", ""))
                    samples[p].append({"lang": rec.get("lang"),
                                       "q": (q or "")[:90], "a": (a or "")[:90]})
    return n, counts, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--fail_pct", type=float, default=1.0,
                    help="exit non-zero if problem rate exceeds this percent")
    a = ap.parse_args()

    worst = 0.0
    for d in a.dirs:
        p = Path(d)
        f = p / "train.jsonl" if p.is_dir() else p
        if not f.exists():
            continue
        n, counts, samples = audit_file(f, a.limit, a.show)
        bad = sum(counts.values())
        pct = 100.0 * bad / n if n else 0.0
        worst = max(worst, pct)
        flag = "  <-- FAIL" if pct > a.fail_pct else ""
        print(f"\n{f}  rows={n}  problems={bad} ({pct:.2f}%){flag}")
        for k, v in counts.most_common():
            print(f"    {k:16s} {v:7d}  ({100.0*v/n:.2f}%)")
            for s in samples.get(k, []):
                print(f"        [{s['lang']}] Q: {s['q']}")
                print(f"                 A: {s['a']}")
    return 1 if worst > a.fail_pct else 0


if __name__ == "__main__":
    sys.exit(main())
