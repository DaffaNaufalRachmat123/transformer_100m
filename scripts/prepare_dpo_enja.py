#!/usr/bin/env python3
"""prepare_dpo_enja.py

Filter DPO data to keep ONLY EN<->JA translation_quality pairs
(and drop ID/KO/other-language DPO that is irrelevant to the EN<->JA goal).

Reads  : sft_data/dpo/  (all *.jsonl)
Writes : sft_data/dpo_enja/dpo_translation_enja.jsonl
Entries keep the same schema used by load_dpo_data() in train_cuda_unified.cu:
  {system, prompt, chosen, rejected, lang, task}
"""

import json
import os
import re

DPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sft_data", "dpo")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sft_data", "dpo_enja")
OUT_FILE = os.path.join(OUT_DIR, "dpo_translation_enja.jsonl")

KEEP_LANG = {"en-ja", "ja-en"}

# Heuristic: reject entries where chosen/rejected are empty, identical, or
# one is a substring of the other. The last case occurs when "rejected" is just
# a truncated "chosen" (found in dpo_ja_preference / dpo_ja_generated) — such
# pairs give the model a degenerate/noisy preference signal.
def is_bad(system, prompt, chosen, rejected):
    if not (chosen and rejected and prompt):
        return True
    c = chosen.strip()
    r = rejected.strip()
    if c == r:
        return True
    # whitespace-normalized
    sc = re.sub(r"\s+", "", c).lower()
    sr = re.sub(r"\s+", "", r).lower()
    if not sc or not sr:
        return True
    # identical scripts (e.g. both stay English) -> bad translation candidate
    if sc == sr:
        return True
    # one is substring of the other (truncated "chosen") -> degenerate pair
    if len(sc) >= 8 and (sc in sr or sr in sc):
        return True
    # repeated phrase inside rejected (e.g. "The meeting starts at three The
    # meeting starts at three") -> noisy/garbled
    if len(sr) >= 16 and max_len_repeat(sr) >= 2:
        return True
    return False


def max_len_repeat(s):
    """Longest sub-phrase that repeats contiguously, in fractional terms."""
    n = len(s)
    best = 0
    for half in range(1, n // 2 + 1):
        if s[:half] and s[:half] * (n // half) == s[: (n // half) * half]:
            best = max(best, n // half)
    return best


def infer_lang(d):
    """Return en-ja / ja-en / None from explicit lang or prompt direction.

    Direction = the TARGET language (the language the text is translated INTO).
      - "to Japanese" / "日本語に翻訳"  -> target is JA  -> en-ja
      - "to English"  / "英語に翻訳"   -> target is EN  -> ja-en
    """
    lang = d.get("lang")
    if lang in KEEP_LANG:
        return lang
    prompt = d.get("prompt", "")
    low = prompt.lower()
    # target = Japanese
    if "to japanese" in low or "japanese translation" in low \
       or "日本語に翻訳" in prompt or "日本語へ翻訳" in prompt:
        return "en-ja"
    # target = English
    if "to english" in low or "english translation" in low \
       or "英語に翻訳" in prompt or "英語へ翻訳" in prompt:
        return "ja-en"
    return None


# Only these files contain actual EN<->JA translation preference pairs.
# The general JA/ID preference files (dpo_ja_*, dpo_id_*) are NOT translation
# data — reading them contaminated the EN<->JA set with conversational pairs.
TRANSLATION_FILES = {
    "dpo_ja_translation_quality.jsonl",
    "dpo_translation_enja.jsonl",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kept = dropped = 0
    seen = set()
    with open(OUT_FILE, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(DPO_DIR)):
            if not fn.endswith(".jsonl"):
                continue
            if fn not in TRANSLATION_FILES:
                continue
            path = os.path.join(DPO_DIR, fn)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        dropped += 1
                        continue
                    lang = infer_lang(d)
                    if lang is None:
                        dropped += 1
                        continue
                    system = d.get("system", "")
                    prompt = d.get("prompt", "")
                    chosen = d.get("chosen", "")
                    rejected = d.get("rejected", "")
                    if is_bad(system, prompt, chosen, rejected):
                        dropped += 1
                        continue
                    h = hash((prompt, chosen, rejected))
                    if h in seen:
                        dropped += 1
                        continue
                    seen.add(h)
                    fo.write(json.dumps({
                        "system": system,
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "lang": lang,
                        "task": "translation_quality",
                    }, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"[dpo_enja] kept={kept} dropped={dropped} -> {OUT_FILE}")


if __name__ == "__main__":
    main()