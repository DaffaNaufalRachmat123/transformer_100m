#!/usr/bin/env python3
"""prepare_sft_enja.py

Build a clean EN<->JA translation + fact SFT dataset for transformer_100m_cuda.

Sources (from sibling project transformer_mobile_claude):
  - sft_translation.jsonl  -> keep only en-ja / ja-en pairs
  - sft_ja.jsonl           -> normalize system prompt (drop KO), keep all
  - sft_en.jsonl           -> extract only fact-bearing rows (no code)

Writes (into this project's sft_data/sft_clean/):
  - sft_translation.jsonl  (EN<->JA only, dedup, quality-filtered)
  - sft_ja.jsonl           (normalized system prompt)
  - sft_en_facts.jsonl     (fact subset)

Format must match train_cuda_unified.cu:
  <|system|>...<|user|>...<|assistant|>RESPONSE<|end|>
  loss is masked on assistant tokens.
"""

import json
import os
import re
import sys

SIBLING = "/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_mobile_claude/sft_data/sft"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sft_data", "sft_clean")
OUT_DIR = os.path.abspath(OUT_DIR)

# ---- Fact keywords (EN + JA) ------------------------------------------------
FACT_KEYS = [
    "capital", "tallest", "highest", "population", "largest", "longest",
    "ibukota", "tertinggi", "terbesar", "首都", "人口", "で一番高い", "で一番",
]

# System-prompt variants that indicate code / irrelevant for facts
CODE_SYS = ["expert programmer", "write clean", "you are an expert", "sql", "python", "javascript"]

# ---- Helpers ----------------------------------------------------------------
def split_parts(text):
    """Return (system, user, assistant) text segments, or None if malformed."""
    m_sys = text.find("<|system|>")
    m_usr = text.find("<|user|>")
    m_ast = text.find("<|assistant|>")
    m_end = text.find("<|end|>")
    if not (0 <= m_sys < m_usr < m_ast < m_end):
        return None
    system = text[m_sys + len("<|system|>"):m_usr]
    user = text[m_usr + len("<|user|>"):m_ast]
    assistant = text[m_ast + len("<|assistant|>"):m_end]
    return system, user, assistant


def normalize_sf_ja(system):
    """Drop KO from multilingual system prompt."""
    return system.replace("（ID/EN/JA/KO）", "（ID/EN/JA）").replace("(ID/EN/JA/KO)", "(ID/EN/JA)")


def is_untranslated(src, tgt, lang):
    """Heuristic: translated target must not be near-identical to source."""
    if not tgt.strip():
        return True
    s = re.sub(r"\s+", "", src).lower()
    t = re.sub(r"\s+", "", tgt).lower()
    if not s or not t:
        return True
    # JA<->EN always differ in script; only flag when target is suspiciously short
    overlap = len(set(s) & set(t)) / max(1, len(set(s)))
    return overlap > 0.9 and len(t) < 120


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Translation: keep en-ja / ja-en only ---------------------------
    trans_in = os.path.join(SIBLING, "sft_translation.jsonl")
    trans_out = os.path.join(OUT_DIR, "sft_translation.jsonl")
    keep_lang = {"en-ja", "ja-en"}

    kept, dropped, untrans = 0, 0, 0
    seen = set()
    with open(trans_in, encoding="utf-8") as f, open(trans_out, "w", encoding="utf-8") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            lang = d.get("lang", "")
            if lang not in keep_lang:
                dropped += 1
                continue
            text = d.get("text", "")
            parts = split_parts(text)
            if parts is None:
                dropped += 1
                continue
            system, user, assistant = parts
            if is_untranslated(user, assistant, lang):
                untrans += 1
                dropped += 1
                continue
            h = hash(text)
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            fo.write(json.dumps({"text": text, "lang": lang, "task": "translation"}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[translation] kept={kept} dropped={dropped} (untranslated={untrans}) -> {trans_out}")

    # ---- 2. JA instructions: normalize system prompt ------------------------
    ja_in = os.path.join(SIBLING, "sft_ja.jsonl")
    ja_out = os.path.join(OUT_DIR, "sft_ja.jsonl")
    kept = dropped = 0
    fact_rows = 0
    seen = set()
    with open(ja_in, encoding="utf-8") as f, open(ja_out, "w", encoding="utf-8") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            text = d.get("text", "")
            parts = split_parts(text)
            if parts is None:
                dropped += 1
                continue
            system, user, assistant = parts
            nsys = normalize_sf_ja(system)
            new_text = f"<|system|>{nsys}<|user|>{user}<|assistant|>{assistant}<|end|>"
            h = hash(new_text)
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            if any(k.lower() in user.lower() for k in FACT_KEYS):
                fact_rows += 1
            fo.write(json.dumps({"text": new_text, "lang": "ja", "task": "instruction"}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[ja] kept={kept} dropped={dropped} fact_rows={fact_rows} -> {ja_out}")

    # ---- 3. EN facts: extract fact-bearing non-code rows --------------------
    en_in = os.path.join(SIBLING, "sft_en.jsonl")
    en_out = os.path.join(OUT_DIR, "sft_en_facts.jsonl")
    kept = dropped = 0
    seen = set()
    with open(en_in, encoding="utf-8") as f, open(en_out, "w", encoding="utf-8") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            text = d.get("text", "")
            parts = split_parts(text)
            if parts is None:
                dropped += 1
                continue
            system, user, assistant = parts
            low = (system + " " + user).lower()
            if any(c in low for c in CODE_SYS):
                dropped += 1
                continue
            if not any(k in low for k in FACT_KEYS):
                dropped += 1
                continue
            h = hash(text)
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            fo.write(json.dumps({"text": text, "lang": "en", "task": "instruction"}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[en_facts] kept={kept} dropped={dropped} -> {en_out}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()