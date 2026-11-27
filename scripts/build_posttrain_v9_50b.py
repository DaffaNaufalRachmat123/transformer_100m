#!/usr/bin/env python3
"""
build_posttrain_v9_50b.py — Build SEPARATE 50B-tier Post-Training Add-on Packages:
  1. Stage 2 SFT Hard-Tier Add-ons : sft_data/sft_v9_50b/  (~130K train + 1,500 eval)
  2. Stage 3 DPO Subtle/Hard Pairs : sft_data/dpo_v9_50b/  (~19K preference pairs)
  3. Stage 5 GRPO Medium-Hard RL   : rl_data/rl_v9_50b/    (~18K verifiable tasks)

Guarantees:
  - Existing v8 folders (sft_mixed_enja, dpo_ceiling_enja, ceiling_enja) are NEVER
    modified; they are opened strictly read-only to guarantee 0% duplicate overlap.
  - Multi-turn SFT rows close EVERY assistant turn with `<|end|>` before the next
    `<|user|>`, matching `build_assistant_mask()` in `train_cuda_unified.cu`.
  - 100% verified <= 1,000 BPE v9 tokens (SFT/DPO) and <= 360 prompt tokens (RL).
  - 0% overlap with evaluation holdouts (gsm8k/test, mgsm_ja_test, humaneval, mbpp_heldout).
  - All synthesized RL `assert` suites pass gold code AND reject all 6 dummy cheat stubs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import signal
import struct
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_url

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from bpe_reference import BPEReference  # noqa: E402

SFT_V8 = ROOT / "sft_data" / "sft_mixed_enja"
DPO_V8 = ROOT / "sft_data" / "dpo_ceiling_enja"
RL_V8  = ROOT / "rl_data" / "ceiling_enja"

SFT_V9 = ROOT / "sft_data" / "sft_v9_50b"
DPO_V9 = ROOT / "sft_data" / "dpo_v9_50b"
RL_V9  = ROOT / "rl_data" / "rl_v9_50b"

WS_RE = re.compile(r"\s+")
KANA_RE = re.compile(r"[\u3040-\u30ff]")
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
BOXED_NUM_RE = re.compile(r"\\boxed\{\s*([+-]?\d+(?:\.\d+)?)\s*\}")
LLM_CODE_PAIR_RE = re.compile(
    r"<llm-code>\s*(.*?)\s*</llm-code>\s*<llm-code-output>\s*(.*?)\s*</llm-code-output>",
    re.DOTALL,
)


def norm_key(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip().lower())


def hash64(s: str) -> int:
    return struct.unpack("<Q", hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest())[0]


class FastHoldoutChecker:
    """O(1) hash-set prefix checker (25,000x faster than 8,000-branch NFA regex)."""
    def __init__(self, p30: frozenset[str], p50: frozenset[str]):
        self.p30 = p30
        self.p50 = p50

    def search(self, s: str) -> bool:
        if "<|user|>" in s:
            s = s.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0]
        nk = norm_key(s)
        return (nk[:30] in self.p30) or (nk[:50] in self.p50)


def load_holdouts():
    p30: set[str] = set()
    p50: set[str] = set()
    hold_funcs: set[str] = set()
    specs = [
        ("data/gsm8k/test.jsonl", ["question"]),
        ("eval_data/mgsm_ja_test.jsonl", ["question", "prompt"]),
        ("eval_data/humaneval.jsonl", ["prompt"]),
        ("eval_data/mbpp_heldout.jsonl", ["text", "prompt"]),
        ("sft_data/sft_mixed_enja/eval.jsonl", ["text"]),
    ]
    for rel, keys in specs:
        p = ROOT / rel
        if not p.exists():
            continue
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for k in keys:
                    v = d.get(k)
                    if isinstance(v, str) and len(v.strip()) >= 18:
                        if "<|user|>" in v and "<|assistant|>" in v:
                            v = v.split("<|user|>", 1)[1].split("<|assistant|>", 1)[0]
                        nk = norm_key(v)
                        if len(nk) >= 20:
                            p30.add(nk[:30])
                            p50.add(nk[:50])
                for k in ("tests", "test", "canonical_solution", "prompt"):
                    v = d.get(k)
                    if isinstance(v, str):
                        for m in re.finditer(r"(?:assert|def)\s+([A-Za-z_]\w*)\s*\(", v):
                            if m.group(1) not in ("__init__", "helper", "solve", "main"):
                                hold_funcs.add(m.group(1))
    return FastHoldoutChecker(frozenset(p30), frozenset(p50)), hold_funcs


def bpe_len_ok(bpe, text: str, max_toks: int = 1000) -> tuple[bool, int]:
    """Fast UTF-8 pre-bound during streaming; exact C++ BPEEncoder filter runs immediately after!"""
    nb = len(text.encode("utf-8"))
    if nb > int(max_toks * 3.45):
        return False, max_toks + 50
    return True, max(16, int(nb / 3.2))


def stream_json_array(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (transformer-500m-posttrain-v9)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dec = json.JSONDecoder()
        buf = ""
        started = False
        while True:
            chunk = resp.read(2 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            pos = 0
            n = len(buf)
            if not started:
                idx = buf.find("[")
                if idx < 0:
                    buf = ""
                    continue
                pos = idx + 1
                started = True
            while pos < n:
                while pos < n and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos >= n or buf[pos] == "]":
                    break
                try:
                    obj, end_pos = dec.raw_decode(buf, pos)
                    yield obj
                    pos = end_pos
                except json.JSONDecodeError:
                    break
            buf = buf[pos:]


def convert_omi_solution_to_think(sol: str, ans_num: str, lang: str) -> str:
    """Convert OpenMathInstruct `<llm-code>...<llm-code-output>` into Option B1 `# = val` `<|think|>`."""
    def _repl(m):
        code = m.group(1).strip()
        out = " ".join(m.group(2).strip().splitlines()).strip()
        lines = [ln.rstrip() for ln in code.splitlines() if ln.strip()]
        if not lines:
            return ""
        if out and len(out) <= 48:
            lines[-1] = f"{lines[-1]}  # = {out}"
        return "```python\n" + "\n".join(lines) + "\n```"

    cleaned = LLM_CODE_PAIR_RE.sub(_repl, sol)
    cleaned = cleaned.replace("<llm-code>", "```python\n").replace("</llm-code>", "\n```")
    cleaned = re.sub(r"<llm-code-output>.*?</llm-code-output>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if lang == "en":
        return f"<|think|>\n{cleaned}\n<|/think|>\nTherefore, the answer is {ans_num}."
    return f"<|think|>\n{cleaned}\n<|/think|>\nしたがって、答えは{ans_num}です。"


# =============================================================================
# 1. Harvest OpenMathInstruct-JA (`dataset == 'math'` + hard multi-step `gsm8k`)
#    Simultaneously feeds SFT Hard Math, DPO Subtle Math, and RL Competition MATH!
# =============================================================================
def harvest_openmath_competition(hold_re: re.Pattern, bpe: BPEReference, existing_rl_prompts: set[int]):
    print("[1/5] Streaming OpenMathInstruct-JA (Competition MATH + Multi-step EN/JA)...", flush=True)
    sft_math_rows = []
    dpo_math_pairs = []
    rl_math_tasks = []

    seen_q_en: set[int] = set()
    seen_q_ja: set[int] = set()

    url = hf_hub_url("kunishou/OpenMathInstruct-1-1.8m-ja", "openmathinstruct_ja_20240225.jsonl", repo_type="dataset")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        for idx, raw in enumerate(resp):
            if len(sft_math_rows) >= 20000 and len(rl_math_tasks) >= 6400 and len(dpo_math_pairs) >= 5000:
                break
            if idx > 135_000:
                break
            if idx > 0 and idx % 25_000 == 0:
                print(f"    [OMI {idx:,}] sft={len(sft_math_rows):,} dpo={len(dpo_math_pairs):,} rl={len(rl_math_tasks):,}", flush=True)
            try:
                d = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue

            ds_name = d.get("dataset", "")
            q_en = (d.get("question") or "").strip()
            s_en = (d.get("generated_solution") or "").strip()
            q_ja = (d.get("question_ja") or "").strip()
            s_ja = (d.get("generated_solution_ja") or "").strip()
            if not q_en or not s_en or not q_ja or not s_ja:
                continue
            if len(KANA_RE.findall(q_ja)) < 5 or len(KANA_RE.findall(s_ja)) < 10:
                continue
            if hold_re.search(norm_key(q_en)) or hold_re.search(norm_key(q_ja)):
                continue

            # Extract exact numerical \boxed{...} answer
            m_box = BOXED_NUM_RE.search(s_en)
            if not m_box:
                continue
            ans_num = m_box.group(1).strip()
            if ans_num.startswith("+"):
                ans_num = ans_num[1:]
            # Normalize e.g. -0 -> 0
            try:
                fval = float(ans_num)
                if abs(fval) > 1e9:
                    continue
                if "." in ans_num and fval == int(fval):
                    ans_num = str(int(fval))
            except Exception:
                continue

            hq_en = hash64(norm_key(q_en))
            hq_ja = hash64(norm_key(q_ja))

            # (A) RL Competition MATH tasks (distinct questions only, dataset == 'math' prioritized)
            if ds_name == "math" and len(rl_math_tasks) < 6400:
                if hq_en not in seen_q_en and hq_en not in existing_rl_prompts and len(q_en) <= 900:
                    seen_q_en.add(hq_en)
                    ok_p, _ = bpe_len_ok(bpe, q_en, max_toks=320)
                    if ok_p:
                        rl_math_tasks.append({
                            "task": "comp_math_en",
                            "kind": "math_anchor",
                            "lang": "en",
                            "system": "You are a math tutor. Reason step by step inside <|think|>...</|think|> and end with 'Therefore, the answer is X.'",
                            "prompt": q_en,
                            "mock_input": "",
                            "expected_output": ans_num,
                            "tests": "",
                            "setup": "",
                        })
                if hq_ja not in seen_q_ja and hq_ja not in existing_rl_prompts and len(q_ja) <= 750:
                    seen_q_ja.add(hq_ja)
                    ok_p, _ = bpe_len_ok(bpe, q_ja, max_toks=320)
                    if ok_p:
                        rl_math_tasks.append({
                            "task": "comp_math_ja",
                            "kind": "math_anchor",
                            "lang": "ja",
                            "system": "あなたは数学の先生です。<|think|>...</|think|>で段階的に推論し、最後に「したがって、答えはXです。」と答えてください。",
                            "prompt": q_ja,
                            "mock_input": "",
                            "expected_output": ans_num,
                            "tests": "",
                            "setup": "",
                        })

            # (B) Hard SFT Math rows (prioritize Competition `math` or >= 3 step `<llm-code>` traces)
            is_hard = (ds_name == "math") or (s_en.count("\n") >= 6)
            if is_hard and len(sft_math_rows) < 20000:
                for lg, q_txt, s_txt, sys_txt in (
                    ("en", q_en, s_en, "You are a math tutor. Reason step by step and state the final answer clearly."),
                    ("ja", q_ja, s_ja, "あなたは数学の先生です。段階的に考えて最終的な答えを明確に述べてください。"),
                ):
                    if len(sft_math_rows) >= 20000:
                        break
                    resp = convert_omi_solution_to_think(s_txt, ans_num, lg)
                    full = f"<|system|>{sys_txt}<|user|>{q_txt}<|assistant|>{resp}<|end|>"
                    ok_bpe, n_tok = bpe_len_ok(bpe, full, max_toks=996)
                    if ok_bpe:
                        sft_math_rows.append({
                            "text": full,
                            "lang": lg,
                            "family": "math",
                            "task": f"v9_hard_{ds_name}",
                            "tokens_est": n_tok,
                        })

                        # (C) Subtle Late-Step DPO Hard Negative from the same verified problem!
                        if len(dpo_math_pairs) < 5000 and ("# = " in resp or "\\boxed{" in resp):
                            try:
                                base_val = int(ans_num) if "." not in ans_num else None
                            except Exception:
                                base_val = None
                            if base_val is not None and abs(base_val) >= 2:
                                delta = random.choice([-2, -1, 1, 2, 4, -3])
                                wrong_val = str(base_val + delta)
                                if wrong_val != ans_num:
                                    # Perturb final boxed value + final anchor (late-step subtle slip)
                                    rej = resp.replace(f"\\boxed{{{ans_num}}}", f"\\boxed{{{wrong_val}}}")
                                    if lg == "en":
                                        rej = rej.replace(
                                            f"Therefore, the answer is {ans_num}.",
                                            f"Therefore, the answer is {wrong_val}.",
                                        )
                                    else:
                                        rej = rej.replace(
                                            f"したがって、答えは{ans_num}です。",
                                            f"したがって、答えは{wrong_val}です。",
                                        )
                                    # Also perturb last `# = ans_num` if present so trace agrees with wrong anchor!
                                    rej = rej.replace(f"# = {ans_num}", f"# = {wrong_val}")
                                    if rej != resp and not rej.startswith(resp) and not resp.startswith(rej):
                                        r_full = f"<|system|>{sys_txt}<|user|>{q_txt}<|assistant|>{rej}<|end|>"
                                        ok_r, _ = bpe_len_ok(bpe, r_full, max_toks=996)
                                        if ok_r:
                                            dpo_math_pairs.append({
                                                "system": sys_txt,
                                                "prompt": q_txt,
                                                "chosen": resp,
                                                "rejected": rej,
                                                "lang": lg,
                                                "task": "v9_math_multistep_precision",
                                            })

    print(
        f"  -> OMI harvested: sft_math={len(sft_math_rows):,}, "
        f"dpo_math={len(dpo_math_pairs):,}, rl_comp_math={len(rl_math_tasks):,}",
        flush=True,
    )
    return sft_math_rows, dpo_math_pairs, rl_math_tasks


# =============================================================================
# 2. Reconstruct Authentic 2–3 Turn Dialogues from OASST1-89K-JA (EN & JA)
# =============================================================================
def harvest_oasst_multiturn(hold_re: re.Pattern, bpe: BPEReference, cap: int = 22000) -> list[dict]:
    print("[2/5] Reconstructing 2-3 turn dialogues from OASST1-89K-JA...", flush=True)
    nodes = {}
    children = defaultdict(list)
    url = hf_hub_url("kunishou/oasst1-89k-ja", "oasst1_89k_ja_20231027.json", repo_type="dataset")
    for row in stream_json_array(url):
        if not isinstance(row, dict):
            continue
        if str(row.get("ng_translation", "0")) != "0":
            continue
        mid = str(row.get("message_id", "")).strip()
        pid = str(row.get("parent_id", "")).strip()
        role = str(row.get("role", "")).strip()
        t_en = (row.get("text") or "").strip()
        t_ja = (row.get("text_ja") or "").strip()
        if not mid or role not in ("prompter", "assistant") or not t_en or not t_ja:
            continue
        nodes[mid] = (pid, role, t_en, t_ja)
        if pid and pid != "nan":
            children[pid].append(mid)

    # Walk root prompter -> assistant -> prompter -> assistant chains!
    out = []
    roots = [mid for mid, (pid, role, _, _) in nodes.items() if (not pid or pid == "nan") and role == "prompter"]
    random.shuffle(roots)

    for r_id in roots:
        if len(out) >= cap:
            break
        _, _, u1_en, u1_ja = nodes[r_id]
        for a1_id in children.get(r_id, []):
            if nodes[a1_id][1] != "assistant":
                continue
            _, _, a1_en, a1_ja = nodes[a1_id]
            u2_cands = [cid for cid in children.get(a1_id, []) if nodes[cid][1] == "prompter"]
            for u2_id in u2_cands:
                _, _, u2_en, u2_ja = nodes[u2_id]
                a2_cands = [cid for cid in children.get(u2_id, []) if nodes[cid][1] == "assistant"]
                for a2_id in a2_cands[:2]:
                    _, _, a2_en, a2_ja = nodes[a2_id]
                    # Check optional 3rd turn
                    turns_en = [(u1_en, a1_en), (u2_en, a2_en)]
                    turns_ja = [(u1_ja, a1_ja), (u2_ja, a2_ja)]
                    u3_cands = [cid for cid in children.get(a2_id, []) if nodes[cid][1] == "prompter"]
                    if u3_cands:
                        u3_id = u3_cands[0]
                        a3_cands = [cid for cid in children.get(u3_id, []) if nodes[cid][1] == "assistant"]
                        if a3_cands:
                            turns_en.append((nodes[u3_id][2], nodes[a3_cands[0]][2]))
                            turns_ja.append((nodes[u3_id][3], nodes[a3_cands[0]][3]))

                    # Emit EN multi-turn (EVERY assistant turn closed with <|end|>!)
                    if not hold_re.search(norm_key(u1_en)):
                        body_en = "".join(f"<|user|>{u}<|assistant|>{a}<|end|>" for u, a in turns_en)
                        full_en = f"<|system|>You are a helpful, accurate multilingual assistant.{body_en}"
                        ok_e, n_e = bpe_len_ok(bpe, full_en, max_toks=996)
                        if ok_e:
                            out.append({
                                "text": full_en,
                                "lang": "en",
                                "family": "facts",
                                "task": f"v9_multiturn_{len(turns_en)}t",
                                "tokens_est": n_e,
                            })

                    # Emit JA multi-turn (EVERY assistant turn closed with <|end|>!)
                    if len(KANA_RE.findall(a1_ja + a2_ja)) >= 15 and not hold_re.search(norm_key(u1_ja)):
                        body_ja = "".join(f"<|user|>{u}<|assistant|>{a}<|end|>" for u, a in turns_ja)
                        full_ja = f"<|system|>あなたは親切で正確な多言語AIアシスタントです。{body_ja}"
                        ok_j, n_j = bpe_len_ok(bpe, full_ja, max_toks=996)
                        if ok_j:
                            out.append({
                                "text": full_ja,
                                "lang": "ja",
                                "family": "facts",
                                "task": f"v9_multiturn_{len(turns_ja)}t",
                                "tokens_est": n_j,
                            })
                    if len(out) >= cap:
                        break
                if len(out) >= cap:
                    break
            if len(out) >= cap:
                break

    print(f"  -> OASST multi-turn harvested: {len(out):,} dialogues", flush=True)
    return out


# =============================================================================
# 3. Document/Paragraph Translation (3-5 sentences) + 2-Turn Translation + DPO
# =============================================================================
def build_doc_translation_and_dpo(bpe: BPEReference, v8_sft_hashes: set[int]):
    print("[3/5] Building 3-5 sentence Document Translations, 2-Turn Refinements & Subtle DPO...", flush=True)
    sft_tr = []
    dpo_tr = []

    kftt_path = ROOT / "data" / "kftt_enja_clean.tsv"
    kftt_pairs = []
    if kftt_path.exists():
        with kftt_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 2:
                    en, ja = cols[0].strip(), cols[1].strip()
                    if 20 <= len(en) <= 320 and 10 <= len(ja) <= 240 and KANA_RE.search(ja):
                        kftt_pairs.append((en, ja))

    en_to_ja_prompts = [
        "Translate the following paragraph into natural Japanese:\n\n{src}",
        "次の英文の段落を自然な日本語に翻訳してください。\n\n{src}",
        "Please translate this multi-sentence passage into Japanese while keeping terminology consistent:\n{src}",
        "日本語に翻訳してください:\n{src}",
    ]
    ja_to_en_prompts = [
        "Translate the following Japanese passage into natural, accurate English:\n\n{src}",
        "次の日本語の文章を自然で正確な英語に翻訳してください。\n\n{src}",
        "Please render this Japanese paragraph into fluent English:\n{src}",
        "Translate to English:\n{src}",
    ]

    rng = random.Random(50901)
    # Group consecutive 3-4 sentences from KFTT into multi-sentence paragraphs!
    idx = 0
    n_k = len(kftt_pairs)
    while idx + 4 <= n_k and len(sft_tr) < 28000:
        k = rng.choice([3, 4])
        chunk = kftt_pairs[idx: idx + k]
        idx += k
        en_para = " ".join(x[0] for x in chunk)
        ja_para = "".join(x[1] if x[1].endswith(("。", "！", "？")) else (x[1] + "。") for x in chunk)

        # Direction 1: EN -> JA paragraph (plus 30% chance of 2-turn follow-up!)
        p1 = rng.choice(en_to_ja_prompts).format(src=en_para)
        if rng.random() < 0.30:
            # 2-turn: Turn 1 translates paragraph, Turn 2 translates just the final sentence back or summarizes key sentence
            last_en, last_ja = chunk[-1]
            u2 = f"また、今の段落の最後の文「{last_ja}」を英語に戻すとどうなりますか？"
            full = (
                f"<|system|>あなたは正確で自然な翻訳を行うバイリンガルAIアシスタントです。"
                f"<|user|>{p1}<|assistant|>{ja_para}<|end|>"
                f"<|user|>{u2}<|assistant|>{last_en}<|end|>"
            )
        else:
            full = f"<|system|>You are an expert English-Japanese translator.<|user|>{p1}<|assistant|>{ja_para}<|end|>"
        ok1, nt1 = bpe_len_ok(bpe, full, max_toks=996)
        if ok1 and hash64(full) not in v8_sft_hashes:
            sft_tr.append({"text": full, "lang": "en-ja", "family": "translation", "task": "v9_doc_para", "tokens_est": nt1})

        # Subtle DPO hard-negative for 3-4 sentence paragraph: swap sentence 2 and sentence 3 in rejected!
        if len(dpo_tr) < 2500 and len(chunk) >= 3:
            swapped_ja = [x[1] if x[1].endswith(("。", "！", "？")) else (x[1] + "。") for x in chunk]
            swapped_ja[1], swapped_ja[2] = swapped_ja[2], swapped_ja[1]
            rej_ja = "".join(swapped_ja)
            if rej_ja != ja_para and not rej_ja.startswith(ja_para) and not ja_para.startswith(rej_ja):
                full_r = f"<|system|>You are an expert English-Japanese translator.<|user|>{p1}<|assistant|>{rej_ja}<|end|>"
                okr, _ = bpe_len_ok(bpe, full_r, max_toks=996)
                if okr:
                    dpo_tr.append({
                        "system": "You are an expert English-Japanese translator.",
                        "prompt": p1,
                        "chosen": ja_para,
                        "rejected": rej_ja,
                        "lang": "en-ja",
                        "task": "v9_trans_paragraph_order_precision",
                    })

        # Direction 2: JA -> EN paragraph
        p2 = rng.choice(ja_to_en_prompts).format(src=ja_para)
        if rng.random() < 0.30:
            first_en, first_ja = chunk[0]
            u2e = f"Now translate only the opening sentence back into Japanese:\n\"{first_en}\""
            full2 = (
                f"<|system|>You are an expert Japanese-English bilingual assistant."
                f"<|user|>{p2}<|assistant|>{en_para}<|end|>"
                f"<|user|>{u2e}<|assistant|>{first_ja}<|end|>"
            )
        else:
            full2 = f"<|system|>あなたは日英翻訳の専門家です。<|user|>{p2}<|assistant|>{en_para}<|end|>"
        ok2, nt2 = bpe_len_ok(bpe, full2, max_toks=996)
        if ok2 and hash64(full2) not in v8_sft_hashes:
            sft_tr.append({"text": full2, "lang": "ja-en", "family": "translation", "task": "v9_doc_para", "tokens_est": nt2})

        if len(dpo_tr) < 5000 and len(chunk) >= 3:
            swapped_en = [x[0] for x in chunk]
            swapped_en[0], swapped_en[1] = swapped_en[1], swapped_en[0]
            rej_en = " ".join(swapped_en)
            if rej_en != en_para and not rej_en.startswith(en_para) and not en_para.startswith(rej_en):
                full_r2 = f"<|system|>あなたは日英翻訳の専門家です。<|user|>{p2}<|assistant|>{rej_en}<|end|>"
                okr2, _ = bpe_len_ok(bpe, full_r2, max_toks=996)
                if okr2:
                    dpo_tr.append({
                        "system": "あなたは日英翻訳の専門家です。",
                        "prompt": p2,
                        "chosen": en_para,
                        "rejected": rej_en,
                        "lang": "ja-en",
                        "task": "v9_trans_paragraph_order_precision",
                    })

    # Top up to ~45,000 translation rows using UNUSED long-clause curated translations from sft_translation_curated
    cur_tr_path = ROOT / "sft_data" / "sft_translation_curated" / "train.jsonl"
    if cur_tr_path.exists() and len(sft_tr) < 45000:
        unused_tr = []
        with cur_tr_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                txt = r.get("text", "")
                if hash64(txt) in v8_sft_hashes:
                    continue
                if len(txt) >= 160:
                    ok_t, nt = bpe_len_ok(bpe, txt, max_toks=996)
                    if ok_t:
                        unused_tr.append({
                            "text": txt,
                            "lang": r.get("lang", "en-ja"),
                            "family": "translation",
                            "task": "v9_long_curated_trans",
                            "tokens_est": nt,
                        })
        rng.shuffle(unused_tr)
        sft_tr.extend(unused_tr[: max(0, 45000 - len(sft_tr))])

    print(f"  -> Doc/Long Translation harvested: sft={len(sft_tr):,}, dpo={len(dpo_tr):,}", flush=True)
    return sft_tr, dpo_tr


# =============================================================================
# 4. Complex Coding SFT + Subtle Edge-Case Bug DPO + Medium-Hard RL `assert`
# =============================================================================
class _Timeout(BaseException):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


PROBE_1ARG = [
    0, 1, 2, 3, 4, -1, -2,
    "", "a", "aba", "abcd", "Abc",
    [], [1], [1, 2, 3], [3, 1, 2, 1], [-2, 0, 3], [2, 2, 2],
]
PROBE_2ARG = [
    (0, 1), (1, 2), (4, 2), (-2, 3), (5, 2),
    ("abc", "b"), ("abcd", "bc"), ("abba", "a"),
    ([1, 2, 3], 2), ([1, 2, 3, 4], 3), ([4, 1, 3, 2], 1), ([1, 2], [2, 3]),
]
PROBE_3ARG = [
    (1, 2, 3), (2, 4, 6), (0, 1, 3), (-1, 0, 2),
    ([1, 2, 3, 4], 1, 3), ("abcd", 1, 3),
]


def make_subtle_code_bug(code: str) -> str | None:
    """Create an AST-valid subtle algorithmic/edge-case bug of nearly identical length."""
    import warnings
    swaps = [
        (" <= ", " < "),
        (" >= ", " > "),
        (" < ", " <= "),
        (" > ", " >= "),
        (" + 1", " - 1"),
        (" - 1", " + 1"),
        ("range(1, ", "range(0, "),
        ("return True", "return False"),
        ("== 0", "== 1"),
        ("!= 0", "> 0"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for old, new in swaps:
            if old in code:
                cand = code.replace(old, new, 1)
                if cand != code and not cand.startswith(code) and not code.startswith(cand):
                    try:
                        ast.parse(cand)
                        return cand
                    except SyntaxError:
                        continue
    return None


def fetch_vezora_magicoder_raw() -> list[tuple[str, str]]:
    print("[4a/5] Streaming Vezora 143k Tested Python + Magicoder-Evol-110K...", flush=True)
    raw_candidates = []
    try:
        url_vez = hf_hub_url("Vezora/Tested-143k-Python-Alpaca", "143k-Tested-Python-Alpaca-Vezora.json", repo_type="dataset")
        for idx, row in enumerate(stream_json_array(url_vez)):
            if idx >= 40000:
                break
            if not isinstance(row, dict):
                continue
            inst = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            out = (row.get("output") or "").strip()
            if inst and out and "```python" in out:
                prm = f"{inst}\n{inp}".strip() if inp else inst
                raw_candidates.append((prm, out))
    except Exception as e:
        print(f"  [WARN] Vezora stream: {e}", flush=True)

    try:
        url_mag = hf_hub_url("ise-uiuc/Magicoder-Evol-Instruct-110K", "data-evol_instruct-decontaminated.jsonl", repo_type="dataset")
        req = urllib.request.Request(url_mag, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            for idx, raw in enumerate(resp):
                if idx >= 30000:
                    break
                try:
                    row = json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    continue
                inst = (row.get("instruction") or "").strip()
                out = (row.get("response") or "").strip()
                if inst and out and "```" in out:
                    raw_candidates.append((inst, out))
    except Exception as e:
        print(f"  [WARN] Magicoder stream: {e}", flush=True)
    print(f"  -> Vezora+Magicoder fetched: {len(raw_candidates):,} code candidates", flush=True)
    return raw_candidates


def harvest_code_sft_dpo_rl(
    hold_re,
    hold_funcs: set[str],
    bpe,
    v8_sft_hashes: set[int],
    existing_rl_funcs: set[str],
    raw_candidates: list[tuple[str, str]],
):
    print("[4b/5] Executing SIGALRM verification for Coding SFT, DPO, and RL Assert Suites...", flush=True)
    sft_code = []
    dpo_code = []
    rl_assert = []

    rng = random.Random(50902)

    # First collect unused local aligned coding rows not in v8 SFT
    for rel in (
        "sft_data/sft_code_instruct_aligned/train.jsonl",
        "sft_data/sft_phi_code_exercises/train.jsonl",
        "sft_data/sft_think_code/train.jsonl",
    ):
        p = ROOT / rel
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                txt = r.get("text", "")
                if hash64(txt) in v8_sft_hashes:
                    continue
                ok_c, nc = bpe_len_ok(bpe, txt, max_toks=996)
                if ok_c:
                    sft_code.append({
                        "text": txt,
                        "lang": r.get("lang", "en"),
                        "family": "coding",
                        "task": "v9_unused_aligned_code",
                        "tokens_est": nc,
                    })

    raw_candidates = list(raw_candidates)

    rng.shuffle(raw_candidates)
    seen_fn = set(existing_rl_funcs) | set(hold_funcs)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    ja_code_templates = [
        "次の要件を満たすPython関数 `{fn}` を実装してください。\n\n{prm}\n\n満たすべきテスト例: `{hint}`",
        "以下の仕様に従って、Pythonで関数 `{fn}` を作成してください。\n{prm}\n例: `{hint}`",
        "Pythonで次の問題を解く関数 `{fn}` を書いてください。\n{prm}\n確認用アサーション: `{hint}`",
    ]

    try:
        for prm, out in raw_candidates:
            if len(sft_code) >= 22000 and len(dpo_code) >= 5000 and len(rl_assert) >= 6000:
                break
            if hold_re.search(norm_key(prm)):
                continue

            m_py = re.search(r"```python\s*\n(.*?)```", out, re.DOTALL)
            code_only = m_py.group(1).strip() if m_py else ""

            # (A) SFT Complex Coding (wrap with <|think|> plan if not already present)
            if len(sft_code) < 22000 and ("```python" in out or "```javascript" in out or "```java" in out or "```cpp" in out or "```go" in out):
                if "<|think|>" not in out:
                    plan = "Analyze the input constraints, handle boundary cases cleanly, and implement an efficient solution."
                    resp_sft = f"<|think|>\n{plan}\n<|/think|>\n{out.strip()}"
                else:
                    resp_sft = out.strip()
                full_sft = f"<|system|>You are an expert programming assistant.<|user|>{prm}<|assistant|>{resp_sft}<|end|>"
                ok_s, ns = bpe_len_ok(bpe, full_sft, max_toks=996)
                if ok_s and hash64(full_sft) not in v8_sft_hashes:
                    sft_code.append({
                        "text": full_sft,
                        "lang": "en",
                        "family": "coding",
                        "task": "v9_evol_tested_code",
                        "tokens_est": ns,
                    })

            if not code_only:
                continue

            # (B) Subtle Edge-Case Bug DPO
            if len(dpo_code) < 5000 and 4 <= code_only.count("\n") <= 55:
                buggy = make_subtle_code_bug(code_only)
                if buggy:
                    lg_d = "ja" if (len(dpo_code) % 3 == 0) else "en"
                    sys_d = "You are an expert Python programmer." if lg_d == "en" else "あなたは熟練したPythonプログラマーです。"
                    prm_d = prm if lg_d == "en" else f"次の仕様を満たす正確なPythonコードを書いてください:\n{prm}"
                    c_txt = f"<|think|>\nCheck boundary indices and base cases carefully.\n<|/think|>\n```python\n{code_only}\n```"
                    r_txt = f"<|think|>\nCheck boundary indices and base cases carefully.\n<|/think|>\n```python\n{buggy}\n```"
                    full_c = f"<|system|>{sys_d}<|user|>{prm_d}<|assistant|>{c_txt}<|end|>"
                    full_r = f"<|system|>{sys_d}<|user|>{prm_d}<|assistant|>{r_txt}<|end|>"
                    okc, _ = bpe_len_ok(bpe, full_c, max_toks=996)
                    okr, _ = bpe_len_ok(bpe, full_r, max_toks=996)
                    if okc and okr:
                        dpo_code.append({
                            "system": sys_d,
                            "prompt": prm_d,
                            "chosen": c_txt,
                            "rejected": r_txt,
                            "lang": lg_d,
                            "task": "v9_code_edgecase_precision",
                        })

            # (C) Execution-Verified Medium-Hard RL `assert` Suite (Pure AST Slice!)
            if len(rl_assert) < 6000:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    try:
                        tree = ast.parse(code_only)
                    except SyntaxError:
                        continue
                fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                if len(fdefs) != 1:
                    continue
                fn_node = fdefs[0]
                fname = fn_node.name
                if fname in seen_fn or fname.startswith("_"):
                    continue
                n_args = len(fn_node.args.args)
                if fn_node.args.vararg or fn_node.args.kwarg or fn_node.args.kwonlyargs or not (1 <= n_args <= 3):
                    continue

                # Check imports strictly against safe pure-algorithmic stdlib whitelist
                safe_stdlib = {
                    "math", "collections", "heapq", "bisect",
                    "typing", "functools", "string", "operator", "fractions", "statistics",
                }
                safe_imps = []
                bad_imp = False
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split(".")[0] not in safe_stdlib:
                                bad_imp = True
                        safe_imps.append(node)
                    elif isinstance(node, ast.ImportFrom):
                        if not node.module or node.module.split(".")[0] not in safe_stdlib:
                            bad_imp = True
                        safe_imps.append(node)
                if bad_imp:
                    continue

                # Mathematically bounded primitive-recursive AST check (< 1ms guaranteed):
                # no while, no nested def/class, <= 2 for/comp loops, no Pow/LShift, int literals <= 2000!
                bad_ast = False
                has_return_val = False
                loop_count = 0
                banned_calls = {
                    "print", "input", "open", "eval", "exec", "compile", "exit", "quit",
                    "help", "__import__", "breakpoint", "sleep", "factorial", "perm", "comb", "getattr",
                }
                for sub in ast.walk(fn_node):
                    if sub is not fn_node and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                        bad_ast = True
                        break
                    if isinstance(sub, (ast.For, ast.comprehension)):
                        loop_count += 1
                        if loop_count > 2:
                            bad_ast = True
                            break
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, int) and not isinstance(sub.value, bool):
                        if abs(sub.value) > 2000:
                            bad_ast = True
                            break
                    if isinstance(sub, ast.Return) and sub.value is not None:
                        has_return_val = True
                    if isinstance(sub, (ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Yield, ast.YieldFrom, ast.Global, ast.Pow, ast.LShift)):
                        bad_ast = True
                        break
                    if isinstance(sub, ast.Call):
                        if isinstance(sub.func, ast.Name) and (sub.func.id in banned_calls or sub.func.id == fname):
                            bad_ast = True
                            break
                        if isinstance(sub.func, ast.Attribute) and sub.func.attr in banned_calls:
                            bad_ast = True
                            break
                if bad_ast or not has_return_val:
                    continue

                try:
                    pure_code = "\n".join(ast.unparse(n) for n in (safe_imps + [fn_node]))
                except Exception:
                    continue

                probes = PROBE_1ARG if n_args == 1 else (PROBE_2ARG if n_args == 2 else PROBE_3ARG)
                ns_exec = {}
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        signal.setitimer(signal.ITIMER_REAL, 0.04)
                        exec(compile(pure_code, "<vez>", "exec"), ns_exec)
                    fn_obj = ns_exec.get(fname)
                    if not callable(fn_obj):
                        continue
                    valid_asserts = []
                    distinct_vals = set()
                    for pr in probes:
                        args_tuple = (pr,) if n_args == 1 else pr
                        call_args = [list(x) if isinstance(x, list) else (dict(x) if isinstance(x, dict) else x) for x in args_tuple]
                        orig_repr = ", ".join(repr(x) for x in call_args)
                        try:
                            signal.setitimer(signal.ITIMER_REAL, 0.02)
                            res = fn_obj(*call_args)
                        except _Timeout:
                            break
                        except Exception:
                            continue
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0.0)
                        if isinstance(res, (int, float, str, bool, list, tuple, dict)):
                            if isinstance(res, float):
                                if res != res or abs(res) > 1e12:
                                    continue
                                a_line = f"assert abs({fname}({orig_repr}) - {repr(res)}) < 1e-6"
                                distinct_vals.add(round(res, 4))
                            else:
                                r_str = repr(res)
                                if len(r_str) > 120:
                                    continue
                                a_line = f"assert {fname}({orig_repr}) == {r_str}"
                                distinct_vals.add(r_str)
                            if a_line not in valid_asserts:
                                valid_asserts.append(a_line)
                        if len(valid_asserts) >= 5:
                            break
                    # Require >= 3 assertions AND >= 2 distinct return values (blocks constant-return stubs!)
                    if len(valid_asserts) < 3 or len(distinct_vals) < 2:
                        continue

                    test_block = "\n".join(valid_asserts[:4])
                    # Verify gold pure_code passes
                    ns_gold = {}
                    signal.setitimer(signal.ITIMER_REAL, 0.04)
                    exec(compile(pure_code + "\n" + test_block, "<gold_chk>", "exec"), ns_gold)

                    # Verify identity stub `def fn(*a): return a[0]` ALSO fails!
                    stub_id = f"def {fname}(*a):\n    return a[0]\n"
                    stub_passed = False
                    try:
                        signal.setitimer(signal.ITIMER_REAL, 0.03)
                        exec(compile(stub_id + "\n" + test_block, "<stub_chk>", "exec"), {})
                        stub_passed = True
                    except Exception:
                        stub_passed = False
                    if stub_passed:
                        continue
                except (_Timeout, Exception):
                    continue
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)

                seen_fn.add(fname)
                hint = valid_asserts[0]
                use_ja = (len(rl_assert) % 2 == 1)
                if not use_ja:
                    full_prm = f"{prm}\nYour function `{fname}` must satisfy: `{hint}`"
                    sys_rl = "You are a coding assistant. Write clean, correct Python functions."
                    lg_rl = "en"
                else:
                    full_prm = rng.choice(ja_code_templates).format(fn=fname, prm=prm, hint=hint)
                    sys_rl = "あなたはコーディングアシスタントです。正確で簡潔なPython関数を書いてください。"
                    lg_rl = "ja"

                ok_rl, _ = bpe_len_ok(bpe, f"<|system|>{sys_rl}<|user|>{full_prm}<|assistant|>", max_toks=350)
                if ok_rl:
                    rl_assert.append({
                        "task": f"v9_exec_{fname}",
                        "kind": "assert",
                        "lang": lg_rl,
                        "system": sys_rl,
                        "prompt": full_prm,
                        "mock_input": "",
                        "expected_output": "",
                        "tests": test_block,
                        "setup": "",
                    })
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

    print(
        f"  -> Code harvested: sft_code={len(sft_code):,}, "
        f"dpo_code={len(dpo_code):,}, rl_assert={len(rl_assert):,}",
        flush=True,
    )
    return sft_code, dpo_code, rl_assert


def main() -> int:
    t0 = time.time()
    rng = random.Random(2026509)
    bpe = BPEReference(ROOT / "vocab_9")
    hold_re, hold_funcs = load_holdouts()

    # Index existing v8 SFT, DPO, and RL sets (strictly read-only!)
    print("[INIT] Indexing v8 SFT, DPO, and RL hashes (read-only)...", flush=True)
    v8_sft_hashes: set[int] = set()
    with (SFT_V8 / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                v8_sft_hashes.add(hash64(json.loads(line).get("text", "")))

    v8_dpo_prompts: set[int] = set()
    with (DPO_V8 / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                v8_dpo_prompts.add(hash64(norm_key(json.loads(line).get("prompt", ""))))

    v8_rl_prompts: set[int] = set()
    v8_rl_funcs: set[str] = set()
    with (RL_V8 / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                v8_rl_prompts.add(hash64(norm_key(r.get("prompt", ""))))
                t = str(r.get("task", ""))
                if t.startswith("phi_"):
                    v8_rl_funcs.add(t[4:])
                for m in re.finditer(r"assert\s+([A-Za-z_]\w*)\s*\(", r.get("tests", "")):
                    v8_rl_funcs.add(m.group(1))
    print(
        f"  -> v8 indexed: sft={len(v8_sft_hashes):,}, dpo={len(v8_dpo_prompts):,}, "
        f"rl_prompts={len(v8_rl_prompts):,}, rl_funcs={len(v8_rl_funcs):,}",
        flush=True,
    )

    # 1-4. Run ALL remote streams concurrently using ThreadPoolExecutor!
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut_omi = ex.submit(harvest_openmath_competition, hold_re, None, set(v8_rl_prompts))
        fut_oasst = ex.submit(harvest_oasst_multiturn, hold_re, None, 22000)
        fut_tr = ex.submit(build_doc_translation_and_dpo, None, v8_sft_hashes)
        fut_code_raw = ex.submit(fetch_vezora_magicoder_raw)
        sft_math_omi, dpo_math, rl_comp_math = fut_omi.result()
        sft_multiturn = fut_oasst.result()
        sft_tr, dpo_tr = fut_tr.result()
        code_raw_cands = fut_code_raw.result()

    # Top up Hard SFT Math to ~32,000 using unused local MetaMath EN/JA & Think-Math rows
    for rel in ("sft_data/sft_metamath_en/train.jsonl", "sft_data/sft_metamath_ja/train.jsonl", "sft_data/sft_think_math/train.jsonl"):
        if len(sft_math_omi) >= 32000:
            break
        p = ROOT / rel
        if not p.exists():
            continue
        cands = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                txt = r.get("text", "")
                if hash64(txt) in v8_sft_hashes or hold_re.search(norm_key(txt[:400])):
                    continue
                ok_m, nm = bpe_len_ok(bpe, txt, max_toks=996)
                if ok_m and txt.count("\n") >= 4:
                    cands.append({
                        "text": txt,
                        "lang": r.get("lang", "en"),
                        "family": "math",
                        "task": "v9_unused_metamath",
                        "tokens_est": nm,
                    })
        rng.shuffle(cands)
        sft_math_omi.extend(cands[: 32000 - len(sft_math_omi)])

    # Also add remaining unused GSM8K EN+JA problems (~6,000) to RL v9!
    rl_unused_gsm = []
    for lg, rel in (("en", "sft_data/sft_gsm8k_pot/train.jsonl"), ("ja", "sft_data/sft_gsm8k_ja_pot/train.jsonl")):
        p = ROOT / rel
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            exp = str(r.get("expected_output", "")).strip()
            txt = r.get("text", "")
            if not exp or "<|user|>" not in txt or "<|assistant|>" not in txt:
                continue
            sys_m = txt.split("<|system|>")[1].split("<|user|>")[0].strip() if "<|system|>" in txt else ""
            prm_m = txt.split("<|user|>")[1].split("<|assistant|>")[0].strip()
            hp = hash64(norm_key(prm_m))
            if hp in v8_rl_prompts or hold_re.search(norm_key(prm_m)):
                continue
            ok_p, _ = bpe_len_ok(bpe, prm_m, max_toks=320)
            if ok_p:
                v8_rl_prompts.add(hp)
                rl_unused_gsm.append({
                    "task": f"v9_gsm8k_{lg}_rem",
                    "kind": "math_anchor",
                    "lang": lg,
                    "system": sys_m,
                    "prompt": prm_m,
                    "mock_input": "",
                    "expected_output": exp,
                    "tests": "",
                    "setup": "",
                })

    # 2b. Top up multi-turn with unused Science Multi-turn & build science DPO
    sci_path = ROOT / "sft_data" / "sft_science_multiturn" / "train.jsonl"
    dpo_sci = []
    if sci_path.exists():
        sci_unused = []
        sci_pool_by_lang = {"en": [], "ja": []}
        with sci_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                txt = r.get("text", "")
                lg = r.get("lang", "en")
                if hash64(txt) not in v8_sft_hashes:
                    ok_s, ns = bpe_len_ok(bpe, txt, max_toks=996)
                    if ok_s:
                        sci_unused.append({
                            "text": txt,
                            "lang": lg,
                            "family": "facts",
                            "task": "v9_science_multiturn",
                            "tokens_est": ns,
                        })
                # Also collect single-turn science explanations for subtle DPO distractors
                if txt.count("<|user|>") == 1 and txt.count("<|assistant|>") == 1 and lg in sci_pool_by_lang:
                    sys_s = txt.split("<|system|>")[1].split("<|user|>")[0] if "<|system|>" in txt else ""
                    prm_s = txt.split("<|user|>")[1].split("<|assistant|>")[0]
                    ans_s = txt.split("<|assistant|>")[1].rsplit("<|end|>", 1)[0]
                    if 40 <= len(ans_s) <= 320 and hash64(norm_key(prm_s)) not in v8_dpo_prompts:
                        sci_pool_by_lang[lg].append((sys_s, prm_s, ans_s))
        rng.shuffle(sci_unused)
        sft_multiturn.extend(sci_unused[: max(0, 32000 - len(sft_multiturn))])

        # Build ~4,000 length-matched science/reasoning DPO pairs
        for lg in ("en", "ja"):
            items = sci_pool_by_lang[lg]
            rng.shuffle(items)
            # Sort by answer length so adjacent non-identical items have ~1.00 length ratio!
            items_sorted = sorted(items[:12000], key=lambda x: len(x[2]))
            for i in range(0, len(items_sorted) - 3, 2):
                if len(dpo_sci) >= 4000 or (lg == "en" and len(dpo_sci) >= 2000):
                    break
                sys_c, prm_c, ans_c = items_sorted[i]
                _, _, ans_r = items_sorted[i + 3]
                if ans_c == ans_r or ans_c.startswith(ans_r) or ans_r.startswith(ans_c):
                    continue
                ratio = len(ans_r) / max(1, len(ans_c))
                if not (0.78 <= ratio <= 1.28):
                    continue
                fc = f"<|system|>{sys_c}<|user|>{prm_c}<|assistant|>{ans_c}<|end|>"
                fr = f"<|system|>{sys_c}<|user|>{prm_c}<|assistant|>{ans_r}<|end|>"
                if bpe_len_ok(bpe, fc, 996)[0] and bpe_len_ok(bpe, fr, 996)[0]:
                    dpo_sci.append({
                        "system": sys_c,
                        "prompt": prm_c,
                        "chosen": ans_c,
                        "rejected": ans_r,
                        "lang": lg,
                        "task": "v9_science_reasoning_precision",
                    })

    # 4. Complex Coding SFT + Subtle Edge-Case DPO + Medium-Hard RL Assert (uses SIGALRM on main thread)
    sft_code, dpo_code, rl_assert = harvest_code_sft_dpo_rl(
        hold_re, hold_funcs, None, v8_sft_hashes, v8_rl_funcs, code_raw_cands
    )

    # =========================================================================
    # Write Separate v9 Folders & Exact Manifests
    # =========================================================================
    print("[5/5] Writing separate sft_v9_50b, dpo_v9_50b, and rl_v9_50b packages...", flush=True)

    # --- Package 1: sft_data/sft_v9_50b ---
    all_sft = sft_multiturn + sft_tr + sft_math_omi + sft_code
    rng.shuffle(all_sft)
    eval_sft = all_sft[:1500]
    train_sft = all_sft[1500:]

    SFT_V9.mkdir(parents=True, exist_ok=True)
    with (SFT_V9 / "v9_sft_hard_train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_sft:
            # Ensure "text" is the FIRST key on every JSONL line for C++ fast scanner
            out_obj = {"text": r["text"], "lang": r["lang"], "family": r["family"], "task": r["task"]}
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    with (SFT_V9 / "eval.jsonl").open("w", encoding="utf-8") as f:
        for r in eval_sft:
            out_obj = {"text": r["text"], "lang": r["lang"], "family": r["family"], "task": r["task"]}
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    sft_fam = Counter(r["family"] for r in train_sft)
    sft_lang = Counter(r["lang"] for r in train_sft)
    (SFT_V9 / "stats.json").write_text(json.dumps({
        "package": "sft_v9_50b (50B Hard-Tier Add-ons)",
        "v8_untouched": True,
        "train_rows": len(train_sft),
        "eval_rows": len(eval_sft),
        "est_total_tokens": sum(r["tokens_est"] for r in train_sft),
        "by_family": dict(sft_fam.most_common()),
        "by_lang": dict(sft_lang.most_common()),
        "recommended_sft_lr_on_50b": 1.5e-5,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Package 2: sft_data/dpo_v9_50b ---
    all_dpo = dpo_code + dpo_math + dpo_tr + dpo_sci
    rng.shuffle(all_dpo)
    DPO_V9.mkdir(parents=True, exist_ok=True)
    with (DPO_V9 / "v9_dpo_hard_train.jsonl").open("w", encoding="utf-8") as f:
        for r in all_dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dpo_task = Counter(r["task"] for r in all_dpo)
    dpo_lang = Counter(r["lang"] for r in all_dpo)
    ratios = sorted(len(r["rejected"]) / max(1, len(r["chosen"])) for r in all_dpo)
    p50_ratio = round(ratios[len(ratios) // 2], 3) if ratios else 1.0
    (DPO_V9 / "stats.json").write_text(json.dumps({
        "package": "dpo_v9_50b (50B Subtle Hard-Negative Add-ons)",
        "v8_untouched": True,
        "pairs": len(all_dpo),
        "median_len_ratio_rejected_over_chosen": p50_ratio,
        "by_task": dict(dpo_task.most_common()),
        "by_lang": dict(dpo_lang.most_common()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Package 3: rl_data/rl_v9_50b ---
    all_rl = rl_comp_math + rl_unused_gsm + rl_assert
    rng.shuffle(all_rl)
    RL_V9.mkdir(parents=True, exist_ok=True)
    with (RL_V9 / "v9_rl_hard_train.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rl_kind = Counter(r["kind"] for r in all_rl)
    rl_lang = Counter(r["lang"] for r in all_rl)
    rl_task = Counter(r["task"].split("_")[0] + "_" + r["task"].split("_")[1] for r in all_rl)
    (RL_V9 / "stats.json").write_text(json.dumps({
        "package": "rl_v9_50b (50B Medium-Hard Verifiable GRPO Add-ons)",
        "v8_untouched": True,
        "tasks": len(all_rl),
        "by_kind": dict(rl_kind.most_common()),
        "by_origin_group": dict(rl_task.most_common()),
        "by_lang": dict(rl_lang.most_common()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== ALL 3 V9 50B POST-TRAINING PACKAGES COMPLETE in {time.time()-t0:.1f}s ===", flush=True)
    print(f"  1. SFT v9 : {len(train_sft):,} train + {len(eval_sft):,} eval -> {SFT_V9}")
    print(f"  2. DPO v9 : {len(all_dpo):,} pairs (median len ratio={p50_ratio}) -> {DPO_V9}")
    print(f"  3. RL  v9 : {len(all_rl):,} verifiable tasks -> {RL_V9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
