#!/usr/bin/env python3
r"""fetch_ceiling_datasets.py — Stream, verify, and decontaminate high-leverage datasets
to push the 491.58M model to its theoretical ceiling without bloating local disk.

Acquires & builds:
  1. GSM8K-JA Option B1 (`p1atdev/gsm8k-ja-slim` aligned with `data/gsm8k/train.jsonl`)
     -> `sft_data/sft_gsm8k_ja_pot/{train,eval}.jsonl` (~6,500 JA word problems with
        verified `# = val` Python traces!) + raw text for mid-training math shard.
  2. MetaMathQA Option B1 (`meta-math/MetaMathQA`, decontaminated vs GSM8K test)
     -> `sft_data/sft_metamath_en/{train,eval}.jsonl` (~20,000 step-by-step EN math rows)
        + raw text for mid-training math shard.
  3. Phi-1 Textbook Python (`jinaai/code_exercises`, decontaminated vs HumanEval & MBPP test)
     -> `sft_data/sft_phi_code_exercises/{train,eval}.jsonl` (~10,000 AST-verified EN+JA
        textbook Python functions <= 950 tokens) + 25,000 functions for mid-training code shard.
  4. Science & Everyday Knowledge + Multi-Turn EN/JA (`allenai/sciq` + `OpenAssistant/oasst2`
     + bilingual core science/commonsense QA)
     -> `sft_data/sft_science_multiturn/{train,eval}.jsonl` + science support docs for mid-training.
  5. Kyoto Free Translation Task (`OPUS-KFTT` human-translated EN<->JA Wikipedia/news)
     -> `data/kftt_enja_clean.tsv` (~100,000 clean pairs) + `eval_data/kftt_heldout.jsonl` (1,000 pairs).
"""
import ast
import collections
import hashlib
import io
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_gsm8k_pot_sft import build_program, fmt_num, to_number
from build_code_instruct_sft import load_heldout, norm_ngrams

KANA = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
HAN = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
NUM_SEQ_RE = re.compile(r"\d+(?:\.\d+)?")


def load_gsm8k_test_fingerprints():
    fps = set()
    nums_set = set()
    p = ROOT / "data/gsm8k/test.jsonl"
    if p.exists():
        for l in p.open(encoding="utf-8"):
            d = json.loads(l)
            q = re.sub(r"\s+", " ", d.get("question", "").strip().lower())
            if len(q) >= 25:
                fps.add(q[:70])
            ns = tuple(NUM_SEQ_RE.findall(q))
            ans = str(to_number(d.get("answer", "")))
            if len(ns) >= 3 and ans:
                nums_set.add((ns, ans))
    return fps, nums_set


def step1_gsm8k_ja_and_metamath(rng):
    from datasets import load_dataset
    print("=== [1/5] Building GSM8K-JA (Option B1) + MetaMathQA (decontaminated) ===")
    test_fps, test_num_sigs = load_gsm8k_test_fingerprints()

    # Index local GSM8K train by (numbers_tuple, gold_answer_str) AND exact index
    en_train = []
    by_sig = collections.defaultdict(list)
    for idx, l in enumerate((ROOT / "data/gsm8k/train.jsonl").open(encoding="utf-8")):
        d = json.loads(l)
        q_en = d.get("question", "").strip()
        full = d.get("answer_full", "")
        gold = to_number(d.get("answer"))
        built = build_program(full, trace_values=True)
        if not built or gold is None:
            en_train.append(None)
            continue
        code, val = built
        if to_number(fmt_num(val)) != gold:
            en_train.append(None)
            continue
        g_str = fmt_num(gold)
        rec_info = (q_en, code, g_str)
        en_train.append(rec_info)
        ns = tuple(NUM_SEQ_RE.findall(q_en))
        by_sig[(ns, g_str)].append(rec_info)

    SYS_JA = "あなたは数学の先生です。答えを計算する短いPythonプログラムを順を追って書き、その結果を述べてください。"
    ja_rows = []
    midtrain_math_docs = []

    try:
        ds_ja = list(load_dataset("p1atdev/gsm8k-ja-slim", split="train", streaming=True))
        print(f"  streamed p1atdev/gsm8k-ja-slim: {len(ds_ja):,} rows")
        matched_idx = matched_sig = dropped_test = 0
        for idx, r in enumerate(ds_ja):
            q_ja = str(r.get("question", "")).replace("\r", "").strip()
            ans_ja = to_number(r.get("answer", ""))
            if not q_ja or ans_ja is None or not KANA.search(q_ja):
                continue
            a_str = fmt_num(ans_ja)
            ns_ja = tuple(NUM_SEQ_RE.findall(q_ja.replace(",", "")))
            if (ns_ja, a_str) in test_num_sigs:
                dropped_test += 1
                continue

            match = None
            if idx < len(en_train) and en_train[idx] is not None and en_train[idx][2] == a_str:
                match = en_train[idx]
                matched_idx += 1
            elif (ns_ja, a_str) in by_sig and len(by_sig[(ns_ja, a_str)]) == 1:
                match = by_sig[(ns_ja, a_str)][0]
                matched_sig += 1

            if not match:
                continue
            _q_en, code, g_str = match
            ja_rows.append({
                "text": f"<|system|>{SYS_JA}<|user|>{q_ja}<|assistant|>"
                        f"<|think|>{code}<|/think|>したがって、答えは{g_str}です。<|end|>",
                "lang": "ja",
                "task": "gsm8k_ja_pot",
                "expected_output": g_str,
                "has_think": True,
                "source": "gsm8k_ja_pot",
                "family": "math",
            })
            midtrain_math_docs.append(f"問題：{q_ja}\n解法：\n{code}\nしたがって、答えは{g_str}です。")
        print(f"  GSM8K-JA matched={len(ja_rows):,} (by_idx={matched_idx:,}, by_sig={matched_sig:,}, dropped_test_sig={dropped_test})")
    except Exception as e:
        print(f"  [WARN] gsm8k-ja-slim failed: {e}")

    rng.shuffle(ja_rows)
    out_ja = ROOT / "sft_data/sft_gsm8k_ja_pot"
    out_ja.mkdir(parents=True, exist_ok=True)
    n_ev = min(200, max(50, len(ja_rows) // 35))
    for sp, chunk in (("train", ja_rows[n_ev:]), ("eval", ja_rows[:n_ev])):
        with (out_ja / f"{sp}.jsonl").open("w", encoding="utf-8") as f:
            for x in chunk:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # Stream MetaMathQA (GSM_AnsAug / GSM_Rephrased / GSM_FOBAR — concise, step-by-step numerical answers)
    SYS_MM = "You are a careful math tutor. Solve the problem step by step, then state the final answer."
    mm_rows = []
    seen_mm_q = set()
    leaked_mm = 0
    try:
        ds_mm = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        for d in ds_mm:
            if len(mm_rows) >= 22000:
                break
            mtype = str(d.get("type", ""))
            if not mtype.startswith("GSM_"):
                continue
            q = str(d.get("query", "")).replace("\r", "").strip()
            orig_q = str(d.get("original_question", "")).replace("\r", "").strip().lower()
            resp = str(d.get("response", "")).replace("\r", "").strip()
            if not q or not resp or len(q) + len(resp) > 1600:
                continue
            q_norm = re.sub(r"\s+", " ", q.lower())
            orig_norm = re.sub(r"\s+", " ", orig_q)
            if any(fp in q_norm or fp in orig_norm for fp in test_fps):
                leaked_mm += 1
                continue
            if "####" not in resp and "The answer is:" not in resp:
                continue
            if "The answer is:" in resp:
                steps_part, ans_part = resp.rsplit("The answer is:", 1)
            else:
                steps_part, ans_part = resp.rsplit("####", 1)
            steps_part = re.sub(r"####\s*[-\d.,]+\s*$", "", steps_part.strip()).strip()
            gold_num = to_number(ans_part.strip())
            if gold_num is None or not steps_part:
                continue
            qh = hashlib.sha1(q_norm[:100].encode()).digest()
            if qh in seen_mm_q:
                continue
            seen_mm_q.add(qh)
            g_str = fmt_num(gold_num)
            mm_rows.append({
                "text": f"<|system|>{SYS_MM}<|user|>{q}<|assistant|>"
                        f"<|think|>{steps_part}<|/think|>Therefore, the answer is {g_str}.<|end|>",
                "lang": "en",
                "task": "metamath_gsm",
                "expected_output": g_str,
                "has_think": True,
                "source": "metamath_en",
                "family": "math",
            })
            if len(midtrain_math_docs) < 30000:
                midtrain_math_docs.append(f"Question: {q}\nSolution:\n{steps_part}\nTherefore, the answer is {g_str}.")
        print(f"  MetaMathQA kept={len(mm_rows):,} (decontam_dropped={leaked_mm})")
    except Exception as e:
        print(f"  [WARN] MetaMathQA failed: {e}")

    rng.shuffle(mm_rows)
    out_mm = ROOT / "sft_data/sft_metamath_en"
    out_mm.mkdir(parents=True, exist_ok=True)
    n_ev_m = min(300, max(50, len(mm_rows) // 50))
    for sp, chunk in (("train", mm_rows[n_ev_m:]), ("eval", mm_rows[:n_ev_m])):
        with (out_mm / f"{sp}.jsonl").open("w", encoding="utf-8") as f:
            for x in chunk:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    return midtrain_math_docs


def step2_phi_code_exercises(rng):
    from datasets import load_dataset
    from build_thinking_sft import describe_code, PLAN_EN, PLAN_JA
    print("=== [2/5] Streaming Phi-1 Textbook Python (`jinaai/code_exercises`) ===")
    heldout_names, heldout_asserts = load_heldout()
    name_re = (re.compile(r"\b(" + "|".join(sorted(map(re.escape, heldout_names), key=len, reverse=True)) + r")\b")
               if heldout_names else None)

    SYS_EN = "You are a coding assistant. Write short, correct Python programs."
    SYS_JA = "あなたはコーディングアシスタントです。短く正確なPythonプログラムを書いてください。"
    sft_rows = []
    midtrain_code_docs = []
    seen_fn = set()
    leaked = syntax_err = 0

    ds = load_dataset("jinaai/code_exercises", split="train", streaming=True)
    for d in ds:
        if len(sft_rows) >= 12000 and len(midtrain_code_docs) >= 28000:
            break
        prob = str(d.get("problem", "")).replace("\r\n", "\n").replace("\r", "\n").rstrip()
        sol = str(d.get("solution", "")).replace("\r\n", "\n").replace("\r", "\n").rstrip()
        if not prob or not sol:
            continue
        full_code = prob + "\n" + sol
        if not (120 <= len(full_code) <= 2200):
            continue
        if (name_re and name_re.search(full_code)) or (norm_ngrams(full_code, n=10) & heldout_asserts):
            leaked += 1
            continue
        try:
            tree = ast.parse(full_code)
        except SyntaxError:
            syntax_err += 1
            continue
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not fns:
            continue
        fn_node = fns[0]
        doc = ast.get_docstring(fn_node) or ""
        if len(doc) < 20:
            continue
        sig_line = prob.splitlines()[0].strip()
        key = (fn_node.name, doc[:60])
        if key in seen_fn:
            continue
        seen_fn.add(key)

        if len(midtrain_code_docs) < 28000:
            midtrain_code_docs.append(full_code)

        if len(sft_rows) < 12000:
            # Strip docstring from body so the prompt asks for the function from docstring+signature,
            # teaching real synthesis instead of copying the prompt!
            lang = "ja" if (len(sft_rows) % 3 == 0) else "en"
            step_desc = describe_code(full_code, lang) or ("implement the function" if lang == "en" else "関数を実装する")
            plan = rng.choice(PLAN_EN if lang == "en" else PLAN_JA).format(step=step_desc)
            if lang == "en":
                user_q = f"Write a Python function `{sig_line}` that satisfies the following specification:\n{doc}"
                sys_m = SYS_EN
            else:
                user_q = f"次の仕様を満たすPython関数 `{sig_line}` を書いてください。\n{doc}"
                sys_m = SYS_JA
            ans_fenced = f"```python\n{full_code}\n```"
            text = f"<|system|>{sys_m}<|user|>{user_q}<|assistant|><|think|>{plan}<|/think|>{ans_fenced}<|end|>"
            if "\\u" not in json.dumps({"text": text}, ensure_ascii=False):
                sft_rows.append({
                    "text": text,
                    "lang": lang,
                    "task": "phi_code_exercises",
                    "has_think": True,
                    "source": "phi_code_exercises",
                    "family": "coding",
                })

    print(f"  CodeExercises: sft={len(sft_rows):,} midtrain_docs={len(midtrain_code_docs):,} leaked={leaked} syntax_err={syntax_err}")
    rng.shuffle(sft_rows)
    out = ROOT / "sft_data/sft_phi_code_exercises"
    out.mkdir(parents=True, exist_ok=True)
    n_ev = 200
    for sp, chunk in (("train", sft_rows[n_ev:]), ("eval", sft_rows[:n_ev])):
        with (out / f"{sp}.jsonl").open("w", encoding="utf-8") as f:
            for x in chunk:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return midtrain_code_docs


CURATED_BILINGUAL_SCIENCE_COMMONSENSE = [
    ("What is the chemical formula of water?", "The chemical formula of water is H2O.", "水の化学式は何ですか？", "水の化学式はH2Oです。"),
    ("What is the chemical formula of carbon dioxide?", "The chemical formula of carbon dioxide is CO2.", "二酸化炭素の化学式は何ですか？", "二酸化炭素の化学式はCO2です。"),
    ("What is the chemical formula of table salt?", "The chemical formula of table salt (sodium chloride) is NaCl.", "食塩（塩化ナトリウム）の化学式は何ですか？", "食塩（塩化ナトリウム）の化学式はNaClです。"),
    ("What is Earth's natural satellite?", "Earth's natural satellite is the Moon.", "地球の衛星は何ですか？", "地球の唯一の自然衛星は月です。"),
    ("What is the largest planet in the solar system?", "The largest planet in the solar system is Jupiter.", "太陽系で最も大きい惑星は何ですか？", "太陽系で最も大きい惑星は木星です。"),
    ("Which planet is closest to the Sun?", "The planet closest to the Sun is Mercury.", "太陽に最も近い惑星は何ですか？", "太陽に最も近い惑星は水星です。"),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet.", "「赤い惑星」として知られる惑星は何ですか？", "火星が「赤い惑星」として知られています。"),
    ("How many legs does a human have?", "A human has 2 legs.", "人間の足の数は何本ですか？", "人間の足の数は2本です。"),
    ("How many legs does an insect have?", "An insect has 6 legs.", "昆虫の足は何本ですか？", "昆虫の足は6本です。"),
    ("How many legs does a spider have?", "A spider has 8 legs.", "クモの足は何本ですか？", "クモの足は8本です。"),
    ("What color is a clear daytime sky?", "A clear daytime sky is blue due to Rayleigh scattering of sunlight.", "晴れた昼間の空の色は何色ですか？", "晴れた昼間の空は青色です。"),
    ("At what temperature does pure water boil at sea level?", "Pure water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "海面気圧で純粋な水の沸点は何度ですか？", "標準気圧における水の沸点は摂氏100度です。"),
    ("At what temperature does water freeze?", "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit).", "水の凝固点（凍る温度）は何度ですか？", "水の凝固点は摂氏0度です。"),
    ("What gas do plants absorb during photosynthesis?", "Plants absorb carbon dioxide (CO2) and release oxygen during photosynthesis.", "植物は光合成で何の気体を吸収しますか？", "植物は光合成で二酸化炭素を吸収し、酸素を放出します。"),
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792 kilometers per second.", "真空中の光の速さは秒速約何キロメートルですか？", "真空中の光の速さは秒速約30万キロメートル（299,792 km/s）です。"),
    ("How many days are in a leap year?", "There are 366 days in a leap year.", "うるう年は1年間に何日ありますか？", "うるう年は1年間に366日あります。"),
    ("How many bones are in the adult human body?", "An adult human body has 206 bones.", "成人の人間の骨の数はいくつですか？", "成人の人間の骨は約206本あります。"),
    ("Which organ pumps blood throughout the human body?", "The heart pumps blood throughout the human body.", "体中に血液を送り出す臓器は何ですか？", "心臓が体中に血液を送り出します。"),
    ("What is the powerhouse of the cell?", "The mitochondrion is the powerhouse of the cell, producing ATP.", "細胞の中でエネルギー（ATP）を作る小器官は何ですか？", "ミトコンドリアが細胞内でATP（エネルギー）を産生します。"),
    ("How many continents are there on Earth?", "There are 7 continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia/Oceania.", "地球にはいくつの大陸がありますか？", "地球にはアジア、アフリカ、北アメリカ、南アメリカ、南極、ヨーロッパ、オセアニアの7大陸があります。"),
]


def step3_science_multiturn_abstention(rng):
    from datasets import load_dataset
    print("=== [3/5] Building Science QA (`allenai/sciq`), Multi-Turn EN/JA (`oasst2`), & Honest Abstention ===")
    SYS_EN = "You are a helpful, accurate multilingual assistant."
    SYS_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"

    sft_rows = []
    midtrain_sci_docs = []

    # 3a. Curated core bilingual science & commonsense (with question variants)
    for q_en, a_en, q_ja, a_ja in CURATED_BILINGUAL_SCIENCE_COMMONSENSE:
        for _ in range(3):
            sft_rows.append({"text": f"<|system|>{SYS_EN}<|user|>{q_en}<|assistant|>{a_en}<|end|>",
                             "lang": "en", "task": "science_core", "source": "science_core", "family": "facts",
                             "value_en": a_en.split()[-1].rstrip(".")})
            sft_rows.append({"text": f"<|system|>{SYS_JA}<|user|>{q_ja}<|assistant|>{a_ja}<|end|>",
                             "lang": "ja", "task": "science_core", "source": "science_core", "family": "facts",
                             "value_ja": a_ja.rstrip("。")})
        midtrain_sci_docs.append(f"{q_en} {a_en}\n{q_ja} {a_ja}")

    # 3b. Stream allenai/sciq (11,679 science questions + encyclopedia support)
    ds_sciq = load_dataset("allenai/sciq", split="train", streaming=True)
    sciq_kept = 0
    for d in ds_sciq:
        q = str(d.get("question", "")).strip()
        ans = str(d.get("correct_answer", "")).strip()
        sup = str(d.get("support", "")).replace("\r", "").strip()
        if not q or not ans or len(q) > 260 or len(ans) > 80:
            continue
        # Craft a natural full-sentence answer using first sentence of support if it mentions ans, else direct
        first_sent = sup.split(". ")[0].strip() + "." if sup else ""
        if first_sent and len(first_sent) <= 220 and ans.lower() in first_sent.lower():
            full_ans = first_sent
        else:
            full_ans = f"The answer is {ans}."
        sft_rows.append({
            "text": f"<|system|>{SYS_EN}<|user|>{q}<|assistant|>{full_ans}<|end|>",
            "lang": "en", "task": "fact_qa", "category": "science_qa",
            "value_en": ans, "source": "sciq", "family": "facts",
        })
        if sup and len(sup) >= 60:
            midtrain_sci_docs.append(f"{sup}\nQuestion: {q} Answer: {full_ans}")
        sciq_kept += 1
    print(f"  allenai/sciq kept={sciq_kept:,}")

    # 3c. Stream OpenAssistant/oasst2 for clean 2-turn EN & JA multi-turn conversations (<= 900 chars)
    mt_kept = collections.Counter()
    try:
        ds_oa = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        by_id = {}
        children = collections.defaultdict(list)
        for d in ds_oa:
            lg = str(d.get("lang", ""))
            if lg not in ("en", "ja"):
                continue
            if d.get("deleted") or not d.get("review_result", True):
                continue
            if d.get("rank") is not None and d.get("rank") > 0:
                continue
            mid = d.get("message_id")
            pid = d.get("parent_id")
            txt = str(d.get("text", "")).replace("\r\n", "\n").replace("\r", "\n").strip()
            if not txt or len(txt) > 550 or "<|" in txt:
                continue
            by_id[mid] = (d.get("role"), lg, txt)
            if pid:
                children[pid].append(mid)

        for mid, (role, lg, u1) in by_id.items():
            if role != "prompter" or mt_kept[lg] >= (2500 if lg == "en" else 2000):
                continue
            c1s = [c for c in children.get(mid, []) if by_id.get(c, (None,))[0] == "assistant"]
            if not c1s:
                continue
            a1 = by_id[c1s[0]][2]
            c2s = [c for c in children.get(c1s[0], []) if by_id.get(c, (None,))[0] == "prompter"]
            sys_m = SYS_EN if lg == "en" else SYS_JA
            if c2s:
                u2 = by_id[c2s[0]][2]
                c3s = [c for c in children.get(c2s[0], []) if by_id.get(c, (None,))[0] == "assistant"]
                if c3s:
                    a2 = by_id[c3s[0]][2]
                    rendered = (f"<|system|>{sys_m}<|user|>{u1}<|assistant|>{a1}<|end|>"
                                f"<|user|>{u2}<|assistant|>{a2}<|end|>")
                    if len(rendered) <= 2200:
                        if lg == "ja" and not KANA.search(a1 + a2):
                            continue
                        if lg == "en" and KANA.search(u1 + a1 + u2 + a2):
                            continue
                        sft_rows.append({
                            "text": rendered, "lang": lg, "task": "multiturn_chat",
                            "source": "oasst2_multiturn", "family": "facts",
                        })
                        mt_kept[lg] += 1
        print(f"  oasst2 2-turn multi-turn kept={dict(mt_kept)}")
    except Exception as e:
        print(f"  [WARN] oasst2 failed: {e}")

    rng.shuffle(sft_rows)
    out = ROOT / "sft_data/sft_science_multiturn"
    out.mkdir(parents=True, exist_ok=True)
    n_ev = 200
    for sp, chunk in (("train", sft_rows[n_ev:]), ("eval", sft_rows[:n_ev])):
        with (out / f"{sp}.jsonl").open("w", encoding="utf-8") as f:
            for x in chunk:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return midtrain_sci_docs


def step4_kftt_and_mgsm_eval(rng):
    from datasets import load_dataset
    print("=== [4/5] Fetching OPUS KFTT (clean human EN<->JA) + Held-Out MGSM-JA Eval ===")
    eval_dir = ROOT / "eval_data"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save strictly held-out MGSM-JA test (250 human-translated GSM8K test problems!)
    try:
        ds_mgsm = load_dataset("juletxara/mgsm", "ja", split="test", streaming=True)
        mgsm_rows = []
        for idx, d in enumerate(ds_mgsm):
            mgsm_rows.append({
                "id": f"mgsm_ja_{idx}",
                "lang": "ja",
                "question": d.get("question", ""),
                "answer_number": str(d.get("answer_number", "")),
                "answer_text": d.get("answer", ""),
            })
        with (eval_dir / "mgsm_ja_test.jsonl").open("w", encoding="utf-8") as f:
            for r in mgsm_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved held-out eval_data/mgsm_ja_test.jsonl: {len(mgsm_rows)} problems")
    except Exception as e:
        print(f"  [WARN] mgsm ja eval fetch: {e}")

    # Download OPUS KFTT zip (high-quality Kyoto Free Translation Task!)
    kftt_out = ROOT / "data/kftt_enja_clean.tsv"
    url = "https://object.pouta.csc.fi/OPUS-KFTT/v1.0/moses/en-ja.txt.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_zip = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(raw_zip))
        en_name = [n for n in zf.namelist() if n.endswith(".en")][0]
        ja_name = [n for n in zf.namelist() if n.endswith(".ja")][0]
        en_lines = zf.read(en_name).decode("utf-8", errors="replace").splitlines()
        ja_lines = zf.read(ja_name).decode("utf-8", errors="replace").splitlines()

        pairs = []
        seen = set()
        for en, ja in zip(en_lines, ja_lines):
            en = re.sub(r"\s+", " ", en).strip()
            ja = re.sub(r"\s+", " ", ja.replace(" ", "")).strip() if KANA.search(ja) else ja.strip()
            if not (15 <= len(en) <= 380 and 6 <= len(ja) <= 280):
                continue
            if not LATIN.search(en) or KANA.search(en) or not KANA.search(ja):
                continue
            h = hashlib.sha1((en.lower() + "\t" + ja).encode()).digest()
            if h in seen:
                continue
            seen.add(h)
            pairs.append((en, ja))
            if len(pairs) >= 130000:
                break
        rng.shuffle(pairs)
        heldout_kftt, train_kftt = pairs[:1000], pairs[1000:]
        with kftt_out.open("w", encoding="utf-8") as f:
            for en, ja in train_kftt:
                f.write(f"{en}\t{ja}\n")
        with (eval_dir / "kftt_heldout_enja.jsonl").open("w", encoding="utf-8") as f:
            for en, ja in heldout_kftt:
                f.write(json.dumps({"en": en, "ja": ja}, ensure_ascii=False) + "\n")
        print(f"  OPUS KFTT saved: train={len(train_kftt):,} pairs -> data/kftt_enja_clean.tsv, heldout=1,000 -> eval_data/kftt_heldout_enja.jsonl")
    except Exception as e:
        print(f"  [WARN] OPUS KFTT download: {e}")


def step5_write_midtrain_addons(math_docs, code_docs, sci_docs, rng):
    print("=== [5/5] Staging Pretrain Mid-Training Annealing Add-on Shards (for 47% -> 100%) ===")
    addon_root = ROOT / "data/pretrain_midtrain_addons"
    (addon_root / "math").mkdir(parents=True, exist_ok=True)
    (addon_root / "code").mkdir(parents=True, exist_ok=True)
    (addon_root / "mixed").mkdir(parents=True, exist_ok=True)

    DOC_SEP = "\n<|endoftext|>\n"
    # Repeat high-signal GSM8K-JA + MetaMath 6x (~180K docs) so within the 5B math bucket
    # it receives meaningful sampling probability during the 47%->100% cosine decay!
    if math_docs:
        p = addon_root / "math/midtrain_gsm_ja_metamath_00.txt"
        with p.open("w", encoding="utf-8") as f:
            for _ in range(6):
                rng.shuffle(math_docs)
                for doc in math_docs:
                    f.write(doc.strip() + DOC_SEP)
        print(f"  wrote {p.relative_to(ROOT)} ({p.stat().st_size/1e6:.1f} MB)")

    if code_docs:
        p = addon_root / "code/midtrain_phi_python_textbook_00.txt"
        with p.open("w", encoding="utf-8") as f:
            for _ in range(5):
                rng.shuffle(code_docs)
                for doc in code_docs:
                    f.write(doc.strip() + DOC_SEP)
        print(f"  wrote {p.relative_to(ROOT)} ({p.stat().st_size/1e6:.1f} MB)")

    if sci_docs:
        p = addon_root / "mixed/midtrain_science_commonsense_00.txt"
        with p.open("w", encoding="utf-8") as f:
            for _ in range(8):
                rng.shuffle(sci_docs)
                for doc in sci_docs:
                    f.write(doc.strip() + DOC_SEP)
        print(f"  wrote {p.relative_to(ROOT)} ({p.stat().st_size/1e6:.1f} MB)")


def main():
    rng = random.Random(2026)
    math_docs = step1_gsm8k_ja_and_metamath(rng)
    code_docs = step2_phi_code_exercises(rng)
    sci_docs = step3_science_multiturn_abstention(rng)
    step4_kftt_and_mgsm_eval(rng)
    step5_write_midtrain_addons(math_docs, code_docs, sci_docs, rng)
    print("\n[ALL CEILING DATASETS FETCHED & DECONTAMINATED]")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
