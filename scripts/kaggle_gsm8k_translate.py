#!/usr/bin/env python3
"""kaggle_gsm8k_translate.py — Translate GSM8K train questions to ID/JA.

Runs on Kaggle (GPU). Uses NLLB-3.3B for quality. The reasoning steps
(answer_full) are translated too so SFT can teach CoT in 3 languages;
the final answer numbers are language-neutral and stay as-is.

Input : /kaggle/input/gsm8k-dataset/train.jsonl  (upload data/gsm8k/)
Output: /kaggle/working/gsm8k_train_tri.jsonl
        rows: {question, question_id, question_ja, answer_full_id, answer_full_ja, answer}

Resume-safe (state file tracks index).
"""

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/kaggle/input/gsm8k-dataset/train.jsonl")
    ap.add_argument("--out", default="/kaggle/working/gsm8k_train_tri.jsonl")
    ap.add_argument("--state", default="/kaggle/working/gsm8k_tri_state.json")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", torch_dtype=torch.float16).to(device).eval()

    LANGS = {"id": tok.convert_tokens_to_ids("ind_Latn"),
             "ja": tok.convert_tokens_to_ids("jpn_Jpan")}

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    print(f"loaded {len(rows)} GSM8K rows")

    done = 0
    st = Path(args.state)
    if st.exists():
        done = json.loads(st.read_text()).get("next", 0)
        print(f"resuming from {done}")

    @torch.no_grad()
    def translate(texts, lang):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        gen = model.generate(**enc, forced_bos_token_id=LANGS[lang],
                             max_new_tokens=256, num_beams=1)
        return tok.batch_decode(gen, skip_special_tokens=True)

    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        i = done
        while i < len(rows):
            batch = rows[i:i + args.batch]
            q = [r["question"] for r in batch]
            a = [r["answer_full"] for r in batch]
            try:
                q_id = translate(q, "id")
                q_ja = translate(q, "ja")
                a_id = translate(a, "id")
                a_ja = translate(a, "ja")
            except RuntimeError as e:  # OOM -> halve batch
                if "out of memory" in str(e).lower() and args.batch > 2:
                    args.batch //= 2
                    torch.cuda.empty_cache()
                    print(f"OOM -> batch {args.batch}", flush=True)
                    continue
                raise
            for r, qi, qj, ai, aj in zip(batch, q_id, q_ja, a_id, a_ja):
                f.write(json.dumps({
                    "question": r["question"], "answer": r["answer"],
                    "question_id": qi, "answer_full_id": ai,
                    "question_ja": qj, "answer_full_ja": aj,
                }, ensure_ascii=False) + "\n")
            i += len(batch)
            f.flush()
            st.write_text(json.dumps({"next": i}))
            if (i // args.batch) % 20 == 0:
                rate = i / max(time.time() - t0, 1)
                print(f"[{i}/{len(rows)}] {rate:.1f} rows/s "
                      f"ETA {(len(rows)-i)/max(rate,.01)/60:.0f}min", flush=True)

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
