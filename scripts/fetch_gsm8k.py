#!/usr/bin/env python3
"""fetch_gsm8k.py — Download GSM8K train/test (HF: openai/gsm8k).

Stores the raw English data; translation to id/ja happens on Kaggle with
NLLB-3.3B (kaggle_gsm8k_translate.py).

Output:
  data/gsm8k/train.jsonl   (~7.47K rows: {question, answer})
  data/gsm8k/test.jsonl    (~1.32K)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pull(split):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for r in ds:
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"question": r["question"], "answer_full": r["answer"],
                     "answer": ans})
    return rows


def main():
    out = ROOT / "data" / "gsm8k"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = pull(split)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"gsm8k {split}: {len(rows)} rows -> {out / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
