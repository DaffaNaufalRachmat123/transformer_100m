#!/usr/bin/env python3
"""kaggle_labse_filter.py — Embedding-quality filter for noisy parallel pairs.

Runs on Kaggle T4 (free). Uses LaBSE (language-agnostic sentence embeddings)
to score cosine similarity of both sides; drops pairs below a per-pair
threshold. Intended for the noisiest sources (wikimatrix/ccmatrix mined data).

Input : 3-col raw TSV  (a \t b \t source)   — upload data/parallel_raw/<pair>/
Output: 2-col clean TSV (a \t b) + rejected report with scores.

Usage (Kaggle):
    !python kaggle_labse_filter.py --raw /kaggle/input/parallel-raw/ja-id/wikimatrix.tsv \
        --pair ja-id --out /kaggle/working/wikimatrix_labse.tsv
"""

import argparse
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="3-col raw TSV (a \\t b \\t source)")
    ap.add_argument("--pair", required=True, choices=["en-id", "ja-id", "en-ja"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="labse_filter_report.json")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="0 = use per-pair default")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel

    defaults = {"en-id": 0.72, "ja-id": 0.65, "en-ja": 0.65}
    thr = args.threshold or defaults[args.pair]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device).eval()

    rows = []
    for line in open(args.raw, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            rows.append((p[0].strip(), p[1].strip()))
    print(f"loaded {len(rows)} raw pairs; threshold={thr:.2f}")

    @torch.no_grad()
    def embed(texts):
        out = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            hid = model(**enc).pooler_output          # (B, 768)
            hid = torch.nn.functional.normalize(hid, dim=1)
            out.append(hid.cpu())
        return torch.cat(out)

    ea = embed([r[0] for r in rows])
    eb = embed([r[1] for r in rows])
    sims = (ea * eb).sum(dim=1).tolist()

    kept, rejected = [], []
    for (a, b), s in zip(rows, sims):
        if s >= thr:
            kept.append((a, b))
        else:
            rejected.append({"a": a, "b": b, "sim": round(s, 4)})

    with open(args.out, "w", encoding="utf-8") as f:
        for a, b in kept:
            f.write(f"{a}\t{b}\n")

    import json
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({
            "raw": len(rows), "kept": len(kept), "rejected": len(rejected),
            "threshold": thr,
            "mean_sim": sum(sims) / max(len(sims), 1),
            "rejected_sample": rejected[:100],
        }, f, ensure_ascii=False, indent=2)

    print(f"kept {len(kept)}/{len(rows)} ({100*len(kept)/max(len(rows),1):.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
