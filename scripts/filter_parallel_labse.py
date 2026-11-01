#!/usr/bin/env python3
"""filter_parallel_labse.py — drop misaligned EN<->JA pairs with LaBSE.

CCMatrix is bitext mined at a permissive threshold, so it is large and noisy:
hand-scoring 28 random pairs put it at ~61% genuinely parallel. A misaligned
pair is worse than no pair because it actively teaches a wrong translation, so
cutting 8.4M pairs down to the ~5M that score well is expected to raise chrF
rather than lower it.

LaBSE is used rather than a paraphrase model because it is trained with a
translation-ranking objective. Measured on 1,500 real CCMatrix rows: garbage
lands at 0.15-0.17, genuine pairs at 0.75-0.95, and 0.70 keeps ~60%.

The heuristic filters in fetch_corpus.pair_ok still apply on top — LaBSE scores
"What You Probably Never Knew About X" against "What あなた Probably Never Knew
About X" at 0.99, because the words really are the same, even though the
Japanese side is untranslated.

Usage:
  python3 scripts/filter_parallel_labse.py --threshold 0.70
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import pair_ok      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/parallel_raw/en-ja/par_ccmatrix.txt")
    ap.add_argument("--out", default="data/parallel_raw/en-ja/par_ccmatrix_labse.txt")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--out_cand", default="",
                    help="optional candidate output: cand_min <= score < threshold")
    ap.add_argument("--cand_min", type=float, default=0.65)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_seq", type=int, default=96,
                    help="pairs are <=300 chars, so the 256 default is wasted work")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fp32", action="store_true",
                   help="disable fp16; fp16 measured 861 vs 656 pair/s and moves"
                        " cosine scores by <0.001, far below the threshold")
    args = ap.parse_args()

    src, out = ROOT / args.src, ROOT / args.out
    if not src.exists():
        print(f"[ERR] missing {src}")
        return 1

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE", device=args.device)
    model.max_seq_length = args.max_seq
    if not args.fp32:
        model = model.half()

    total = kept = pre_dropped = cand = 0
    t0 = time.time()
    buf = []

    f_cand = None
    if args.out_cand:
        out_cand = ROOT / args.out_cand
        out_cand.parent.mkdir(parents=True, exist_ok=True)
        f_cand = out_cand.open("w", encoding="utf-8")

    def flush(f, f_cand, buf):
        nonlocal kept, cand
        if not buf:
            return
        en = [b[0] for b in buf]
        ja = [b[1] for b in buf]
        a = model.encode(en, batch_size=args.batch, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
        b = model.encode(ja, batch_size=args.batch, convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
        for (e, j), s in zip(buf, (a * b).sum(-1).tolist()):
            if s >= args.threshold:
                f.write(f"{e}\t{j}\n")
                kept += 1
            elif f_cand is not None and s >= args.cand_min:
                f_cand.write(f"{e}\t{j}\n")
                cand += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
            for line in fin:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                total += 1
                e, j = cols[0].strip(), cols[1].strip()
                if not pair_ok(e, j):
                    pre_dropped += 1
                    continue
                buf.append((e, j))
                if len(buf) >= args.batch * 8:
                    flush(fout, f_cand, buf)
                    buf = []
                    el = time.time() - t0
                    rate = total / max(el, 1)
                    print(f"  {total:>9,} read  core {kept:>9,}  cand {cand:>8,}  "
                          f"{rate:,.0f} pair/s  eta {(8_400_000-total)/max(rate,1)/3600:.1f} h",
                          flush=True)
            flush(fout, f_cand, buf)
    finally:
        if f_cand is not None:
            f_cand.close()

    print(f"\n[done] {out}" + (f" (+ {args.out_cand})" if f_cand is not None else ""))
    print(f"  read {total:,}  heuristic-dropped {pre_dropped:,}  "
          f"core {kept:,}  cand {cand:,}  in {(time.time()-t0)/3600:.2f} h")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
