#!/usr/bin/env python3
"""kaggle_nllb_jaid_distill.py — Synthesize JA<->ID pairs via NLLB-200 (pivot EN).

Runs on Kaggle T4x2 (free). Pipeline:
  1. Load cleaned EN<->ID pairs (data/parallel_en_id.tsv, uploaded as dataset).
  2. Translate the EN side to JA with NLLB-200-distilled-600M (fp16, batched).
  3. Post-filter: JA script check, length ratio, LID via fastText (optional).
  4. Output ja<TAB>id TSV (data/parallel_ja_id_nllb.tsv) + checkpoint/resume.

Run as Kaggle notebook cell or script with GPU enabled:
    !python kaggle_nllb_jaid_distill.py --max_pairs 500000 --start 0

Resume: the script writes output incrementally and tracks its position in
logs/nllb_jaid_state.json, so re-running picks up where it stopped.
"""

import argparse
import json
import re
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enid", default="/kaggle/input/parallel-enid/parallel_en_id.tsv")
    ap.add_argument("--out", default="/kaggle/working/parallel_ja_id_nllb.tsv")
    ap.add_argument("--state", default="/kaggle/working/nllb_jaid_state.json")
    ap.add_argument("--max_pairs", type=int, default=500_000)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--max_chars", type=int, default=140)
    ap.add_argument("--min_chars", type=int, default=3)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, gpus: {torch.cuda.device_count()}")

    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", torch_dtype=torch.float16)
    model = model.to(device).eval()

    JA = tok.convert_tokens_to_ids("jpn_Jpan")

    # ── load pairs
    pairs = []
    for line in open(args.enid, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 2:
            continue
        en, iid = p[0].strip(), p[1].strip()
        if args.min_chars <= len(en) <= args.max_chars and \
           args.min_chars <= len(iid) <= args.max_chars:
            pairs.append((en, iid))
    print(f"loaded {len(pairs)} usable en-id pairs")

    # ── resume state
    start = 0
    st = Path(args.state)
    if st.exists():
        start = json.loads(st.read_text()).get("next", 0)
        print(f"resuming from {start}")

    def has_ja(s):
        return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in s)

    t0 = time.time()
    n_written = start
    with open(args.out, "a", encoding="utf-8") as out_f:
        batch_src, batch_id = [], []

        def flush():
            nonlocal n_written
            if not batch_src:
                return
            enc = tok(batch_src, return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=JA,
                                     max_new_tokens=128, num_beams=1)
            jas = tok.batch_decode(gen, skip_special_tokens=True)
            for ja, iid in zip(jas, batch_id):
                ja = ja.strip()
                if not (args.min_chars <= len(ja) <= args.max_chars):
                    continue
                if not has_ja(ja):
                    continue
                r = len(ja) / max(len(iid), 1)
                if not (0.3 <= r <= 3.2):
                    continue
                out_f.write(f"{ja}\t{iid}\n")
                n_written += 1
            out_f.flush()
            batch_src.clear()
            batch_id.clear()
            st.write_text(json.dumps({"next": n_written}))
            rate = n_written / max(time.time() - t0, 1)
            print(f"[{n_written}/{args.max_pairs}] {rate:.1f} pairs/s "
                  f"ETA {(args.max_pairs - n_written) / max(rate, 0.01) / 60:.0f} min", flush=True)

        for en, iid in pairs[start:]:
            if n_written >= args.max_pairs:
                break
            batch_src.append(en)
            batch_id.append(iid)
            if len(batch_src) >= args.batch_size:
                flush()
        flush()

    print(f"done: {n_written} synthetic ja-id pairs -> {args.out}")

if __name__ == "__main__":
    main()
