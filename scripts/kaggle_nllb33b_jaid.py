#!/usr/bin/env python3
"""kaggle_nllb33b_jaid.py — JA<->ID synthesis at scale with NLLB-200-3.3B.

Kaggle T4x2 (or P100). Steps:
  1. Load cleaned EN-ID pairs (upload data/parallel_en_id.tsv).
  2. Translate the EN side to JA with NLLB-3.3B (fp16, batched, resume).
  3. Filter: JA script, length ratio, dedup.
  4. Output 2-col TSV: ja \\t id  (data/parallel_ja_id_nllb.tsv)

Also supports --mode backtranslate: translate ID monolingual lines
(data/culturax sample, upload as mono_id.txt) to EN -> synthetic id->en pairs.

Usage:
  !python kaggle_nllb33b_jaid.py --mode distill --max_pairs 1000000
  !python kaggle_nllb33b_jaid.py --mode backtranslate --mono /kaggle/input/mono-id/mono_id.txt
"""

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["distill", "backtranslate"], default="distill")
    ap.add_argument("--enid", default="/kaggle/input/parallel-enid/parallel_en_id.tsv")
    ap.add_argument("--mono", default="/kaggle/input/mono-id/mono_id.txt")
    ap.add_argument("--out", default="")
    ap.add_argument("--state", default="")
    ap.add_argument("--max_pairs", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_chars", type=int, default=140)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if args.mode == "distill":
        out = args.out or "/kaggle/working/parallel_ja_id_nllb.tsv"
        state = args.state or "/kaggle/working/jaid_state.json"
    else:
        out = args.out or "/kaggle/working/parallel_id_en_bt.tsv"
        state = args.state or "/kaggle/working/bt_state.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} gpus={torch.cuda.device_count()}")
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", torch_dtype=torch.float16).to(device).eval()

    JA = tok.convert_tokens_to_ids("jpn_Jpan")
    EN = tok.convert_tokens_to_ids("eng_Latn")

    def has_ja(s):
        return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in s)

    # load source pool
    if args.mode == "distill":
        pairs = []
        for line in open(args.enid, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            en, iid = p[0].strip(), p[1].strip()
            if 3 <= len(en) <= args.max_chars and 3 <= len(iid) <= args.max_chars:
                pairs.append((en, iid))
    else:
        pairs = []
        for line in open(args.mono, encoding="utf-8"):
            t = line.strip()
            if 3 <= len(t) <= args.max_chars:
                pairs.append((t, t))  # (src=id, ref=id); EN side synthesized
    print(f"pool: {len(pairs)}")

    start = 0
    st = Path(state)
    if st.exists():
        start = json.loads(st.read_text()).get("next", 0)
        print(f"resuming from {start}")

    t0 = time.time()
    written = start
    with open(out, "a", encoding="utf-8") as f:
        i = start
        while i < len(pairs) and written < start + args.max_pairs:
            batch = pairs[i:i + args.batch]
            src_texts = [b[0] for b in batch]
            try:
                enc = tok(src_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        forced_bos_token_id=(JA if args.mode == "distill" else EN),
                        max_new_tokens=128, num_beams=1)
                outs = tok.batch_decode(gen, skip_special_tokens=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and args.batch > 4:
                    torch.cuda.empty_cache()
                    args.batch = max(4, args.batch // 2)
                    print(f"OOM -> batch {args.batch}", flush=True)
                    continue
                raise
            for (src, ref), o in zip(batch, outs):
                o = o.strip()
                if not (3 <= len(o) <= args.max_chars):
                    continue
                if args.mode == "distill":
                    if not has_ja(o):
                        continue
                    r = len(o) / max(len(ref), 1)
                    if not (0.3 <= r <= 3.2):
                        continue
                    f.write(f"{o}\t{ref}\n")
                else:
                    r = len(o) / max(len(src), 1)
                    if not (0.4 <= r <= 2.5):
                        continue
                    f.write(f"{src}\t{o}\n")
                written += 1
            i += len(batch)
            f.flush()
            st.write_text(json.dumps({"next": i}))
            if (i // args.batch) % 50 == 0:
                rate = (i - start) / max(time.time() - t0, 1)
                eta = (len(pairs) - i) / max(rate, .01) / 60
                print(f"[{i}/{len(pairs)}] {rate:.1f} src/s ETA {eta:.0f}min "
                      f"(kept {written})", flush=True)

    print(f"done: kept {written} -> {out}")


if __name__ == "__main__":
    main()
