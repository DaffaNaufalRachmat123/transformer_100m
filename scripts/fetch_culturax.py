#!/usr/bin/env python3
"""fetch_culturax.py — Stream-download CulturaX (id/ja) to one-doc-per-line txt.

Runs locally (overnight-friendly): resumable, byte-capped, spam-light filter.

Output: data/culturax_raw/culturax_{lang}.txt  (one document per line)
State: logs/culturax_{lang}_state.json (bytes written + skip count)

Usage:
  python3 scripts/fetch_culturax.py --lang id --max_bytes 2684354560
  python3 scripts/fetch_culturax.py --lang ja --max_bytes 2684354560
"""

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# light pre-filter before writing (heavy cleaning is a separate stage)
BAD_PATTERNS = re.compile(
    r"agen\s+(penjual|obat)|obat\s+kuat|dewasa\s+18|poker\s+online|"
    r"judi\s+(online| bola)|bandar\s+(qq|togel|casino)|situsslot|"
    r"ボーナス胴元|風俗|出会い系|アダルト|http[s]?://\S{200,}"
)


def doc_ok(text):
    if not text:
        return False
    n = len(text)
    if n < 200 or n > 200_000:
        return False
    if BAD_PATTERNS.search(text):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["id", "ja"])
    ap.add_argument("--max_bytes", type=int, default=2_684_354_560)  # 2.5 GiB
    ap.add_argument("--out_dir", default="data/culturax_raw")
    ap.add_argument("--max_docs", type=int, default=3_000_000)
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"culturax_{args.lang}.txt"
    state_path = ROOT / "logs" / f"culturax_{args.lang}_state.json"

    bytes_written = 0
    docs_written = 0
    docs_skipped = 0
    if state_path.exists():
        st = json.loads(state_path.read_text())
        if st.get("done"):
            print(f"[{args.lang}] already done ({st.get('bytes')} bytes) — skipping")
            return
        bytes_written = st.get("bytes", 0)
        docs_written = st.get("docs", 0)
        docs_skipped = st.get("skipped", 0)
        print(f"[{args.lang}] resuming at {bytes_written} bytes / {docs_written} docs")

    print(f"[{args.lang}] streaming uonlp/CulturaX ...")
    ds = load_dataset("uonlp/CulturaX", args.lang, split="train", streaming=True)

    t0 = time.time()
    n_chunk = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(ds):
            if bytes_written >= args.max_bytes or docs_written >= args.max_docs:
                break
            text = (rec.get("text") or "").strip()
            if not doc_ok(text):
                docs_skipped += 1
                continue
            line = text.replace("\n", " ").replace("\r", " ")
            line = re.sub(r"[ \t]{2,}", " ", line)
            if not doc_ok(line):
                docs_skipped += 1
                continue
            f.write(line + "\n")
            bytes_written += len(line.encode("utf-8")) + 1
            docs_written += 1
            n_chunk += 1
            if n_chunk >= 50_000:
                rate = docs_written / max(time.time() - t0, 1)
                print(f"[{args.lang}] {docs_written} docs, {bytes_written/1e9:.2f}GB "
                      f"({rate:.0f} docs/s, skipped {docs_skipped})", flush=True)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps({
                    "bytes": bytes_written, "docs": docs_written,
                    "skipped": docs_skipped, "done": False}))
                n_chunk = 0

    done = bytes_written >= args.max_bytes or docs_written >= args.max_docs
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "bytes": bytes_written, "docs": docs_written, "skipped": docs_skipped,
        "done": done}))
    print(f"\n[{args.lang}] DONE: {docs_written} docs, {bytes_written/1e9:.2f}GB, "
          f"skipped {docs_skipped} -> {out_path}")


if __name__ == "__main__":
    main()
