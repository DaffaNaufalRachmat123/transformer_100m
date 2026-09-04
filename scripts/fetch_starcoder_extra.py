#!/usr/bin/env python3
"""fetch_starcoder_extra.py — Stream additional curated Python from
bigcode/starcoderdata for the 1B code bucket.

Filters (quality-focused, loader-compatible output):
  - Python only (lang == 'Python')
  - 20 .. 512 lines (teachable size, not minified dumps)
  - has a docstring or >=3 comments (readable code)
  - median line length < 120 (no generated data blobs)
  - no extremely long lines (>2000 chars)

Output: data/corpus_1b/code/starcoder_extra_clean.txt
(chunked to <=900-byte lines by clean_code_corpus.py conventions)

Usage:
  python3 scripts/fetch_starcoder_extra.py --max_bytes 2684354560
"""

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LONG_LINE = re.compile(r"^.{2000,}")


def code_ok(text):
    if not text:
        return False
    lines = text.split("\n")
    n = len(lines)
    if not (20 <= n <= 512):
        return False
    # docstring or comments
    has_doc = '"""' in text or "'''" in text
    n_comments = sum(1 for l in lines if l.strip().startswith("#"))
    if not has_doc and n_comments < 3:
        return False
    # median line length
    lens = sorted(len(l) for l in lines)
    if lens[n // 2] > 120:
        return False
    for l in lines:
        if len(l) > 2000:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_bytes", type=int, default=2_684_354_560)
    ap.add_argument("--out", default="data/corpus_1b/code/starcoder_extra_clean.txt")
    args = ap.parse_args()

    from datasets import load_dataset

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_path = ROOT / "logs" / "starcoder_extra_state.json"
    written = skipped = docs = 0
    if state_path.exists():
        st = json.loads(state_path.read_text())
        if st.get("done"):
            print(f"done already ({st.get('bytes')} bytes)")
            return
        written = st.get("bytes", 0)
        docs = st.get("docs", 0)
        skipped = st.get("skipped", 0)

    print("[starcoder] streaming bigcode/starcoderdata (python)...")
    ds = load_dataset("bigcode/starcoderdata", data_files=["python/*.parquet"],
                      split="train", streaming=True)

    t0 = time.time()
    n_chunk = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for rec in ds:
            if written >= args.max_bytes:
                break
            docs += 1
            text = rec.get("content") or ""
            if not code_ok(text):
                skipped += 1
                continue
            # one document per loader line: join code with \n -> single line?
            # Loader line cap is ~1024B — chunk like clean_code_corpus:
            for chunk_start in range(0, len(text), 850):
                chunk = text[chunk_start:chunk_start + 850].replace("\n", "⏎")
                f.write(chunk + "\n")
                written += len(chunk.encode("utf-8")) + 1
            n_chunk += 1
            if n_chunk % 10_000 == 0:
                rate = docs / max(time.time() - t0, 1)
                print(f"  {docs} docs, {written/1e9:.2f}GB, skipped {skipped} "
                      f"({rate:.0f} docs/s)", flush=True)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps({
                    "bytes": written, "docs": docs, "skipped": skipped,
                    "done": written >= args.max_bytes}))

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "bytes": written, "docs": docs, "skipped": skipped, "done": True}))
    print(f"\nDONE: {written/1e9:.2f}GB from {docs} docs "
          f"(skipped {skipped}) -> {out_path}")


if __name__ == "__main__":
    main()
