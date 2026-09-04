#!/usr/bin/env python3
"""clean_code_corpus.py — Clean + chunk the starcoder corpus for the pretrain
loader.

The pretrain loader reads one document per line with a ~1024-byte line cap,
so multi-line code files must be chunked into <=900-byte lines split at
newline boundaries (each chunk becomes one "document"; BOS/EOS separators
land between chunks, preserving local code structure).

Cleaning rules:
  - <reponame>...</reponame> markers become document boundaries
  - pathological lines dropped: >1000 chars (minified/data), high binary-byte
    ratio, control chars other than \n\t
  - repo-level cap: 20k lines per repo (truncated)

Usage:
  python3 scripts/clean_code_corpus.py \
      --in data/corpus_1b/code/starcoder_raw.txt \
      --out data/corpus_1b/code/starcoder_clean.txt
"""

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPONAME = re.compile(r"^<reponame>\s*(\S+)\s*$")
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def line_ok(line):
    if len(line) > 1000:
        return False
    if CTRL.search(line):
        return False
    # binary-ish: too many non-printable-ish bytes
    if line:
        weird = sum(1 for c in line if ord(c) > 0xFFFD or (ord(c) < 32 and c not in "\t"))
        if weird / len(line) > 0.05:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_bytes", type=int, default=900)
    ap.add_argument("--max_repo_lines", type=int, default=20_000)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    t0 = time.time()
    in_path = Path(args.inp)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"repos": 0, "lines_in": 0, "lines_dropped": 0, "chunks_out": 0,
             "bytes_out": 0}

    def flush(buf, fo):
        if not buf:
            return
        text = "\n".join(buf)
        # chunk at newline boundaries, <= chunk_bytes per output line
        cur = ""
        for l in buf:
            candidate = (cur + "\n" + l) if cur else l
            if len(candidate.encode("utf-8")) > args.chunk_bytes and cur:
                fo.write(cur + "\n")
                stats["chunks_out"] += 1
                stats["bytes_out"] += len(cur.encode("utf-8")) + 1
                cur = l
            else:
                cur = candidate
        if cur:
            fo.write(cur + "\n")
            stats["chunks_out"] += 1
            stats["bytes_out"] += len(cur.encode("utf-8")) + 1

    buf = []
    repo_lines = 0
    with open(in_path, encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fo:
        for line in fin:
            line = line.rstrip("\r\n")
            stats["lines_in"] += 1
            m = REPONAME.match(line)
            if m:
                flush(buf, fo)
                buf = []
                repo_lines = 0
                stats["repos"] += 1
                continue
            repo_lines += 1
            if repo_lines > args.max_repo_lines:
                continue
            if not line_ok(line):
                stats["lines_dropped"] += 1
                continue
            buf.append(line)
        flush(buf, fo)

    el = time.time() - t0
    print(f"{in_path.name}: {stats}")
    print(f"  -> {out_path} ({stats['bytes_out']/1e9:.2f}GB, {el:.0f}s)")
    if args.report:
        rp = ROOT / args.report
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"input": str(in_path), "output": str(out_path),
                                  "stats": stats, "seconds": round(el, 1)}, indent=2))


if __name__ == "__main__":
    main()
