#!/usr/bin/env python3
"""fetch_nllb_enid.py — Download + filter OPUS-NLLB en-id (5.9GB zip).

The zip is too large to read into RAM. Strategy:
  1. resumable download to disk (curl -C -)
  2. stream-decompress the two side files from the zip via zipfile
     (disk-backed, bounded memory)
  3. light filter + cap -> data/parallel_raw/en-id/nllb.tsv (3-col)
  4. delete the zip

Usage:
  python3 scripts/fetch_nllb_enid.py --cap 2000000
"""

import argparse
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_parallel import light_pair_ok

ROOT = Path(__file__).resolve().parent.parent

URL = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-id.txt.zip"
ZIP_PATH = ROOT / "data" / "parallel_raw" / "en-id" / "nllb.zip"
OUT_PATH = ROOT / "data" / "parallel_raw" / "en-id" / "nllb.tsv"


def download():
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size >= 5_933_570_516:
        print("[dl] zip already complete")
        return True
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[dl] downloading (resumable)...", flush=True)
    rc = subprocess.call(["curl", "-sL", "-C", "-", "--retry", "10",
                          "--retry-delay", "5", "-o", str(ZIP_PATH), URL])
    return rc == 0


def extract(cap):
    print("[extract] streaming zip members...", flush=True)
    zf = zipfile.ZipFile(ZIP_PATH)
    names = zf.namelist()
    en_name = [n for n in names if n.endswith(".en")][0]
    id_name = [n for n in names if n.endswith(".id")][0]
    # stream both members in lockstep, chunked
    en_stream = io.TextIOWrapper(zf.open(en_name), encoding="utf-8", errors="replace")
    id_stream = io.TextIOWrapper(zf.open(id_name), encoding="utf-8", errors="replace")

    kept = 0
    t0 = time.time()
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        while kept < cap:
            a = en_stream.readline()
            b = id_stream.readline()
            if not a or not b:
                break
            pr = light_pair_ok(a, b, "en-id")
            if not pr:
                continue
            out.write(f"{pr[0]}\t{pr[1]}\tnllb\n")
            kept += 1
            if kept % 500_000 == 0:
                rate = kept / max(time.time() - t0, 1)
                print(f"  {kept} pairs ({rate:.0f}/s)", flush=True)
    en_stream.close()
    id_stream.close()
    zf.close()
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2_000_000)
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()

    if not download():
        print("[dl] FAILED")
        sys.exit(1)

    kept = extract(args.cap)
    print(f"[done] {kept} pairs -> {OUT_PATH}")

    if not args.keep_zip:
        ZIP_PATH.unlink()
        print(f"[cleanup] zip deleted (freed ~5.9GB)")


if __name__ == "__main__":
    main()
