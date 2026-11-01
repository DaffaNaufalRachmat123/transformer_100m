#!/usr/bin/env python3
"""measure_bytes_per_token.py — re-measure BYTES_PER_TOKEN for a tokenizer.

The balance report and the Chinchilla epoch estimate both divide bucket bytes by
a per-bucket bytes/token constant. Those constants are tokenizer-specific: the
vocab_7 -> vocab_8 change moved Japanese by +22%, and splitting digits into one
token each moves every bucket again. Stale constants silently misreport how many
epochs the run will do.

  python3 scripts/measure_bytes_per_token.py --vocab_dir vocab_9
  python3 scripts/measure_bytes_per_token.py --vocab_dir vocab_9 --write
"""

import argparse
import glob
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = ("english", "japanese", "code", "math", "mixed")
TARGETS = ("scripts/build_pretrain_corpus.py", "scripts/preflight_pretrain.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_dir", default="vocab_9")
    ap.add_argument("--corpus", default="data/corpus_pretrain_v8")
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--samples", type=int, default=140)
    ap.add_argument("--size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--write", action="store_true",
                    help="patch BYTES_PER_TOKEN in the builder and preflight")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    import bpe_reference as BR
    cls = [getattr(BR, n) for n in dir(BR)
           if n.lower().startswith("bpe") and isinstance(getattr(BR, n), type)][0]
    tok = cls(str(ROOT / args.vocab_dir))

    rng = random.Random(args.seed)
    rates = {}
    for b in BUCKETS:
        fs = sorted(glob.glob(str(ROOT / args.corpus / b / "*.txt")))
        if not fs:
            print(f"  {b:9} no files — skipped")
            continue
        rng.shuffle(fs)
        nb = nt = 0
        for f in fs[:args.files]:
            sz = os.path.getsize(f)
            with open(f, "rb") as fh:
                for _ in range(args.samples):
                    fh.seek(rng.randrange(0, max(1, sz - args.size)))
                    fh.readline()
                    try:
                        s = fh.read(args.size).decode("utf-8")
                    except Exception:
                        continue
                    nb += len(s.encode())
                    nt += len(tok.encode(s))
        rates[b] = nb / max(nt, 1)
        print(f"  {b:9} {rates[b]:.2f} bytes/token")

    if not args.write:
        print("\n(dry run — pass --write to patch the scripts)")
        return 0

    lit = ("BYTES_PER_TOKEN = {" +
           ", ".join(f'"{b}": {rates[b]:.2f}' for b in BUCKETS if b in rates) + "}")
    pat = re.compile(r"BYTES_PER_TOKEN = \{[^}]*\}", re.S)
    for rel in TARGETS:
        p = ROOT / rel
        src = p.read_text(encoding="utf-8")
        if not pat.search(src):
            print(f"  [WARN] no BYTES_PER_TOKEN in {rel}")
            continue
        p.write_text(pat.sub(lit, src, count=1), encoding="utf-8")
        print(f"  patched {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
