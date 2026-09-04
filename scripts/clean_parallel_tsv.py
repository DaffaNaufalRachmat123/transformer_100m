#!/usr/bin/env python3
"""clean_parallel_tsv.py — Apply filter_parallel_v2 quality rules to an
existing 2-column parallel TSV (no raw source files needed).

Used for data/parallel_enja.tsv (1.96M pairs fetched before v2.1 filters
existed): junk patterns, LID, number-consistency, length ratio, dedup.

Usage:
  python3 scripts/clean_parallel_tsv.py --in data/parallel_enja.tsv \
      --out data/parallel_en_ja.tsv --pair en-ja
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_parallel_v2 import (MOJIBAKE, LONG_TOKEN, LONG_TOKEN_OK, PLACEHOLDER,
                                CONTROL_CODE, RATIO_BOUNDS, SMALL_NUM, FT_LANG)
import re
import html


def numbers_ok(a, b):
    def get_numbers(s):
        out = set()
        for m in re.finditer(r"\d[\d,\.]*", s):
            raw = m.group(0).replace(",", "").rstrip(".")
            if not raw:
                continue
            try:
                out.add(int(float(raw)))
            except ValueError:
                pass
        return out
    for n in get_numbers(a):
        if n not in SMALL_NUM and n not in get_numbers(b):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", required=True, choices=["en-ja", "ja-en", "en-id", "id-en", "ja-id", "id-ja"])
    ap.add_argument("--lid_model", default="data/lid.176.bin")
    ap.add_argument("--min_score", type=float, default=0.70)
    ap.add_argument("--light", action="store_true",
                    help="junk/mojibake/html/dedup only — keep LID/ratio/number "
                         "checks off (for corpora already pre-filtered by their "
                         "fetcher, e.g. subtitle pairs with legitimately wild "
                         "length ratios)")
    args = ap.parse_args()

    la, lb = args.pair.split("-")
    lo, hi = RATIO_BOUNDS.get(args.pair, (0.4, 2.5))

    ft = None
    try:
        import fasttext
        ft = fasttext.load_model(str(Path(args.lid_model)))
    except Exception as e:
        print(f"WARN: no LID model ({e}); script-heuristic only")

    def ft_probs(text, k=3):
        try:
            labels, probs = ft.predict(text, k=k)
            return {l.replace("__label__", ""): s for l, s in zip(labels, probs)}
        except ValueError:
            pass
        try:
            res = ft.f.predict(text, k, 0.0, "strict")
            return {l.replace("__label__", ""): p for p, l in res}
        except Exception:
            return {}

    def lid_ok(text, want):
        if want == "ja":
            return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text)
        if ft is None:
            return True
        best = ft_probs(text.replace("\n", " ").strip())
        if want == "id":
            return (best.get("id", 0.0) + best.get("ms", 0.0)) >= 0.55 or best.get("id", 0.0) >= args.min_score
        return best.get(want, 0.0) >= args.min_score

    def text_junk(a, b):
        for s in (a, b):
            if CONTROL_CODE.search(s) or PLACEHOLDER.search(s) or MOJIBAKE.search(s):
                return True
            for m in LONG_TOKEN.finditer(s):
                if m.group(0).lower() not in LONG_TOKEN_OK:
                    return True
        return False

    in_path = Path(args.inp)
    out_path = Path(args.out)
    n_in = n_junk = n_lid = n_num = n_ratio = n_dup = n_kept = 0
    seen = set()
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fo:
        for line in fin:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            a, b = p[0].strip(), p[1].strip()
            n_in += 1
            if "&" in a or "&" in b:
                a, b = html.unescape(a), html.unescape(b)
            if text_junk(a, b):
                n_junk += 1
                continue
            if not args.light:
                if not lid_ok(a, la) or not lid_ok(b, lb):
                    n_lid += 1
                    continue
                if not numbers_ok(a, b):
                    n_num += 1
                    continue
                ra = len(a) / max(len(b), 1)
                if not (lo <= ra <= hi):
                    n_ratio += 1
                    continue
            key = (a.casefold(), b)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            fo.write(f"{a}\t{b}\n")
            n_kept += 1

    print(f"{in_path.name}: in={n_in} junk={n_junk} lid={n_lid} num={n_num} "
          f"ratio={n_ratio} dup={n_dup} -> kept {n_kept} ({100*n_kept/max(n_in,1):.1f}%) -> {out_path}")


if __name__ == "__main__":
    main()
