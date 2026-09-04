#!/usr/bin/env python3
"""filter_parallel_v2.py — Heavy quality filter for parallel corpora (v2).

Pipeline per pair (reads data/parallel_raw/{pair}/*.tsv, 3-col: a \\t b \\t src):
  1. fastText LID (lid.176.bin) on both sides — hard filter
  2. number-consistency: numbers >= 13 in source must appear in target
  3. adaptive length ratio (per pair)
  4. global dedup (casefolded)
  5. source tags preserved + per-source pass-rate stats

Output:
  data/parallel_{pair}.tsv          (2-col: a \\t b)  — builder-compatible
  logs/parallel_filter_{pair}.json  (stats per source)

Usage:
  python3 scripts/filter_parallel_v2.py --pair en-id
  python3 scripts/filter_parallel_v2.py --pair ja-id
"""

import argparse
import glob
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# fastText language codes -> our codes
FT_LANG = {"en": "en", "id": "id", "ja": "ja", "eng": "en", "ind": "id", "jpn": "ja"}

# Length-ratio bounds per pair (chars, side_a / side_b)
RATIO_BOUNDS = {
    "en-id": (0.45, 2.2),
    "ja-id": (0.30, 3.2),
    "en-ja": (0.35, 2.8),
}

SMALL_NUM = set(range(0, 13))  # numbers below 13 may legitimately differ

# ── v2.1 junk filters (audit findings) ─────────────────────────────────────
# Missing-space junk: long alphabetic token without hyphens/digits
# ("Kenapakautidakmenjawab?") — real words are never this long.
LONG_TOKEN = re.compile(r"[A-Za-z]{20,}")
# legitimate long technical words that must not trip the filter
LONG_TOKEN_OK = {
    "internationalization", "interdisciplinary", "unconstitutional",
    "misunderstandings", "counterrevolutionary", "characteristically",
    "uncompromisingly", "internationalisation", "responsibilities",
    "ketidakbertanggungjawaban", "memperhitungkan", "mempertanggungjawabkan",
}
# Subtitle control codes / karaoke tags: {Y: i}, (\ I1), [ flashback ]
CONTROL_CODE = re.compile(r"\{[A-Za-z]{1,4}:\s*[^}]*\}|\(\\\s*\w+\)|</?[a-z]+>")
# Placeholder junk from subtitle sources
PLACEHOLDER = re.compile(r"\*{3,}|_{5,}|%[sd]")
# Mojibake markers: double-encoded UTF-8 (Ã©, Ã¨, Â , â€™, Ã¶, ...)
MOJIBAKE = re.compile(r"Ã[©¨¤¶®«»°¼½¾¢]|Â[\xa0°]|â€[˜™œ›""]|\bÃ\b")


def text_junk(a, b):
    """Return True if either side carries audit-found junk patterns."""
    for s in (a, b):
        if CONTROL_CODE.search(s):
            return True
        if PLACEHOLDER.search(s):
            return True
        if MOJIBAKE.search(s):
            return True
        for m in LONG_TOKEN.finditer(s):
            if m.group(0).lower() not in LONG_TOKEN_OK:
                return True
    return False


def get_numbers(s):
    """All integer values appearing in the text (commas stripped)."""
    out = set()
    for m in re.finditer(r"\d[\d,\.]*", s):
        raw = m.group(0).replace(",", "")
        # avoid trailing punctuation like periods at sentence end
        raw = raw.rstrip(".")
        if not raw:
            continue
        try:
            out.add(int(float(raw)))
        except ValueError:
            continue
    return out


def numbers_ok(a, b):
    na, nb = get_numbers(a), get_numbers(b)
    for n in na:
        if n in SMALL_NUM:
            continue
        if n not in nb:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--raw_dir", default="data/parallel_raw")
    ap.add_argument("--lid_model", default="data/lid.176.bin")
    ap.add_argument("--min_score", type=float, default=0.70,
                    help="fastText probability threshold per side")
    ap.add_argument("--max_rows", type=int, default=2_500_000)
    args = ap.parse_args()

    pair = args.pair
    la, lb = pair.split("-")
    raw_files = sorted(glob.glob(str(ROOT / args.raw_dir / pair / "*.tsv")))
    if not raw_files:
        print(f"No raw files for {pair}")
        return

    # ── fastText LID
    import fasttext
    model_path = ROOT / args.lid_model
    ft = None
    if model_path.exists():
        ft = fasttext.load_model(str(model_path))
        print(f"LID model loaded: {model_path}")
    else:
        print("WARN: lid.176.bin not found — falling back to script-heuristic LID")

    def ft_probs(text, k=3):
        """fastText label->prob dict, robust to numpy-2.x wrapper breakage."""
        try:
            labels, probs = ft.predict(text, k=k)
            return {l.replace("__label__", ""): s for l, s in zip(labels, probs)}
        except ValueError:
            pass
        try:
            res = ft.f.predict(text, k, 0.0, "strict")
            d = {}
            for p, l in res:  # list of (prob, label)
                d[l.replace("__label__", "")] = p
            return d
        except Exception:
            return {}

    def lid_ok(text, want):
        if want == "ja":
            # script check is cheaper and more reliable than LID for Japanese
            return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'
                       for c in text)
        if ft is not None:
            text = text.replace("\n", " ").strip()
            best = ft_probs(text)
            if want == "id":
                # fastText confuses id/ms heavily (they are near-identical);
                # sources are genuine *-id corpora, so accept id+ms mass.
                return (best.get("id", 0.0) + best.get("ms", 0.0)) >= 0.55 \
                    or best.get("id", 0.0) >= args.min_score
            return best.get(want, 0.0) >= args.min_score
        # fallback (no model): Latin langs pass — ratios filter later
        return True

    lo_ratio, hi_ratio = RATIO_BOUNDS.get(pair, (0.4, 2.5))

    stats = Counter()
    kept_by_src = Counter()
    seen = set()
    out_rows = []

    for path in raw_files:
        src = Path(path).stem
        n_in = n_lid = n_num = n_ratio = n_dup = n_junk = 0
        for line in open(path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            a, b = p[0].strip(), p[1].strip()
            n_in += 1
            # v2.1: decode HTML entities first so &amp;-style rows get a fair
            # chance at passing the remaining checks
            if "&" in a or "&" in b:
                a, b = html.unescape(a), html.unescape(b)
            if text_junk(a, b):
                n_junk += 1
                continue
            if not lid_ok(a, la) or not lid_ok(b, lb):
                n_lid += 1
                continue
            if not numbers_ok(a, b):
                n_num += 1
                continue
            ra = len(a) / max(len(b), 1)
            if not (lo_ratio <= ra <= hi_ratio):
                n_ratio += 1
                continue
            key = (a.casefold(), b)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            out_rows.append((a, b, src))
            kept_by_src[src] += 1
        stats[f"{src}:in"] = n_in
        stats[f"{src}:junk_rej"] = n_junk
        stats[f"{src}:lid_rej"] = n_lid
        stats[f"{src}:num_rej"] = n_num
        stats[f"{src}:ratio_rej"] = n_ratio
        stats[f"{src}:dup_rej"] = n_dup
        stats[f"{src}:kept"] = kept_by_src[src]
        print(f"[{src}] in={n_in} junk={n_junk} lid_rej={n_lid} num_rej={n_num} "
              f"ratio_rej={n_ratio} dup={n_dup} kept={kept_by_src[src]}")

    if len(out_rows) > args.max_rows:
        import random
        rng = random.Random(42)
        rng.shuffle(out_rows)
        out_rows = out_rows[: args.max_rows]

    out_path = ROOT / f"data/parallel_{pair.replace('-', '_')}.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        for a, b, _src in out_rows:
            f.write(f"{a}\t{b}\n")

    rep = {
        "pair": pair,
        "total_kept": len(out_rows),
        "by_source": {s: kept_by_src[s] for s in sorted(kept_by_src)},
        "stats": dict(stats),
    }
    rep_path = ROOT / "logs" / f"parallel_filter_{pair.replace('-', '_')}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print(f"\nDone: {len(out_rows)} clean {pair} pairs -> {out_path}")
    print(f"  by source: {dict(kept_by_src)}")


if __name__ == "__main__":
    main()
