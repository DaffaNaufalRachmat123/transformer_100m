#!/usr/bin/env python3
"""fetch_parallel.py — Generalized parallel-corpus fetcher for any language pair.

Sources: Tatoeba (sentence-link join) + OPUS moses .txt.zip corpora.
Streams each OPUS zip once, applies a light writing-system filter, and writes
one 3-column TSV per source:  side_a \t side_b \t source_tag

Heavy quality filtering (LID, number-consistency, ratio, dedup) is a separate
stage: scripts/filter_parallel_v2.py.

Usage:
  python3 scripts/fetch_parallel.py --pair en-id
  python3 scripts/fetch_parallel.py --pair ja-id --sources tatoeba,wikimatrix
"""

import argparse
import bz2
import io
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language"
OPUS_BASE = "https://object.pouta.csc.fi"

# pair -> tatoeba lang codes (side_a, side_b) + link file stem candidates
TATOEBA = {
    "en-id": {"langs": ("eng", "ind"), "link_stems": ["ind-eng", "eng-ind"]},
    "ja-id": {"langs": ("jpn", "ind"), "link_stems": ["jpn-ind", "ind-jpn"]},
    "en-ja": {"langs": ("eng", "jpn"), "link_stems": ["jpn-eng", "eng-jpn"]},
}

# pair -> {source_name: (opus_relpath, keep_cap)}
OPUS = {
    "en-id": {
        "nllb":          ("OPUS-NLLB/v1/moses/en-id.txt.zip",            1_000_000),
        "opensubtitles": ("OPUS-OpenSubtitles/v2018/moses/en-id.txt.zip",   800_000),
        "qed":           ("OPUS-QED/v2.0a/moses/en-id.txt.zip",             250_000),
        "globalvoices":  ("OPUS-GlobalVoices/v2018q4/moses/en-id.txt.zip",   60_000),
        "wikimatrix":    ("OPUS-WikiMatrix/v1/moses/en-id.txt.zip",         200_000),
    },
    "ja-id": {
        "wikimatrix":    ("OPUS-WikiMatrix/v1/moses/ja-id.txt.zip",         200_000),
        "ccmatrix":      ("OPUS-CCMatrix/v1/moses/ja-id.txt.zip",           300_000),
    },
}


def has_ja(s):
    return any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'
               or '\u3000' <= c <= '\u303f' or '\uff00' <= c <= '\uffef' for c in s)


def bad_text(s):
    if any(c in s for c in ("\t", "\n", "\r", "\x00")):
        return True
    if re.search(r"%[0-9]|@action|(^|\s)\{\{", s):
        return True
    return False


def light_pair_ok(a, b, pair, min_len=3, max_len=140):
    a, b = a.strip(), b.strip()
    if not a or not b:
        return None
    if len(a) < min_len or len(b) < min_len:
        return None
    if len(a) > max_len or len(b) > max_len:
        return None
    if bad_text(a) or bad_text(b):
        return None
    if a.casefold() == b.casefold():
        return None
    if pair.startswith("ja") and not has_ja(a):
        return None
    if pair.endswith("ja") and not has_ja(b):
        return None
    if pair == "en-ja":
        if sum(1 for c in a if ord(c) < 128) / len(a) < 0.7:
            return None
    return (a, b)


def stream_bz2_lines(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "parallel-fetcher/2.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            text = bz2.decompress(data).decode("utf-8", errors="replace")
            return text.split("\n")
        except Exception as e:
            print(f"    [retry {attempt+1}] {e}")
            time.sleep(5 * (attempt + 1))
    return []


def load_tatoeba_sentences(lang):
    url = f"{TATOEBA_BASE}/{lang}/{lang}_sentences.tsv.bz2"
    print(f"  [tatoeba] loading {lang} sentences...")
    d = {}
    for line in stream_bz2_lines(url):
        p = line.split("\t")
        if len(p) >= 3:
            try:
                d[int(p[0])] = p[2].strip()
            except ValueError:
                pass
    print(f"  [tatoeba]   {lang}: {len(d)}")
    return d


def fetch_tatoeba(pair, out_path):
    cfg = TATOEBA[pair]
    la, lb = cfg["langs"]
    sents_a = load_tatoeba_sentences(la)
    sents_b = load_tatoeba_sentences(lb)
    pairs = set()
    for stem in cfg["link_stems"]:
        url = f"{TATOEBA_BASE}/{stem.split('-')[0]}/{stem}_links.tsv.bz2"
        print(f"  [tatoeba] links {stem}...")
        n0 = len(pairs)
        for line in stream_bz2_lines(url):
            p = line.split("\t")
            if len(p) < 2:
                continue
            try:
                x, y = int(p[0]), int(p[1])
            except ValueError:
                continue
            # try both orientations — the link file may store either order
            for (aid, bid) in ((x, y), (y, x)):
                sa, sb = sents_a.get(aid), sents_b.get(bid)
                if sa and sb:
                    pr = light_pair_ok(sa, sb, pair)
                    if pr:
                        pairs.add(pr)
        print(f"  [tatoeba]   +{len(pairs)-n0} from {stem}")
    with open(out_path, "w", encoding="utf-8") as f:
        for a, b in sorted(pairs):
            f.write(f"{a}\t{b}\ttatoeba\n")
    print(f"  [tatoeba] kept {len(pairs)} -> {out_path}")
    return len(pairs)


def fetch_opus(pair, name, rel, cap, out_path):
    url = f"{OPUS_BASE}/{rel}"
    la, lb = pair.split("-")
    print(f"  [opus:{name}] downloading {url}")
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "parallel-fetcher/2.0"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
            break
        except Exception as e:
            print(f"    [retry {attempt+1}] {e}")
            time.sleep(10 * (attempt + 1))
    else:
        print(f"  [opus:{name}] FAILED — skipped")
        return 0
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    a_name = [n for n in names if n.endswith(f".{la}")][0]
    b_name = [n for n in names if n.endswith(f".{lb}")][0]
    a_lines = zf.read(a_name).decode("utf-8", errors="replace").split("\n")
    b_lines = zf.read(b_name).decode("utf-8", errors="replace").split("\n")
    kept = 0
    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for a, b in zip(a_lines, b_lines):
            pr = light_pair_ok(a, b, pair)
            if not pr:
                continue
            key = (pr[0].casefold(), pr[1])
            if key in seen:
                continue
            seen.add(key)
            f.write(f"{pr[0]}\t{pr[1]}\t{name}\n")
            kept += 1
            if kept >= cap:
                break
    print(f"  [opus:{name}] kept {kept} -> {out_path}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(OPUS.keys()) + list(TATOEBA.keys()))
    ap.add_argument("--sources", default="all", help="comma list or 'all'")
    ap.add_argument("--out_dir", default="data/parallel_raw")
    args = ap.parse_args()

    pair = args.pair
    out_dir = ROOT / args.out_dir / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = None if args.sources == "all" else {s.strip() for s in args.sources.split(",")}

    if wanted is None or "tatoeba" in wanted:
        fetch_tatoeba(pair, out_dir / "tatoeba.tsv")
    for name, (rel, cap) in OPUS.get(pair, {}).items():
        if wanted is not None and name not in wanted:
            continue
        fetch_opus(pair, name, rel, cap, out_dir / f"{name}.tsv")
        time.sleep(3)
    print(f"\nDone fetching pair {pair}.")


if __name__ == "__main__":
    main()
