#!/usr/bin/env python3
"""clean_corpus_v2.py — Heavy cleaner for Indonesian web corpora (mc4_id_extra,
CulturaX pulls) with cross-corpus exact-line dedup.

Adapted from clean_mc4_id.py filters + audit findings. One doc per line in,
one doc per line out.

Stages per line:
  1. length window (200..100k chars)
  2. spam/adult/gambling/SEO keyword scoring (hard drop on adult/gambling,
     soft cap on SEO density)
  3. non-Latin script ratio (Indonesian is Latin; CJK/Arabic/Thai junk out)
  4. URL domination, comment/timestamp-only fragments
  5. cross-corpus exact-line dedup (vs earlier-loaded corpora hashes)

Usage:
  python3 scripts/clean_corpus_v2.py --lang id \
      --in  "/path/to/mc4_id_extra.txt" \
      --out data/corpus_1b/indonesian/mc4_id_clean.txt \
      --dedup_vs data/corpus_pretrain_id/indonesian/culturax_id.txt,data/corpus_pretrain_id/indonesian/wiki_id.txt
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ADULT = re.compile(
    r"bugil|telanjang|porno|bokep|ngentot|sange|memek|kontol|bispak|lonte|"
    r"pelacur|jablay|colmek|crot|link\s*bocil|asianwiki|bokepid", re.I)
GAMBLING = re.compile(
    r"poker\s*(online|\d)|judi\s*(online| bola|slot)|slot\s*(gacor|deposit|online|dana)|"
    r"togel\s*(online|hk|sgp|hari)|bandar\s*(qq|togel|casino|judi)|situsslot|"
    r"capsa\s*susun|sicbo|roulette|baccarat|tembak\s*ikan|daftar\s*slot|"
    r"rtp\s*slot|bonus\s*member\s*baru\s*\d+", re.I)
SEO = re.compile(
    r"gratis\s*ongkir|cod\b|bayar\s*di\s*tempat|shopee|tokopedia|lazada|"
    r"bukalapak|blibli|harga\s*(murah|spesial)|promo\s*(terbatas|hari\s*ini)|"
    r"wa\s*[:/]\s*08|whatsapp\s*[:/]\s*08|hubungi\s*08\d", re.I)
NON_LATIN = re.compile(r"[^\x00-\x7F\u0100-\u024F\u2010-\u203A\u20AC]")
URL_RE = re.compile(r"https?://[^\s]+")
TIMESTAMP_ONLY = re.compile(r"^[\d\s:/\-.|,]+$")


def line_score_ok(text):
    n = len(text)
    if n < 200 or n > 100_000:
        return False
    if ADULT.search(text) or GAMBLING.search(text):
        return False
    # SEO density: >3 distinct e-commerce/spam signals -> junk promo page
    if len(set(SEO.findall(text))) > 3:
        return False
    # non-Latin ratio (CJK/Arabic/Thai junk)
    nl = len(NON_LATIN.findall(text))
    if nl / max(n, 1) > 0.10:
        return False
    # URL domination
    if len(URL_RE.findall(text)) > 10:
        return False
    if len(URL_RE.findall(text)) >= 3:
        url_chars = sum(len(m.group(0)) for m in URL_RE.finditer(text))
        if url_chars / n > 0.3:
            return False
    if TIMESTAMP_ONLY.match(text):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", required=True, choices=["id", "ja"])
    ap.add_argument("--dedup_vs", default="",
                    help="comma-separated earlier corpora to exact-dedupe against")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    t0 = time.time()
    # load dedup hashes
    seen = set()
    for vs in filter(None, args.dedup_vs.split(",")):
        p = Path(vs)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"WARN dedup source missing: {p}")
            continue
        n = 0
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                h = hashlib.md5(line.strip().casefold().encode("utf-8", errors="ignore")).digest()
                seen.add(h)
                n += 1
        print(f"dedup base: {p.name} -> {n} lines ({len(seen)} hashes)")

    stats = {"in": 0, "kept": 0, "len": 0, "spam": 0, "script": 0, "url": 0,
             "dup": 0}
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.inp)
    with open(in_path, encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fo:
        for line in fin:
            text = line.rstrip("\n")
            stats["in"] += 1
            n = len(text)
            if n < 200 or n > 100_000:
                stats["len"] += 1
                continue
            if ADULT.search(text) or GAMBLING.search(text) or \
               len(set(SEO.findall(text))) > 3:
                stats["spam"] += 1
                continue
            if args.lang == "id":
                # Indonesian is Latin-script: CJK/Arabic/Thai junk out
                if len(NON_LATIN.findall(text)) / max(n, 1) > 0.10:
                    stats["script"] += 1
                    continue
            elif args.lang == "ja":
                # Japanese must be dominated by kana/kanji; Latin-heavy docs
                # are misfiled/mojibake
                latin = len(re.findall(r"[A-Za-z]", text))
                if latin / max(n, 1) > 0.5:
                    stats["script"] += 1
                    continue
            urls = URL_RE.findall(text)
            if len(urls) > 10 or (len(urls) >= 3 and
                                  sum(len(u) for u in urls) / n > 0.3) or \
               TIMESTAMP_ONLY.match(text):
                stats["url"] += 1
                continue
            h = hashlib.md5(text.strip().casefold().encode("utf-8", errors="ignore")).digest()
            if h in seen:
                stats["dup"] += 1
                continue
            seen.add(h)
            fo.write(text + "\n")
            stats["kept"] += 1
            if stats["in"] % 500_000 == 0:
                el = time.time() - t0
                print(f"  {stats['in']} in, {stats['kept']} kept "
                      f"({stats['kept']/max(stats['in'],1)*100:.1f}%), {el:.0f}s", flush=True)

    el = time.time() - t0
    print(f"\n{in_path.name}: {stats}")
    print(f"  -> {out_path} ({el:.0f}s)")
    if args.report:
        rp = ROOT / args.report
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps({"input": str(in_path), "output": str(out_path),
                                  "stats": stats, "seconds": round(el, 1)}, indent=2))


if __name__ == "__main__":
    main()
