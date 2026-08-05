#!/usr/bin/env python3
"""
clean_corpus.py — Deep-cleaning pipeline for data/corpus_final (EN + JA).

Reads one-document-per-line text files, applies configurable cleaning stages,
and writes cleaned copies to an output directory. Originals are preserved.

Design notes
------------
* The C++ trainer (pretrain_dataset.cpp) reads one document per line and
  truncates each line to 1024 bytes. Long concatenated lines therefore lose
  content silently, so we split mC4 JA multi-documents into one-doc-per-line
  and keep lines reasonably bounded.
* Streaming, memory-bounded; MinHash/LSH dedup uses numpy and is bounded by
  per-file signature arrays.

Usage
-----
    python3 scripts/clean_corpus.py --all
    python3 scripts/clean_corpus.py --split-mc4 --outdir data/corpus_final/cleaned
    python3 scripts/clean_corpus.py --no-dedup --no-safety
Config flags (CLI):
    --all            run the full recommended pipeline
    --split-mc4      split mC4 JA multi-documents + strip blog metadata
    --dedup / --no-dedup        enable/disable near-duplicate removal (default on)
    --safety / --no-safety      enable/disable content-safety spam/adult filters (default OFF)
    --pii / --no-pii            scrub emails/phones/IDs (default OFF)
    --min-len N      drop docs shorter than N chars (default 80)
    --max-len N      drop docs longer than N chars (default 0 = off)
    --outdir DIR     output directory (default data/corpus_final/cleaned)
    --keep-originals keep originals (default: only write cleaned copies)
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from hashlib import md5

import numpy as np
import zlib

SRC = os.path.join("data", "corpus_final")
FILES = {
    "EN": [
        "english/fineweb_edu.txt",
        "english/wiki_en.txt",
        "english/openwebtext.txt",
    ],
    "JA": [
        "japanese/mc4_ja.txt",
        "japanese/wiki_ja.txt",
    ],
}

# ---------------------------------------------------------------------------
# Unicode hygiene
# ---------------------------------------------------------------------------
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
REPL_RE = re.compile(r"\ufffd")
HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);")
URL_RE = re.compile(r"https?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+|www\.[\w\-._~:/?#@!$&'()*+,;=%]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:0\d{1,4}[-－]?\d{1,4}[-－]?\d{3,4}|(?:\+?\d{1,3}[-.\s]?)?\d{2,4}[-.\s]\d{2,4}[-.\s]\d{3,4})(?!\d)")
WS_RE = re.compile(r"[ \t]+")
REPEAT_RE = re.compile(r"(.)\1{7,}")  # 8+ identical chars in a row
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
HIRAGANA = re.compile(r"[\u3040-\u309f]")
KATAKANA = re.compile(r"[\u30a0-\u30ff]")
KANJI = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# ---- mC4 JA blog metadata / boilerplate markers ---------------------------
JA_BLOG_TAGS = ["エキサイトブログ", "タグまとめ", "アメーバブログ", "ライブドアブログ",
                "はてなブログ", "レビュー対象商品", "ベストレビュワーランキング",
                "スポンサーサイト", "ブログトップ", "この記事へのトラックバック"]
JA_DATE = re.compile(r"(?P<date>\d{4}/\d{1,2}/\d{1,2}(?:[ \t]*\d{1,2}:\d{2})?)")
# date followed by an author/title dash acts as a document boundary
JA_BOUNDARY = re.compile(r"\d{4}/\d{1,2}/\d{1,2}(?:[ \t]*\d{1,2}:\d{2})?[ \t]*-[ \t]")

# ---- wiki boilerplate ------------------------------------------------------
WIKI_CITE = re.compile(r"\[(?:要出典|要検証|要説明|独自研究|リンク切れ|要更新|要改名|要加筆)\]")
WIKI_NAV = re.compile(r"^(?:カテゴリ|Category|Template|テンプレート|外部リンク|関連項目|参考文献|脚注|注釈|出典)")
WIKI_EN_NAV = re.compile(r"^(?:Category|Template|References|External links|See also|Further reading|Notes|Citations):")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nfc(s):
    return unicodedata.normalize("NFC", s)


def clean_unicode(text):
    text = nfc(text)
    text = CONTROL_RE.sub("", text)
    text = INVISIBLE_RE.sub("", text)
    text = REPL_RE.sub("", text)
    return text


def strip_html(text):
    text = HTML_ENTITY_RE.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def collapse_repeats(text):
    return REPEAT_RE.sub(lambda m: m.group(1) * 3, text)


def compress_ws(text):
    text = re.sub(r"[ \t\r\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_gibberish(text):
    """Low-entropy / over-repetitive lines."""
    if len(text) < 40:
        return False
    uniq = len(set(text))
    if uniq == 0:
        return True
    if uniq / len(text) < 0.05:
        return True
    # repeated-character runs
    if REPEAT_RE.search(text):
        return True
    # single char accounting for >60% (e.g. "aaaa...")
    from collections import Counter as _C
    top = _C(text).most_common(1)[0][1]
    if top / len(text) > 0.6:
        return True
    return False


def lang_ok(text, lang):
    if lang == "EN":
        # keep docs that are mostly latin; allow embedded CJK glosses/terms.
        if not re.search(r"[A-Za-z]", text):
            return False
        letters = sum(1 for c in text if c.isalpha())
        if letters == 0:
            return False
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        if latin / letters < 0.5:
            return False
        return True
    else:  # JA
        if not CJK.search(text):
            # JA-ish text may sometimes be pure kana-in-latin... require some CJK
            return False
        return True


def should_filter_safety(text, lang):
    """Content-safety (spam/adult) filters. Best-effort; low precision intent."""
    t = text if lang == "JA" else text.lower()
    if lang == "JA":
        spam = sum(1 for k in SPAM_JA if k in t)
        if spam >= 3:
            return True
        return False
    else:
        spam = sum(1 for k in SPAM_EN if k in t)
        if spam >= 4:
            return True
        return False


SPAM_EN = ["buy now", "click here", "limited offer", "discount code", "free trial",
           "order now", "best price", "subscribe now", "risk free", "make money",
           "wire transfer", "act now", "satisfaction guaranteed", "100% free",
           "congratulations you", "you have been selected"]
SPAM_JA = ["送料無料", "最安値", "激安", "購入はこちら", "今すぐ購入", "カートに入れる",
           "即日発送", "限定品", "品質保証", "全国無料", "ご応募お待ち"]


def scrub_pii(text):
    text = EMAIL_RE.sub("<email>", text)
    # phone: only mask when it looks like a standalone phone, not a long digit run
    text = PHONE_RE.sub("<phone>", text)
    return text


# ---------------------------------------------------------------------------
# mC4 JA document splitter
# ---------------------------------------------------------------------------
def split_mc4_ja(line):
    """Split a concatenated mC4 JA line into a list of documents.

    Heuristic doc boundaries:
      1. A date (YYYY/MM/DD [HH:MM]) followed by ' - ' (blog post header).
      2. A date followed by a space and a short title (review/entry header).
    Returns a list of (title_prefix, body) strings; trims trailing metadata.
    """
    text = line.strip()
    # find boundary positions
    bounds = [m.start() for m in JA_BOUNDARY.finditer(text)]
    if not bounds:
        return [text] if len(text) >= 20 else []

    docs = []
    cuts = bounds + [len(text)]
    for i in range(len(bounds)):
        seg = text[bounds[i]:cut] if (cut := cuts[i + 1]) else text[bounds[i]:]
        seg = seg.strip()
        if len(seg) >= 20:
            docs.append(seg)
    return docs


def strip_ja_metadata(doc):
    """Remove inline blog metadata / boilerplate from a JA doc."""
    for tag in JA_BLOG_TAGS:
        doc = doc.replace(tag, " ")
    # strip a leading date+'-' header
    doc = JA_BOUNDARY.sub(" ", doc, count=1)
    # drop trailing date-only crumbs
    doc = JA_DATE.sub(" ", doc)
    doc = compress_ws(doc)
    return doc


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def process_line(line, lang, cfg, stats=None):
    """Return None to drop, or a cleaned string. `stats` is a Counter for audit."""
    def _drop(reason):
        if stats is not None:
            stats["drop_" + reason] += 1
        return None

    text = line.rstrip("\n\r")
    if not text:
        return _drop("empty")
    text = clean_unicode(text)
    if not text.strip():
        return _drop("blank_after_unicode")

    # lang gate
    if cfg.get("lang_check") and not lang_ok(text, lang):
        return _drop("lang")

    # length gates
    if cfg.get("min_len") and len(text) < cfg["min_len"]:
        return _drop("short")
    if cfg.get("max_len") and len(text) > cfg["max_len"]:
        return _drop("long")

    # html / urls
    if cfg.get("strip_html"):
        text = strip_html(text)
    if cfg.get("strip_urls") and lang == "EN":
        text = URL_RE.sub(" ", text)

    # wiki boilerplate
    if cfg.get("wiki_boilerplate") and "wiki" in cfg.get("is_wiki", ""):
        text = WIKI_CITE.sub("", text)
        text = WIKI_NAV.sub("", text)
        text = WIKI_EN_NAV.sub("", text)

    # emoji / repeats
    if cfg.get("collapse_repeats"):
        text = collapse_repeats(text)

    # safety
    if cfg.get("safety") and should_filter_safety(text, lang):
        return _drop("safety")

    # gibberish
    if cfg.get("gibberish") and is_gibberish(text):
        return _drop("gibberish")

    # pii
    if cfg.get("pii"):
        text = scrub_pii(text)

    text = compress_ws(text)
    if len(text) < (cfg.get("min_len") or 1):
        return _drop("short_after_clean")
    return text


def run_file(path, lang, cfg, report):
    """Process a single file. Returns list of cleaned docs (dedup applied)."""
    print(f"[{lang}] reading {path}", flush=True)
    docs = []
    stats = Counter()
    n_in = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_in += 1
            if cfg.get("split_mc4") and "mc4" in path:
                subs = split_mc4_ja(line)
                for s in subs:
                    s = strip_ja_metadata(s)
                    c = process_line(s if s else s, lang, {**cfg, "is_wiki": ""}, stats)
                    if c:
                        docs.append(c)
                        stats["kept"] += 1
                    else:
                        stats["dropped"] += 1
                continue
            c = process_line(line, lang, cfg, stats)
            if c:
                docs.append(c)
                stats["kept"] += 1
            else:
                stats["dropped"] += 1

    report[path] = {"in_lines": n_in, **dict(stats)}
    return docs


# ---------------------------------------------------------------------------
# Near-duplicate removal
# ---------------------------------------------------------------------------
def dedup_streaming(docs, prefix_len=80, exact=True):
    """Streaming near-dup removal.

    * exact: drop exact duplicates (hash of whole doc).
    * prefix: drop docs whose normalized first `prefix_len` chars already seen
      (catches boilerplate / truncated / near-identical leading fragments).

    O(N) memory-bounded, suitable for multi-million-doc files.
    """
    if len(docs) < 2:
        return docs, 0
    seen_exact = set()
    seen_prefix = set()
    out = []
    removed = 0
    for d in docs:
        h = d.encode("utf-8")
        if exact:
            if h in seen_exact:
                removed += 1
                continue
            seen_exact.add(h)
        prefix = h[:prefix_len]
        if prefix in seen_prefix:
            removed += 1
            continue
        seen_prefix.add(prefix)
        out.append(d)
    return out, removed


def _shingle_hashes(text, n, a, b, mod):
    """Vectorized minhash via fast crc32 shingle hashing."""
    if len(text) < 5:
        text = text + " " * (5 - len(text))
    N = len(text) - 4
    g = np.empty(N, dtype=np.uint32)
    for i in range(N):
        g[i] = zlib.crc32(text[i:i + 5].encode("utf-8")) & 0xffffffff
    if N == 0:
        return np.zeros(n, dtype=np.uint32)
    gu = g.astype(np.uint64)
    h = np.empty((n, N), dtype=np.uint64)
    av = np.asarray(a, dtype=np.uint64)
    bv = np.asarray(b, dtype=np.uint64)
    for i in range(n):
        h[i] = (av[i] * gu + bv[i]) % mod
    return h.min(axis=1).astype(np.uint32)


def near_dedup_minhash(docs, threshold=0.75, n=24, bands=8):
    """MinHash LSH near-dup removal. Slower; use only on smaller files."""
    if len(docs) < 2:
        return docs, 0
    rng = np.random.RandomState(42)
    mod = np.uint64((1 << 32) - 5)
    a = rng.randint(1, 2 ** 30, size=n)
    b = rng.randint(0, 2 ** 30, size=n)
    rows_per_band = n // bands

    sigs = np.stack([_shingle_hashes(d, n, a, b, mod) for d in docs])
    keep = np.ones(len(docs), dtype=bool)
    for band in range(bands):
        cols = slice(band * rows_per_band, (band + 1) * rows_per_band)
        band_sig = sigs[:, cols]
        order = np.lexsort(tuple(band_sig[:, k] for k in range(band_sig.shape[1])[::-1]))
        sorted_sig = band_sig[order]
        diffs = np.any(sorted_sig[1:] != sorted_sig[:-1], axis=1)
        starts = np.concatenate(([0], np.flatnonzero(diffs) + 1))
        for s in starts:
            e = s + 1
            while e < len(order) and not diffs[e - 1]:
                e += 1
            grp = order[s:e]
            if len(grp) <= 1:
                continue
            kept_rep = None
            for idx in grp:
                if not keep[idx]:
                    continue
                if kept_rep is not None and jaccard(docs[idx], docs[kept_rep]) >= threshold:
                    keep[idx] = False
                    continue
                kept_rep = idx
    removed = int((~keep).sum())
    return [d for i, d in enumerate(docs) if keep[i]], removed


def jaccard(a, b):
    sa = set(a[i:i + 5] for i in range(len(a) - 4)) if len(a) >= 5 else set(a)
    sb = set(b[i:i + 5] for i in range(len(b) - 4)) if len(b) >= 5 else set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run full recommended pipeline")
    ap.add_argument("--split-mc4", action="store_true")
    ap.add_argument("--dedup", action="store_true", default=False)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.set_defaults(dedup=True)
    ap.add_argument("--minhash", action="store_true", help="use MinHash LSH dedup (slow; only for small files)")
    ap.add_argument("--safety", action="store_true", help="enable spam/adult filters (default off)")
    ap.add_argument("--no-safety", dest="safety", action="store_false")
    ap.set_defaults(safety=False)
    ap.add_argument("--pii", action="store_true")
    ap.add_argument("--no-pii", dest="pii", action="store_false")
    ap.set_defaults(pii=False)
    ap.add_argument("--min-len", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=0, help="0 = off")
    ap.add_argument("--outdir", default=os.path.join("data", "corpus_final", "cleaned"))
    ap.add_argument("--files", nargs="*", default=None, help="only these basenames")
    args = ap.parse_args()

    if args.all:
        args.split_mc4 = True

    cfg = {
        "lang_check": True,
        "strip_html": True,
        "strip_urls": True,
        "wiki_boilerplate": True,
        "collapse_repeats": True,
        "gibberish": True,
        "safety": args.safety,
        "pii": args.pii,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "split_mc4": args.split_mc4,
    }

    os.makedirs(args.outdir, exist_ok=True)
    report = {"config": cfg, "files": {}}
    total_in = total_out = 0

    for lang, rels in FILES.items():
        for rel in rels:
            src = os.path.join(SRC, rel)
            if not os.path.exists(src):
                print(f"SKIP missing: {src}")
                continue
            if args.files and not any(a in os.path.basename(src) for a in args.files):
                continue
            base = os.path.basename(src)
            docs = run_file(src, lang, cfg, report["files"])
            n_before = len(docs)
            if args.dedup:
                if args.minhash:
                    docs, drem = near_dedup_minhash(docs)
                else:
                    docs, drem = dedup_streaming(docs)
                n_after = len(docs)
                report["files"][src]["dedup_removed"] = drem
            else:
                n_after = len(docs)
                report["files"][src]["dedup_removed"] = 0

            lang_dir = os.path.join(args.outdir, "english" if lang == "EN" else "japanese")
            os.makedirs(lang_dir, exist_ok=True)
            outp = os.path.join(lang_dir, base)
            # write username + doc per line
            with open(outp, "w", encoding="utf-8") as o:
                for d in docs:
                    o.write(d + "\n")
            total_in += report["files"][src].get("in_lines", 0)
            total_out += n_after
            print(f"[{lang}] {base}: in={report['files'][src]['in_lines']:,} "
                  f"kept={n_after:,} dropped={report['files'][src]['dropped']:,} "
                  f"dedup={n_before - n_after:,} -> {outp}", flush=True)

    report["summary"] = {"total_in": total_in, "total_out": total_out}
    report_path = os.path.join(args.outdir, "cleaning_report.json")
    with open(report_path, "w", encoding="utf-8") as o:
        json.dump(report, o, ensure_ascii=False, indent=2)
    print("\nReport:", report_path)
    print(f"TOTAL: {total_in:,} lines -> {total_out:,} docs ({total_out / max(total_in,1) * 100:.1f}%)")


if __name__ == "__main__":
    main()