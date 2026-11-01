#!/usr/bin/env python3
"""fetch_corpus.py — stream pretraining corpora from HuggingFace to plain text.

Closes the gaps the corpus builder reports. Measured on the current
data/corpus_pretrain_v8 build:

    bucket     real GB   tokens   share   target
    english       6.55    1.86B   34.6%    32.0%   ok
    japanese      7.15    2.25B   41.9%    36.0%   ok
    code          4.53    1.24B   23.2%    20.0%   ok, but 94.6% of it is Python
    math          0.05    0.01B    0.2%    12.0%   effectively empty

So the two things worth downloading are math and non-Python code; Japanese is
listed as well because more of it removes the 1.86-epoch repetition.

Output: data/{math,ja,code,en}_raw/<name>.txt — documents separated by a line
holding exactly "<|endoftext|>", with their newlines intact.
build_pretrain_corpus.py detects that layout automatically, then filters and
deduplicates against everything already in the corpus. Only a light pre-filter
runs here; the strict pass belongs to one place, and that place is the builder.

Newlines matter: an earlier version joined every document onto a single line,
and the builder then rejected 100% of the fetched code as "too_few_lines"
(6,664 code_contests + 79,057 csn_javascript + 144,278 gh_javascript + 48,664
rosetta documents).

State: logs/fetch_<name>_state.json — resumable, re-run to continue.

Usage:
    pip install datasets
    python3 scripts/fetch_corpus.py --list
    python3 scripts/fetch_corpus.py --dataset finemath   --max_bytes 3000000000
    python3 scripts/fetch_corpus.py --dataset stack_js   --max_bytes 800000000
    python3 scripts/fetch_corpus.py --dataset fineweb2_ja --max_bytes 5000000000

Then rebuild:
    python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Spec:
    def __init__(self, bucket, repo, config, split, field, extract=None):
        self.bucket = bucket
        self.repo = repo
        self.config = config
        self.split = split
        self.field = field
        # Optional row -> list[str]; for datasets whose text is not a plain
        # column (code_contests keeps solutions in a parallel-array struct).
        self.extract = extract


# code_contests language ids: 1 PYTHON2, 2 CPP, 3 PYTHON3, 4 JAVA.
# Only C++ and Java are taken — the bucket already has all the Python it needs.
_CC_WANT = {2, 4}


def _extract_code_contests(row):
    out = []
    for key in ("solutions", "incorrect_solutions"):
        sol = row.get(key) or {}
        langs = sol.get("language") or []
        srcs = sol.get("solution") or []
        if key == "incorrect_solutions":
            continue
        for lang, src in zip(langs, srcs):
            if lang in _CC_WANT and isinstance(src, str) and len(src) > 300:
                out.append(src)
    return out[:4]      # a problem has hundreds of near-identical solutions


# Parallel corpora come in two shapes. Both are emitted as one "en\tja" line so
# the downstream builder sees a single format.
def _extract_pair_st(row):
    """sentence-transformers/parallel-sentences-*: {english, non_english}."""
    en, ja = row.get("english"), row.get("non_english")
    if not isinstance(en, str) or not isinstance(ja, str):
        return []
    return [en.replace("\t", " ") + "\t" + ja.replace("\t", " ")]


def _extract_pair_tr(row):
    """Helsinki-NLP style: {translation: {en: ..., ja: ...}}."""
    t = row.get("translation") or {}
    en, ja = t.get("en"), t.get("ja")
    if not isinstance(en, str) or not isinstance(ja, str):
        return []
    return [en.replace("\t", " ") + "\t" + ja.replace("\t", " ")]


# name -> Spec. Every entry here was verified to stream without an auth token on
# datasets 5.x. Notable exclusions and why:
#   bigcode/the-stack*        gated  (needs an HF token and accepting the terms)
#   bigcode/commitpackft      loader script, removed in datasets 5.x
#   EleutherAI/proof-pile-2   loader script, removed in datasets 5.x
#   codeparrot/github-code*   loader script, removed in datasets 5.x
# If you have an HF token, `huggingface-cli login` then add the-stack-v2-dedup
# back — it is by far the best multi-language code source.
DATASETS = {
    # ---- math ------------------------------------------------------------
    "finemath":    Spec("math", "HuggingFaceTB/finemath", "finemath-3plus", "train", "text"),
    "openwebmath": Spec("math", "open-web-math/open-web-math", None, "train", "text"),

    # ---- japanese --------------------------------------------------------
    "fineweb2_ja": Spec("japanese", "HuggingFaceFW/fineweb-2", "jpn_Jpan", "train", "text"),
    "culturax_ja": Spec("japanese", "uonlp/CulturaX", "ja", "train", "text"),

    # ---- code ------------------------------------------------------------
    # The existing bucket is 94.6% Python, so only non-Python entries are here.
    "gh_javascript": Spec("code", "angie-chen55/javascript-github-code", None, "train", "code"),
    "csn_javascript": Spec("code", "Nan-Do/code-search-net-javascript", None, "train", "code"),
    "csn_java":     Spec("code", "Nan-Do/code-search-net-java", None, "train", "code"),
    "csn_go":       Spec("code", "Nan-Do/code-search-net-go", None, "train", "code"),
    "csn_php":      Spec("code", "Nan-Do/code-search-net-php", None, "train", "code"),
    "csn_ruby":     Spec("code", "Nan-Do/code-search-net-ruby", None, "train", "code"),
    "kstack_kotlin": Spec("code", "JetBrains/KStack", None, "train", "content"),
    "rosetta":      Spec("code", "cakiki/rosetta-code", None, "train", "code"),
    "code_contests": Spec("code", "deepmind/code_contests", None, "train", None,
                          extract=_extract_code_contests),

    # ---- english ---------------------------------------------------------
    "fineweb_edu": Spec("english", "HuggingFaceFW/fineweb-edu", "sample-100BT", "train", "text"),

    # ---- EN<->JA parallel -------------------------------------------------
    # Ordered best-aligned first, because build_mixed_bucket.py deduplicates
    # first-wins. ccmatrix and wikimatrix are bitext-mined and genuinely noisy
    # (a sampled WikiMatrix row paired "Keep us out of Temptation" with an
    # unrelated Japanese line), so pair_ok below is deliberately strict.
    "par_tatoeba":  Spec("parallel", "sentence-transformers/parallel-sentences-tatoeba", "en-ja", "train", None, extract=_extract_pair_st),
    "par_talks":    Spec("parallel", "sentence-transformers/parallel-sentences-talks", "en-ja", "train", None, extract=_extract_pair_st),
    "par_news":     Spec("parallel", "Helsinki-NLP/news_commentary", "en-ja", "train", None, extract=_extract_pair_tr),
    "par_opus100":  Spec("parallel", "Helsinki-NLP/opus-100", "en-ja", "train", None, extract=_extract_pair_tr),
    "par_subs":     Spec("parallel", "sentence-transformers/parallel-sentences-opensubtitles", "en-ja", "train", None, extract=_extract_pair_st),
    "par_wikimatrix": Spec("parallel", "sentence-transformers/parallel-sentences-wikimatrix", "en-ja", "train", None, extract=_extract_pair_st),
    "par_ccmatrix": Spec("parallel", "sentence-transformers/parallel-sentences-ccmatrix", "en-ja", "train", None, extract=_extract_pair_st),
}

DOC_SEP = "<|endoftext|>"

RAW_DIR = {"math": "data/math_raw", "japanese": "data/ja_raw",
           "code": "data/code_raw", "english": "data/en_raw",
           "parallel": "data/parallel_raw/en-ja"}

# Light pre-filter only — the builder does the real cleaning, and doing it twice
# would just make the download slower.
BAD_RE = re.compile(
    r"\b(?:porn|xxx|hentai|onlyfans|casino|online ?gambling|escort ?service|viagra)\b"
    r"|http[s]?://\S{200,}",
    re.I,
)
MATH_HINT_RE = re.compile(
    r"\\(?:frac|sum|int|sqrt|begin\{)|\$\$?[^$\n]{3,}\$|"
    r"[\u2211\u220f\u222b\u221a\u2264\u2265\u2260\u2248\u2261\u2202]|"
    r"\b(?:theorem|lemma|proof|derivative|integral|polynomial|equation)\b",
    re.I,
)
KANA_RE = re.compile(r"[\u3040-\u30ff]")
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGITS_RE = re.compile(r"\d+")


def pair_ok(en: str, ja: str) -> bool:
    """Reject misaligned pairs. Mined bitext is full of them, and a bad pair is
    worse than no pair: it actively teaches the model to mistranslate."""
    en, ja = en.strip(), ja.strip()
    if not (8 <= len(en) <= 400) or not (4 <= len(ja) <= 300):
        return False
    if "\ufffd" in en or "\ufffd" in ja:
        return False
    if not LATIN_RE.search(en) or not CJK_RE.search(ja):
        return False
    # A "Japanese" side with no kana is usually a bare noun list or Chinese.
    if not KANA_RE.search(ja):
        return False
    # An English side carrying CJK means the columns are swapped or merged.
    if len(CJK_RE.findall(en)) > 2:
        return False
    # Japanese is roughly 2x denser than English per character. Anything far
    # outside that band is an alignment error, not a translation.
    ratio = len(en) / max(1, len(ja))
    if not (0.7 <= ratio <= 6.0):
        return False
    # Numbers survive translation. If they disagree, the sentences do not match.
    de, dj = set(DIGITS_RE.findall(en)), set(DIGITS_RE.findall(ja))
    if de and dj and not (de & dj):
        return False
    if len(de) >= 2 and not dj:
        return False
    return not (BAD_RE.search(en) or BAD_RE.search(ja))


def doc_ok(text: str, bucket: str) -> bool:
    if not text:
        return False
    if bucket == "parallel":
        parts = text.split("\t")
        return len(parts) == 2 and pair_ok(parts[0], parts[1])
    n = len(text)
    if n < 300 or n > 400_000:
        return False
    if "\ufffd" in text:
        return False
    if BAD_RE.search(text):
        return False
    if bucket == "math":
        # A math-corpus row with no notation at all is web boilerplate that
        # leaked through the upstream classifier.
        return bool(MATH_HINT_RE.search(text))
    if bucket == "japanese":
        return len(KANA_RE.findall(text[:2000])) >= 20
    if bucket == "code":
        return text.count("\n") >= 5
    return True


def clean_doc(text: str) -> str:
    """Normalize line endings and neutralize any literal separator line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if DOC_SEP in text:
        text = text.replace(DOC_SEP, "<|eot|>")
    return text.strip("\n")


def clean_pair(text: str) -> str:
    """Parallel rows are one line each, so newlines and zero-width marks (jw300
    and OpenSubtitles are full of U+200B) have to go rather than be preserved."""
    text = text.replace("\r", " ").replace("\n", " ")
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(z, "")
    if DOC_SEP in text:
        text = text.replace(DOC_SEP, " ")
    return " ".join(text.split(" ")).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finemath", choices=sorted(DATASETS))
    ap.add_argument("--max_bytes", type=int, default=2_000_000_000)
    ap.add_argument("--out_dir", default="", help="override the bucket default")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'name':14} {'bucket':9} repo")
        for k in sorted(DATASETS, key=lambda x: (DATASETS[x].bucket, x)):
            s = DATASETS[k]
            cfg = f" [{s.config}]" if s.config else ""
            print(f"{k:14} {s.bucket:9} {s.repo}{cfg}")
        print("\nGaps in the current build: math is 0.2% of tokens against a 12% "
              "target, and the code bucket is 94.6% Python.")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERR] pip install datasets", file=sys.stderr)
        return 1

    spec = DATASETS[args.dataset]
    out_dir = ROOT / (args.out_dir or RAW_DIR[spec.bucket])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.txt"
    state_path = ROOT / "logs" / f"fetch_{args.dataset}_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state = {"rows_seen": 0, "bytes": 0, "kept": 0, "skipped": 0}
    if state_path.exists():
        state.update(json.loads(state_path.read_text()))
    resume_after = state["rows_seen"]
    if out_path.exists() and resume_after:
        print(f"[resume] {out_path.name}: {state['bytes']/1e9:.2f} GB, "
              f"skipping first {resume_after:,} rows")
    elif out_path.exists():
        out_path.unlink()

    print(f"[fetch] {args.dataset} -> {out_path}  "
          f"(bucket={spec.bucket}, target {args.max_bytes/1e9:.1f} GB)")
    ds = load_dataset(spec.repo, spec.config, split=spec.split, streaming=True)
    is_par = spec.bucket == "parallel"
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i < resume_after:
                continue
            state["rows_seen"] = i + 1
            if spec.extract:
                texts = spec.extract(row)
            else:
                texts = [row.get(spec.field) or ""]
            for text in texts:
                text = clean_pair(text) if is_par else clean_doc(text)
                if not doc_ok(text, spec.bucket):
                    state["skipped"] += 1
                    continue
                if is_par:
                    f.write(text + "\n")
                    state["bytes"] += len(text.encode("utf-8")) + 1
                else:
                    f.write(text + "\n" + DOC_SEP + "\n")
                    state["bytes"] += len(text.encode("utf-8")) + len(DOC_SEP) + 2
                state["kept"] += 1
            if (i + 1) % 20_000 == 0:
                f.flush()
                state_path.write_text(json.dumps(state, indent=2))
                el = time.time() - t0
                print(f"  rows={i+1:,} kept={state['kept']:,} "
                      f"{state['bytes']/1e9:.2f}/{args.max_bytes/1e9:.1f} GB "
                      f"{state['bytes']/1e6/max(el,1):.1f} MB/s", flush=True)
            if state["bytes"] >= args.max_bytes:
                break

    state_path.write_text(json.dumps(state, indent=2))
    print(f"[done] {out_path} — {state['bytes']/1e9:.2f} GB, "
          f"{state['kept']:,} docs, {state['skipped']:,} skipped")
    print("[next] python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # datasets' streaming reader leaves non-daemon worker threads alive, so a
    # normal return hangs the interpreter after the download has finished.
    # Do not pipe this script into another command when chaining several runs:
    # _exit skips the interpreter's fd teardown, and the next python in the
    # chain can inherit a closed stdout and die in init_sys_streams before it
    # fetches anything. Redirect to a file and filter the file afterwards.
    os._exit(rc)
