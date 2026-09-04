#!/usr/bin/env python3
"""audit_corpus.py — Gate A: per-source corpus quality audit (sampling-based).

For each corpus file (one doc per line) — reads a bounded SAMPLE, not the
whole file, so even 3GB corpora audit in seconds:
  1. LID purity: fastText on sampled docs (must be >= threshold)
  2. Spam pattern density: hits per sampled doc
  3. Empty/short-line share, average doc length (from the sample)
  4. bytes/token estimate via calibrated bytes-per-token table

Verdict per file: PASS / WARN / FAIL. Exit non-zero on any FAIL.

Usage:
  python3 scripts/audit_corpus.py data/corpus_1b/indonesian/mc4_id_clean.txt --lang id
  python3 scripts/audit_corpus.py --dir data/corpus_1b   # all files
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPAM = re.compile(
    r"bugil|telanjang|porno|bokep|ngentot|memek|kontol|poker\s*online|"
    r"judi\s*(online| bola|slot)|slot\s*gacor|togel\s*(online|hk|sgp)|"
    r"bandar\s*(qq|togel|casino)|situsslot| gratis\s*ongkir |wa\s*[:/]\s*08\d|"
    r"アダルト|風俗|出会い系", re.I)

THRESHOLDS = {
    "lid_min": 0.95,       # sampled docs matching expected language
    "spam_max_rate": 0.01, # spam-marker share among sampled docs
    "short_max": 0.10,     # share of sampled docs < 50 chars
}

BPT = {"english": 4.2, "japanese": 3.3, "indonesian": 4.5, "code": 3.8,
       "math": 4.0, "mixed": 4.2}


def detect_lang(path):
    name = str(path)
    if "/mixed/" in name:
        return "mixed"  # bilingual concat docs — LID not applicable
    if "/code/" in name or "starcoder" in name:
        return "code"
    if "/math/" in name or "math" in Path(name).stem:
        return "math"
    if "/japanese/" in name or Path(name).stem.endswith(("_ja", "ja")):
        return "ja"
    if "/indonesian/" in name or Path(name).stem.endswith(("_id", "id")):
        return "id"
    if "/english/" in name or Path(name).stem.endswith(("_en", "en")):
        return "en"
    return "unknown"


def sample_lines(path, n_samples=1500, per_offset=150):
    """Read lines from evenly-spaced offsets (bounded total work)."""
    size = path.stat().st_size
    lines = []
    if size < 20_000_000:  # small file: read all, reservoir-sample
        rng = random.Random(42)
        all_lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
        rng.shuffle(all_lines)
        return all_lines[:n_samples], len(all_lines)
    rng = random.Random(42)
    n_offsets = max(1, n_samples // per_offset)
    est_lines = None
    for i in range(n_offsets):
        frac = (i + 0.5) / n_offsets
        with open(path, "rb") as f:
            f.seek(int(size * frac))
            f.readline()  # skip partial line
            chunk = f.read(per_offset * 4096).decode("utf-8", errors="ignore")
            chunk_lines = chunk.split("\n")[:-1]
            lines.extend(chunk_lines[:per_offset])
    # estimate total line count from first-offset density
    return lines, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dir", default="")
    ap.add_argument("--lang", default="")
    ap.add_argument("--report", default="logs/gate_a_audit.json")
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths]
    if args.dir:
        d = ROOT / args.dir
        for lang_dir in sorted(d.iterdir()):
            if lang_dir.is_dir():
                paths.extend(sorted(lang_dir.glob("*.txt")))
    paths = [p for p in paths if not p.name.endswith("_raw.txt")]
    if not paths:
        print("no files to audit")
        return 0

    ft = None
    try:
        import fasttext
        ft = fasttext.load_model(str(ROOT / "data" / "lid.176.bin"))
    except Exception as e:
        print(f"WARN: LID model unavailable ({e}) — spam/length checks only")

    def lid_of(text):
        if ft is None:
            return None
        try:
            res = ft.f.predict(text[:2000].replace("\n", " "), 1, 0.0, "strict")
            if res:
                return res[0][1].replace("__label__", "")
        except Exception:
            pass
        try:
            labels, _ = ft.predict(text[:2000].replace("\n", " "), k=1)
            return labels[0].replace("__label__", "")
        except Exception:
            return None

    expected = {"id": {"id", "ms"}, "ja": {"ja"}, "en": {"en"},
                "code": None, "math": None, "mixed": None}

    report = {}
    any_fail = False
    for path in paths:
        lang = args.lang or detect_lang(path)
        size = path.stat().st_size
        lines, _n = sample_lines(path)
        lines = [l for l in lines if l is not None]
        if not lines:
            report[str(path)] = {"error": "no sample"}
            any_fail = True
            continue

        spam_docs = 0
        short_docs = 0
        lid_good = 0
        lid_testable = 0
        for t in lines:
            if len(t) < 50:
                short_docs += 1
            if SPAM.search(t[:3000]):
                spam_docs += 1
            # LID only on substantial docs — titles/short fragments are
            # unclassifiable and would produce false FAILs on real corpora
            if ft is not None and expected.get(lang) is not None and len(t.strip()) >= 80:
                lid_testable += 1
                if lid_of(t) in expected[lang]:
                    lid_good += 1

        n = len(lines)
        lid_purity = round(lid_good / lid_testable, 3) if lid_testable else None
        spam_rate = spam_docs / n
        short_share = short_docs / n
        avg_len = sum(len(l) for l in lines) / n
        est_tokens = size / BPT.get(lang if lang in BPT else "english", 4.2)

        verdict = "PASS"
        reasons = []
        if lid_purity is not None and lid_purity < THRESHOLDS["lid_min"]:
            verdict = "FAIL" if lid_purity < 0.85 else "WARN"
            reasons.append(f"lid={lid_purity}")
        if spam_rate > THRESHOLDS["spam_max_rate"]:
            verdict = "FAIL" if spam_rate > 0.05 else "WARN"
            reasons.append(f"spam={spam_rate:.3f}")
        if short_share > THRESHOLDS["short_max"]:
            verdict = "WARN" if verdict == "PASS" else verdict
            reasons.append(f"short={short_share:.1%}")
        if verdict == "FAIL":
            any_fail = True

        report[str(path)] = {
            "lang": lang, "bytes": size, "est_tokens": int(est_tokens),
            "sampled": n, "avg_doc_chars": round(avg_len, 0),
            "lid_purity": lid_purity, "spam_rate": round(spam_rate, 4),
            "short_share": round(short_share, 4),
            "verdict": verdict, "reasons": reasons,
        }
        print(f"[{verdict:<4}] {path.name:<26} {lang:<5} {size/1e9:5.2f}GB "
              f"~{est_tokens/1e9:5.2f}Btok lid={lid_purity} spam={spam_rate:.3f} "
              f"{' '.join(reasons)}")

    rp = ROOT / args.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nGate A: {'FAIL' if any_fail else 'PASS'} -> {rp}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
