#!/usr/bin/env python3
"""build_mixed_bucket.py — EN<->JA parallel corpus for the pretraining mix.

Renders sentence pairs two ways and interleaves them:

  plain   "{en} {ja}"  and  "{ja} {en}"
          Bare bilingual concatenation is the alignment signal that makes
          translation work at small scale; the model has to attend across the
          language boundary to predict the second half.

  marked  "{src}\\n<|translate|><|lang_ja|>\\n{tgt}"
          Same content wearing the inference-time costume. `<|translate|>` and
          `<|lang_en|>` / `<|lang_ja|>` are real vocabulary entries (ids 7, 12,
          13), so this teaches "after the translate marker and a language tag,
          emit that language" during pretraining rather than leaving all of it
          to SFT. That behaviour is what the T1 compliance metric measures.

Several pairs go into one document, because a document holding one 20-token
pair spends most of its length on the BOS/EOS boundary.

Output: data/corpus_inject_parallel/parallel_enja.txt
        (documents separated by a "<|endoftext|>" line)

Usage:
  python3 scripts/build_mixed_bucket.py --max_pairs 1500000
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import pair_ok      # noqa: E402  same rules as the fetcher

DOC_SEP = "<|endoftext|>"

# Order matters: deduplication is first-wins, so the best-aligned corpora go
# first and the mined ones only contribute what is new. par_ccmatrix_labse.txt
# is the LaBSE-filtered CCMatrix (cosine >= 0.70); the raw file is deliberately
# NOT listed, because listing both would re-admit exactly the pairs the filter
# rejected.
SOURCES = [
    "data/parallel_en_ja.tsv",
    "data/parallel_raw/en-ja/par_tatoeba.txt",
    "data/parallel_raw/en-ja/par_talks.txt",
    "data/parallel_raw/en-ja/par_news.txt",
    "data/parallel_raw/en-ja/par_opus100.txt",
    "data/parallel_raw/en-ja/par_subs.txt",
    "data/parallel_raw/en-ja/par_wikimatrix.txt",
    "data/parallel_raw/en-ja/par_ccmatrix_labse.txt",
    "data/parallel_raw/en-ja/par_ccmatrix_cand.txt",
]


def load_pairs(paths, max_pairs):
    pairs, seen = [], set()
    stats = {}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            print(f"  {rel:44} MISSING")
            continue
        kept = dropped = dup = 0
        for line in p.open(encoding="utf-8", errors="replace"):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            en, ja = cols[0].strip(), cols[1].strip()
            if not pair_ok(en, ja):
                dropped += 1
                continue
            key = (en[:60].lower(), ja[:40])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            pairs.append((en, ja))
            kept += 1
            if len(pairs) >= max_pairs:
                break
        tot = kept + dropped + dup
        stats[rel] = kept
        print(f"  {Path(rel).name:44} kept {kept:>9,}  "
              f"dropped {100*dropped/max(1,tot):5.1f}%  dup {100*dup/max(1,tot):5.1f}%")
        if len(pairs) >= max_pairs:
            print(f"  (cap of {max_pairs:,} reached)")
            break
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/corpus_inject_parallel")
    ap.add_argument("--max_pairs", type=int, default=20_000_000)
    ap.add_argument("--pairs_per_doc", type=int, default=5)
    ap.add_argument("--marked_frac", type=float, default=0.40)
    ap.add_argument("--repeats", type=int, default=1,
                    help="passes over the pair set; each pass re-randomizes form")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("loading parallel sources:")
    pairs, stats = load_pairs(SOURCES, args.max_pairs)
    print(f"\n{len(pairs):,} unique usable pairs")
    if not pairs:
        return 1

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "parallel_enja.txt"

    def render(en, ja):
        if rng.random() < args.marked_frac:
            if rng.random() < 0.5:
                return f"{en}\n<|translate|><|lang_ja|>\n{ja}"
            return f"{ja}\n<|translate|><|lang_en|>\n{en}"
        return f"{en} {ja}" if rng.random() < 0.5 else f"{ja} {en}"

    docs = nbytes = 0
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.repeats):
            order = list(range(len(pairs)))
            rng.shuffle(order)
            for i in range(0, len(order), args.pairs_per_doc):
                chunk = [pairs[j] for j in order[i:i + args.pairs_per_doc]]
                body = "\n".join(render(en, ja) for en, ja in chunk)
                f.write(body + "\n" + DOC_SEP + "\n")
                docs += 1
                nbytes += len(body.encode("utf-8")) + len(DOC_SEP) + 2

    est_tok = nbytes / 3.35   # mixed EN/JA, between the two per-language rates
    print(f"{out}: {docs:,} documents, {nbytes/1e6:.0f} MB, ~{est_tok/1e6:.0f}M tokens")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "mixed_bucket_stats.json").write_text(
        json.dumps({"pairs": len(pairs), "per_source": stats, "documents": docs,
                    "bytes": nbytes, "est_tokens": int(est_tok),
                    "path": str(out)}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
