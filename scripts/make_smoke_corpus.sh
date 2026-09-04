#!/usr/bin/env bash
# ============================================================================
# make_smoke_corpus.sh — Build the trimmed corpus for the $15 vast.ai smoke
# (892M model end-to-end proof, ~250M pretrain tokens, vocab_6).
#
# Creates data/corpus_smoke/ with size-truncated copies (~8.4GB upload) —
# sized generously above what ~250M tokens at smoke weights consumes.
#
# Smoke weights (set at runtime via env, see run_smoke_vast_15usd.sh):
#   EN 0.33 / JA 0.25 / ID 0.20 / CODE 0.00 / MATH 0.00 / MIXED 0.22
#   (code zeroed: vocab_6 strips newlines -> code would be garbage tokens;
#    math zeroed: openwebmath source no longer on disk;
#    mixed boosted: ~55M tokens through the injection corpus at 250M total
#    gives each fact rendering ~130 exposures — above the ~100x memorization
#    threshold, which is exactly what this smoke must prove)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=data/corpus_1b
OUT=data/corpus_smoke
rm -rf "$OUT"
mkdir -p "$OUT"/{english,japanese,indonesian,math,mixed,code}

# code + math buckets: 1-byte placeholders so six-bucket mode activates
# (code/ AND math/ must exist) while weight 0 keeps them out of every batch.
# (openwebmath source was removed from the sibling project between sessions;
#  re-fetch it before the full 21B run if math is wanted there.)
printf 'x' > "$OUT/code/placeholder.txt"
printf 'x' > "$OUT/math/placeholder.txt"

# english: 2.58GB (~0.6B tok available vs 429M needed)
head -c 1550000000 "$SRC/english/fineweb_edu.txt"   > "$OUT/english/fineweb_edu.txt"
head -c 1030000000 "$SRC/english/openwebtext.txt"   > "$OUT/english/openwebtext.txt"

# japanese: 1.5GB of wiki_ja (~450M tok vs 325M needed)
head -c 1500000000 "$SRC/japanese/wiki_ja.txt"      > "$OUT/japanese/wiki_ja.txt"

# indonesian: 2.14GB (~500M tok vs 260M needed)
head -c 1200000000 "$SRC/indonesian/culturax_id.txt" > "$OUT/indonesian/culturax_id.txt"
cp    "$SRC/indonesian/wiki_id.txt"                  "$OUT/indonesian/wiki_id.txt"

# mixed: full (injection corpus is the point of this run)
cp "$SRC"/mixed/*.txt "$OUT/mixed/"

du -sh "$OUT"/*
echo "----"
du -sh "$OUT"
