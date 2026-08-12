#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_parallel_enja.sh — Rebuild the EN<->JA parallel corpus + SFT data.
#
#  Steps:
#    1. Download + filter + dedup the parallel corpus (Tatoeba + OPUS).
#    2. Build SFT rows (both directions) + a held-out eval split.
#    3. Build the final balanced SFT dataset (translation-dominant).
#    4. Run the deep quality filter.
#
#  Usage:
#     bash scripts/run_parallel_enja.sh
#
#  Flags (optional, via env):
#     MAX_PARALLEL   cap on unique pairs used for SFT (default: 300000)
#     MAX_LEN        max sentence length in chars (default: 80)
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

MAX_PARALLEL="${MAX_PARALLEL:-300000}"
MAX_LEN="${MAX_LEN:-80}"

echo "═══ [1/4] Fetch parallel corpus (Tatoeba + OpenSubtitles + KFTT + wikimedia) ═══"
python3 scripts/fetch_parallel_enja.py \
    --sources tatoeba,opensubtitles,kftt,wikimedia \
    --max_len "$MAX_LEN" \
    --out data/parallel_enja.tsv

echo ""
echo "═══ [2/4] Build SFT rows (both directions) + held-out eval ═══"
python3 scripts/build_sft_translation.py \
    --in data/parallel_enja.tsv \
    --out sft_data/sft_clean/sft_translation_large.jsonl \
    --eval_out sft_data/sft_clean/sft_translation_eval.jsonl \
    --max_pairs "$MAX_PARALLEL" \
    --eval_fraction 0.05

echo ""
echo "═══ [3/4] Build final balanced SFT dataset ═══"
python3 scripts/prepare_sft_translation_dominant.py

echo ""
echo "═══ [4/4] Deep quality filter ═══"
python3 scripts/clean_translation_quality.py

echo ""
echo "✓ Done. Final dataset: sft_data/sft_translation_focused/train.jsonl"
echo "  Held-out eval:       sft_data/sft_clean/sft_translation_eval.jsonl"