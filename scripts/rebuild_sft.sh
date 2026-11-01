#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  rebuild_sft.sh — regenerate every EN/JA SFT dataset, in dependency order.
#
#  Order matters: the thinking sets read the coding sets, and the mixer reads
#  the thinking sets. Running the mixer against stale inputs silently produces
#  a dataset that no longer matches its own stats.json.
#
#  Usage:
#    bash scripts/rebuild_sft.sh              # rebuild everything
#    SKIP_TRANSLATION=1 bash scripts/rebuild_sft.sh
#
#  Tunable (env):
#    TOTAL        rows requested from the mixer  [400000, capped by data]
#    RATIO        family mix  [facts=.3875,translation=.3875,coding=.075,math=.15]
#    MAX_PAIRS    translation pairs to keep      [900000]
#    PHRASINGS    MBPP rewordings per language   [4]
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

TOTAL="${TOTAL:-400000}"
RATIO="${RATIO:-facts=0.3875,translation=0.3875,coding=0.075,math=0.15}"
MAX_PAIRS="${MAX_PAIRS:-900000}"
PHRASINGS="${PHRASINGS:-4}"
CODE_INSTRUCT="${CODE_INSTRUCT:-150000}"
OUT="${OUT:-sft_data/sft_mixed_enja}"

echo "═══ 1/5  facts ═══"
python3 scripts/build_facts_expanded.py

if [ "${SKIP_TRANSLATION:-0}" = "1" ]; then
    echo "═══ 2/5  translation — skipped ═══"
else
    echo "═══ 2/5  translation ═══"
    python3 scripts/build_translation_enja.py --max_pairs "$MAX_PAIRS"
fi

echo "═══ 3/5  coding (MBPP; HumanEval stays held out) ═══"
python3 scripts/build_coding_expanded.py \
    --mbpp /tmp/mbpp.jsonl --humaneval /tmp/he.jsonl \
    --phrasings "$PHRASINGS"

# Decontaminated Magicoder rows. Without these the coding pool is 9,065 unique
# rows, which at a 0.075 share caps the whole mixed set at 120,080.
if [ ! -s sft_data/sft_code_instruct/train.jsonl ]; then
    python3 scripts/build_code_instruct_sft.py --max_rows "$CODE_INSTRUCT"
fi

echo "═══ 4/5  reasoning spans ═══"
python3 scripts/build_thinking_sft.py
python3 scripts/build_gsm8k_pot_sft.py

echo "═══ 5/5  mixing ═══"
rm -rf "$OUT"
python3 scripts/build_mixed_sft.py --out "$OUT" --total "$TOTAL" \
    --strict_ratio --ratio "$RATIO"

echo ""
echo "═══ audit (must report 0.00%) ═══"
python3 scripts/audit_sft.py "$OUT" --fail_pct 0.01
