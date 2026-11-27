#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  rebuild_sft.sh — Ceiling Edition v3 SFT + DPO + Verifiable RL Regeneration
#
#  Config (aligned with 491.58M @ 10B corpus_pretrain_v8, vocab_9):
#    * Math        : Option B1 (GSM8K EN + GSM8K JA + MetaMathQA + sft_think_math)
#    * Coding      : Option A2 (max_seq=1024, Phi-1 textbook Python + sft_think_code + aligned)
#    * Facts       : 41 Wikidata categories + SciQ Science QA + OASST2 EN/JA multi-turn
#    * Translation : OPUS KFTT + Conversational + LaBSE + 2-sent paragraphs + 50+ freeform
#                    non-template styles + anti-QA-collapse question translation pairs
#    * Mixing      : Masked-token budget (41% trans, 21% facts, 21% math, 17% coding)
#    * DPO         : Prefix-safe length-matched hard negatives (`sft_data/dpo_ceiling_enja`)
#    * RL (GRPO)   : Decontaminated EN+JA MBPP asserts + multi-case code + verifiable GSM8K
# ═══════════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

TARGET_MTOK="${TARGET_MTOK:-24500000}"
RATIO="${RATIO:-translation=0.34,facts=0.24,math=0.22,coding=0.20}"
OUT="${OUT:-sft_data/sft_mixed_enja}"

echo "═══ 1/6  facts (41 categories, holdout-clean) ═══"
python3 scripts/build_facts_expanded.py

echo "═══ 2/6  translation ceiling (KFTT + conversational + LaBSE + freeform + q-guard) ═══"
python3 scripts/build_translation_curated.py

echo "═══ 3/6  math & coding reasoning spans (Option B1 + CRLF/fence clean) ═══"
python3 scripts/build_gsm8k_pot_sft.py
python3 scripts/build_thinking_sft.py

echo "═══ 4/6  coding instruct filter (Option A2 <= 1000 BPE v9 tokens, pretrain langs) ═══"
python3 scripts/filter_code_instruct_aligned.py

echo "═══ 5/6  token-budget mixing (Ceiling v3) ═══"
rm -rf "$OUT"
python3 scripts/build_mixed_sft.py --out "$OUT" \
    --target_masked_tokens "$TARGET_MTOK" \
    --strict_ratio --ratio "$RATIO"

echo "═══ 6/6  prefix-safe DPO ceiling + verifiable RL ceiling ═══"
python3 scripts/build_dpo_ceiling.py
python3 scripts/build_rl_ceiling.py

echo ""
echo "═══ audit (must report 0.00%) ═══"
python3 scripts/audit_sft.py "$OUT" --fail_pct 0.01
