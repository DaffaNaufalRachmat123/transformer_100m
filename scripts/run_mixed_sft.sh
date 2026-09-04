#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_mixed_sft.sh — single-stage multi-task SFT (EN/JA).
#
#  Replaces the staged run_facts_sft.sh (JA facts -> EN facts -> translation).
#  Each stage there fine-tunes on top of the previous one and overwrites it:
#  the facts checkpoint scores 0.003 chrF on translation, while the translation
#  checkpoint answers 0/10 EN facts. One interleaved pass removes the overwrite.
#
#  Usage:
#    bash scripts/run_mixed_sft.sh
#    ARCH_CONFIG=config_492m.json BASE_MODEL=checkpoints/ckpt_pretrain_final.safetensors \
#      bash scripts/run_mixed_sft.sh
#
#  Tunable (env):
#    BASE_MODEL   : checkpoint to fine-tune  [translation_resources/facts/ckpt_sft_final.safetensors]
#    ARCH_CONFIG  : architecture JSON        [empty = trainer default]
#    CKPT_DIR     : output dir               [checkpoints_mixed]
#    MIX_DIR      : dataset dir              [sft_data/sft_mixed_enja]
#    MIX_TOTAL    : rows to sample           [400000, capped by data ceiling]
#    MIX_RATIO    : family mix               [facts=0.40,translation=0.40,coding=0.05,math=0.15]
#    STEPS        : max steps                [-1 = full epochs]
#    LR / EPOCHS / BATCH / MAX_SEQ / WARMUP
#    FORCE_RESTART=1 : wipe CKPT_DIR first
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-translation_resources/facts/ckpt_sft_final.safetensors}"
ARCH_CONFIG="${ARCH_CONFIG:-}"
CKPT_DIR="${CKPT_DIR:-checkpoints_mixed}"
MIX_DIR="${MIX_DIR:-sft_data/sft_mixed_enja}"
MIX_TOTAL="${MIX_TOTAL:-400000}"
MIX_RATIO="${MIX_RATIO:-facts=0.40,translation=0.40,coding=0.05,math=0.15}"
STEPS="${STEPS:--1}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-16}"
MAX_SEQ="${MAX_SEQ:-512}"
WARMUP="${WARMUP:-200}"
VOCAB_DIR="${VOCAB_DIR:-vocab_6}"
BPE="${BPE:-bpe_v6.bin}"

# ── Dataset ────────────────────────────────────────────────────────────────
if [ ! -f "$MIX_DIR/train.jsonl" ]; then
    echo "═══ Building mixed EN/JA dataset ═══"
    python3 scripts/build_mixed_sft.py \
        --out "$MIX_DIR" --total "$MIX_TOTAL" \
        --ratio "$MIX_RATIO" --strict_ratio
else
    echo "  [reuse] $MIX_DIR ($(wc -l < "$MIX_DIR/train.jsonl") train rows)"
fi

if [ ! -f "$BASE_MODEL" ]; then
    echo "[ERROR] base model not found: $BASE_MODEL"
    exit 1
fi

# ── Checkpoint dir: keep state so an interrupted run can --resume ──────────
if [ "${FORCE_RESTART:-0}" = "1" ]; then
    rm -rf "$CKPT_DIR"; mkdir -p "$CKPT_DIR"
    echo "  [force restart] cleared $CKPT_DIR"
elif [ -f "$CKPT_DIR/train_state_sft.txt" ]; then
    echo "  [resume] keeping existing state in $CKPT_DIR"
else
    mkdir -p "$CKPT_DIR"
fi

ARCH_ARGS=()
[ -n "$ARCH_CONFIG" ] && ARCH_ARGS=(--arch_config "$ARCH_CONFIG")

echo ""
echo "═══ Mixed SFT ═══"
echo "  base model : $BASE_MODEL"
echo "  arch       : ${ARCH_CONFIG:-<trainer default>}"
echo "  data       : $MIX_DIR"
echo "  ckpt dir   : $CKPT_DIR"
echo "  lr=$LR epochs=$EPOCHS batch=$BATCH seq=$MAX_SEQ steps=$STEPS"

python3 scripts/train_pipeline.py \
    --stages sft \
    --no_skip_done \
    --checkpoint_dir "$CKPT_DIR" \
    --vocab_dir "$VOCAB_DIR" \
    --bpe "$BPE" \
    --sft_dir "$MIX_DIR" \
    --sft_base_model "$BASE_MODEL" \
    --sft_lr "$LR" \
    --sft_epochs "$EPOCHS" \
    --sft_batch_size "$BATCH" \
    --sft_max_seq "$MAX_SEQ" \
    --sft_warmup "$WARMUP" \
    --sft_max_steps "$STEPS" \
    --sft_save_every 200 \
    "${ARCH_ARGS[@]}"

OUT="$CKPT_DIR/ckpt_sft_final.safetensors"
if [ ! -f "$OUT" ]; then
    echo "[ERROR] SFT did not produce $OUT"
    exit 1
fi

echo ""
echo "═══ Multi-skill eval (the point: ONE checkpoint must pass ALL skills) ═══"
python3 scripts/eval_multiskill.py \
    --model "$OUT" \
    --mix_dir "$MIX_DIR" \
    ${ARCH_CONFIG:+--config "$ARCH_CONFIG"} \
    --out "results/mixed_sft_$(basename "$CKPT_DIR").json"
