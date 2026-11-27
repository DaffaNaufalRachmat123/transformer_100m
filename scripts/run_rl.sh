#!/bin/bash
# =============================================================================
# run_rl.sh — Launch Stage 5 (RL / GRPO-lite) on the 491.58M EN/JA/Code/Math model.
#
# Auto-adapts batch_size x n_samples to GPU VRAM:
#   - Tesla T4 (14.6 GB VRAM, e.g. Kaggle T4 x2): batch_size=1, n_samples=4 (G=4)
#   - >= 22 GB VRAM (RTX 4090 / RTX 5090 / A100): batch_size=2, n_samples=6 (G=12)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -n "$RL_BASE_MODEL" ]; then
    BASE="$RL_BASE_MODEL"
elif [ -f "$PROJECT_DIR/checkpoints/ckpt_dpo_final.safetensors" ]; then
    BASE="$PROJECT_DIR/checkpoints/ckpt_dpo_final.safetensors"
else
    BASE="$PROJECT_DIR/checkpoints/ckpt_sft_final.safetensors"
fi

RL_DIR="${RL_DIR:-$PROJECT_DIR/rl_data/ceiling_enja}"
VERIFIER="${RL_VERIFIER:-python3 $PROJECT_DIR/scripts/rl_verifier.py}"
ARCH_CFG="${ARCH_CFG:-$PROJECT_DIR/config_500m.json}"
VOCAB_DIR="${VOCAB_DIR:-$PROJECT_DIR/vocab_9}"
BPE_BIN="${BPE_BIN:-$PROJECT_DIR/bpe_v9.bin}"

VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 15360)
if [ "${VRAM_MIB:-15360}" -lt 20000 ]; then
    DEF_BS=1
    DEF_NS=4
else
    DEF_BS=2
    DEF_NS=6
fi

ARGS=(
  --stage 5
  --arch_config "$ARCH_CFG"
  --vocab_dir "$VOCAB_DIR"
  --bpe "$BPE_BIN"
  --base_model "$BASE"
  --ref_model "$BASE"
  --rl_dir "$RL_DIR"
  --verifier_cmd "$VERIFIER"
  --rl_workdir "$PROJECT_DIR/rl_work"
  --batch_size "${RL_BATCH_SIZE:-$DEF_BS}"
  --n_samples "${RL_N_SAMPLES:-$DEF_NS}"
  --rl_max_new 384
  --rl_beta 0.08
  --rl_temp 0.75
  --rl_top_k 40
  --max_seq 768
  --epochs 1
  --lr 5e-6
  --grad_clip 1.0
  --save_every 50
  --upload_every "${RL_UPLOAD_EVERY:-300}"
  --log_every 5
  "$@"
)

echo "═══════════════════════════════════════════════════════"
echo " Stage 5: RL (GRPO-lite) — policy=$BASE"
echo " Data   : $RL_DIR | VRAM=${VRAM_MIB}MiB -> B=${RL_BATCH_SIZE:-$DEF_BS} x N=${RL_N_SAMPLES:-$DEF_NS}"
echo "═══════════════════════════════════════════════════════"

BINARY="$PROJECT_DIR/build/train_sft_cuda"
if [ ! -f "$BINARY" ]; then
    echo "[ERROR] Build first: bash scripts/build_cuda.sh"
    exit 1
fi

if [[ "$*" == *"--ddp"* ]]; then
    "$PROJECT_DIR/scripts/launch_ddp.sh" "$BINARY" "${ARGS[@]}"
else
    "$BINARY" "${ARGS[@]}"
fi
