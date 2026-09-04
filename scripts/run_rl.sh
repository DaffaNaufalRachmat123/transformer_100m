#!/bin/bash
# =============================================================================
# run_rl.sh — Launch Stage 5 (RL / GRPO-lite) on the 892M model.
#
# Requires:
#   - A SFT checkpoint as the policy base (--base_model) and the reference
#     for the KL penalty (--ref_model, defaults to --base_model).
#   - An RL dataset built by scripts/prepare_rl_coding.py (--rl_dir).
#
# Usage:
#   ./scripts/run_rl.sh                 # single GPU
#   ./scripts/run_rl.sh --ddp           # 2 GPUs (T4 x2)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BASE="${RL_BASE_MODEL:-$PROJECT_DIR/checkpoints/ckpt_sft_final.safetensors}"
RL_DIR="${RL_DIR:-$PROJECT_DIR/rl_data}"
VERIFIER="${RL_VERIFIER:-python3 $PROJECT_DIR/scripts/rl_verifier.py}"

ARGS=(
  --stage 5
  --base_model "$BASE"
  --ref_model "$BASE"
  --rl_dir "$RL_DIR"
  --verifier_cmd "$VERIFIER"
  --rl_workdir "$PROJECT_DIR/rl_work"
  --batch_size 2
  --n_samples 6
  --rl_max_new 80
  --rl_beta 0.1
  --rl_temp 0.8
  --rl_top_k 50
  --max_seq 384
  --epochs 3
  --lr 1e-5
  --grad_clip 1.0
  --save_every 50
  --log_every 5
  "$@"
)

echo "═══════════════════════════════════════════════════════"
echo " Stage 5: RL (GRPO-lite) — policy=$BASE"
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