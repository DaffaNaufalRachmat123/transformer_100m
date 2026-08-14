#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_rl_pipeline.sh — Auto-run the RL pipeline for the 204M model.
#
#  Flow (adds coding on top of the existing JA/EN-facts + translation model):
#
#    Stage A: coding SFT   from the translation model
#             -> checkpoints_rl_sft/ckpt_sft_final.safetensors
#             (teaches the policy to write Python programs — REQUIRED precursor)
#    Stage B: RL (GRPO-lite)  from the coding-SFT model
#             -> checkpoints_rl/ckpt_rl_final.safetensors
#             (rule-based verifier amplifies correct code)
#    Stage C: copy final -> checkpoints/ckpt_rl_final.safetensors
#
#  ⚠️ Important (no BPE / pretrain rebuild needed):
#    The 65536-token BPE already contains all code tokens (def, print, input,
#    ==, (, ), {, }, _ ...) because English prose contains those ASCII chars,
#    and Python is plain ASCII. So code encodes fine with the existing vocab.
#    What is REQUIRED is Stage A (coding SFT) so RL has a passable base.
#
#  Requirements:
#    - Base model: checkpoints/ckpt_sft_final.safetensors (output of the
#      existing JA/EN-facts + translation flow) or RL_BASE_MODEL override.
#    - Build done: bash scripts/build_cuda.sh
#
#  Usage:
#    bash scripts/run_rl_pipeline.sh                # single GPU (Colab)
#    bash scripts/run_rl_pipeline.sh --ddp          # T4 x2 (Kaggle)
#  Tunable (env):
#    RL_BASE_MODEL, RL_SFT_DIR, RL_SFT_CKPT, RL_CKPT,
#    RL_N, RL_BETA, RL_MAX_NEW, RL_STEPS
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-checkpoints}"
RL_BASE_MODEL="${RL_BASE_MODEL:-$CKPT_DIR/ckpt_sft_final.safetensors}"
RL_SFT_DIR="${RL_SFT_DIR:-sft_data/sft_coding}"
RL_DATA_DIR="${RL_DATA_DIR:-rl_data}"
RL_SFT_CKPT="${RL_SFT_CKPT:-checkpoints_rl_sft}"
RL_CKPT="${RL_CKPT:-checkpoints_rl}"
RL_N="${RL_N:-4000}"                 # examples per dataset
RL_STEPS="${RL_STEPS:--1}"           # -1 = run to completion / epochs
RL_BETA="${RL_BETA:-0.1}"
RL_MAX_NEW="${RL_MAX_NEW:-80}"
RL_N_SAMPLES="${RL_N_SAMPLES:-6}"

JOB_FLAGS=()
for a in "$@"; do JOB_FLAGS+=("$a"); done

echo "═══════════════════════════════════════════════════════"
echo " RL pipeline — base model: $RL_BASE_MODEL"
echo "═══════════════════════════════════════════════════════"

if [ ! -f "$RL_BASE_MODEL" ]; then
    echo "[ERROR] base model not found: $RL_BASE_MODEL"
    echo "        Run the JA/EN-facts + translation SFT flow first"
    echo "        (it produces $CKPT_DIR/ckpt_sft_final.safetensors),"
    echo "        or set RL_BASE_MODEL to an existing SFT checkpoint."
    exit 1
fi
if [ ! -f "build/train_sft_cuda" ]; then
    echo "[ERROR] build/train_sft_cuda missing. Run: bash scripts/build_cuda.sh"
    exit 1
fi

# ── 0. Prepare coding SFT + RL datasets ────────────────────────────────────
if [ ! -f "$RL_SFT_DIR/train.jsonl" ] || [ "${FORCE_RESTART:-0}" = "1" ]; then
    echo ""
    echo "═══ [0] Building high-quality coding SFT + RL datasets (self-verified) ═══"
    python3 scripts/build_sft_coding_dataset.py --n "$RL_N" --seed 42 \
        --out_sft "$RL_SFT_DIR/train.jsonl" \
        --out_rl "$RL_DATA_DIR/coding.jsonl"
else
    echo "[0] coding SFT data present: $RL_SFT_DIR/train.jsonl (reuse)"
fi

# ── 1. Coding SFT (from translation model) ─────────────────────────────────
echo ""
echo "═══ [1] Coding SFT (steps=$RL_STEPS) ═══"
echo "  base model : $RL_BASE_MODEL"
echo "  ckpt dir   : $RL_SFT_CKPT"
if [ -f "$RL_SFT_CKPT/ckpt_sft_final.safetensors" ] && \
   [ ! -f "$RL_SFT_CKPT/train_state_sft.txt" ]; then
    echo "  -> coding SFT final already present, skipping."
else
    python3 scripts/train_pipeline.py \
        --stages "sft" \
        --no_skip_done \
        --checkpoint_dir "$RL_SFT_CKPT" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --sft_dir "$RL_SFT_DIR" \
        --sft_base_model "$RL_BASE_MODEL" \
        --sft_lr 2e-5 \
        --sft_epochs 20 \
        --sft_batch_size 16 \
        --sft_max_seq 512 \
        --sft_warmup 50 \
        --sft_save_every 200 \
        --sft_max_steps "$RL_STEPS" \
        "${JOB_FLAGS[@]}"
    if [ ! -f "$RL_SFT_CKPT/ckpt_sft_final.safetensors" ]; then
        # train_pipeline saves ckpt_sft_final only at the very end; if blocked
        # on max_steps use the latest step checkpoint as the coding-SFT base.
        CKPT_RL_SFT_BASE=$(ls -t "$RL_SFT_CKPT"/ckpt_sft_step*.safetensors 2>/dev/null | head -1)
        if [ -n "$CKPT_RL_SFT_BASE" ]; then
            echo "[RL] max_steps hit; using latest coding-SFT step: $CKPT_RL_SFT_BASE"
            cp "$CKPT_RL_SFT_BASE" "$RL_SFT_CKPT/ckpt_sft_final.safetensors"
        else
            echo "[ERROR] coding SFT produced no checkpoint."
            exit 1
        fi
    fi
fi

# ── 2. RL (GRPO-lite, from coding-SFT model) ───────────────────────────────
echo ""
echo "═══ [2] RL (GRPO-lite, beta=$RL_BETA, n_samples=$RL_N_SAMPLES) ═══"
echo "  policy + reference : $RL_SFT_CKPT/ckpt_sft_final.safetensors"
echo "  rl dir             : $RL_DATA_DIR"
echo "  ckpt dir           : $RL_CKPT"
mkdir -p "$RL_CKPT"
if [ -f "$RL_CKPT/ckpt_rl_final.safetensors" ] && \
   [ ! -f "$RL_CKPT/train_state_rl.txt" ]; then
    echo "  -> RL final already present, skipping."
else
    ./build/train_sft_cuda --stage 5 \
        --base_model "$RL_SFT_CKPT/ckpt_sft_final.safetensors" \
        --ref_model "$RL_SFT_CKPT/ckpt_sft_final.safetensors" \
        --rl_dir "$RL_DATA_DIR" \
        --rl_workdir "$RL_CKPT/rl_work" \
        --checkpoint_dir "$RL_CKPT" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --batch_size 2 \
        --n_samples "$RL_N_SAMPLES" \
        --rl_max_new "$RL_MAX_NEW" \
        --rl_beta "$RL_BETA" \
        --rl_temp 0.8 \
        --rl_top_k 50 \
        --max_seq 384 \
        --epochs 20 \
        --lr 1e-5 \
        --grad_clip 1.0 \
        --save_every 50 \
        --log_every 5 \
        --verifier_cmd "python3 scripts/rl_verifier.py" \
        "${JOB_FLAGS[@]}"
    if [ ! -f "$RL_CKPT/ckpt_rl_final.safetensors" ]; then
        echo "[ERROR] RL did not produce ckpt_rl_final.safetensors."
        exit 1
    fi
fi

# ── 3. Copy final to active location ───────────────────────────────────────
echo ""
echo "═══ [3] Copying RL model to active location ═══"
mkdir -p "$CKPT_DIR"
cp "$RL_CKPT/ckpt_rl_final.safetensors" "$CKPT_DIR/ckpt_rl_final.safetensors"
echo "  ✅ $CKPT_DIR/ckpt_rl_final.safetensors"

echo ""
echo "✓ RL pipeline complete."
echo "  Inspect progress: cat $RL_CKPT/loss_log.txt (if present)"
echo "  Verify: run the verifier on a few prompts, or eval with tools/infer"