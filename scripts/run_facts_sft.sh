#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_facts_sft.sh — Three-stage SFT, one checkpoint folder per language.
#
#  Stage 1 (facts JA-only):  from ckpt_pretrain_final -> checkpoints_facts_ja
#  Stage 2 (facts EN-dominant): from Stage 1 JA model -> checkpoints_facts_en
#  Stage 3 (translation):       from Stage 2 EN model -> checkpoints_transl
#
#  Each stage writes to its OWN checkpoint dir so the pipeline's "stage already
#  done" skip (--skip_done, default on) never fires. A stage is skipped ONLY if
#  its own *_final already exists AND no resume state is present; otherwise it
#  resumes from the last checkpoint. If a stage's ckpt_sft_final.safetensors
#  already exists, the script moves on to the next stage (auto-advance).
#
#  Requirements:
#    - sft_data/sft_facts_ja/train.jsonl      (build_facts_dataset.py --mode ja_only)
#    - sft_data/sft_facts_en/train.jsonl      (build_facts_dataset.py --mode en_dominant)
#    - sft_data/sft_translation_focused/train.jsonl
#    - A base model to start Stage 1 (FACTS_BASE_MODEL).
#
#  Usage:
#    bash scripts/run_facts_sft.sh
#
#  Tunable (env):
#    FACTS_BASE_MODEL : base to start Stage 1 (default: CKPT_DIR/ckpt_pretrain_final.safetensors)
#    JA_STEPS         : max steps Stage 1 (default: 15000)
#    EN_STEPS         : max steps Stage 2 (default: 15000)
#    TRANSL_STEPS     : max steps Stage 3 (default: 15000)
#    CKPT_DIR         : final output dir (default: checkpoints)
#    CKPT_JA_DIR      : Stage 1 dir (default: checkpoints_facts_ja)
#    CKPT_EN_DIR      : Stage 2 dir (default: checkpoints_facts_en)
#    CKPT_TRANSL_DIR  : Stage 3 dir (default: checkpoints_transl)
#    COPY_TO_TRANSL   : also copy final to translation_resources/ (default: 1)
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-checkpoints}"
CKPT_JA_DIR="${CKPT_JA_DIR:-checkpoints_facts_ja}"
CKPT_EN_DIR="${CKPT_EN_DIR:-checkpoints_facts_en}"
CKPT_TRANSL_DIR="${CKPT_TRANSL_DIR:-checkpoints_transl}"
JA_STEPS="${JA_STEPS:-15000}"
EN_STEPS="${EN_STEPS:-15000}"
TRANSL_STEPS="${TRANSL_STEPS:-15000}"
COPY_TO_TRANSL="${COPY_TO_TRANSL:-1}"

FACTS_BASE_MODEL="${FACTS_BASE_MODEL:-$CKPT_DIR/ckpt_pretrain_final.safetensors}"

JA_DIR="sft_data/sft_facts_ja"
EN_DIR="sft_data/sft_facts_en"
TRANSL_DIR="sft_data/sft_translation_focused"

# ── Validation ─────────────────────────────────────────────────────────────
for d in "$JA_DIR" "$EN_DIR" "$TRANSL_DIR"; do
    if [ ! -f "$d/train.jsonl" ]; then
        echo "[ERROR] $d/train.jsonl not found."
        echo "        Stage1: python3 scripts/build_facts_dataset.py --mode ja_only"
        echo "        Stage2: python3 scripts/build_facts_dataset.py --mode en_dominant"
        echo "        Stage3: run the translation data pipeline first."
        exit 1
    fi
done
if [ ! -f "$FACTS_BASE_MODEL" ]; then
    echo "[ERROR] base model not found: $FACTS_BASE_MODEL"
    exit 1
fi

# ── Helpers ────────────────────────────────────────────────────────────────
# Clean a stage dir ONLY if there is no training state (fresh start). If a
# train_state exists we keep it so the pipeline can --resume. FORCE_RESTART=1
# wipes everything. Symlink-aware: clears contents but keeps the Drive link.
has_state() { [ -f "$1/train_state_sft.txt" ] || [ -f "$1/train_state_dpo.txt" ]; }
prepare_dir() {
    local d="$1"
    if [ "${FORCE_RESTART:-0}" = "1" ]; then
        [ -L "$d" ] && find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} + || { rm -rf "$d"; mkdir -p "$d"; }
        echo "  [force restart] cleared $d"
    elif has_state "$d"; then
        echo "  [resume] keeping existing state in $d"
    elif [ -L "$d" ]; then
        find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        echo "  cleared symlinked dir: $d"
    else
        rm -rf "$d"; mkdir -p "$d"
        echo "  prepared local dir: $d"
    fi
}
# Does the stage already have a final checkpoint (-> auto-advance)?
has_final() { [ -f "$1/ckpt_sft_final.safetensors" ]; }

run_sft() {
    local stage_name="$1" ckpt="$2" base_model="$3" sft_dir="$4" lr="$5" max_steps="$6"
    echo ""
    echo "═══ [$stage_name] SFT (steps=$max_steps) ═══"
    echo "  base model : $base_model"
    echo "  ckpt dir   : $ckpt"
    # If final already exists and no state to resume, auto-advance.
    if has_final "$ckpt" && ! has_state "$ckpt"; then
        echo "  -> final already present, skipping to next stage."
        return 0
    fi
    prepare_dir "$ckpt"
    python3 scripts/train_pipeline.py \
        --stages "sft" \
        --no_skip_done \
        --checkpoint_dir "$ckpt" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --sft_dir "$sft_dir" \
        --sft_base_model "$base_model" \
        --sft_lr "$lr" \
        --sft_epochs 50 \
        --sft_batch_size 20 \
        --sft_max_seq 256 \
        --sft_warmup 50 \
        --sft_max_steps "$max_steps" \
        --sft_save_every 200 \
        --sft_upload_every 600
    if [ ! -f "$ckpt/ckpt_sft_final.safetensors" ]; then
        echo "[ERROR] $stage_name did not produce ckpt_sft_final.safetensors"
        exit 1
    fi
}

# ── Stage 1: JA facts (from pretrain) ─────────────────────────────────────
run_sft "Stage 1: FACTS JA" "$CKPT_JA_DIR" "$FACTS_BASE_MODEL" "$JA_DIR" "1e-5" "$JA_STEPS"

# ── Stage 2: EN facts (from Stage 1 JA model) ─────────────────────────────
run_sft "Stage 2: FACTS EN" "$CKPT_EN_DIR" "$CKPT_JA_DIR/ckpt_sft_final.safetensors" "$EN_DIR" "1e-5" "$EN_STEPS"

# ── Stage 3: translation (from Stage 2 EN model) ──────────────────────────
run_sft "Stage 3: TRANSLATION" "$CKPT_TRANSL_DIR" "$CKPT_EN_DIR/ckpt_sft_final.safetensors" "$TRANSL_DIR" "3e-5" "$TRANSL_STEPS"

# ── Copy final to active locations ────────────────────────────────────────
echo ""
echo "═══ Copying final model to active locations ═══"
mkdir -p "$CKPT_DIR"
cp "$CKPT_TRANSL_DIR/ckpt_sft_final.safetensors" "$CKPT_DIR/ckpt_sft_final.safetensors"
echo "  ✅ $CKPT_DIR/ckpt_sft_final.safetensors"

if [ "$COPY_TO_TRANSL" = "1" ]; then
    mkdir -p translation_resources
    cp "$CKPT_TRANSL_DIR/ckpt_sft_final.safetensors" "translation_resources/ckpt_sft_final.safetensors"
    echo "  ✅ translation_resources/ckpt_sft_final.safetensors"
fi

echo ""
echo "✓ Done. Final model: $CKPT_DIR/ckpt_sft_final.safetensors"
echo "  Evaluate: python3 scripts/eval_facts.py --model $CKPT_EN_DIR/ckpt_sft_final.safetensors"
echo "            python3 scripts/eval_translation.py --model $CKPT_DIR/ckpt_sft_final.safetensors"