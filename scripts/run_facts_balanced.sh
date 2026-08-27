#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_facts_balanced.sh — Option A: Oversampled EN+JA facts stage
#
#  Pipeline:
#    Stage 1 (Oversampled Facts): from ckpt_pretrain_final -> checkpoints_facts
#          strict category prompts, EN boost, hard-category weighting
#    Stage 2 (Translation):    from Stage 1              -> checkpoints_transl
#    Stage 3 (Coding SFT):     from Stage 2              -> checkpoints_rl_sft
#    Stage 4 (RL Coding):      from Stage 3              -> checkpoints_rl
#
#  Each stage auto-skips if its ckpt_*_final.safetensors already exists
#  and no train_state is present (completed run). Use FORCE_RESTART=1
#  to wipe and restart from scratch.
#
#  Usage:
#    bash scripts/run_facts_balanced.sh
#    FORCE_RESTART=1 bash scripts/run_facts_balanced.sh
#
#  Tunable (env):
#    FACTS_STEPS      : max steps Stage 1 (default: 20000)
#    FACTS_TARGET_ROWS: generated train rows if FACTS_DIR missing (default: 500000)
#    FACTS_EN_SHARE   : generated EN share if FACTS_DIR missing (default: 0.34)
#    FACTS_EVAL_ROWS  : generated eval rows if FACTS_DIR missing (default: 6000)
#    FACTS_PROFILE    : oversampling profile (default: v10_trilingual)
#    FACTS_DIR        : facts dataset dir (default: sft_data/sft_facts_combined_300k_v8_balanced_repair)
#    TRANSL_DIR       : translation replay dataset dir (default: sft_data/sft_translation_focused_v4_balanced_repair)
#    TRANSLATION_FRAC : translation share in Stage 2 replay dataset (default: 0.86)
#    TRANSL_FACT_FILES: curated fact files for Stage 2 replay (default: sft_facts_curated.jsonl)
#    TRANSL_EXTRA_FACT_DIRS: generated fact dirs sampled into Stage 2 replay (default: FACTS_DIR)
#    TRANSL_EN_REPAIR_CATEGORIES: EN categories to prioritize in Stage 2 replay
#    TRANSL_JA_PROTECT_CATEGORIES: JA categories to keep protected in Stage 2 replay
#    TRANSL_EN_REPAIR_SHARE: share of non-curated fact replay budget reserved for EN repair
#    TRANSL_JA_PROTECT_SHARE: share of non-curated fact replay budget reserved for JA protection
#    TRANSL_REBUILD_IF_MISSING: auto-generate TRANSL_DIR if missing (default: 1)
#    TRANSL_FORCE_REBUILD: wipe and rebuild TRANSL_DIR before Stage 2 prep (default: 0)
#    TRANSL_STEPS     : max steps Stage 2 (default: 15000)
#    CODING_STEPS     : max steps Stage 3 (default: -1 = run to completion)
#    RL_STEPS         : max steps Stage 4 (default: -1)
#    BASE_MODEL       : base model for Stage 1 (default: checkpoints/ckpt_pretrain_final.safetensors)
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-checkpoints}"
CKPT_FACTS="${CKPT_FACTS:-checkpoints_facts}"
CKPT_TRANSL="${CKPT_TRANSL:-checkpoints_transl}"
CKPT_CODING="${CKPT_CODING:-checkpoints_rl_sft}"
CKPT_RL="${CKPT_RL:-checkpoints_rl}"

FACTS_STEPS="${FACTS_STEPS:-20000}"
TRANSL_STEPS="${TRANSL_STEPS:-15000}"
CODING_STEPS="${CODING_STEPS:--1}"
RL_STEPS="${RL_STEPS:--1}"
FACTS_TARGET_ROWS="${FACTS_TARGET_ROWS:-500000}"
FACTS_EN_SHARE="${FACTS_EN_SHARE:-0.34}"
FACTS_EVAL_ROWS="${FACTS_EVAL_ROWS:-6000}"
FACTS_PROFILE="${FACTS_PROFILE:-v10_trilingual}"
TRANSLATION_FRAC="${TRANSLATION_FRAC:-0.86}"
TRANSL_FACT_FILES="${TRANSL_FACT_FILES:-sft_facts_curated.jsonl}"
TRANSL_EN_REPAIR_CATEGORIES="${TRANSL_EN_REPAIR_CATEGORIES:-official_language,highest_point,currency}"
TRANSL_JA_PROTECT_CATEGORIES="${TRANSL_JA_PROTECT_CATEGORIES:-population,capital,element_symbol,element_number,continent_of,highest_point,area}"
TRANSL_EN_REPAIR_SHARE="${TRANSL_EN_REPAIR_SHARE:-0.40}"
TRANSL_JA_PROTECT_SHARE="${TRANSL_JA_PROTECT_SHARE:-0.35}"
TRANSL_REBUILD_IF_MISSING="${TRANSL_REBUILD_IF_MISSING:-1}"
TRANSL_FORCE_REBUILD="${TRANSL_FORCE_REBUILD:-0}"

BASE_MODEL="${BASE_MODEL:-$CKPT_DIR/ckpt_pretrain_final.safetensors}"

FACTS_DIR="${FACTS_DIR:-sft_data/sft_facts_v10_trilingual}"
TRANSL_DIR="${TRANSL_DIR:-sft_data/sft_translation_focused_v4_balanced_repair}"
TRANSL_EXTRA_FACT_DIRS="${TRANSL_EXTRA_FACT_DIRS:-$FACTS_DIR}"
CODING_DIR="sft_data/sft_coding"
RL_DIR="rl_data"

# ── Validation ─────────────────────────────────────────────────────────────
if [ ! -f "$FACTS_DIR/train.jsonl" ]; then
    echo "[facts] $FACTS_DIR/train.jsonl not found; generating oversampled facts..."
    python3 scripts/build_facts_dataset.py \
        --mode oversample \
        --target_rows "$FACTS_TARGET_ROWS" \
        --en_share "$FACTS_EN_SHARE" \
        --eval_rows "$FACTS_EVAL_ROWS" \
        --profile "$FACTS_PROFILE" \
        --id_share 0.5 \
        --out_dir "$FACTS_DIR"
fi
if [ -f "scripts/audit_facts_dataset.py" ] && [ -f "$FACTS_DIR/eval.jsonl" ]; then
    python3 scripts/audit_facts_dataset.py "$FACTS_DIR/train.jsonl" "$FACTS_DIR/eval.jsonl" --max_category_share 0.30
fi

if [ "$TRANSL_FORCE_REBUILD" = "1" ] && [ -d "$TRANSL_DIR" ]; then
    rm -rf "$TRANSL_DIR"
    echo "[transl] force rebuild cleared $TRANSL_DIR"
fi
if [ ! -f "$TRANSL_DIR/train.jsonl" ]; then
    if [ "$TRANSL_REBUILD_IF_MISSING" != "1" ]; then
        echo "[ERROR] $TRANSL_DIR/train.jsonl not found and TRANSL_REBUILD_IF_MISSING=$TRANSL_REBUILD_IF_MISSING"
        exit 1
    fi
    echo "[transl] $TRANSL_DIR/train.jsonl not found; generating translation replay dataset..."
    python3 scripts/prepare_sft_translation_dominant.py \
        --out_dir "$TRANSL_DIR" \
        --translation_frac "$TRANSLATION_FRAC" \
        --fact_files "$TRANSL_FACT_FILES" \
        --extra_fact_dirs "$TRANSL_EXTRA_FACT_DIRS" \
        --en_repair_categories "$TRANSL_EN_REPAIR_CATEGORIES" \
        --ja_protect_categories "$TRANSL_JA_PROTECT_CATEGORIES" \
        --en_repair_share "$TRANSL_EN_REPAIR_SHARE" \
        --ja_protect_share "$TRANSL_JA_PROTECT_SHARE"
    if [ -f "scripts/clean_translation_quality.py" ]; then
        python3 scripts/clean_translation_quality.py --sft "$TRANSL_DIR/train.jsonl"
    fi
fi

for d in "$FACTS_DIR" "$TRANSL_DIR" "$CODING_DIR"; do
    if [ ! -f "$d/train.jsonl" ]; then
        echo "[ERROR] $d/train.jsonl not found."
        echo "        Stage1: python3 scripts/build_facts_dataset.py --mode oversample --target_rows $FACTS_TARGET_ROWS --profile $FACTS_PROFILE --out_dir $FACTS_DIR"
        echo "        Stage2: python3 scripts/prepare_sft_translation_dominant.py --out_dir $TRANSL_DIR --translation_frac $TRANSLATION_FRAC --extra_fact_dirs $TRANSL_EXTRA_FACT_DIRS --en_repair_categories $TRANSL_EN_REPAIR_CATEGORIES --ja_protect_categories $TRANSL_JA_PROTECT_CATEGORIES"
        echo "        Stage3: python3 scripts/build_sft_coding_dataset.py --n 4000"
        exit 1
    fi
done
if [ ! -f "$BASE_MODEL" ]; then
    echo "[ERROR] base model not found: $BASE_MODEL"
    exit 1
fi
if [ ! -f "$RL_DIR/coding.jsonl" ]; then
    echo "[ERROR] $RL_DIR/coding.jsonl not found."
    echo "        Run: python3 scripts/build_sft_coding_dataset.py --n 4000"
    exit 1
fi

count_rows() {
    if [ -f "$1/train.jsonl" ]; then
        wc -l < "$1/train.jsonl" | tr -d ' '
    else
        echo "0"
    fi
}

FACTS_ROWS="$(count_rows "$FACTS_DIR")"
TRANSL_ROWS="$(count_rows "$TRANSL_DIR")"

# ── Helpers ────────────────────────────────────────────────────────────────
has_state() { [ -f "$1/train_state_sft.txt" ] || [ -f "$1/train_state_rl.txt" ]; }
prepare_dir() {
    local d="$1"
    if [ "${FORCE_RESTART:-0}" = "1" ]; then
        if [ -L "$d" ]; then
            find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
        else
            rm -rf "$d"; mkdir -p "$d"
        fi
        echo "  [force restart] cleared $d"
    elif has_state "$d"; then
        echo "  [resume] keeping existing state in $d"
    elif [ -L "$d" ]; then
        find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
        echo "  cleared symlinked dir: $d"
    else
        rm -rf "$d"; mkdir -p "$d"
        echo "  prepared local dir: $d"
    fi
}
has_final() { [ -f "$1/ckpt_sft_final.safetensors" ] || [ -f "$1/ckpt_rl_final.safetensors" ]; }

# ── Stage 1: Oversampled EN+JA Facts (from pretrained) ────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Stage 1: Oversampled Facts (strict prompts, EN boost, from pretrained)"
echo "═══════════════════════════════════════════════════════════════════"
echo "  base model : $BASE_MODEL"
echo "  ckpt dir   : $CKPT_FACTS"
echo "  data       : $FACTS_DIR ($FACTS_ROWS rows)"
echo "  max steps  : $FACTS_STEPS"
echo "  target rows: $FACTS_TARGET_ROWS (EN share $FACTS_EN_SHARE, profile $FACTS_PROFILE)"

if has_final "$CKPT_FACTS" && ! has_state "$CKPT_FACTS"; then
    echo "  -> final already present, skipping."
else
    prepare_dir "$CKPT_FACTS"
    python3 scripts/train_pipeline.py \
        --stages "sft" \
        --no_skip_done \
        --checkpoint_dir "$CKPT_FACTS" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --sft_dir "$FACTS_DIR" \
        --sft_base_model "$BASE_MODEL" \
        --sft_lr 1e-5 \
        --sft_epochs 2 \
        --sft_batch_size 20 \
        --sft_max_seq 256 \
        --sft_warmup 50 \
        --sft_max_steps "$FACTS_STEPS" \
        --sft_save_every 200 \
        --sft_upload_every 600
    if [ ! -f "$CKPT_FACTS/ckpt_sft_final.safetensors" ]; then
        echo "[ERROR] Stage 1 did not produce ckpt_sft_final.safetensors"
        exit 1
    fi
fi

# ── Stage 2: Translation (from Stage 1) ───────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Stage 2: Translation (EN↔JA, from facts model)"
echo "═══════════════════════════════════════════════════════════════════"
echo "  base model : $CKPT_FACTS/ckpt_sft_final.safetensors"
echo "  ckpt dir   : $CKPT_TRANSL"
echo "  data       : $TRANSL_DIR ($TRANSL_ROWS rows, translation frac $TRANSLATION_FRAC)"
echo "  max steps  : $TRANSL_STEPS"

if has_final "$CKPT_TRANSL" && ! has_state "$CKPT_TRANSL"; then
    echo "  -> final already present, skipping."
else
    prepare_dir "$CKPT_TRANSL"
    python3 scripts/train_pipeline.py \
        --stages "sft" \
        --no_skip_done \
        --checkpoint_dir "$CKPT_TRANSL" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --sft_dir "$TRANSL_DIR" \
        --sft_base_model "$CKPT_FACTS/ckpt_sft_final.safetensors" \
        --sft_lr 3e-5 \
        --sft_epochs 1 \
        --sft_batch_size 20 \
        --sft_max_seq 256 \
        --sft_warmup 50 \
        --sft_max_steps "$TRANSL_STEPS" \
        --sft_save_every 200 \
        --sft_upload_every 600
    if [ ! -f "$CKPT_TRANSL/ckpt_sft_final.safetensors" ]; then
        echo "[ERROR] Stage 2 did not produce ckpt_sft_final.safetensors"
        exit 1
    fi
fi

# ── Stage 3: Coding SFT (from Stage 2) ────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Stage 3: Coding SFT (Python code generation)"
echo "═══════════════════════════════════════════════════════════════════"
echo "  base model : $CKPT_TRANSL/ckpt_sft_final.safetensors"
echo "  ckpt dir   : $CKPT_CODING"
echo "  data       : $CODING_DIR (312 rows)"
echo "  max steps  : $CODING_STEPS"

if has_final "$CKPT_CODING" && ! has_state "$CKPT_CODING"; then
    echo "  -> final already present, skipping."
else
    prepare_dir "$CKPT_CODING"
    python3 scripts/train_pipeline.py \
        --stages "sft" \
        --no_skip_done \
        --checkpoint_dir "$CKPT_CODING" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --sft_dir "$CODING_DIR" \
        --sft_base_model "$CKPT_TRANSL/ckpt_sft_final.safetensors" \
        --sft_lr 2e-5 \
        --sft_epochs 20 \
        --sft_batch_size 16 \
        --sft_max_seq 512 \
        --sft_warmup 50 \
        --sft_save_every 200 \
        --sft_max_steps "$CODING_STEPS" \
        "${JOB_FLAGS[@]}"
    if [ ! -f "$CKPT_CODING/ckpt_sft_final.safetensors" ]; then
        CKPT_CODING_BASE=$(ls -t "$CKPT_CODING"/ckpt_sft_step*.safetensors 2>/dev/null | head -1)
        if [ -n "$CKPT_CODING_BASE" ]; then
            echo "[Stage 3] max_steps hit; using latest step: $CKPT_CODING_BASE"
            cp "$CKPT_CODING_BASE" "$CKPT_CODING/ckpt_sft_final.safetensors"
        else
            echo "[ERROR] Stage 3 produced no checkpoint."
            exit 1
        fi
    fi
fi

# ── Stage 4: RL Coding (GRPO-lite, from Stage 3) ──────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Stage 4: RL Coding (GRPO-lite)"
echo "═══════════════════════════════════════════════════════════════════"
echo "  policy+ref : $CKPT_CODING/ckpt_sft_final.safetensors"
echo "  ckpt dir   : $CKPT_RL"
echo "  data       : $RL_DIR/coding.jsonl"
echo "  max steps  : $RL_STEPS"

if has_final "$CKPT_RL" && ! has_state "$CKPT_RL"; then
    echo "  -> final already present, skipping."
else
    mkdir -p "$CKPT_RL"
    if [ "${FORCE_RESTART:-0}" = "1" ]; then
        rm -rf "$CKPT_RL/rl_work"
        echo "  [force restart] cleared rl_work"
    fi
    ./build/train_sft_cuda --stage 5 \
        --base_model "$CKPT_CODING/ckpt_sft_final.safetensors" \
        --ref_model "$CKPT_CODING/ckpt_sft_final.safetensors" \
        --rl_dir "$RL_DIR" \
        --rl_workdir "$CKPT_RL/rl_work" \
        --checkpoint_dir "$CKPT_RL" \
        --vocab_dir vocab_6 \
        --bpe bpe_v6.bin \
        --batch_size 2 \
        --n_samples 6 \
        --rl_max_new 80 \
        --rl_beta 0.1 \
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
    if [ ! -f "$CKPT_RL/ckpt_rl_final.safetensors" ]; then
        echo "[ERROR] Stage 4 did not produce ckpt_rl_final.safetensors"
        exit 1
    fi
fi

# ── Copy finals to active locations ────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " Copying final models"
echo "═══════════════════════════════════════════════════════════════════"

mkdir -p "$CKPT_DIR" translation_resources

cp "$CKPT_FACTS/ckpt_sft_final.safetensors" "$CKPT_DIR/ckpt_sft_final.safetensors"
echo "  ✅ $CKPT_DIR/ckpt_sft_final.safetensors"

cp "$CKPT_FACTS/ckpt_sft_final.safetensors" "translation_resources/ckpt_sft_final.safetensors"
echo "  ✅ translation_resources/ckpt_sft_final.safetensors"

if [ -f "$CKPT_RL/ckpt_rl_final.safetensors" ]; then
    cp "$CKPT_RL/ckpt_rl_final.safetensors" "$CKPT_DIR/ckpt_rl_final.safetensors"
    echo "  ✅ $CKPT_DIR/ckpt_rl_final.safetensors"
fi

echo ""
echo "✓ Pipeline complete."
echo "  Evaluate facts:  python3 scripts/eval_facts.py --model $CKPT_FACTS/ckpt_sft_final.safetensors --eval $FACTS_DIR/eval.jsonl --max_test 200"
echo "  Evaluate translation: python3 scripts/eval_translation.py --model $CKPT_TRANSL/ckpt_sft_final.safetensors"
