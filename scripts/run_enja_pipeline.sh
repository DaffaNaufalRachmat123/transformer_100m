#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  run_enja_pipeline.sh — Re-run SFT (translation-focused) ONLY.
#
#  DPO is intentionally SKIPPED: the EN<->JA DPO preference data is tiny and
#  low-quality, so it hurt the model more than it helped (margin exploded).
#  For a 892M translation-focused model, SFT with clean parallel data is the
#  right call.
#
#  Manual steps BEFORE running (CRITICAL):
#
#    1. REBUILD train_sft_cuda with the unified source.
#
#    2. Remove the old SFT final checkpoint so the pipeline re-runs SFT:
#         rm -f checkpoints/ckpt_sft_final.safetensors \
#               checkpoints/train_state_sft.txt
#
#  Usage:
#    bash scripts/run_enja_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════

set -e
cd /kaggle/working/transformer_100m_cuda

# ── Sanity: ensure the translation-focused data exists ─────────────────────
if [ ! -f "sft_data/sft_translation_focused/train.jsonl" ]; then
    echo "[ERROR] sft_data/sft_translation_focused/train.jsonl not found."
    echo "        Run: python3 scripts/prepare_sft_translation_dominant.py"
    exit 1
fi

# ── Run SFT (translation-focused) ──────────────────────────────────────────
python3 scripts/train_pipeline.py \
    --stages "sft" \
    --checkpoint_dir /kaggle/working/checkpoints \
    --vocab_dir vocab_6 \
    --bpe bpe_v6.bin \
    --log_every 5 \
    --upload_every 600 \
    --ddp --num_gpus 2 \
    --resume \
    --sft_dir sft_data/sft_translation_focused \
    --sft_lr 3e-5 \
    --sft_epochs 1 \
    --sft_batch_size 8 \
    --sft_max_seq 512 \
    --sft_warmup 100 \
    --sft_save_every 300 \
    --sft_upload_every 600