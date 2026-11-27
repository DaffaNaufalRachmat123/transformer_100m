#!/usr/bin/env bash
# =============================================================================
#  apply_midtrain_addons.sh — Inject high-signal Mid-Training Annealing shards
#  (GSM8K-JA, MetaMathQA, Phi-1 Python CodeExercises, Science/Commonsense)
#  into corpus_pretrain_v8 for the remaining 47% -> 100% cosine decay.
#
#  Modes:
#    1. Local / Vast.ai (writable corpus directory):
#         bash scripts/apply_midtrain_addons.sh [TARGET_CORPUS_DIR]
#       Copies the 3 mid-training .txt shards (~278 MB) directly into math/,
#       code/, and mixed/ inside TARGET_CORPUS_DIR (default: data/corpus_pretrain_v8).
#
#    2. Kaggle Read-Only Input Overlay (0 MB copy of 26.4 GB corpus!):
#         bash scripts/apply_midtrain_addons.sh --kaggle-overlay \
#           /kaggle/input/datasets/naufalrachmat/datatraining/data/corpus_pretrain_v8 \
#           /kaggle/working/corpus_pretrain_v8_ceiling
#       Symlinks every existing bucket file from read-only /kaggle/input and links
#       the 3 new midtrain_*.txt shards so PretrainDataset::discover_files() reads
#       all 26.4 GB + 278 MB without re-uploading the 26 GB Kaggle dataset!
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

ADDON_DIR="${ADDON_DIR:-$PROJECT_DIR/data/pretrain_midtrain_addons}"

if [ ! -d "$ADDON_DIR" ]; then
    echo "[ERR] $ADDON_DIR not found."
    exit 1
fi

if [ "${1:-}" = "--kaggle-overlay" ]; then
    RO_SRC="${2:?Usage: $0 --kaggle-overlay <READONLY_CORPUS_V8> <WRITABLE_OVERLAY_DIR>}"
    OV_DST="${3:?Usage: $0 --kaggle-overlay <READONLY_CORPUS_V8> <WRITABLE_OVERLAY_DIR>}"
    for bucket in english japanese code math mixed; do
        mkdir -p "$OV_DST/$bucket"
        if [ -d "$RO_SRC/$bucket" ]; then
            for f in "$RO_SRC/$bucket"/*; do
                [ -e "$f" ] || continue
                ln -sf "$f" "$OV_DST/$bucket/$(basename "$f")"
            done
        fi
    done
    for bucket in math code mixed; do
        if [ -d "$ADDON_DIR/$bucket" ]; then
            for f in "$ADDON_DIR/$bucket"/*.txt; do
                [ -f "$f" ] || continue
                ln -sf "$f" "$OV_DST/$bucket/$(basename "$f")"
                echo "[OK] Linked midtrain shard $(basename "$f") -> $OV_DST/$bucket/"
            done
        fi
    done
    echo "[DONE] Kaggle 0-byte symlink overlay ready at: $OV_DST"
    echo "       Pass --data_dir $OV_DST to train_pipeline.py / train_pretrain_cuda."
    exit 0
fi

TARGET_DIR="${1:-$PROJECT_DIR/data/corpus_pretrain_v8}"
for bucket in math code mixed; do
    mkdir -p "$TARGET_DIR/$bucket"
    if [ -d "$ADDON_DIR/$bucket" ]; then
        for f in "$ADDON_DIR/$bucket"/*.txt; do
            [ -f "$f" ] || continue
            cp -f "$f" "$TARGET_DIR/$bucket/$(basename "$f")"
            echo "[OK] Installed $(basename "$f") -> $TARGET_DIR/$bucket/ ($(du -h "$f" | cut -f1))"
        done
    fi
done

echo ""
echo "[DONE] Mid-training shards installed into $TARGET_DIR."
