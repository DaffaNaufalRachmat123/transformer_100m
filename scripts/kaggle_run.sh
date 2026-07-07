#!/bin/bash
# =============================================================================
# kaggle_run.sh — Copy-paste this into a Kaggle notebook cell
#
# Prerequisites:
#   - GPU: T4 x 2
#   - Dataset: uploaded as "transformer-100m-cuda" by user naufalrachmat
# =============================================================================

# ─── 1. Copy project from input (read-only) to working (read-write) ────────
# NOTE: Only copy code + sft/dpo data. Pretrain corpus stays in /kaggle/input (read-only is fine for streaming)
echo "[1/4] Copying project to /kaggle/working..."
INPUT_DIR="/kaggle/input/transformercuda100m/transformer_100m_cuda"
WORK_DIR="/kaggle/working/transformer_100m_cuda"

mkdir -p "$WORK_DIR"
# Copy everything EXCEPT the large corpus data
rsync -a --exclude='data/corpus_final' "$INPUT_DIR/" "$WORK_DIR/"
# Remove data folder if it got copied (save disk space)
rm -rf "$WORK_DIR/data"
chmod +x "$WORK_DIR/scripts/"*.sh

cd "$WORK_DIR"
echo "  Done. Size: $(du -sh . | cut -f1)"
echo "  Corpus data (read from input): $INPUT_DIR/data/corpus_final"

# ─── 2. Install dependencies ───────────────────────────────────────────────
echo ""
echo "[2/4] Checking dependencies..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvcc --version | grep release

# cuDNN & NCCL (usually pre-installed on Kaggle)
if ! ldconfig -p | grep -q libcudnn; then
    apt-get update -qq && apt-get install -y -qq libcudnn8-dev
fi
if ! ldconfig -p | grep -q libnccl; then
    apt-get install -y -qq libnccl-dev libnccl2
fi

# ─── 3. Build ──────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Building (DDP enabled)..."
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DUSE_DDP=ON \
    2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -3
cd ..

echo ""
echo "[4/4] Build complete!"
ls -la build/train_pretrain_cuda build/train_sft_cuda
echo ""
echo "════════════════════════════════════════════════════"
echo " Ready to train! Run the cells below."
echo "════════════════════════════════════════════════════"
