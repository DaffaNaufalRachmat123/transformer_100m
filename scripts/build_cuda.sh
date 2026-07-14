#!/bin/bash
# =============================================================================
# build_cuda.sh — Build transformer_100m for CUDA (T4 / sm_75)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "═══════════════════════════════════════════════════════"
echo " Building transformer_100m_cuda (sm_75 / T4)"
echo "═══════════════════════════════════════════════════════"

cd "$PROJECT_DIR"
mkdir -p build && cd build

# Detect if DDP should be enabled
DDP_FLAG=""
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${USE_DDP:-auto}" = "auto" ]; then
    if [ "$GPU_COUNT" -ge 2 ] 2>/dev/null; then
        DDP_FLAG="-DUSE_DDP=ON"
        echo "[DDP] Multi-GPU detected ($GPU_COUNT GPUs) — DDP enabled"
    else
        echo "[GPU] Single GPU detected — DDP disabled"
    fi
elif [ "${USE_DDP}" = "1" ]; then
    DDP_FLAG="-DUSE_DDP=ON"
    echo "[DDP] Multi-GPU enabled (forced)"
else
    echo "[GPU] DDP disabled (USE_DDP=0)"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    ${DDP_FLAG}

make -j$(nproc)

echo ""
echo "✓ Build complete. Binaries:"
echo "  build/train_pretrain_cuda  — Stage 1: Pretraining"
echo "  build/train_sft_cuda       — Stage 2-4: SFT/DPO/Phone"
echo ""
