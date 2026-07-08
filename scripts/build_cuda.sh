#!/bin/bash
# =============================================================================
# build_cuda.sh — Build transformer_100m for CUDA
#
# Usage:
#   ./scripts/build_cuda.sh              # Default: T4 (Kaggle)
#   ./scripts/build_cuda.sh A100         # NVIDIA A100
#   ./scripts/build_cuda.sh RTX4090      # RTX 4090
#   ./scripts/build_cuda.sh ALL          # Fat binary (all GPUs)
#
# Supported GPU_TARGET: T4, V100, A100, RTX3090, RTX4090, H100, ALL
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# GPU target from argument or env, default T4
GPU_TARGET="${1:-${GPU_TARGET:-T4}}"

echo "═══════════════════════════════════════════════════════"
echo " Building transformer_100m_cuda"
echo " GPU Target: ${GPU_TARGET}"
echo "═══════════════════════════════════════════════════════"

cd "$PROJECT_DIR"
mkdir -p build && cd build

# Detect if DDP should be enabled
DDP_FLAG=""
if [ "${USE_DDP:-0}" = "1" ]; then
    DDP_FLAG="-DUSE_DDP=ON"
    echo "[DDP] Multi-GPU enabled"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGET=${GPU_TARGET} \
    ${DDP_FLAG}

make -j$(nproc)

echo ""
echo "✓ Build complete. Binaries:"
echo "  build/train_pretrain_cuda  — Stage 1: Pretraining"
echo "  build/train_sft_cuda       — Stage 2-4: SFT/DPO/Phone"
echo ""
