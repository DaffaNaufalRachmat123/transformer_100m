#!/bin/bash
# =============================================================================
# build_cuda.sh — Build transformer_100m_cuda for any CUDA GPU
#
# Arch is auto-detected from nvidia-smi compute capability; override with:
#   CUDA_ARCH=80 bash scripts/build_cuda.sh        (A100)
#   CUDA_ARCH=89 bash scripts/build_cuda.sh        (RTX 4090 / L40S)
#   CUDA_ARCH=75 bash scripts/build_cuda.sh        (T4)
#   CUDA_ARCH=90 bash scripts/build_cuda.sh        (H100)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── GPU architecture detection ─────────────────────────────────────────────
ARCH="${CUDA_ARCH:-}"
if [ -z "$ARCH" ]; then
    # nvidia-smi returns e.g. "8.0", "8.9", "9.0" -> strip the dot -> "80"/"89"/"90"
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    ARCH=$(echo "$CC" | tr -d '.')
fi
if [ -z "$ARCH" ]; then
    echo "[WARN] cannot detect compute capability — defaulting to 80 (A100)"
    ARCH=80
fi
echo "═══════════════════════════════════════════════════════"
echo " Building transformer_100m_cuda (sm_${ARCH})"
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
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ${DDP_FLAG}

make -j$(nproc)

echo ""
echo "✓ Build complete. Binaries:"
echo "  build/train_pretrain_cuda  — Stage 1: Pretraining"
echo "  build/train_sft_cuda       — Stage 2-5: SFT/DPO/Phone/RL"
echo ""
