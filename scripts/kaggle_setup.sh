#!/bin/bash
# =============================================================================
# kaggle_setup.sh — Full setup + build + train on Kaggle (T4 x 2)
#
# Usage: Upload this project as a Kaggle dataset, then run this script
#        in a GPU T4 x2 notebook.
# =============================================================================
set -e

echo "═══════════════════════════════════════════════════════"
echo " Kaggle T4 x2 Setup — transformer_100m_cuda"
echo "═══════════════════════════════════════════════════════"

# ─── 1. Verify CUDA environment ─────────────────────────────────────────────
echo ""
echo "[1/5] Checking CUDA..."
nvidia-smi
nvcc --version

# ─── 2. Install dependencies (cuDNN should be pre-installed on Kaggle) ──────
echo ""
echo "[2/5] Checking dependencies..."
if ! ldconfig -p | grep -q libcudnn; then
    echo "Installing cuDNN..."
    apt-get update -qq && apt-get install -y -qq libcudnn8-dev
fi

if ! ldconfig -p | grep -q libnccl; then
    echo "Installing NCCL..."
    apt-get install -y -qq libnccl-dev
fi

echo "  cuDNN: $(ldconfig -p | grep cudnn | head -1)"
echo "  NCCL:  $(ldconfig -p | grep nccl | head -1)"

# ─── 3. Build ───────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Building project..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DUSE_DDP=ON
make -j$(nproc)
cd "$PROJECT_DIR"

# ─── 4. Verify build ────────────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying..."
ls -la build/train_pretrain_cuda build/train_sft_cuda

# ─── 5. Print usage ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] Ready! Example commands:"
echo ""
echo "  # Stage 1: Pretrain (2B tokens, DDP 2 GPU via launcher)"
echo "  ./scripts/launch_ddp.sh ./build/train_pretrain_cuda \\"
echo "    --data_dir data/corpus_final \\"
echo "    --vocab_dir vocab_6 \\"
echo "    --bpe bpe_v6.bin \\"
echo "    --checkpoint_dir /kaggle/working/checkpoints \\"
echo "    --batch_size 8 --max_seq 256 --max_tokens 2000000000 \\"
echo "    --save_every 500"
echo ""
echo "  # Stage 2: SFT (translation-focused, DDP 2 GPU)"
echo "  ./scripts/launch_ddp.sh ./build/train_sft_cuda --stage 2 \\"
echo "    --base_model /kaggle/working/checkpoints/ckpt_pretrain_final.safetensors \\"
echo "    --sft_dir sft_data/sft_translation_focused \\"
echo "    --vocab_dir vocab_6 --bpe bpe_v6.bin \\"
echo "    --batch_size 8 --max_seq 512 --epochs 1 --lr 3e-5 --grad_clip 1.0"
echo ""
echo "  # Stage 3: DPO (translation quality alignment, DDP 2 GPU)"
echo "  ./scripts/launch_ddp.sh ./build/train_sft_cuda --stage 3 \\"
echo "    --base_model /kaggle/working/checkpoints/ckpt_sft_final.safetensors \\"
echo "    --dpo_dir sft_data/dpo \\"
echo "    --vocab_dir vocab_6 --bpe bpe_v6.bin \\"
echo "    --batch_size 2 --max_seq 512 --epochs 1 --lr 1e-5 --dpo_beta 0.1 --grad_clip 1.0"
echo ""
echo "═══════════════════════════════════════════════════════"
echo " DDP Strategy: Multi-process via launch_ddp.sh (NCCL)"
echo "   - 2 processes, each owns 1 GPU"
echo "   - Both GPUs hold full model replica"
echo "   - Batch split across GPUs (effective_batch = batch × 2)"
echo "   - AllReduce gradients via NCCL after backward"
echo "   - ~1.8x throughput vs single GPU"
echo ""
echo " GPU memory budget (A100 40GB, FP16, per GPU):"
echo "   892M model FP16:  ~1.8 GB"
echo "   Optimizer states: ~10.7 GB (FP32 master + m + v)"
echo "   Activations:      ~2-4 GB (batch=6, seq=2048)"
echo "   DPO mode:         ~4-6 GB (2x forward, ref model)"
echo "   Peak per GPU:     ~5-8 GB / 16 GB available"
echo ""
echo " Data summary:"
echo "   Pretrain: data/corpus_final/ (ID/EN/JA/KO ~16GB)"
echo "   SFT:      sft_data/sft_clean/ (9 files, ~93MB)"
echo "   DPO:      sft_data/dpo/ (9 files, ~68MB)"
echo "   BPE:      bpe_v6.bin + vocab_6/"
echo ""
echo " Checkpoint format: FP16 safetensors (~200MB each)"
echo "═══════════════════════════════════════════════════════"
