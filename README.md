# Transformer 204M — CUDA Edition

Standalone CUDA C++ implementation of a 204M parameter quadrilingual transformer.
Designed to run on **Kaggle T4 x 2** GPUs.

## Architecture

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| Layers | 18 |
| Query Heads | 12 (GQA) |
| KV Heads | 3 (4:1 ratio) |
| d_head | 64 |
| FFN dim | 3072 (SwiGLU) |
| Vocab | 65,536 (BPE) |
| Max seq | 2048 |
| Languages | EN, JA |
| RoPE θ | 500,000 |
| Weight tying | Yes |
| Parameters | 204.3M

## Quick Start (Kaggle)

```bash
# 1. Build
bash scripts/kaggle_setup.sh

# 2. Pretrain
./build/train_pretrain_cuda \
  --data_dir /kaggle/input/corpus/corpus_final \
  --vocab_dir vocab_6 \
  --checkpoint_dir /kaggle/working/checkpoints \
  --batch_size 8 --max_tokens 1000000000 \
  --ddp --num_gpus 2

# 3. SFT
./build/train_sft_cuda --stage 2 \
  --base_model /kaggle/working/checkpoints/ckpt_pretrain_final.safetensors \
  --sft_dir /kaggle/input/sft_data/sft_clean \
  --batch_size 4 --epochs 3

# 4. DPO
./build/train_sft_cuda --stage 3 \
  --base_model /kaggle/working/checkpoints/ckpt_sft.safetensors \
  --dpo_dir /kaggle/input/sft_data/dpo \
  --batch_size 2 --epochs 1
```

## Project Structure

```
transformer_100m_cuda/
├── CMakeLists.txt           # Top-level build (CUDA + CXX)
├── cuda/                    # CUDA backend (tensor, kernels, ops, context)
│   ├── cuda_tensor.h        # GPU tensor (replaces mx::array)
│   ├── cuda_context.h/cu    # Device init, cuBLAS/cuDNN handles
│   ├── cuda_ops.h/cu        # High-level ops (matmul, RMSNorm, RoPE, SwiGLU)
│   ├── cuda_kernels.cuh/cu  # Fused CUDA kernels
│   ├── cuda_autograd.h      # Manual backward with gradient checkpointing
│   ├── cuda_monitor.h/cu    # NVML GPU monitoring
│   └── cuda_allocator.h     # Memory pool
├── bpe/                     # BPE tokenizer (pure C++)
│   ├── bpe_tokenizer.h
│   └── bpe_tokenizer.cpp
├── transformer/             # Model + training
│   ├── train_cuda.cu        # Stage 1: Pretrain (100M arch)
│   ├── train_cuda_unified.cu # Stage 2-4: SFT/DPO/Phone
│   ├── adamw_cuda.cuh       # Fused AdamW optimizer
│   ├── model_config.h/cpp   # Architecture config
│   ├── pretrain_dataset.h/cpp # Streaming corpus loader
│   ├── safetensors.h        # Checkpoint I/O
│   └── lora.h/cpp           # LoRA adapters
├── tools/                   # Utilities
│   └── infer.cpp            # Inference engine
├── vocab_6/                 # BPE vocabulary + merges
│   ├── vocab.txt
│   └── merges.txt
└── scripts/
    ├── build_cuda.sh        # Build script
    └── kaggle_setup.sh      # Full Kaggle setup + build
```

## Hardware Requirements

- **GPU**: NVIDIA T4 (16GB, sm_75) × 1 or 2
- **CUDA**: 11.8+ (12.x recommended)
- **cuDNN**: 8.9+
- **NCCL**: 2.18+ (for DDP multi-GPU)
- **RAM**: 16GB+ system memory
- **Disk**: ~50GB for corpus data

## Memory Budget (T4 16GB)

| Component | Size |
|-----------|------|
| Model params (FP16) | ~409 MB |
| Master weights (FP32) | ~817 MB |
| Optimizer states (2×FP32) | ~1,635 MB |
| Activations (batch=24, seq=256) | ~3 GB |
| KV cache + workspace | ~1 GB |
| **Total** | **~6.9 GB** |

Leaves ~11.6 GB headroom per GPU for larger batches or longer sequences.

## Training Pipeline

1. **Pretrain** (Stage 1): Curriculum learning with seq lengths 256→512→1024→2048
2. **SFT** (Stage 2): Supervised fine-tuning on assistant responses
3. **DPO** (Stage 3): Direct Preference Optimization with chosen/rejected pairs
4. **Phone** (Stage 4): Phone assistant adapter (conservative LR)

## Differences from Original (MLX) Project

- All Apple MLX / Metal dependencies removed
- ARM NEON intrinsics replaced with CUDA kernels
- IOKit GPU monitoring replaced with NVML
- Architecture constants fixed to 204M (was 476M in CUDA files)
- Proper residual connections in forward pass
- Cosine LR schedule with warmup
- Training state save/restore for Kaggle session resumption
