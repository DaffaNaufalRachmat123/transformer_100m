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

# 5. RL (GRPO-lite, rule-based verifier)
python3 scripts/prepare_sft_coding.py --out_sft sft_data/sft_coding/train.jsonl --out_rl rl_data/coding.jsonl --n 4000
./build/train_sft_cuda --stage 5 \
  --base_model /kaggle/working/checkpoints/ckpt_sft_final.safetensors \
  --ref_model /kaggle/working/checkpoints/ckpt_sft_final.safetensors \
  --rl_dir rl_data \
  --verifier_cmd "python3 scripts/rl_verifier.py" \
  --rl_workdir rl_work \
  --batch_size 2 --n_samples 6 --rl_max_new 80 --rl_beta 0.1 \
  --epochs 3 --lr 1e-5
```

> **Coding SFT first (required for RL):** before running RL, fine-tune the policy
> on coding examples so it can actually write programs. One command does both:
> `bash scripts/run_rl_pipeline.sh` (coding SFT → RL → copy final).
> No BPE rebuild or pretrain rerun is needed — the 65536-token vocab already
> contains every code token (`def`, `print`, `input`, `==`, `(`, `)`, `{`, `}`, ...).

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
│   ├── train_cuda_unified.cu # Stage 2-5: SFT/DPO/Phone/RL
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
5. **RL** (Stage 5): GRPO-lite with a rule-based code verifier

## RL (Stage 5) — GRPO-lite

Stage 5 trains the policy with **online RL** using a rule-based reward (no neural
reward model / critic), the same recipe as DeepSeek-R1 / Qwen3 for verifiable tasks.

Per training step:
1. Sample **N completions per prompt** from the policy (top-k + temperature).
2. Decode each completion and run it through a **rule-based verifier**
   (`scripts/rl_verifier.py` executes the generated Python and compares stdout
   to the expected output).
3. Compute a **group-normalised advantage** `A[g] = (r - mean)/(std+eps)` per prompt.
4. Add a **per-token KL penalty** vs the frozen reference (SFT) model.
5. Backprop the masked-CE gradient scaled by `(A[g] - β·kl_t)/Nt`.

```
prepare_rl_coding.py  →  rl_data/*.jsonl   (prompt + mock_input + expected_output)
train_sft_cuda --stage 5 →  rl_work/req.jsonl → rl_verifier.py → rl_work/reward.jsonl
```

Key flags: `--n_samples` (default 6), `--rl_max_new` (80), `--rl_beta` (0.1),
`--rl_temp` (0.8), `--rl_top_k` (50), `--rl_dir`, `--rl_workdir`, `--verifier_cmd`.

> **Why it works at 204M:** RL amplifies completions the model can *already* get
> right after SFT. For simple, deterministic coding tasks with a dense verifiable
> reward this is achievable at ~200M. It is **not** a substitute for SFT — start
> from a good SFT checkpoint and use `--ref_model` for the KL anchor.

## Differences from Original (MLX) Project

- All Apple MLX / Metal dependencies removed
- ARM NEON intrinsics replaced with CUDA kernels
- IOKit GPU monitoring replaced with NVML
- Architecture constants fixed to 204M (was 476M in CUDA files)
- Proper residual connections in forward pass
- Cosine LR schedule with warmup
- Training state save/restore for Kaggle session resumption
