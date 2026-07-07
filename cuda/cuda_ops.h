#pragma once
// =============================================================================
// cuda_ops.h — High-level CUDA operation declarations
// Replaces all mx:: math operations with cuBLAS + custom kernel calls.
// =============================================================================
#include "cuda_tensor.h"
#include "cuda_context.h"
#include "cuda_kernels.cuh"

namespace cuda {

// ─── Allocation ────────────────────────────────────────────────────────────
CudaTensor zeros_like(const CudaTensor& t);
CudaTensor ones_like(const CudaTensor& t);

// ─── Type conversion ───────────────────────────────────────────────────────
CudaTensor astype(const CudaTensor& src, DType target_dtype);

// ─── Core math ─────────────────────────────────────────────────────────────
CudaTensor matmul(const CudaTensor& A, const CudaTensor& B,
                  bool trans_a = false, bool trans_b = false);

CudaTensor matmul_transb(const CudaTensor& A, const CudaTensor& B);

CudaTensor rms_norm(const CudaTensor& x, const CudaTensor& gamma, float eps,
                    CudaTensor& saved_rsqrt, CudaTensor& saved_normed);

void rms_norm_backward(const CudaTensor& dy, const CudaTensor& gamma,
                       const CudaTensor& saved_rsqrt, const CudaTensor& saved_normed,
                       CudaTensor& dx, CudaTensor& dgamma, float eps);

void rope_inplace(CudaTensor& x, int num_heads, int d_head, int offset,
                  const float* cos_table, const float* sin_table);

void rope_backward_inplace(CudaTensor& dx, int num_heads, int d_head, int offset,
                           const float* cos_table, const float* sin_table);

// ─── Flash attention (via cuDNN) ───────────────────────────────────────────
CudaTensor flash_attention(const CudaTensor& Q, const CudaTensor& K, const CudaTensor& V,
                           float scale, bool causal);

// ─── SwiGLU ────────────────────────────────────────────────────────────────
CudaTensor swiglu(const CudaTensor& gate, const CudaTensor& up,
                  CudaTensor& saved_silu);

void swiglu_backward(const CudaTensor& dy, const CudaTensor& gate,
                     const CudaTensor& up, const CudaTensor& saved_silu,
                     CudaTensor& dgate, CudaTensor& dup);

// ─── Cross-entropy ─────────────────────────────────────────────────────────
CudaTensor cross_entropy_loss(const CudaTensor& logits, const CudaTensor& targets,
                              const CudaTensor& mask);

void cross_entropy_backward(const CudaTensor& logits, const CudaTensor& targets,
                            const CudaTensor& mask, CudaTensor& d_logits);

// ─── DPO utilities ─────────────────────────────────────────────────────────
// Compute sum of masked log probabilities: sum(mask * log_softmax(logits)[targets])
CudaTensor masked_log_probs(const CudaTensor& logits, const CudaTensor& targets,
                            const CudaTensor& mask);

// Scale all FP16 elements in-place by a scalar
void scale_fp16(CudaTensor& t, float scale);

// ─── Embedding ─────────────────────────────────────────────────────────────
CudaTensor embedding_forward(const CudaTensor& embed, const CudaTensor& token_ids);

// ─── Shape ops (zero-copy) ─────────────────────────────────────────────────
// reshape() is defined inline in cuda_tensor.h
CudaTensor transpose(CudaTensor& src, const std::vector<int>& axes);
CudaTensor slice(CudaTensor& src, int64_t start, int64_t end, int axis = 0);

// ─── Reduction ─────────────────────────────────────────────────────────────
CudaTensor sum_all(const CudaTensor& t);
CudaTensor sum_axis(const CudaTensor& t, int axis);

// ─── AdamW ─────────────────────────────────────────────────────────────────
void adamw_step(CudaTensor& param, CudaTensor& master_param,
                CudaTensor& m, CudaTensor& v,
                const CudaTensor& grad,
                float lr, float beta1, float beta2,
                float eps, float wd, int step);

} // namespace cuda
