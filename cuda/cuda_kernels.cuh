#pragma once
// =============================================================================
// cuda_kernels.cuh — Fused CUDA kernel declarations
//
// All kernels are designed to minimize launch overhead by fusing multiple
// operations into single kernel launches.
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ─── RMSNorm fused (forward + backward) ────────────────────────────────────
// Forward:  y[i] = x[i] * rsqrt(mean(x²) + eps) * gamma[col]
// Backward: computes dgamma and dx in same kernel
// Launch:   block=(256,1,1), grid=(B*S, 1, 1)  — one block per row
void rms_norm_fused_kernel_forward(
    const __half* x, const __half* gamma,
    __half* y, __half* saved_rsqrt, __half* saved_normed,
    int B, int S, int D, float eps,
    cudaStream_t stream);

void rms_norm_fused_kernel_backward(
    const __half* dy, const __half* gamma,
    const __half* saved_rsqrt, const __half* saved_normed,
    __half* dx, __half* dgamma,
    int B, int S, int D, float eps,
    cudaStream_t stream);

// ─── RoPE fused (in-place Q and K rotation) ────────────────────────────────
// Uses pre-computed cos/sin tables in device memory
// Launch: block=(256,1,1), grid=((B*S*NUM_HEADS*D_HEAD/2 + 255)/256, 1, 1)
void rope_fused_forward(
    __half* x, const float* cos_table, const float* sin_table,
    int B, int S, int num_heads, int d_head, int offset,
    cudaStream_t stream);

void rope_fused_backward(
    __half* dx, const float* cos_table, const float* sin_table,
    int B, int S, int num_heads, int d_head, int offset,
    cudaStream_t stream);

// ─── SwiGLU fused (gate * silu(up)) ────────────────────────────────────────
// Forward:  y = gate * sigmoid(up) * up
// Backward: dgate, dup
// Launch: block=(256,1,1), grid=((B*S*FFN_DIM + 255)/256, 1, 1)
void swiglu_fused_forward(
    const __half* gate, const __half* up,
    __half* y, __half* saved_silu,
    int B, int S, int FD,
    cudaStream_t stream);

void swiglu_fused_backward(
    const __half* dy, const __half* gate, const __half* up,
    const __half* saved_silu,
    __half* dgate, __half* dup,
    int B, int S, int FD,
    cudaStream_t stream);

// ─── Cross-entropy fused (log_softmax + NLL + mask + mean) ─────────────────
// In:  logits [B,S,V], targets [B,S], mask [B,S]
// Out: loss scalar, d_logits [B,S,V]
// Launch: block=(256), grid=(B*S/4)
void cross_entropy_fused_forward(
    const __half* logits, const int* targets, const float* mask,
    float* loss_out, int B, int S, int V,
    cudaStream_t stream);

void cross_entropy_fused_backward(
    const __half* logits, const int* targets, const float* mask,
    __half* d_logits, int B, int S, int V,
    cudaStream_t stream);

// ─── AdamW fused update ────────────────────────────────────────────────────
// m = b1*m + (1-b1)*g | v = b2*v + (1-b2)*g² | p -= lr * m_hat/(√v_hat + eps) - wd*p
// Launch: block=(256,1,1), grid=((numel + 255)/256, 1, 1)
void adamw_fused_update(
    __half* param, float* master_param,
    float* m, float* v,
    const __half* grad,
    size_t numel, float lr, float beta1, float beta2,
    float eps, float wd, int step,
    cudaStream_t stream);

// ─── Adafactor fused update ────────────────────────────────────────────────
void adafactor_fused_update_2d(
    __half* param, float* master_param,
    float* Vr, float* Vc,
    const __half* grad,
    int rows, int cols, float lr, float beta2, float eps1,
    float rms_scale, int step,
    cudaStream_t stream);

// ─── Embedding gather ──────────────────────────────────────────────────────
// out[i] = embed[token_ids[i]]
// Launch: block=(256), grid=((B*S + 255)/256)
void embedding_gather_forward(
    const __half* embed, const int* token_ids,
    __half* out, int B, int S, int D,
    cudaStream_t stream);

// ─── Type conversion ───────────────────────────────────────────────────────
void fp16_to_fp32_kernel(
    const __half* in, float* out, size_t n, cudaStream_t stream);
void fp32_to_fp16_kernel(
    const float* in, __half* out, size_t n, cudaStream_t stream);

// ─── Element-wise helpers ─────────────────────────────────────────────────
void elementwise_add_fp16(
    const __half* a, const __half* b, __half* out,
    size_t n, cudaStream_t stream);
void elementwise_mul_fp16(
    const __half* a, const __half* b, __half* out,
    size_t n, cudaStream_t stream);
void elementwise_sqrt_fp16(
    const __half* in, __half* out, size_t n, cudaStream_t stream);

// ─── Reduction ─────────────────────────────────────────────────────────────
void sum_all_fp16(const __half* in, float* out, size_t n, cudaStream_t stream);
void row_sum_fp16(const __half* in, float* out, int rows, int cols, cudaStream_t stream);
