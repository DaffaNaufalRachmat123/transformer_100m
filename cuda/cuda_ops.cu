// =============================================================================
// cuda_ops.cu — High-level CUDA operation implementations
// =============================================================================
#include "cuda_ops.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

// ─── Helper kernels (must be free functions, not lambdas) ──────────────────
__global__ void fill_fp16_kernel(__half* d, __half v, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}

__global__ void arange_i32_kernel(int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = i;
}

namespace cuda {

// Forward declaration (defined later in this file)
void sum_all_fp16_impl(const __half* in, float* out, size_t n, cudaStream_t stream);

// ─── Helpers ───────────────────────────────────────────────────────────────
inline cudaDataType_t cu_dtype(DType dt) {
    switch (dt) {
        case DType::F32:  return CUDA_R_32F;
        case DType::FP16: return CUDA_R_16F;
        case DType::BF16: return CUDA_R_16BF;
        default:          return CUDA_R_32F;
    }
}

inline cublasComputeType_t cu_compute(DType dt) {
    return (dt == DType::F32) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_16F;
}

CudaTensor zeros_like(const CudaTensor& t) {
    CudaTensor out(t.shape_vec(), t.dtype);
    out.fill_zero(ctx.compute_stream);
    cuda::sync_compute();
    return out;
}

CudaTensor ones_like(const CudaTensor& t) {
    CudaTensor out(t.shape_vec(), t.dtype);
    if (t.dtype == DType::FP16) {
        int block = 256;
        int grid = (int)((out.numel() + block - 1) / block);
        fill_fp16_kernel<<<grid, block, 0, ctx.compute_stream>>>(
            (__half*)out.data, __float2half(1.0f), out.numel());
    } else {
        cudaMemsetAsync(out.data, 0, out.nbytes, ctx.compute_stream);
    }
    cuda::sync_compute();
    return out;
}

CudaTensor ones(const std::vector<int64_t>& shape, DType dtype) {
    CudaTensor t(shape, dtype);
    int block = 256;
    int grid = (int)((t.numel() + block - 1) / block);
    if (dtype == DType::FP16) {
        fill_fp16_kernel<<<grid, block, 0, ctx.compute_stream>>>(
            (__half*)t.data, __float2half(1.0f), t.numel());
    }
    cuda::sync_compute();
    return t;
}

CudaTensor zeros(const std::vector<int64_t>& shape, DType dtype) {
    CudaTensor t(shape, dtype);
    t.fill_zero(ctx.compute_stream);
    cuda::sync_compute();
    return t;
}

CudaTensor arange(int n, DType dtype) {
    CudaTensor t({n}, dtype);
    if (dtype == DType::I32) {
        int block = 256;
        int grid = (n + block - 1) / block;
        arange_i32_kernel<<<grid, block, 0, ctx.compute_stream>>>((int*)t.data, n);
    }
    cuda::sync_compute();
    return t;
}

// ─── Type conversion ───────────────────────────────────────────────────────
CudaTensor astype(const CudaTensor& src, DType target_dtype) {
    if (src.dtype == target_dtype) {
        CudaTensor out(src.shape_vec(), target_dtype);
        cudaMemcpyAsync(out.data, src.data, src.nbytes,
                        cudaMemcpyDeviceToDevice, ctx.compute_stream);
        cuda::sync_compute();
        return out;
    }

    CudaTensor out(src.shape_vec(), target_dtype);

    if (src.dtype == DType::FP16 && target_dtype == DType::F32) {
        fp16_to_fp32_kernel((const __half*)src.data, (float*)out.data,
                            src.numel(), ctx.compute_stream);
    } else if (src.dtype == DType::F32 && target_dtype == DType::FP16) {
        fp32_to_fp16_kernel((const float*)src.data, (__half*)out.data,
                            src.numel(), ctx.compute_stream);
    }
    cuda::sync_compute();
    return out;
}

// ─── Matmul (cuBLAS) ───────────────────────────────────────────────────────
CudaTensor matmul(const CudaTensor& A, const CudaTensor& B,
                  bool trans_a, bool trans_b) {
    int m = trans_a ? (int)A.shape[1] : (int)A.shape[0];
    int n = trans_b ? (int)B.shape[0] : (int)B.shape[1];
    int k = trans_a ? (int)A.shape[0] : (int)A.shape[1];

    CudaTensor C({m, n}, A.dtype == DType::F32 ? DType::F32 : DType::FP16);
    float alpha = 1.0f, beta = 0.0f;

    cublasGemmEx(ctx.cublas_handle,
                 trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                 trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                 n, m, k,
                 &alpha,
                 B.data, cu_dtype(B.dtype), trans_b ? (int)B.shape[1] : n,
                 A.data, cu_dtype(A.dtype), trans_a ? (int)A.shape[1] : k,
                 &beta,
                 C.data, cu_dtype(C.dtype), n,
                 cu_compute(A.dtype),
                 CUBLAS_GEMM_DEFAULT);

    return C;
}

CudaTensor matmul_transb(const CudaTensor& A, const CudaTensor& B) {
    return matmul(A, B, false, true);
}

// ─── RMSNorm ───────────────────────────────────────────────────────────────
CudaTensor rms_norm(const CudaTensor& x, const CudaTensor& gamma, float eps,
                    CudaTensor& saved_rsqrt, CudaTensor& saved_normed) {
    int B = (int)(x.ndim >= 2 ? x.shape[0] : 1);
    int S = (int)(x.ndim >= 2 ? x.shape[1] : 1);
    int D = (int)(x.ndim >= 2 ? x.shape[2] : x.shape[0]);

    saved_rsqrt = CudaTensor({B, S}, DType::FP16);
    saved_normed = CudaTensor(x.shape_vec(), DType::FP16);
    CudaTensor y(x.shape_vec(), DType::FP16);

    rms_norm_fused_kernel_forward(
        (const __half*)x.data, (const __half*)gamma.data,
        (__half*)y.data, (__half*)saved_rsqrt.data, (__half*)saved_normed.data,
        B, S, D, eps, ctx.compute_stream);
    return y;
}

void rms_norm_backward(const CudaTensor& dy, const CudaTensor& gamma,
                       const CudaTensor& saved_rsqrt, const CudaTensor& saved_normed,
                       CudaTensor& dx, CudaTensor& dgamma, float eps) {
    int B = (int)(dy.ndim >= 2 ? dy.shape[0] : 1);
    int S = (int)(dy.ndim >= 2 ? dy.shape[1] : 1);
    int D = (int)(dy.ndim >= 2 ? dy.shape[2] : dy.shape[0]);

    dx.fill_zero(ctx.compute_stream);
    dgamma.fill_zero(ctx.compute_stream);

    rms_norm_fused_kernel_backward(
        (const __half*)dy.data, (const __half*)gamma.data,
        (const __half*)saved_rsqrt.data, (const __half*)saved_normed.data,
        (__half*)dx.data, (__half*)dgamma.data,
        B, S, D, eps, ctx.compute_stream);
}

// ─── RoPE ──────────────────────────────────────────────────────────────────
void rope_inplace(CudaTensor& x, int num_heads, int d_head, int offset,
                  const float* cos_table, const float* sin_table) {
    int B = (int)x.shape[0];
    int S = (int)x.shape[1];
    rope_fused_forward((__half*)x.data, cos_table, sin_table,
                       B, S, num_heads, d_head, offset, ctx.compute_stream);
}

void rope_backward_inplace(CudaTensor& dx, int num_heads, int d_head, int offset,
                           const float* cos_table, const float* sin_table) {
    int B = (int)dx.shape[0];
    int S = (int)dx.shape[1];
    rope_fused_backward((__half*)dx.data, cos_table, sin_table,
                        B, S, num_heads, d_head, offset, ctx.compute_stream);
}

// ─── SwiGLU ────────────────────────────────────────────────────────────────
CudaTensor swiglu(const CudaTensor& gate, const CudaTensor& up,
                  CudaTensor& saved_silu) {
    int B = (int)gate.shape[0], S = (int)gate.shape[1], FD = (int)gate.shape[2];
    CudaTensor y({B, S, FD}, DType::FP16);
    saved_silu = CudaTensor({B, S, FD}, DType::FP16);

    swiglu_fused_forward(
        (const __half*)gate.data, (const __half*)up.data,
        (__half*)y.data, (__half*)saved_silu.data,
        B, S, FD, ctx.compute_stream);
    return y;
}

void swiglu_backward(const CudaTensor& dy, const CudaTensor& gate,
                     const CudaTensor& up, const CudaTensor& saved_silu,
                     CudaTensor& dgate, CudaTensor& dup) {
    int B = (int)dy.shape[0], S = (int)dy.shape[1], FD = (int)dy.shape[2];
    swiglu_fused_backward(
        (const __half*)dy.data, (const __half*)gate.data,
        (const __half*)up.data, (const __half*)saved_silu.data,
        (__half*)dgate.data, (__half*)dup.data,
        B, S, FD, ctx.compute_stream);
}

// ─── Cross-Entropy ─────────────────────────────────────────────────────────
CudaTensor cross_entropy_loss(const CudaTensor& logits,
                               const CudaTensor& targets,
                               const CudaTensor& mask) {
    int B = (int)logits.shape[0], S = (int)logits.shape[1], V = (int)logits.shape[2];
    CudaTensor loss({1}, DType::F32);

    // Temporary per-token loss buffer (for backward)
    CudaTensor token_loss({B * S}, DType::F32);

    cross_entropy_fused_forward(
        (const __half*)logits.data, (const int*)targets.data,
        (const float*)mask.data,
        (float*)token_loss.data, B, S, V, ctx.compute_stream);

    // Sum per-token losses
    sum_all_fp16_impl((const __half*)token_loss.data, (float*)loss.data,
                 B * S, ctx.compute_stream);
    cuda::sync_compute();
    return loss;
}

void cross_entropy_backward(const CudaTensor& logits, const CudaTensor& targets,
                            const CudaTensor& mask, CudaTensor& d_logits) {
    int B = (int)logits.shape[0], S = (int)logits.shape[1], V = (int)logits.shape[2];
    cross_entropy_fused_backward(
        (const __half*)logits.data, (const int*)targets.data,
        (const float*)mask.data,
        (__half*)d_logits.data, B, S, V, ctx.compute_stream);
    cuda::sync_compute();
}

// ─── Embedding ─────────────────────────────────────────────────────────────
CudaTensor embedding_forward(const CudaTensor& embed, const CudaTensor& token_ids) {
    int B = 1, S = (int)token_ids.numel(), D = (int)embed.shape[1], V = (int)embed.shape[0];
    if (token_ids.ndim >= 2) { B = (int)token_ids.shape[0]; S = (int)token_ids.shape[1]; }

    CudaTensor out({B, S, D}, DType::FP16);
    embedding_gather_forward(
        (const __half*)embed.data, (const int*)token_ids.data,
        (__half*)out.data, B, S, D, V, ctx.compute_stream);
    return out;
}

// ─── Reshape / Transpose (zero-copy views) ─────────────────────────────────
// reshape() is already defined inline in cuda_tensor.h — no redefinition here

CudaTensor transpose(CudaTensor& src, const std::vector<int>& axes) {
    CudaTensor v;
    v.data   = src.data;
    v.ndim   = src.ndim;
    v.nbytes = src.nbytes;
    v.dtype  = src.dtype;
    v.owns   = false;
    for (int i = 0; i < v.ndim && i < 4; i++) {
        v.shape[i] = src.shape[axes[i]];
    }
    return v;
}

// ─── Sum ───────────────────────────────────────────────────────────────────
CudaTensor sum_all(const CudaTensor& t) {
    CudaTensor out({1}, DType::F32);
    if (t.dtype == DType::FP16) {
        sum_all_fp16_impl((const __half*)t.data, (float*)out.data, t.numel(),
                     ctx.compute_stream);
    }
    cuda::sync_compute();
    return out;
}

// ─── AdamW ─────────────────────────────────────────────────────────────────
void adamw_step(CudaTensor& param, CudaTensor& master_param,
                CudaTensor& m, CudaTensor& v,
                const CudaTensor& grad,
                float lr, float beta1, float beta2,
                float eps, float wd, int step) {
    adamw_fused_update(
        (__half*)param.data, (float*)master_param.data,
        (float*)m.data, (float*)v.data,
        (const __half*)grad.data,
        param.numel(), lr, beta1, beta2, eps, wd, step,
        ctx.compute_stream);
}

// ─── Reduction kernels (needed by cross_entropy) ──────────────────────────
__global__ void sum_reduce_fp16(const __half* in, float* out, size_t n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x)
        sum += __half2float(in[i]);
    s[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = s[0];
}

void sum_all_fp16_impl(const __half* in, float* out, size_t n, cudaStream_t stream) {
    int block = 256;
    sum_reduce_fp16<<<1, block, block * sizeof(float), stream>>>(in, out, n);
}

// ─── DPO: masked_log_probs ─────────────────────────────────────────────────
// Computes MEAN of log_softmax(logits)[targets] over masked positions.
// logits: [B*S, V], targets: [B*S], mask: [B*S]
// Returns scalar tensor with the per-token mean log-prob.
__global__ void masked_log_probs_kernel(
    const __half* logits, const int* targets, const float* mask,
    float* output, int BS, int V)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_count = 0.0f;

    for (int pos = tid; pos < BS; pos += blockDim.x) {
        if (mask[pos] <= 0.0f) continue;

        const __half* row = logits + (size_t)pos * V;
        int target = targets[pos];
        if (target < 0 || target >= V) continue;

        float max_val = -1e30f;
        for (int j = 0; j < V; j++) {
            float v = __half2float(row[j]);
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < V; j++)
            sum_exp += expf(__half2float(row[j]) - max_val);
        float log_prob = __half2float(row[target]) - max_val - logf(sum_exp);
        local_sum += log_prob;
        local_count += 1.0f;
    }

    // Reduce sum
    sdata[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float total_sum = sdata[0];

    // Reduce count
    sdata[tid] = local_count;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float total_count = sdata[0];

    if (tid == 0)
        output[0] = (total_count > 0.0f) ? (total_sum / total_count) : 0.0f;
}

CudaTensor masked_log_probs(const CudaTensor& logits, const CudaTensor& targets,
                            const CudaTensor& mask) {
    // logits: [B, S, V] or [B*S, V], targets: [B, S], mask: [B, S]
    int64_t BS = targets.numel();
    int64_t V;
    if (logits.ndim == 3) V = logits.shape[2];
    else V = logits.shape[logits.ndim - 1];  // last dimension is vocab

    CudaTensor out({1}, DType::F32);
    int block = 256;
    masked_log_probs_kernel<<<1, block, block * sizeof(float), ctx.compute_stream>>>(
        (const __half*)logits.data, (const int*)targets.data, (const float*)mask.data,
        (float*)out.data, (int)BS, (int)V);
    cuda::sync_compute();
    return out;
}

// ─── DPO: scale_fp16 ──────────────────────────────────────────────────────
__global__ void scale_fp16_kernel(const __half* in, __half* out, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(in[idx]) * scale);
    }
}

void scale_fp16(CudaTensor& t, float scale) {
    size_t n = t.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_fp16_kernel<<<grid, block, 0, ctx.compute_stream>>>(
        (const __half*)t.data, (__half*)t.data, scale, n);
    cuda::sync_compute();
}

} // namespace cuda

// ─── Free-function wrappers (declared in cuda_kernels.cuh, no namespace) ──
void sum_all_fp16(const __half* in, float* out, size_t n, cudaStream_t stream) {
    cuda::sum_all_fp16_impl(in, out, n, stream);
}
