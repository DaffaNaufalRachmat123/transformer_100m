#pragma once
// =============================================================================
// ce_embed_kernels.cuh — Correct cross-entropy (no buffer overflow) +
//                        GPU embedding backward (scatter-add).
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// ─── Cross-entropy forward+backward in one pass ─────────────────────────────
// logits  : [BS, V] FP16 (read-only)
// targets : [BS] int
// mask    : [BS] float
// losses  : [BS] float  (per-token loss, 0 where masked)
// d_logits: [BS, V] FP16 (output gradient = (softmax - onehot) * mask)
// One block per token row.
__global__ void ce_fwd_bwd_kernel(
    const __half* __restrict__ logits,
    const int*    __restrict__ targets,
    const float*  __restrict__ mask,
    float* __restrict__ losses,
    __half* __restrict__ d_logits,
    int BS, int V)
{
    int pos = blockIdx.x;
    if (pos >= BS) return;

    const __half* row  = logits   + (size_t)pos * V;
    __half*       drow = d_logits + (size_t)pos * V;
    int   tgt = targets[pos];
    float m   = mask[pos];
    int   tid = threadIdx.x;

    extern __shared__ float sm[];   // [blockDim.x]

    // max
    float local_max = -1e30f;
    for (int j = tid; j < V; j += blockDim.x) {
        float v = __half2float(row[j]);
        if (v > local_max) local_max = v;
    }
    sm[tid] = local_max; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && sm[tid+s] > sm[tid]) sm[tid] = sm[tid+s];
        __syncthreads();
    }
    float max_val = sm[0]; __syncthreads();

    // sum exp
    float local_sum = 0.0f;
    for (int j = tid; j < V; j += blockDim.x)
        local_sum += expf(__half2float(row[j]) - max_val);
    sm[tid] = local_sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid+s];
        __syncthreads();
    }
    float sum_exp = sm[0];
    float log_sum_exp = logf(sum_exp);

    // loss
    if (tid == 0) {
        if (m > 0.0f && tgt >= 0 && tgt < V)
            losses[pos] = -(__half2float(row[tgt]) - max_val - log_sum_exp) * m;
        else
            losses[pos] = 0.0f;
    }

    // gradient (softmax - onehot) * mask  — written to SEPARATE fp16 buffer
    for (int j = tid; j < V; j += blockDim.x) {
        float prob = expf(__half2float(row[j]) - max_val) / sum_exp;
        float grad = (prob - (j == tgt ? 1.0f : 0.0f)) * m;
        drow[j] = __float2half(grad);
    }
}

static inline void launch_ce_fwd_bwd(
    const __half* logits, const int* targets, const float* mask,
    float* losses, __half* d_logits, int BS, int V, cudaStream_t stream)
{
    int block = 256;
    size_t shared = block * sizeof(float);
    ce_fwd_bwd_kernel<<<BS, block, shared, stream>>>(
        logits, targets, mask, losses, d_logits, BS, V);
}

// ─── Scale fp16 buffer in place by scalar ───────────────────────────────────
__global__ void scale_inplace_fp16_kernel(__half* d, float s, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(d[i]);
        d[i] = __float2half(std::isfinite(v) ? v * s : 0.0f);
    }
}

static inline void launch_scale_fp16(__half* d, float s, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    scale_inplace_fp16_kernel<<<grid, block, 0, stream>>>(d, s, n);
}

// ─── Scale f32 buffer in place by scalar ────────────────────────────────────
__global__ void scale_inplace_f32_kernel(float* d, float s, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = d[i];
        d[i] = std::isfinite(v) ? v * s : 0.0f;
    }
}

static inline void launch_scale_f32(float* d, float s, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    scale_inplace_f32_kernel<<<grid, block, 0, stream>>>(d, s, n);
}

// ─── Clamp fp16 buffer in place to [-max_val, +max_val] ─────────────────────
// Available as utility; not used in default forward pass.
__global__ void clamp_inplace_fp16_kernel(__half* d, float max_val, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __half2float(d[i]);
        v = fminf(fmaxf(v, -max_val), max_val);
        d[i] = __float2half(v);
    }
}

static inline void launch_clamp_fp16(__half* d, float max_val, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    clamp_inplace_fp16_kernel<<<grid, block, 0, stream>>>(d, max_val, n);
}

// ─── Embedding backward (scatter-add into FP32 accumulator) ─────────────────
// d_x   : [BS, D] FP16 gradient w.r.t. embedded output
// ids   : [BS] int token ids
// accum : [V, D] FP32 gradient accumulator (must be zeroed first)
// One thread per (row, d).
__global__ void embed_scatter_add_kernel(
    const __half* __restrict__ d_x,
    const int*    __restrict__ ids,
    float* __restrict__ accum,
    int BS, int D, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BS * D) return;
    int row = idx / D;
    int d   = idx % D;
    int tid = ids[row];
    if (tid < 0 || tid >= V) return;
    atomicAdd(&accum[(size_t)tid * D + d], __half2float(d_x[idx]));
}

static inline void launch_embed_scatter_add(
    const __half* d_x, const int* ids, float* accum,
    int BS, int D, int V, cudaStream_t stream)
{
    int block = 256;
    int grid = (int)(((size_t)BS * D + block - 1) / block);
    embed_scatter_add_kernel<<<grid, block, 0, stream>>>(d_x, ids, accum, BS, D, V);
}

// ─── FP32 accumulator -> FP16 (with optional weight-tie add) ────────────────
__global__ void f32_to_f16_add_kernel(
    const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

static inline void launch_f32_to_f16(const float* src, __half* dst, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    f32_to_f16_add_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

// ─── Cast FP32 scratch -> FP16 out (for dQ/dK/dV) ───────────────────────────
static inline void launch_cast_f32_f16(const float* src, __half* dst, size_t n, cudaStream_t stream) {
    launch_f32_to_f16(src, dst, n, stream);
}

// ─── Seed FP32 accumulator from FP16 source (accum = fp32(src)) ─────────────
__global__ void f16_to_f32_seed_kernel(const __half* __restrict__ src,
                                       float* __restrict__ accum, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = __half2float(src[i]);
}

static inline void launch_f16_to_f32_seed(const __half* src, float* accum, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    f16_to_f32_seed_kernel<<<grid, block, 0, stream>>>(src, accum, n);
}

// ─── Element-wise add for FP32 buffers: dst[i] += src[i] ────────────────────
__global__ void f32_add_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

static inline void launch_f32_add(const float* src, float* dst, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    f32_add_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

// ─── L2 norm squared of FP16 buffer (for gradient clipping) ─────────────────
// Accumulates sum-of-squares into *out (must be zeroed before first call).
// Multiple calls accumulate into the same output for global norm computation.
__global__ void l2_norm_sq_fp16_kernel(const __half* __restrict__ d, float* out, size_t n) {
    extern __shared__ float shmem[];
    float val = 0.0f;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        float v = __half2float(d[idx]);
        val += v * v;
    }
    shmem[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, shmem[0]);
}

static inline void launch_l2_norm_sq(const __half* d, float* out_sq, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)(((n + block - 1) / block) < 1024 ? (n + block - 1) / block : 1024);
    l2_norm_sq_fp16_kernel<<<grid, block, block * sizeof(float), stream>>>(d, out_sq, n);
}

// ─── NaN/Inf detection kernels ──────────────────────────────────────────────
__global__ void count_nan_inf_kernel(const __half* d, unsigned int* nan_cnt,
                                      unsigned int* inf_cnt, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(d[i]);
    if (isnan(v)) atomicAdd(nan_cnt, 1);
    else if (isinf(v)) atomicAdd(inf_cnt, 1);
}

static inline void launch_count_nan_inf(const __half* d, unsigned int* nan_out,
                                         unsigned int* inf_out, size_t n,
                                         cudaStream_t stream) {
    if (n == 0) return;
    cudaMemsetAsync(nan_out, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(inf_out, 0, sizeof(unsigned int), stream);
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    count_nan_inf_kernel<<<grid, block, 0, stream>>>(d, nan_out, inf_out, n);
}

// FP32 atomicMin/atomicMax (not natively supported on sm_75; must be defined
// before min_max_fp16_kernel which calls them).
__device__ inline void atomicMin_f32(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        float f_old = __int_as_float(assumed);
        float f_new = fminf(f_old, val);
        old = atomicCAS(iaddr, assumed, __float_as_int(f_new));
    } while (old != assumed);
}
__device__ inline void atomicMax_f32(float* addr, float val) {
    int* iaddr = (int*)addr;
    int old = *iaddr, assumed;
    do {
        assumed = old;
        float f_old = __int_as_float(assumed);
        float f_new = fmaxf(f_old, val);
        old = atomicCAS(iaddr, assumed, __float_as_int(f_new));
    } while (old != assumed);
}

__global__ void min_max_fp16_kernel(const __half* d, float* min_out,
                                     float* max_out, size_t n) {
    extern __shared__ float sm[];
    float* sm_min = sm;
    float* sm_max = sm + blockDim.x;
    int tid = threadIdx.x;
    float local_min = 1e30f, local_max = -1e30f;
    for (size_t i = tid; i < n; i += blockDim.x) {
        float v = __half2float(d[i]);
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }
    sm_min[tid] = local_min; sm_max[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sm_min[tid + s] < sm_min[tid]) sm_min[tid] = sm_min[tid + s];
            if (sm_max[tid + s] > sm_max[tid]) sm_max[tid] = sm_max[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMin_f32(min_out, sm_min[0]);
        atomicMax_f32(max_out, sm_max[0]);
    }
}

static inline void launch_min_max_fp16(const __half* d, float* min_out,
                                        float* max_out, size_t n,
                                        cudaStream_t stream) {
    if (n == 0) return;
    float init_min = 1e30f, init_max = -1e30f;
    cudaMemcpyAsync(min_out, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_out, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    min_max_fp16_kernel<<<grid, block, block * 2 * sizeof(float), stream>>>(
        d, min_out, max_out, n);
}
