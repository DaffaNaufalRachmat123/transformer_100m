// =============================================================================
// cuda_kernels.cu — Fused CUDA kernel implementations
// =============================================================================
#include "cuda_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// ─── Kernel utilities ──────────────────────────────────────────────────────
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-wide sum reduction.
//
// REQUIRES blockDim.x to be a power of two and >= 64. Both call sites launch
// with 256 threads (see rms_norm_fused_kernel_forward / _backward below).
//
// The tree loop must run down to s == 32 *inclusive* so that exactly 32 live
// partials remain for the warp-shuffle stage to fold. Using `s > 32` leaves 64
// live partials while the warp stage only folds shared[0..31], silently
// discarding shared[32..63] — i.e. half of every reduction. That bug made
// RMSNorm sum only 384 of 768 dims while still dividing by the full D.
__device__ inline float block_reduce_sum(float val, float* shared, int tid) {
    shared[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float w = shared[tid];
        w += __shfl_down_sync(0xffffffff, w, 16);
        w += __shfl_down_sync(0xffffffff, w, 8);
        w += __shfl_down_sync(0xffffffff, w, 4);
        w += __shfl_down_sync(0xffffffff, w, 2);
        w += __shfl_down_sync(0xffffffff, w, 1);
        if (tid == 0) shared[0] = w;
    }
    __syncthreads();
    return shared[0];
}

// =============================================================================
// RMSNorm Fused — Forward
// =============================================================================
__global__ void rms_norm_forward_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    __half* __restrict__ y,
    __half* __restrict__ saved_rsqrt,
    __half* __restrict__ saved_normed,
    int B, int S, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= B * S) return;

    extern __shared__ float sdata[];
    float* smean = sdata;
    int tid = threadIdx.x;

    const __half* x_row = x + row * D;
    __half* y_row = y + row * D;
    __half* n_row = saved_normed + row * D;

    // Compute mean(x^2) via warp reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }
    float total_sq = block_reduce_sum(sum_sq, smean, tid);
    float rsqrt = rsqrtf(total_sq / D + eps);

    if (tid == 0) saved_rsqrt[row] = __float2half(rsqrt);

    // Compute y = x * rsqrt * gamma
    for (int i = tid; i < D; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        float g = __half2float(gamma[i]);
        float normed = v * rsqrt;
        y_row[i] = __float2half(normed * g);
        n_row[i] = __float2half(normed);
    }
}

void rms_norm_fused_kernel_forward(
    const __half* x, const __half* gamma,
    __half* y, __half* saved_rsqrt, __half* saved_normed,
    int B, int S, int D, float eps,
    cudaStream_t stream)
{
    size_t shared_bytes = 256 * sizeof(float);
    rms_norm_forward_kernel<<<B * S, 256, shared_bytes, stream>>>(
        x, gamma, y, saved_rsqrt, saved_normed, B, S, D, eps);
}

// =============================================================================
// RMSNorm Fused — Backward
//   y      = normed * gamma,  normed = x * rsqrt,  rsqrt = 1/sqrt(mean(x^2)+eps)
//   dgamma = sum_rows(dy * normed)
//   dx     = rsqrt * (dy*gamma - normed * mean(dy*gamma*normed))
//
// NOTE: gamma belongs *inside* the mean-subtraction term. Applying it after
// the reduction (i.e. rsqrt*(dy - normed*mean(dy*normed))*gamma) is only
// equivalent when gamma is a uniform scalar, which it is not here.
// =============================================================================
__global__ void rms_norm_backward_kernel(
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma,
    const __half* __restrict__ saved_rsqrt,
    const __half* __restrict__ saved_normed,
    __half* __restrict__ dx,
    float* __restrict__ dgamma_f32,
    int B, int S, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= B * S) return;

    extern __shared__ float sdata[];
    float* shared = sdata;
    int tid = threadIdx.x;

    const __half* dy_row = dy + row * D;
    const __half* n_row = saved_normed + row * D;
    __half* dx_row = dx + row * D;
    int col = blockIdx.y;

    if (col == 0) {
        // Compute sum(dy * gamma * normed_x) for this row
        float sum_dyn = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) {
            sum_dyn += __half2float(dy_row[i]) * __half2float(gamma[i])
                     * __half2float(n_row[i]);
        }
        float total = block_reduce_sum(sum_dyn, shared, tid) / D;

        float rsqrt = __half2float(saved_rsqrt[row]);

        // dx = rsqrt * (dy*gamma - normed_x * mean(dy*gamma*normed_x))
        for (int i = tid; i < D; i += blockDim.x) {
            float dyg = __half2float(dy_row[i]) * __half2float(gamma[i]);
            float d = dyg - __half2float(n_row[i]) * total;
            dx_row[i] = __float2half(d * rsqrt);
        }
    } else if (col == 1) {
        // Accumulate dgamma across all rows into an FP32 buffer (sm_75 has no
        // fp16 atomicAdd(__half*, float); and fp32 accum is far more accurate).
        for (int i = tid; i < D; i += blockDim.x) {
            float d = __half2float(dy_row[i]) * __half2float(n_row[i]);
            atomicAdd(dgamma_f32 + i, d);
        }
    }
}

// tiny converter: fp32 dgamma scratch -> fp16 output
__global__ void rmsnorm_dgamma_f32_to_f16(const float* __restrict__ src,
                                          __half* __restrict__ dst, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) dst[i] = __float2half(src[i]);
}

void rms_norm_fused_kernel_backward(
    const __half* dy, const __half* gamma,
    const __half* saved_rsqrt, const __half* saved_normed,
    __half* dx, __half* dgamma,
    int B, int S, int D, float eps,
    cudaStream_t stream)
{
    size_t shared_bytes = 256 * sizeof(float);
    // FP32 scratch for dgamma accumulation (allocated per call; D is small).
    float* dgamma_f32 = nullptr;
    cudaMallocAsync((void**)&dgamma_f32, (size_t)D * sizeof(float), stream);
    cudaMemsetAsync(dgamma_f32, 0, (size_t)D * sizeof(float), stream);

    dim3 grid(B * S, 2);
    rms_norm_backward_kernel<<<grid, 256, shared_bytes, stream>>>(
        dy, gamma, saved_rsqrt, saved_normed, dx, dgamma_f32, B, S, D, eps);

    int blk = 256, g = (D + blk - 1) / blk;
    rmsnorm_dgamma_f32_to_f16<<<g, blk, 0, stream>>>(dgamma_f32, dgamma, D);
    cudaFreeAsync(dgamma_f32, stream);
}

// =============================================================================
// RoPE Fused — Forward (in-place)
// =============================================================================
__global__ void rope_forward_kernel(
    __half* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int B, int S, int num_heads, int d_head, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = d_head / 2;
    int total = B * S * num_heads * half_dim;
    if (idx >= total) return;

    // Map flat index to (b, s, h, half_j)
    int b = idx / (S * num_heads * half_dim);
    int rem = idx % (S * num_heads * half_dim);
    int s = rem / (num_heads * half_dim);
    int rem2 = rem % (num_heads * half_dim);
    int h = rem2 / half_dim;
    int j = rem2 % half_dim;

    int pos = s + offset;
    float cos_val = cos_table[pos * half_dim + j];
    float sin_val = sin_table[pos * half_dim + j];

    // x_{2j} = x_{2j} * cos - x_{2j+1} * sin
    // x_{2j+1} = x_{2j} * sin + x_{2j+1} * cos
    int base = b * (S * num_heads * d_head) + s * (num_heads * d_head) + h * d_head;
    float x0 = __half2float(x[base + j * 2]);
    float x1 = __half2float(x[base + j * 2 + 1]);

    x[base + j * 2]     = __float2half(x0 * cos_val - x1 * sin_val);
    x[base + j * 2 + 1] = __float2half(x0 * sin_val + x1 * cos_val);
}

void rope_fused_forward(
    __half* x, const float* cos_table, const float* sin_table,
    int B, int S, int num_heads, int d_head, int offset,
    cudaStream_t stream)
{
    int half_dim = d_head / 2;
    int total = B * S * num_heads * half_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_forward_kernel<<<grid, block, 0, stream>>>(
        x, cos_table, sin_table, B, S, num_heads, d_head, offset);
}

// =============================================================================
// RoPE — Backward (same rotation with negated sin)
// =============================================================================
__global__ void rope_backward_kernel(
    __half* __restrict__ dx,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int B, int S, int num_heads, int d_head, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = d_head / 2;
    int total = B * S * num_heads * half_dim;
    if (idx >= total) return;

    int b = idx / (S * num_heads * half_dim);
    int rem = idx % (S * num_heads * half_dim);
    int s = rem / (num_heads * half_dim);
    int rem2 = rem % (num_heads * half_dim);
    int h = rem2 / half_dim;
    int j = rem2 % half_dim;

    int pos = s + offset;
    float cos_val = cos_table[pos * half_dim + j];
    float sin_val = -sin_table[pos * half_dim + j]; // NEGATED for backward

    int base = b * (S * num_heads * d_head) + s * (num_heads * d_head) + h * d_head;
    float x0 = __half2float(dx[base + j * 2]);
    float x1 = __half2float(dx[base + j * 2 + 1]);

    dx[base + j * 2]     = __float2half(x0 * cos_val - x1 * sin_val);
    dx[base + j * 2 + 1] = __float2half(x0 * sin_val + x1 * cos_val);
}

void rope_fused_backward(
    __half* dx, const float* cos_table, const float* sin_table,
    int B, int S, int num_heads, int d_head, int offset,
    cudaStream_t stream)
{
    int half_dim = d_head / 2;
    int total = B * S * num_heads * half_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    rope_backward_kernel<<<grid, block, 0, stream>>>(
        dx, cos_table, sin_table, B, S, num_heads, d_head, offset);
}

// =============================================================================
// SwiGLU Fused — Forward: y = gate * silu(up)
// =============================================================================
__device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_forward_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ y,
    __half* __restrict__ saved_silu,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float gv = __half2float(gate[idx]);
    float uv = __half2float(up[idx]);
    float su = silu(uv);

    y[idx] = __float2half(gv * su);
    saved_silu[idx] = __float2half(su);
}

void swiglu_fused_forward(
    const __half* gate, const __half* up,
    __half* y, __half* saved_silu,
    int B, int S, int FD,
    cudaStream_t stream)
{
    int N = B * S * FD;
    int block = 256;
    int grid = (N + block - 1) / block;
    swiglu_forward_kernel<<<grid, block, 0, stream>>>(
        gate, up, y, saved_silu, N);
}

// =============================================================================
// SwiGLU Fused — Backward
// =============================================================================
__global__ void swiglu_backward_kernel(
    const __half* __restrict__ dy,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const __half* __restrict__ saved_silu,
    __half* __restrict__ dgate,
    __half* __restrict__ dup,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float dy_v = __half2float(dy[idx]);
    float gv = __half2float(gate[idx]);
    float uv = __half2float(up[idx]);
    float su = __half2float(saved_silu[idx]);

    // dgate = dy * silu(up)
    dgate[idx] = __float2half(dy_v * su);

    // dup = dy * gate * silu'(up)
    // silu'(x) = silu(x) + sigmoid(x)*(1 - silu(x))
    float sig = 1.0f / (1.0f + expf(-uv));
    float d_silu = su + sig * (1.0f - su);
    dup[idx] = __float2half(dy_v * gv * d_silu);
}

void swiglu_fused_backward(
    const __half* dy, const __half* gate, const __half* up,
    const __half* saved_silu,
    __half* dgate, __half* dup,
    int B, int S, int FD,
    cudaStream_t stream)
{
    int N = B * S * FD;
    int block = 256;
    int grid = (N + block - 1) / block;
    swiglu_backward_kernel<<<grid, block, 0, stream>>>(
        dy, gate, up, saved_silu, dgate, dup, N);
}

// =============================================================================
// Cross-Entropy Fused — Forward: log_softmax + NLL + mask
//
// NOTE: This computes ONLY the per-token loss (read-only over logits). It does
// NOT write gradients back into the logits buffer (the previous version did an
// illegal ((float*)row)[i]=grad write that overflowed the FP16 buffer). The
// backward pass recomputes softmax into a SEPARATE d_logits buffer.
// =============================================================================
__global__ void cross_entropy_forward_kernel(
    const __half* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ mask,
    float* __restrict__ loss_out,
    int B, int S, int V)
{
    int bid = blockIdx.x;
    if (bid >= B * S) return;

    extern __shared__ float shared[];
    float* smax = shared;
    float* ssum = shared + blockDim.x;
    int tid = threadIdx.x;

    const __half* row = logits + (size_t)bid * V;
    int tgt = targets[bid];
    float m = mask[bid];

    // Find max for numerical stability
    float max_val = -1e30f;
    for (int i = tid; i < V; i += blockDim.x) {
        float v = __half2float(row[i]);
        if (v > max_val) max_val = v;
    }
    smax[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s]; }
        __syncthreads();
    }
    max_val = smax[0];
    __syncthreads();

    // Compute sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) {
        sum_exp += expf(__half2float(row[i]) - max_val);
    }
    ssum[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float log_sum = logf(ssum[0]);

    if (tid == 0) {
        if (m > 0.0f && tgt >= 0 && tgt < V) {
            float target_logit = __half2float(row[tgt]);
            loss_out[bid] = -(target_logit - max_val - log_sum) * m;
        } else {
            loss_out[bid] = 0.0f;
        }
    }
}

void cross_entropy_fused_forward(
    const __half* logits, const int* targets, const float* mask,
    float* loss_out, int B, int S, int V,
    cudaStream_t stream)
{
    int N = B * S;
    size_t shared_bytes = 256 * 2 * sizeof(float);
    cross_entropy_forward_kernel<<<N, 256, shared_bytes, stream>>>(
        logits, targets, mask, loss_out, B, S, V);
}

// =============================================================================
// Cross-Entropy Fused — Backward (recomputes softmax into SEPARATE d_logits)
// d_logits[i,j] = (softmax(logits[i])[j] - onehot(target)) * mask / N
// One block per token row; reads logits, writes d_logits. No in-place overwrite.
// =============================================================================
__global__ void cross_entropy_backward_kernel(
    const __half* __restrict__ logits,
    const int* __restrict__ targets,
    const float* __restrict__ mask,
    __half* __restrict__ d_logits,
    int B, int S, int V, float inv_n)
{
    int bid = blockIdx.x;
    if (bid >= B * S) return;

    extern __shared__ float sm[];
    int tid = threadIdx.x;

    const __half* row  = logits   + (size_t)bid * V;
    __half*       drow = d_logits + (size_t)bid * V;
    int   tgt = targets[bid];
    float m   = mask[bid];

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

    for (int j = tid; j < V; j += blockDim.x) {
        float prob = expf(__half2float(row[j]) - max_val) / sum_exp;
        float grad = (prob - (j == tgt ? 1.0f : 0.0f)) * m * inv_n;
        drow[j] = __float2half(grad);
    }
}

void cross_entropy_fused_backward(
    const __half* logits, const int* targets, const float* mask,
    __half* d_logits, int B, int S, int V,
    cudaStream_t stream)
{
    int N = B * S;
    float inv_n = 1.0f / (float)N;
    size_t shared_bytes = 256 * sizeof(float);
    cross_entropy_backward_kernel<<<N, 256, shared_bytes, stream>>>(
        logits, targets, mask, d_logits, B, S, V, inv_n);
}

// =============================================================================
// AdamW Fused Update
// =============================================================================
__global__ void adamw_fused_kernel(
    __half* __restrict__ param,
    float* __restrict__ master_param,
    float* __restrict__ m,
    float* __restrict__ v,
    const __half* __restrict__ grad,
    size_t n, float lr, float beta1, float beta2,
    float eps, float wd, int step)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __half2float(grad[idx]);
    if (!isfinite(g)) return;   // NaN/Inf gradient: skip this element entirely
    float m_val = m[idx] * beta1 + g * (1.0f - beta1);
    float v_val = v[idx] * beta2 + g * g * (1.0f - beta2);

    m[idx] = m_val;
    v[idx] = v_val;

    float bias_corr1 = 1.0f - powf(beta1, (float)step);
    float bias_corr2 = 1.0f - powf(beta2, (float)step);
    float m_hat = m_val / bias_corr1;
    float v_hat = v_val / bias_corr2;

    float mp = master_param[idx];
    mp = mp - lr * m_hat / (sqrtf(v_hat) + eps) - lr * wd * mp;

    master_param[idx] = mp;
    param[idx] = __float2half(mp);
}

void adamw_fused_update(
    __half* param, float* master_param,
    float* m, float* v,
    const __half* grad,
    size_t numel, float lr, float beta1, float beta2,
    float eps, float wd, int step,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((numel + block - 1) / block);
    adamw_fused_kernel<<<grid, block, 0, stream>>>(
        param, master_param, m, v, grad,
        numel, lr, beta1, beta2, eps, wd, step);
}

// =============================================================================
// Type Conversion Kernels
// =============================================================================
__global__ void fp16_to_fp32(const __half* in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

void fp16_to_fp32_kernel(const __half* in, float* out, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    fp16_to_fp32<<<grid, block, 0, stream>>>(in, out, n);
}

__global__ void fp32_to_fp16(const float* in, __half* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

void fp32_to_fp16_kernel(const float* in, __half* out, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    fp32_to_fp16<<<grid, block, 0, stream>>>(in, out, n);
}

// =============================================================================
// Element-wise kernels
// =============================================================================
__global__ void add_fp16(const __half* a, const __half* b, __half* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(a[i], b[i]);
}

void elementwise_add_fp16(const __half* a, const __half* b, __half* out, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    add_fp16<<<grid, block, 0, stream>>>(a, b, out, n);
}

__global__ void mul_fp16(const __half* a, const __half* b, __half* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hmul(a[i], b[i]);
}

void elementwise_mul_fp16(const __half* a, const __half* b, __half* out, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    mul_fp16<<<grid, block, 0, stream>>>(a, b, out, n);
}

__global__ void sqrt_fp16(const __half* in, __half* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = hsqrt(in[i]);
}

void elementwise_sqrt_fp16(const __half* in, __half* out, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    sqrt_fp16<<<grid, block, 0, stream>>>(in, out, n);
}

// =============================================================================
// Embedding gather (with bounds check)
// =============================================================================
__global__ void embedding_gather(const __half* embed, const int* ids,
                                  __half* out, int N, int D, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int token_id = ids[idx];
    if (token_id < 0 || token_id >= V) token_id = 0;
    const __half* row = embed + token_id * D;
    __half* out_row = out + idx * D;
    for (int d = 0; d < D; d++) {
        out_row[d] = row[d];
    }
}

void embedding_gather_forward(
    const __half* embed, const int* token_ids,
    __half* out, int B, int S, int D, int V,
    cudaStream_t stream)
{
    int N = B * S;
    int block = 256;
    int grid = (N + block - 1) / block;
    embedding_gather<<<grid, block, 0, stream>>>(embed, token_ids, out, N, D, V);
}
