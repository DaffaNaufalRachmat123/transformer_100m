#pragma once
// =============================================================================
// attn_kernels.cuh — GQA scaled-dot-product attention (forward + backward)
//
// Efficient version: ONE WARP (32 lanes) per (b, hq, query_i).
//   - DH=64 head dim -> each lane owns 2 elements (e = lane, lane+32).
//   - Dot products use warp-shuffle reduction (no __syncthreads, no shared
//     reduction buffer). This is the hot path, so it must be cheap.
//   - Online softmax over causal keys (j <= i).
//   - LSE (max + log sum) saved for a correct backward.
//
// Layout:
//   Q : [B, S, HQ, DH]   K/V : [B, S, HKV, DH]   O : [B, S, HQ, DH]
//   GQA: hkv = hq / (HQ / HKV)
//
// GPU Compatibility:
//   - Requires sm >= 30 (warp shuffle: __shfl_down_sync)
//   - Works on: T4(sm75), V100(sm70), A100(sm80), RTX3090(sm86), RTX4090(sm89), H100(sm90)
//   - DH constraint: DH must be ≤ 64 (each lane handles DH/32 ≤ 2 elements)
//     For DH=128, use a 2-warp variant or tile the head dimension.
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#ifndef WARP_FULL_MASK
#define WARP_FULL_MASK 0xffffffffu
#endif

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(WARP_FULL_MASK, v, o);
    // broadcast lane 0 result to all lanes
    return __shfl_sync(WARP_FULL_MASK, v, 0);
}

// ─── Forward: one warp per (b, hq, i). blockDim.x must be 32. ───────────────
__global__ void gqa_attn_fwd_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ LSE,
    int B, int S, int HQ, int HKV, int DH, float scale)
{
    int blk = blockIdx.x;                 // over B*HQ*S
    int i   = blk % S;
    int hq  = (blk / S) % HQ;
    int b   = blk / (S * HQ);
    if (b >= B) return;

    int hkv  = hq / (HQ / HKV);
    int lane = threadIdx.x;               // 0..31
    int half = DH >> 1;                   // 32 for DH=64

    const __half* q_ptr = Q + (((size_t)b * S + i) * HQ + hq) * DH;
    // per-lane query components (2 elements)
    float q0 = __half2float(q_ptr[lane]);
    float q1 = (lane + half < DH) ? __half2float(q_ptr[lane + half]) : 0.0f;

    float m = -1e30f;   // running max
    float l = 0.0f;     // running denom
    float a0 = 0.0f, a1 = 0.0f;  // output accumulator (2 elems per lane)

    for (int j = 0; j <= i; j++) {
        const __half* k_ptr = K + (((size_t)b * S + j) * HKV + hkv) * DH;
        float k0 = __half2float(k_ptr[lane]);
        float k1 = (lane + half < DH) ? __half2float(k_ptr[lane + half]) : 0.0f;
        float score = warp_sum(q0 * k0 + q1 * k1) * scale;

        float m_new = fmaxf(m, score);
        float corr  = __expf(m - m_new);
        float p     = __expf(score - m_new);
        l = l * corr + p;

        const __half* v_ptr = V + (((size_t)b * S + j) * HKV + hkv) * DH;
        float v0 = __half2float(v_ptr[lane]);
        float v1 = (lane + half < DH) ? __half2float(v_ptr[lane + half]) : 0.0f;
        a0 = a0 * corr + p * v0;
        a1 = a1 * corr + p * v1;
        m = m_new;
    }

    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    __half* o_ptr = O + (((size_t)b * S + i) * HQ + hq) * DH;
    o_ptr[lane] = __float2half(a0 * inv_l);
    if (lane + half < DH) o_ptr[lane + half] = __float2half(a1 * inv_l);

    if (lane == 0)
        LSE[((size_t)b * HQ + hq) * S + i] = m + __logf(l > 0.0f ? l : 1.0f);
}

static inline void launch_gqa_attn_fwd(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* LSE,
    int B, int S, int HQ, int HKV, int DH, float scale, cudaStream_t stream)
{
    int blocks = B * HQ * S;
    gqa_attn_fwd_kernel<<<blocks, 32, 0, stream>>>(
        Q, K, V, O, LSE, B, S, HQ, HKV, DH, scale);
}

// ─── Backward: one warp per (b, hq, i). ─────────────────────────────────────
// p_j = exp(scale*q.k - LSE);  D = sum_j p_j (dO.V_j)
// dV_j += p_j * dO_i ; dscore = p_j*(dp_j - D)*scale
// dQ_i += dscore*K_j ; dK_j += dscore*Q_i
__global__ void gqa_attn_bwd_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ dO,
    const float*  __restrict__ LSE,
    float* __restrict__ dQ,   // FP32 scratch [B,S,HQ,DH]
    float* __restrict__ dK,   // FP32 scratch [B,S,HKV,DH]
    float* __restrict__ dV,   // FP32 scratch [B,S,HKV,DH]
    int B, int S, int HQ, int HKV, int DH, float scale)
{
    int blk = blockIdx.x;
    int i   = blk % S;
    int hq  = (blk / S) % HQ;
    int b   = blk / (S * HQ);
    if (b >= B) return;

    int hkv  = hq / (HQ / HKV);
    int lane = threadIdx.x;
    int half = DH >> 1;

    const __half* q_ptr  = Q  + (((size_t)b * S + i) * HQ + hq) * DH;
    const __half* dO_ptr = dO + (((size_t)b * S + i) * HQ + hq) * DH;
    float lse = LSE[((size_t)b * HQ + hq) * S + i];

    float q0 = __half2float(q_ptr[lane]);
    float q1 = (lane + half < DH) ? __half2float(q_ptr[lane + half]) : 0.0f;
    float d0 = __half2float(dO_ptr[lane]);
    float d1 = (lane + half < DH) ? __half2float(dO_ptr[lane + half]) : 0.0f;

    // Pass 1: D = sum_j p_j (dO.V_j)
    float Dacc = 0.0f;
    for (int j = 0; j <= i; j++) {
        const __half* k_ptr = K + (((size_t)b * S + j) * HKV + hkv) * DH;
        const __half* v_ptr = V + (((size_t)b * S + j) * HKV + hkv) * DH;
        float k0 = __half2float(k_ptr[lane]);
        float k1 = (lane + half < DH) ? __half2float(k_ptr[lane + half]) : 0.0f;
        float p  = __expf(warp_sum(q0*k0 + q1*k1) * scale - lse);
        float v0 = __half2float(v_ptr[lane]);
        float v1 = (lane + half < DH) ? __half2float(v_ptr[lane + half]) : 0.0f;
        float dp = warp_sum(d0*v0 + d1*v1);
        Dacc += p * dp;
    }

    // Pass 2: accumulate grads
    float dq0 = 0.0f, dq1 = 0.0f;
    for (int j = 0; j <= i; j++) {
        const __half* k_ptr = K + (((size_t)b * S + j) * HKV + hkv) * DH;
        const __half* v_ptr = V + (((size_t)b * S + j) * HKV + hkv) * DH;
        float k0 = __half2float(k_ptr[lane]);
        float k1 = (lane + half < DH) ? __half2float(k_ptr[lane + half]) : 0.0f;
        float p  = __expf(warp_sum(q0*k0 + q1*k1) * scale - lse);
        float v0 = __half2float(v_ptr[lane]);
        float v1 = (lane + half < DH) ? __half2float(v_ptr[lane + half]) : 0.0f;
        float dp = warp_sum(d0*v0 + d1*v1);
        float dscore = p * (dp - Dacc) * scale;

        size_t kv_base = (((size_t)b * S + j) * HKV + hkv) * DH;
        atomicAdd(&dV[kv_base + lane], p * d0);
        if (lane + half < DH) atomicAdd(&dV[kv_base + lane + half], p * d1);
        atomicAdd(&dK[kv_base + lane], dscore * q0);
        if (lane + half < DH) atomicAdd(&dK[kv_base + lane + half], dscore * q1);

        dq0 += dscore * k0;
        dq1 += dscore * k1;
    }

    size_t q_base = (((size_t)b * S + i) * HQ + hq) * DH;
    dQ[q_base + lane] = dq0;
    if (lane + half < DH) dQ[q_base + lane + half] = dq1;
}

static inline void launch_gqa_attn_bwd(
    const __half* Q, const __half* K, const __half* V, const __half* dO,
    const float* LSE, float* dQ, float* dK, float* dV,
    int B, int S, int HQ, int HKV, int DH, float scale, cudaStream_t stream)
{
    int blocks = B * HQ * S;
    gqa_attn_bwd_kernel<<<blocks, 32, 0, stream>>>(
        Q, K, V, dO, LSE, dQ, dK, dV, B, S, HQ, HKV, DH, scale);
}
