#pragma once
// =============================================================================
// train_step.cuh — Correct, self-contained forward + backward for the 100M
// GQA transformer. Stores all activations (no recompute). All math on GPU.
//
// This replaces the broken cuda_autograd stubs (flash_attention / training_backward).
// =============================================================================
#include "cuda/cuda_tensor.h"
#include "cuda/cuda_context.h"
#include "cuda/cuda_ops.h"
#include "cuda/cuda_kernels.cuh"
#include "cuda/attn_kernels.cuh"
#include "cuda/ce_embed_kernels.cuh"
#include <vector>

namespace ts {

using cuda::CudaTensor;
using cuda::DType;

// ─── CUDA error check macro ────────────────────────────────────────────────
#define CUDA_CHK(msg) do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) \
        printf("[CUDA-ERR] %s: %s\n", msg, cudaGetErrorString(_e)); \
} while(0)

// ─── Forward NaN debug (active for first N steps) ───────────────────────────
static int g_debug_fwd_count = 3;
static inline bool check_fwd_nan(const CudaTensor& t, const char* label,
                                  int layer, const char* loc) {
    if (g_debug_fwd_count <= 0) return false;
    size_t n = t.numel();
    if (n == 0) return false;
    unsigned int *d_nan = nullptr, *d_inf = nullptr;
    cudaMalloc(&d_nan, sizeof(unsigned int));
    cudaMalloc(&d_inf, sizeof(unsigned int));
    launch_count_nan_inf((const __half*)t.data, d_nan, d_inf, n,
                         cuda::ctx.compute_stream);
    cuda::sync_compute();
    unsigned int h_nan, h_inf;
    cudaMemcpy(&h_nan, d_nan, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_inf, d_inf, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_nan); cudaFree(d_inf);
    if (h_nan > 0 || h_inf > 0) {
        printf("[FWD-NAN] L%d %s: NaN=%u Inf=%u\n", layer, loc, h_nan, h_inf);
        g_debug_fwd_count = 0;
        return true;
    }
    if (layer < 0) g_debug_fwd_count--;
    return false;
}

struct ModelDims {
    int B, S, D, L, HQ, HKV, DH, FFN, V;
    float rope_theta;
};

// Per-layer saved activations for backward
struct LayerCache {
    CudaTensor x_in;        // residual-stream input to layer  [B,S,D]
    CudaTensor rsqrt_a, normed_a;
    CudaTensor n1;          // rmsnorm output (attn)           [B,S,D]
    CudaTensor Q, K, V;     // post-rope Q,K, and V            [B,S,H*,DH]
    CudaTensor attn;        // attention output flat           [B,S,D]
    CudaTensor LSE;         // [B,HQ,S]
    CudaTensor x_mid;       // x + o                           [B,S,D]
    CudaTensor rsqrt_f, normed_f;
    CudaTensor n2;          // rmsnorm output (ffn)            [B,S,D]
    CudaTensor gate, up;    // [B,S,FFN]
    CudaTensor silu;        // saved silu                      [B,S,FFN]
    CudaTensor ff;          // swiglu output                   [B,S,FFN]
};

struct FwdCache {
    CudaTensor embed_out;   // [B,S,D]
    std::vector<LayerCache> layers;
    CudaTensor rsqrt_final, normed_final;
    CudaTensor x_final;     // final normed hidden             [B,S,D]
};

// ─── helper: make a non-owning view with an explicit shape ──────────────────
static inline CudaTensor view_as(const CudaTensor& src, std::vector<int64_t> shape) {
    CudaTensor v;
    v.data = src.data; v.owns = false; v.dtype = src.dtype; v.nbytes = src.nbytes;
    v.ndim = (int)shape.size();
    for (int i = 0; i < 4; i++) v.shape[i] = (i < v.ndim) ? shape[i] : 1;
    return v;
}

// ─── Forward ────────────────────────────────────────────────────────────────
// params order per layer (base=1+l*9): wq,wk,wv,wo,w1,w2,w3,rms_attn,rms_ffn
// params[0] = embed [V,D]; params[1+L*9] = final_rms [D]
static inline CudaTensor forward(
    const std::vector<CudaTensor>& params,
    const CudaTensor& d_tokens,
    const ModelDims& m,
    const float* rope_cos, const float* rope_sin,
    FwdCache& cache)
{
    int B=m.B,S=m.S,D=m.D,HQ=m.HQ,HKV=m.HKV,DH=m.DH,FFN=m.FFN;
    float attn_scale = 1.0f / sqrtf((float)DH);

    cache.embed_out = cuda::embedding_forward(params[0], d_tokens); // [B,S,D]
    CUDA_CHK("embed_fwd");
    check_fwd_nan(cache.embed_out, "embed", -1, "embed_out");

    // running residual stream (owned)
    CudaTensor x(cache.embed_out.shape_vec(), DType::FP16);
    cudaMemcpyAsync(x.data, cache.embed_out.data, x.nbytes,
                    cudaMemcpyDeviceToDevice, cuda::ctx.compute_stream);

    cache.layers.resize(m.L);
    for (int l = 0; l < m.L; l++) {
        int base = 1 + l*9;
        LayerCache& lc = cache.layers[l];

        // save layer input
        lc.x_in = CudaTensor(x.shape_vec(), DType::FP16);
        cudaMemcpyAsync(lc.x_in.data, x.data, x.nbytes, cudaMemcpyDeviceToDevice, cuda::ctx.compute_stream);

        // --- attention ---
        lc.n1 = cuda::rms_norm(x, params[base+7], 1e-6f, lc.rsqrt_a, lc.normed_a); // [B,S,D]
        check_fwd_nan(lc.n1, "rms_attn", l, "n1");
        CudaTensor n1_2d = view_as(lc.n1, {(int64_t)B*S, D});
        auto Qf = cuda::matmul(n1_2d, params[base+0]);        // [B*S, D]
        auto Kf = cuda::matmul(n1_2d, params[base+1]);        // [B*S, HKV*DH]
        auto Vf = cuda::matmul(n1_2d, params[base+2]);        // [B*S, HKV*DH]

        lc.Q = view_as(Qf, {B,S,HQ,DH});
        lc.K = view_as(Kf, {B,S,HKV,DH});
        lc.V = view_as(Vf, {B,S,HKV,DH});
        // own the buffers by moving the flat tensors into the cache
        lc.Q = std::move(Qf); lc.Q.ndim=4; lc.Q.shape[0]=B; lc.Q.shape[1]=S; lc.Q.shape[2]=HQ; lc.Q.shape[3]=DH;
        lc.K = std::move(Kf); lc.K.ndim=4; lc.K.shape[0]=B; lc.K.shape[1]=S; lc.K.shape[2]=HKV; lc.K.shape[3]=DH;
        lc.V = std::move(Vf); lc.V.ndim=4; lc.V.shape[0]=B; lc.V.shape[1]=S; lc.V.shape[2]=HKV; lc.V.shape[3]=DH;

        cuda::rope_inplace(lc.Q, HQ, DH, 0, rope_cos, rope_sin);
        cuda::rope_inplace(lc.K, HKV, DH, 0, rope_cos, rope_sin);

        // attention forward
        CudaTensor attn4({B,S,HQ,DH}, DType::FP16);
        lc.LSE = CudaTensor({B,HQ,S}, DType::F32);
        launch_gqa_attn_fwd((const __half*)lc.Q.data,(const __half*)lc.K.data,(const __half*)lc.V.data,
                            (__half*)attn4.data,(float*)lc.LSE.data,
                            B,S,HQ,HKV,DH,attn_scale,cuda::ctx.compute_stream);
        if (l == 0) CUDA_CHK("attn_fwd");
        lc.attn = view_as(attn4, {(int64_t)B*S, D});
        lc.attn = std::move(attn4); lc.attn.ndim=2; lc.attn.shape[0]=B*S; lc.attn.shape[1]=D; lc.attn.shape[2]=1; lc.attn.shape[3]=1;

        auto o = cuda::matmul(lc.attn, params[base+3]);       // [B*S, D]
        check_fwd_nan(o, "attn_out", l, "o");
        // x_mid = x + o
        cuda::CudaTensor x_mid(x.shape_vec(), DType::FP16);
        elementwise_add_fp16((__half*)x.data,(__half*)o.data,(__half*)x_mid.data,x.numel(),cuda::ctx.compute_stream);
        check_fwd_nan(x_mid, "resid1", l, "x_mid");
        lc.x_mid = CudaTensor(x_mid.shape_vec(), DType::FP16);
        cudaMemcpyAsync(lc.x_mid.data, x_mid.data, x_mid.nbytes, cudaMemcpyDeviceToDevice, cuda::ctx.compute_stream);

        // --- ffn ---
        lc.n2 = cuda::rms_norm(x_mid, params[base+8], 1e-6f, lc.rsqrt_f, lc.normed_f);
        check_fwd_nan(lc.n2, "rms_ffn", l, "n2");
        CudaTensor n2_2d = view_as(lc.n2, {(int64_t)B*S, D});
        auto gate = cuda::matmul(n2_2d, params[base+4]);      // [B*S, FFN]
        auto up   = cuda::matmul(n2_2d, params[base+6]);      // [B*S, FFN]
        lc.gate = std::move(gate); lc.gate.ndim=3; lc.gate.shape[0]=B; lc.gate.shape[1]=S; lc.gate.shape[2]=FFN;
        lc.up   = std::move(up);   lc.up.ndim=3;   lc.up.shape[0]=B;   lc.up.shape[1]=S;   lc.up.shape[2]=FFN;

        lc.ff = cuda::swiglu(lc.gate, lc.up, lc.silu);        // [B,S,FFN]
        check_fwd_nan(lc.ff, "swiglu", l, "ff");
        CudaTensor ff_2d = view_as(lc.ff, {(int64_t)B*S, FFN});
        auto down = cuda::matmul(ff_2d, params[base+5]);      // [B*S, D]

        // x = x_mid + down
        elementwise_add_fp16((__half*)x_mid.data,(__half*)down.data,(__half*)x.data,x.numel(),cuda::ctx.compute_stream);
        check_fwd_nan(x, "resid2", l, "x");
    }

    // final norm
    cache.x_final = cuda::rms_norm(x, params[1+m.L*9], 1e-6f, cache.rsqrt_final, cache.normed_final);
    check_fwd_nan(cache.x_final, "final_rms", -2, "x_final");
    // return a non-owning view; cache retains ownership (needed for weight-tie grad)
    return view_as(cache.x_final, {(int64_t)B*S, D});
}

// ─── Backward ───────────────────────────────────────────────────────────────
// grads layout mirrors params. dx_final = gradient wrt final normed hidden.
struct Grads {
    CudaTensor d_embed;      // [V,D] FP16
    CudaTensor d_final_rms;  // [D] FP16
    struct LG { CudaTensor dwq,dwk,dwv,dwo,dw1,dw2,dw3,drms_attn,drms_ffn; };
    std::vector<LG> layers;
    CudaTensor embed_accum;  // [V,D] FP32 scratch
};

static inline void backward(
    const std::vector<CudaTensor>& params,
    const CudaTensor& d_tokens,
    const CudaTensor& x_final_hidden, // hidden going into final rmsnorm (== residual stream after last layer)
    const CudaTensor& dx_final,       // grad wrt final rmsnorm output [B*S,D]
    const ModelDims& m,
    const float* rope_cos, const float* rope_sin,
    FwdCache& cache,
    Grads& g)
{
    int B=m.B,S=m.S,D=m.D,HQ=m.HQ,HKV=m.HKV,DH=m.DH,FFN=m.FFN,V=m.V;
    float attn_scale = 1.0f / sqrtf((float)DH);

    // final rmsnorm backward
    CudaTensor dx({(int64_t)B*S, D}, DType::FP16);
    {
        CudaTensor dxf3 = view_as(dx_final, {B,S,D});
        CudaTensor dx3  = view_as(dx, {B,S,D});
        cuda::rms_norm_backward(dxf3, params[1+m.L*9],
                                cache.rsqrt_final, cache.normed_final,
                                dx3, g.d_final_rms, 1e-6f);
    }

    // iterate layers in reverse
    for (int l = m.L-1; l >= 0; l--) {
        int base = 1 + l*9;
        LayerCache& lc = cache.layers[l];
        Grads::LG& lg = g.layers[l];

        // dx currently = grad wrt output of layer l (== x after layer)
        // residual: x = x_mid + down  → d_x_mid = dx, d_down = dx
        CudaTensor d_xmid({(int64_t)B*S,D}, DType::FP16);
        cudaMemcpyAsync(d_xmid.data, dx.data, dx.nbytes, cudaMemcpyDeviceToDevice, cuda::ctx.compute_stream);
        CudaTensor& d_down = dx; // reuse as d_down [B*S,D]

        // down = ff @ W2  → dW2 = ff^T @ d_down ; d_ff = d_down @ W2^T
        CudaTensor ff_2d = view_as(lc.ff, {(int64_t)B*S, FFN});
        lg.dw2 = cuda::matmul(ff_2d, d_down, true, false);         // [FFN, D]
        auto d_ff = cuda::matmul(d_down, params[base+5], false, true); // [B*S, FFN]

        // swiglu backward
        CudaTensor d_ff3 = view_as(d_ff, {B,S,FFN});
        CudaTensor d_gate({B,S,FFN}, DType::FP16), d_up({B,S,FFN}, DType::FP16);
        cuda::swiglu_backward(d_ff3, lc.gate, lc.up, lc.silu, d_gate, d_up);

        // gate = n2 @ W1 ; up = n2 @ W3
        CudaTensor n2_2d = view_as(lc.n2, {(int64_t)B*S, D});
        CudaTensor dgate_2d = view_as(d_gate, {(int64_t)B*S, FFN});
        CudaTensor dup_2d   = view_as(d_up,   {(int64_t)B*S, FFN});
        lg.dw1 = cuda::matmul(n2_2d, dgate_2d, true, false);       // [D, FFN]
        lg.dw3 = cuda::matmul(n2_2d, dup_2d,   true, false);       // [D, FFN]
        auto dn2_a = cuda::matmul(dgate_2d, params[base+4], false, true); // [B*S, D]
        auto dn2_b = cuda::matmul(dup_2d,   params[base+6], false, true); // [B*S, D]
        CudaTensor dn2({(int64_t)B*S,D}, DType::FP16);
        elementwise_add_fp16((__half*)dn2_a.data,(__half*)dn2_b.data,(__half*)dn2.data,dn2.numel(),cuda::ctx.compute_stream);

        // rmsnorm(ffn) backward: dn2 -> d_xmid_contrib, drms_ffn
        CudaTensor d_xmid_from_norm({(int64_t)B*S,D}, DType::FP16);
        CudaTensor dn2_3 = view_as(dn2, {B,S,D});
        cuda::rms_norm_backward(dn2_3, params[base+8], lc.rsqrt_f, lc.normed_f,
                                d_xmid_from_norm, lg.drms_ffn, 1e-6f);
        // d_xmid += contribution
        elementwise_add_fp16((__half*)d_xmid.data,(__half*)d_xmid_from_norm.data,
                             (__half*)d_xmid.data, d_xmid.numel(), cuda::ctx.compute_stream);

        // now d_xmid is grad wrt x_mid.  x_mid = x + o → d_x = d_xmid, d_o = d_xmid
        CudaTensor& d_o = d_xmid; // [B*S,D]

        // o = attn @ Wo → dWo = attn^T @ d_o ; d_attn = d_o @ Wo^T
        lg.dwo = cuda::matmul(lc.attn, d_o, true, false);          // [D, D]
        auto d_attn = cuda::matmul(d_o, params[base+3], false, true); // [B*S, D]
        CudaTensor d_attn4 = view_as(d_attn, {B,S,HQ,DH});

        // attention backward → dQ,dK,dV (FP32 scratch)
        CudaTensor dQf({B,S,HQ,DH}, DType::F32);
        CudaTensor dKf({B,S,HKV,DH}, DType::F32);
        CudaTensor dVf({B,S,HKV,DH}, DType::F32);
        dQf.fill_zero(cuda::ctx.compute_stream);
        dKf.fill_zero(cuda::ctx.compute_stream);
        dVf.fill_zero(cuda::ctx.compute_stream);
        launch_gqa_attn_bwd((const __half*)lc.Q.data,(const __half*)lc.K.data,(const __half*)lc.V.data,
                            (const __half*)d_attn4.data,(const float*)lc.LSE.data,
                            (float*)dQf.data,(float*)dKf.data,(float*)dVf.data,
                            B,S,HQ,HKV,DH,attn_scale,cuda::ctx.compute_stream);

        // convert to fp16
        CudaTensor dQ({B,S,HQ,DH}, DType::FP16);
        CudaTensor dK({B,S,HKV,DH}, DType::FP16);
        CudaTensor dV({B,S,HKV,DH}, DType::FP16);
        launch_cast_f32_f16((float*)dQf.data,(__half*)dQ.data,dQ.numel(),cuda::ctx.compute_stream);
        launch_cast_f32_f16((float*)dKf.data,(__half*)dK.data,dK.numel(),cuda::ctx.compute_stream);
        launch_cast_f32_f16((float*)dVf.data,(__half*)dV.data,dV.numel(),cuda::ctx.compute_stream);

        // rope backward on dQ,dK (rotation applied to gradient)
        cuda::rope_backward_inplace(dQ, HQ, DH, 0, rope_cos, rope_sin);
        cuda::rope_backward_inplace(dK, HKV, DH, 0, rope_cos, rope_sin);

        // Q = n1 @ Wq etc → dWq = n1^T @ dQ ; dn1 += dQ @ Wq^T
        CudaTensor n1_2d = view_as(lc.n1, {(int64_t)B*S, D});
        CudaTensor dQ_2d = view_as(dQ, {(int64_t)B*S, HQ*DH});
        CudaTensor dK_2d = view_as(dK, {(int64_t)B*S, HKV*DH});
        CudaTensor dV_2d = view_as(dV, {(int64_t)B*S, HKV*DH});
        lg.dwq = cuda::matmul(n1_2d, dQ_2d, true, false);          // [D, D]
        lg.dwk = cuda::matmul(n1_2d, dK_2d, true, false);          // [D, HKV*DH]
        lg.dwv = cuda::matmul(n1_2d, dV_2d, true, false);          // [D, HKV*DH]
        auto dn1_q = cuda::matmul(dQ_2d, params[base+0], false, true); // [B*S,D]
        auto dn1_k = cuda::matmul(dK_2d, params[base+1], false, true); // [B*S,D]
        auto dn1_v = cuda::matmul(dV_2d, params[base+2], false, true); // [B*S,D]
        CudaTensor dn1({(int64_t)B*S,D}, DType::FP16);
        elementwise_add_fp16((__half*)dn1_q.data,(__half*)dn1_k.data,(__half*)dn1.data,dn1.numel(),cuda::ctx.compute_stream);
        elementwise_add_fp16((__half*)dn1.data,(__half*)dn1_v.data,(__half*)dn1.data,dn1.numel(),cuda::ctx.compute_stream);

        // rmsnorm(attn) backward: dn1 -> d_x_contrib, drms_attn
        CudaTensor d_x_from_norm({(int64_t)B*S,D}, DType::FP16);
        CudaTensor dn1_3 = view_as(dn1, {B,S,D});
        cuda::rms_norm_backward(dn1_3, params[base+7], lc.rsqrt_a, lc.normed_a,
                                d_x_from_norm, lg.drms_attn, 1e-6f);

        // d_x = d_o(=d_xmid as residual) + d_x_from_norm
        CudaTensor d_x_next({(int64_t)B*S,D}, DType::FP16);
        elementwise_add_fp16((__half*)d_o.data,(__half*)d_x_from_norm.data,
                             (__half*)d_x_next.data, d_x_next.numel(), cuda::ctx.compute_stream);

        // set dx for next (lower) layer
        dx = std::move(d_x_next);
    }

    // embedding backward: dx (== grad wrt embed_out) scatter-add into accum.
    // NOTE: caller must have already seeded g.embed_accum with the weight-tied
    // output-projection gradient (d_logits^T @ x_final) BEFORE calling backward().
    launch_embed_scatter_add((const __half*)dx.data,(const int*)d_tokens.data,
                             (float*)g.embed_accum.data, B*S, D, V, cuda::ctx.compute_stream);
    launch_f32_to_f16((float*)g.embed_accum.data,(__half*)g.d_embed.data, (size_t)V*D, cuda::ctx.compute_stream);
}

} // namespace ts
