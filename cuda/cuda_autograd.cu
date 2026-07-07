// =============================================================================
// cuda_autograd.cu — Backward pass implementations (gradient checkpointing)
// =============================================================================
#include "cuda_autograd.h"
#include "cuda_ops.h"
#include "cuda_kernels.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

namespace cuda {

// ─── Initialize gradient tensors (zero-filled, matching param shapes) ──────
AutogradGrads init_grads(const AutogradParams& params) {
    AutogradGrads g;
    g.d_embed = CudaTensor(params.embed.shape_vec(), params.embed.dtype);
    g.d_embed.fill_zero(ctx.compute_stream);

    g.d_final_rms = CudaTensor(params.final_rms.shape_vec(), params.final_rms.dtype);
    g.d_final_rms.fill_zero(ctx.compute_stream);

    for (const auto& layer : params.layers) {
        AutogradGrads::LayerGrads lg;
        lg.dwq = CudaTensor(layer.wq.shape_vec(), layer.wq.dtype); lg.dwq.fill_zero(ctx.compute_stream);
        lg.dwk = CudaTensor(layer.wk.shape_vec(), layer.wk.dtype); lg.dwk.fill_zero(ctx.compute_stream);
        lg.dwv = CudaTensor(layer.wv.shape_vec(), layer.wv.dtype); lg.dwv.fill_zero(ctx.compute_stream);
        lg.dwo = CudaTensor(layer.wo.shape_vec(), layer.wo.dtype); lg.dwo.fill_zero(ctx.compute_stream);
        lg.dw1 = CudaTensor(layer.w1.shape_vec(), layer.w1.dtype); lg.dw1.fill_zero(ctx.compute_stream);
        lg.dw2 = CudaTensor(layer.w2.shape_vec(), layer.w2.dtype); lg.dw2.fill_zero(ctx.compute_stream);
        lg.dw3 = CudaTensor(layer.w3.shape_vec(), layer.w3.dtype); lg.dw3.fill_zero(ctx.compute_stream);
        lg.drms_attn = CudaTensor(layer.rms_attn.shape_vec(), layer.rms_attn.dtype); lg.drms_attn.fill_zero(ctx.compute_stream);
        lg.drms_ffn = CudaTensor(layer.rms_ffn.shape_vec(), layer.rms_ffn.dtype); lg.drms_ffn.fill_zero(ctx.compute_stream);
        g.layers.push_back(std::move(lg));
    }
    sync_compute();
    return g;
}

// ─── Training backward (DEPRECATED / DISABLED) ─────────────────────────────
// This was a placeholder implementation that produced INCORRECT gradients
// (identity/outer-product placeholders + CPU embedding loop). It is superseded
// by transformer/train_step.cuh (ts::forward / ts::backward), which computes
// the full, correct backward pass. Calling this is a bug; abort loudly so it
// can never silently corrupt training again.
void training_backward(
    const CudaTensor& /*d_logits*/,
    const CudaTensor& /*tokens*/,
    const AutogradParams& /*params*/,
    const ForwardCache& /*cache*/,
    AutogradGrads& /*grads*/)
{
    fprintf(stderr,
        "[FATAL] cuda::training_backward is deprecated and produces WRONG "
        "gradients. Use ts::backward from transformer/train_step.cuh instead.\n");
    abort();
}

// ─── Flash Attention (DEPRECATED / DISABLED) ───────────────────────────────
// This was a stub that merely copied Q into the output (a no-op attention). It
// is superseded by launch_gqa_attn_fwd in cuda/attn_kernels.cuh. Abort loudly.
CudaTensor flash_attention(const CudaTensor& /*Q*/, const CudaTensor& /*K*/,
                           const CudaTensor& /*V*/, float /*scale*/, bool /*causal*/) {
    fprintf(stderr,
        "[FATAL] cuda::flash_attention is a deprecated no-op stub. Use "
        "launch_gqa_attn_fwd from cuda/attn_kernels.cuh instead.\n");
    abort();
    return CudaTensor();
}

} // namespace cuda
