#pragma once
// =============================================================================
// cuda_autograd.h — Manual backward pass for training
//
// Implements the full training backward: given d_logits (from cross_entropy),
// computes all weight gradients. Uses gradient checkpointing to minimize memory.
//
// Architecture order:
//   Embed → [RMSNorm→QKV→RoPE→FlashAttn→Wo→RMSNorm→SwiGLU→W2]×20 → RMSNorm → Logits
//
// Backward order:
//   d_logits → d_x_final → for L=19..0: recompute forward → backward layer → dW[L]
// =============================================================================
#include "cuda_tensor.h"
#include "cuda_ops.h"
#include <vector>

namespace cuda {

struct AutogradParams {
    CudaTensor embed;       // [V, D] FP16
    CudaTensor final_rms;   // [D] FP16

    struct Layer {
        CudaTensor wq, wk, wv, wo;     // QKV weights
        CudaTensor w1, w2, w3;         // FFN weights
        CudaTensor rms_attn, rms_ffn;  // RMS gamma
    };
    std::vector<Layer> layers;
};

struct AutogradGrads {
    CudaTensor d_embed;
    CudaTensor d_final_rms;
    struct LayerGrads {
        CudaTensor dwq, dwk, dwv, dwo;
        CudaTensor dw1, dw2, dw3;
        CudaTensor drms_attn, drms_ffn;
    };
    std::vector<LayerGrads> layers;
};

// ─── Forward pass (training) ───────────────────────────────────────────────
struct ForwardCache {
    std::vector<CudaTensor> hidden_states; // input to each layer + final
    CudaTensor final_normed;
};

void training_forward(
    const CudaTensor& tokens,
    const AutogradParams& params,
    CudaTensor& logits,
    ForwardCache& cache);

// ─── Backward pass (full) ──────────────────────────────────────────────────
// d_logits comes from cross_entropy backward
// Returns gradients into 'grads' struct
void training_backward(
    const CudaTensor& d_logits,
    const CudaTensor& tokens,
    const AutogradParams& params,
    const ForwardCache& cache,
    AutogradGrads& grads);

// ─── Gradient checkpointing ────────────────────────────────────────────────
// Stores only layer inputs. Recomputation happens during backward.
void backward_with_checkpointing(
    const CudaTensor& d_logits,
    const CudaTensor& tokens,
    const AutogradParams& params,
    const std::vector<CudaTensor>& hidden_inputs,
    AutogradGrads& grads);

// ─── Initialize gradients ──────────────────────────────────────────────────
AutogradGrads init_grads(const AutogradParams& params);

// ─── Initialize parameters ─────────────────────────────────────────────────
AutogradParams init_params(int d_model, int n_layers, int n_q_heads,
                           int n_kv_heads, int d_head, int ffn_dim,
                           int vocab_size, uint64_t seed);

} // namespace cuda
