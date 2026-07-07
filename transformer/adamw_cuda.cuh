#pragma once
// =============================================================================
// adamw_cuda.cuh — Fused AdamW optimizer for CUDA
//
// Uses the adamw_fused_kernel from cuda_kernels.cuh for a single-kernel
// parameter update. Stores moments in FP32 for numerical stability.
// =============================================================================
#include "cuda/cuda_tensor.h"
#include "cuda/cuda_kernels.cuh"
#include <cstdio>
#include <vector>

namespace cuda {

struct AdamWState {
    CudaTensor m;  // First moment (FP32)
    CudaTensor v;  // Second moment (FP32)

    void init(const CudaTensor& param) {
        m = zeros(param.shape_vec(), DType::F32);
        v = zeros(param.shape_vec(), DType::F32);
    }

    void update(CudaTensor& param, CudaTensor& master_param,
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
};

// Apply AdamW to all parameters with optional gradient clipping
inline void apply_adamw(
    std::vector<CudaTensor>& params,
    std::vector<CudaTensor>& master_params,
    const std::vector<CudaTensor>& grads,
    std::vector<AdamWState>& states,
    float lr, float beta1, float beta2,
    float eps, float wd, int step,
    float grad_clip = 1.0f)
{
    // Gradient clipping (simplified — full impl computes global norm)
    if (grad_clip > 0.0f) {
        float total_norm_sq = 0.0f;
        for (auto& g : grads) {
            auto g_sq = sum_all(g);
            total_norm_sq += g_sq.item_f32();
        }
        float total_norm = sqrtf(total_norm_sq);
        float scale = grad_clip / fmaxf(total_norm, grad_clip);

        if (scale < 1.0f) {
            printf("[CLIP] grad_norm=%.3f → scale=%.4f\n", total_norm, scale);
        }
        // Apply scale to gradients before optimizer step (simplified)
        (void)scale; // FIXME: apply scale to all grads
    }

    for (size_t i = 0; i < params.size(); i++) {
        states[i].update(params[i], master_params[i], grads[i],
                         lr, beta1, beta2, eps, wd, step);
    }
    sync_compute();
}

// Initialize optimizer states for all parameters
inline std::vector<AdamWState> init_adamw_states(const std::vector<CudaTensor>& params) {
    std::vector<AdamWState> states(params.size());
    for (size_t i = 0; i < params.size(); i++) {
        states[i].init(params[i]);
    }
    return states;
}

} // namespace cuda
