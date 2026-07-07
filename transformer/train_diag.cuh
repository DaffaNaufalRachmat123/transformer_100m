#pragma once
// =============================================================================
// train_diag.cuh — Training diagnostics (compact output + file logging)
// --debug_steps N: prints summary to stdout, writes details to file
// =============================================================================
#include "cuda/cuda_tensor.h"
#include "cuda/cuda_context.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

namespace diag {

struct TensorStats {
    float min_val = 0, max_val = 0, mean_val = 0, std_val = 0;
    int inf_count = 0, nan_count = 0;
    size_t numel = 0;
};

static inline TensorStats compute_fp16_stats(const cuda::CudaTensor& t) {
    TensorStats s = {};
    s.numel = t.numel();
    if (s.numel == 0 || !t.data) return s;
    std::vector<__half> h(s.numel);
    cudaMemcpy(h.data(), t.data, s.numel * sizeof(__half), cudaMemcpyDeviceToHost);
    double sum = 0.0, sum_sq = 0.0;
    s.min_val = 1e30f; s.max_val = -1e30f;
    for (size_t i = 0; i < s.numel; i++) {
        float v = __half2float(h[i]);
        if (std::isinf(v)) { s.inf_count++; continue; }
        if (std::isnan(v)) { s.nan_count++; continue; }
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        sum += v; sum_sq += (double)v * v;
    }
    size_t valid = s.numel - s.inf_count - s.nan_count;
    if (valid > 0) {
        s.mean_val = (float)(sum / valid);
        double var = sum_sq / valid - (double)s.mean_val * s.mean_val;
        s.std_val = (float)sqrt(var > 0 ? var : 0);
    }
    return s;
}

static inline TensorStats compute_f32_stats(const cuda::CudaTensor& t) {
    TensorStats s = {};
    s.numel = t.numel();
    if (s.numel == 0 || !t.data) return s;
    std::vector<float> h(s.numel);
    cudaMemcpy(h.data(), t.data, s.numel * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0.0, sum_sq = 0.0;
    s.min_val = 1e30f; s.max_val = -1e30f;
    for (size_t i = 0; i < s.numel; i++) {
        float v = h[i];
        if (std::isinf(v)) { s.inf_count++; continue; }
        if (std::isnan(v)) { s.nan_count++; continue; }
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        sum += v; sum_sq += (double)v * v;
    }
    size_t valid = s.numel - s.inf_count - s.nan_count;
    if (valid > 0) {
        s.mean_val = (float)(sum / valid);
        double var = sum_sq / valid - (double)s.mean_val * s.mean_val;
        s.std_val = (float)sqrt(var > 0 ? var : 0);
    }
    return s;
}

static inline TensorStats compute_stats(const cuda::CudaTensor& t) {
    if (t.dtype == cuda::DType::F32) return compute_f32_stats(t);
    return compute_fp16_stats(t);
}

// Print compact summary to stdout (only key metrics)
static inline void print_step_summary(int step, float loss, float prev_loss,
                                       float lr, float global_norm, float clip_coeff,
                                       bool skipped) {
    float delta = loss - prev_loss;
    bool spike = (delta > 0.3f && step > 5);
    printf("[DIAG] step=%d loss=%.4f(%+.3f) lr=%.2e gnorm=%.2f clip=%.4f%s%s\n",
           step, loss, delta, lr, global_norm, clip_coeff,
           skipped ? " SKIP" : "", spike ? " ***SPIKE***" : "");
}

// Write full diagnostics to file (details for offline analysis)
static inline void write_full_diag(
    FILE* f, int step, float loss, float prev_loss, float lr,
    float global_norm, float clip_coeff, bool skipped,
    const std::vector<cuda::CudaTensor>& params,
    const std::vector<std::string>& grad_names,
    const std::vector<TensorStats>& grad_stats,
    const std::vector<std::string>& fwd_names,
    const std::vector<TensorStats>& fwd_stats)
{
    fprintf(f, "STEP %d loss=%.6f delta=%+.4f lr=%.2e gnorm=%.4f clip=%.6f skip=%d\n",
            step, loss, loss - prev_loss, lr, global_norm, clip_coeff, skipped ? 1 : 0);

    // Forward stats
    fprintf(f, "FWD ");
    for (size_t i = 0; i < fwd_names.size(); i++) {
        auto& s = fwd_stats[i];
        fprintf(f, "%s:%.2f/%.2f/%.4f/%d/%d ", fwd_names[i].c_str(),
                s.min_val, s.max_val, s.std_val, s.inf_count, s.nan_count);
    }
    fprintf(f, "\n");

    // Gradient stats
    fprintf(f, "GRAD ");
    for (size_t i = 0; i < grad_names.size(); i++) {
        auto& s = grad_stats[i];
        fprintf(f, "%s:%.4e/%.4e/%.4e/%d/%d ", grad_names[i].c_str(),
                s.min_val, s.max_val, s.std_val, s.inf_count, s.nan_count);
    }
    fprintf(f, "\n");

    // Param health (just max abs for each)
    fprintf(f, "PARAM ");
    for (size_t i = 0; i < std::min(params.size(), (size_t)20); i++) {
        auto s = compute_stats(params[i]);
        fprintf(f, "p%zu:%.4f/%.4f/%d/%d ", i, s.min_val, s.max_val, s.inf_count, s.nan_count);
    }
    fprintf(f, "\n---\n");
    fflush(f);
}

} // namespace diag
