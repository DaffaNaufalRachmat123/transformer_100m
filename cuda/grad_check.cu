// =============================================================================
// grad_check.cu — Numeric gradient / reduction correctness harness
//
// Opt-in target. Not part of the default build:
//     cmake --build build --target grad_check && ./build/grad_check
//
// Motivation: two real bugs shipped for ~37k training steps and were invisible
// because the reported loss stayed accurate (the cross-entropy kernels use
// their own correct reduction, so only RMSNorm was affected):
//
//   1. block_reduce_sum used `s > 32`, leaving 64 live partials while the
//      warp-shuffle stage only folded shared[0..31]. Half of every reduction
//      was discarded -> RMSNorm summed 384 of 768 dims but divided by D.
//      Caught here by REDUCER all-ones (exact).
//
//   2. RMSNorm backward applied gamma *outside* the mean-subtraction. Only
//      equivalent when gamma is uniform, so the test below deliberately uses
//      a NON-UNIFORM gamma. A uniform gamma hides this bug entirely.
//
// Tolerances: the project builds with -use_fast_math and all kernel I/O is
// FP16, so exact equality is not meaningful for the finite-difference tests.
// We require cosine similarity > 0.99 and relative L2 error < 0.02. The
// reducer test uses integral values and is checked exactly.
// =============================================================================
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>

// ─── Small helpers ─────────────────────────────────────────────────────────
static int g_failures = 0;

#define CUDA_OK(expr)                                                          \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            printf("  [CUDA ERROR] %s -> %s\n", #expr, cudaGetErrorString(_e)); \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

static float* dev_f32(const std::vector<float>& h) {
    float* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(float));
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

static __half* dev_f16(const std::vector<float>& h) {
    std::vector<__half> tmp(h.size());
    for (size_t i = 0; i < h.size(); i++) tmp[i] = __float2half(h[i]);
    __half* d = nullptr;
    cudaMalloc(&d, tmp.size() * sizeof(__half));
    cudaMemcpy(d, tmp.data(), tmp.size() * sizeof(__half), cudaMemcpyHostToDevice);
    return d;
}

static __half* dev_f16_empty(size_t n) {
    __half* d = nullptr;
    cudaMalloc(&d, n * sizeof(__half));
    cudaMemset(d, 0, n * sizeof(__half));
    return d;
}

static std::vector<float> host_f16(const __half* d, size_t n) {
    std::vector<__half> tmp(n);
    cudaMemcpy(tmp.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) out[i] = __half2float(tmp[i]);
    return out;
}

// Compare against a reference: cosine similarity + relative L2 error.
static void check_vec(const char* name,
                      const std::vector<float>& got,
                      const std::vector<float>& ref,
                      double cos_min = 0.99, double rel_max = 0.02) {
    double dot = 0, na = 0, nb = 0, diff = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        dot  += (double)got[i] * ref[i];
        na   += (double)got[i] * got[i];
        nb   += (double)ref[i] * ref[i];
        diff += (double)(got[i] - ref[i]) * (got[i] - ref[i]);
    }
    double cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    double rel = (nb > 0) ? std::sqrt(diff) / std::sqrt(nb) : std::sqrt(diff);
    bool ok = (cos > cos_min) && (rel < rel_max);
    printf("  %-34s cos=%.6f relerr=%.5f  %s\n",
           name, cos, rel, ok ? "PASS" : "*** FAIL ***");
    if (!ok) g_failures++;
}

// =============================================================================
// TEST 1 — block_reduce_sum via RMSNorm forward, all-ones input (EXACT)
//
// With x[i] = 1 and gamma[i] = 1:  mean(x^2) = 1  ->  rsqrt = 1/sqrt(1+eps).
// If the reducer drops partials, mean(x^2) < 1 and rsqrt grows. Recovering
// the summed element count is exact:  n_summed = D / rsqrt^2  (eps ~ 0).
//
// The historical bug reported 384 of 768. This runs across several D values
// and both call-site-relevant shapes.
// =============================================================================
static void test_reducer_exact() {
    printf("[TEST 1] block_reduce_sum — all-ones exactness (RMSNorm fwd)\n");
    const float eps = 0.0f;
    const int Ds[] = {64, 128, 256, 512, 768, 1024, 3072};

    for (int D : Ds) {
        const int rows = 4;
        std::vector<float> x(rows * D, 1.0f), gamma(D, 1.0f);
        __half* d_x = dev_f16(x);
        __half* d_g = dev_f16(gamma);
        __half* d_y = dev_f16_empty(rows * D);
        __half* d_rs = dev_f16_empty(rows);
        __half* d_nm = dev_f16_empty(rows * D);

        rms_norm_fused_kernel_forward(d_x, d_g, d_y, d_rs, d_nm,
                                      rows, 1, D, eps, 0);
        CUDA_OK(cudaDeviceSynchronize());

        auto rs = host_f16(d_rs, rows);
        // n_summed = D * mean(x^2) = D / rsqrt^2
        double n_summed = (double)D / ((double)rs[0] * rs[0]);
        long   rounded  = std::lround(n_summed);
        bool ok = (rounded == D);
        printf("    D=%-5d summed %4ld / %-5d dims   rsqrt=%.5f  %s\n",
               D, rounded, D, rs[0], ok ? "PASS" : "*** FAIL ***");
        if (!ok) g_failures++;

        cudaFree(d_x); cudaFree(d_g); cudaFree(d_y);
        cudaFree(d_rs); cudaFree(d_nm);
    }
}

// =============================================================================
// TEST 2 — RMSNorm backward vs finite differences, NON-UNIFORM gamma
//
// Two properties are required for this test to actually catch bug #2
// (gamma applied outside the mean-subtraction):
//
//   a) gamma must be NON-UNIFORM. With uniform gamma the two formulations are
//      algebraically identical.
//
//   b) dy must be CORRELATED with the normalized activation. The erroneous
//      term is rank-1 along `normed` and scales with mean(dy*gamma*normed).
//      For i.i.d. random dy that mean concentrates at 0 as D grows (measured
//      relerr at D=768 is only ~0.015 — under the 0.02 threshold, i.e. the
//      bug would PASS). Real backprop dy is strongly correlated with the
//      activation, where the same variant shows ~0.34 relerr. We therefore
//      inject an explicit component along `normed`.
//
// Finite differences are evaluated in FP64 on the host against the same
// mathematical forward the kernel implements.
// =============================================================================
static std::vector<double> rmsnorm_fwd_host(const std::vector<double>& x,
                                            const std::vector<double>& g,
                                            int D, double eps) {
    double ss = 0;
    for (int i = 0; i < D; i++) ss += x[i] * x[i];
    double r = 1.0 / std::sqrt(ss / D + eps);
    std::vector<double> y(D);
    for (int i = 0; i < D; i++) y[i] = x[i] * r * g[i];
    return y;
}

static void test_rmsnorm_backward() {
    printf("[TEST 2] RMSNorm backward vs finite differences (non-uniform gamma)\n");
    const int D = 768;
    const double eps = 1e-5;
    std::mt19937 rng(1234);
    std::normal_distribution<double> nrm(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.6, 3.2);   // non-uniform gamma

    std::vector<double> x(D), g(D), dy(D);
    for (int i = 0; i < D; i++) {
        x[i]  = nrm(rng);
        g[i]  = uni(rng);
    }
    // dy must correlate with `normed` — see (b) in the comment above.
    {
        double ss0 = 0;
        for (int i = 0; i < D; i++) ss0 += x[i] * x[i];
        double r0 = 1.0 / std::sqrt(ss0 / D + eps);
        for (int i = 0; i < D; i++)
            dy[i] = 0.05 * nrm(rng) + 0.05 * (x[i] * r0);
    }

    // ── Host finite-difference reference for dx ──
    std::vector<double> dx_ref(D);
    const double h = 1e-5;
    for (int i = 0; i < D; i++) {
        std::vector<double> xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        auto yp = rmsnorm_fwd_host(xp, g, D, eps);
        auto ym = rmsnorm_fwd_host(xm, g, D, eps);
        double acc = 0;
        for (int j = 0; j < D; j++) acc += (yp[j] - ym[j]) / (2 * h) * dy[j];
        dx_ref[i] = acc;
    }
    // dgamma reference: dy * normed
    double ss = 0;
    for (int i = 0; i < D; i++) ss += x[i] * x[i];
    double r = 1.0 / std::sqrt(ss / D + eps);
    std::vector<double> dgamma_ref(D);
    for (int i = 0; i < D; i++) dgamma_ref[i] = dy[i] * (x[i] * r);

    // ── Run the real kernels ──
    std::vector<float> xf(D), gf(D), dyf(D);
    for (int i = 0; i < D; i++) { xf[i] = (float)x[i]; gf[i] = (float)g[i]; dyf[i] = (float)dy[i]; }

    __half* d_x  = dev_f16(xf);
    __half* d_g  = dev_f16(gf);
    __half* d_dy = dev_f16(dyf);
    __half* d_y  = dev_f16_empty(D);
    __half* d_rs = dev_f16_empty(1);
    __half* d_nm = dev_f16_empty(D);

    rms_norm_fused_kernel_forward(d_x, d_g, d_y, d_rs, d_nm, 1, 1, D, (float)eps, 0);
    CUDA_OK(cudaDeviceSynchronize());

    __half* d_dx = dev_f16_empty(D);
    __half* d_dg = dev_f16_empty(D);
    rms_norm_fused_kernel_backward(d_dy, d_g, d_rs, d_nm, d_dx, d_dg,
                                   1, 1, D, (float)eps, 0);
    CUDA_OK(cudaDeviceSynchronize());

    auto dx_got = host_f16(d_dx, D);
    auto dg_got = host_f16(d_dg, D);

    std::vector<float> dx_r(D), dg_r(D);
    for (int i = 0; i < D; i++) { dx_r[i] = (float)dx_ref[i]; dg_r[i] = (float)dgamma_ref[i]; }

    check_vec("dx   vs finite-difference", dx_got, dx_r);
    check_vec("dgamma vs analytic",        dg_got, dg_r);

    // Explicit guard against the historical bug: gamma applied outside the
    // mean-subtraction. Report how that variant would score, so a regression
    // is obvious rather than silent.
    {
        double mean_dyn = 0;
        for (int i = 0; i < D; i++) mean_dyn += dy[i] * (x[i] * r);
        mean_dyn /= D;
        std::vector<float> buggy(D);
        for (int i = 0; i < D; i++)
            buggy[i] = (float)(r * (dy[i] - (x[i] * r) * mean_dyn) * g[i]);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < D; i++) {
            dot += (double)buggy[i] * dx_r[i];
            na  += (double)buggy[i] * buggy[i];
            nb  += (double)dx_r[i]  * dx_r[i];
        }
        printf("    (reference: gamma-outside variant would score cos=%.6f)\n",
               dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_dy); cudaFree(d_y);
    cudaFree(d_rs); cudaFree(d_nm); cudaFree(d_dx); cudaFree(d_dg);
}

// =============================================================================
// TEST 3 — SwiGLU backward vs finite differences
// Convention in this codebase: y = gate * silu(up)   (see train_step.cuh)
// =============================================================================
static void test_swiglu_backward() {
    printf("[TEST 3] SwiGLU backward vs finite differences\n");
    const int FD = 1024;
    std::mt19937 rng(99);
    std::normal_distribution<double> nrm(0.0, 1.0);

    std::vector<double> gate(FD), up(FD), dy(FD);
    for (int i = 0; i < FD; i++) {
        gate[i] = nrm(rng);
        up[i]   = nrm(rng);
        dy[i]   = nrm(rng) * 0.1;
    }
    auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };

    // Elementwise -> analytic reference is exact and cheap.
    std::vector<float> dgate_ref(FD), dup_ref(FD);
    for (int i = 0; i < FD; i++) {
        double s   = 1.0 / (1.0 + std::exp(-up[i]));
        double sl  = up[i] * s;
        double dsl = s * (1.0 + up[i] * (1.0 - s));      // d/du [u*sigmoid(u)]
        dgate_ref[i] = (float)(dy[i] * sl);
        dup_ref[i]   = (float)(dy[i] * gate[i] * dsl);
    }

    std::vector<float> gf(FD), uf(FD), dyf(FD);
    for (int i = 0; i < FD; i++) { gf[i]=(float)gate[i]; uf[i]=(float)up[i]; dyf[i]=(float)dy[i]; }

    __half* d_gate = dev_f16(gf);
    __half* d_up   = dev_f16(uf);
    __half* d_dy   = dev_f16(dyf);
    __half* d_y    = dev_f16_empty(FD);
    __half* d_silu = dev_f16_empty(FD);

    swiglu_fused_forward(d_gate, d_up, d_y, d_silu, 1, 1, FD, 0);
    CUDA_OK(cudaDeviceSynchronize());

    // Sanity: forward must equal gate * silu(up)
    {
        auto y = host_f16(d_y, FD);
        std::vector<float> yref(FD);
        for (int i = 0; i < FD; i++) yref[i] = (float)(gate[i] * silu(up[i]));
        check_vec("forward y = gate*silu(up)", y, yref);
    }

    __half* d_dgate = dev_f16_empty(FD);
    __half* d_dup   = dev_f16_empty(FD);
    swiglu_fused_backward(d_dy, d_gate, d_up, d_silu, d_dgate, d_dup, 1, 1, FD, 0);
    CUDA_OK(cudaDeviceSynchronize());

    check_vec("dgate vs analytic", host_f16(d_dgate, FD), dgate_ref);
    check_vec("dup   vs analytic", host_f16(d_dup,   FD), dup_ref);

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_dy); cudaFree(d_y);
    cudaFree(d_silu); cudaFree(d_dgate); cudaFree(d_dup);
}

// =============================================================================
// TEST 4 — RoPE backward is the transpose of the forward rotation
//
// RoPE forward is an orthogonal rotation R. Backward must apply R^T, so
// backward(forward(v)) == v for any v. This is a strong, tolerance-free
// structural property that catches sign/pairing mistakes.
// =============================================================================
static void test_rope_transpose() {
    printf("[TEST 4] RoPE backward == transpose of forward (round-trip)\n");
    const int B = 2, S = 8, H = 4, DH = 64;
    const int n = B * S * H * DH;
    const int half = DH / 2;
    const float theta = 500000.0f;

    std::vector<float> cosv(S * half), sinv(S * half);
    for (int p = 0; p < S; p++)
        for (int j = 0; j < half; j++) {
            float freq = 1.0f / std::pow(theta, (2.0f * j) / DH);
            cosv[p * half + j] = std::cos(p * freq);
            sinv[p * half + j] = std::sin(p * freq);
        }

    std::mt19937 rng(7);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) v[i] = nrm(rng);

    float* d_cos = dev_f32(cosv);
    float* d_sin = dev_f32(sinv);
    __half* d_v  = dev_f16(v);

    rope_fused_forward(d_v, d_cos, d_sin, B, S, H, DH, 0, 0);
    CUDA_OK(cudaDeviceSynchronize());
    rope_fused_backward(d_v, d_cos, d_sin, B, S, H, DH, 0, 0);
    CUDA_OK(cudaDeviceSynchronize());

    check_vec("R^T(R(v)) == v", host_f16(d_v, n), v);

    cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_v);
}

// =============================================================================
int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("No CUDA device available — cannot run grad_check.\n");
        return 77;   // distinct from a real failure
    }
    cudaSetDevice(0);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    printf("==========================================================\n");
    printf(" grad_check — kernel correctness harness\n");
    printf(" device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("==========================================================\n\n");

    test_reducer_exact();   printf("\n");
    test_rmsnorm_backward(); printf("\n");
    test_swiglu_backward();  printf("\n");
    test_rope_transpose();   printf("\n");

    printf("==========================================================\n");
    if (g_failures == 0) printf(" ALL CHECKS PASSED\n");
    else                 printf(" %d CHECK(S) FAILED\n", g_failures);
    printf("==========================================================\n");
    return g_failures == 0 ? 0 : 1;
}
