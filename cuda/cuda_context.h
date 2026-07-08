#pragma once
// =============================================================================
// cuda_context.h — CUDA device initialization, streams, handles
// =============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdint>
#include <cstdio>

namespace cuda {

struct GpuCapabilities {
    char     name[256]      = {};
    int      sm_version     = 0;   // e.g. 75 for T4, 80 for A100
    int      sm_count       = 0;   // number of streaming multiprocessors
    size_t   vram_bytes     = 0;
    int      max_threads_per_block = 0;
    int      warp_size      = 0;
    bool     has_bf16       = false;  // sm >= 80
    bool     has_fp8        = false;  // sm >= 89
    bool     has_tensor_cores = false; // sm >= 70
};

struct CudaContext {
    int            device_id       = 0;
    cudaStream_t   compute_stream  = nullptr;
    cudaStream_t   copy_stream     = nullptr;
    cudaStream_t   loss_stream     = nullptr;
    cublasHandle_t cublas_handle   = nullptr;
    cudnnHandle_t  cudnn_handle    = nullptr;

    size_t total_memory   = 0;
    size_t free_memory    = 0;
    int    compute_cap_major = 0;
    int    compute_cap_minor = 0;

    GpuCapabilities gpu_caps;

    bool initialized = false;

    void init(int device = 0);
    void sync();
    void sync_compute();
    void sync_copy();
    void get_memory_info();
    void free_unused();
    void print_gpu_info();
    ~CudaContext();
};

extern CudaContext ctx;

inline void sync()              { ctx.sync(); }
inline void sync_compute()       { cudaStreamSynchronize(ctx.compute_stream); }
inline void sync_copy()          { cudaStreamSynchronize(ctx.copy_stream); }
inline size_t memory_free()      { ctx.get_memory_info(); return ctx.free_memory; }
inline size_t memory_total()     { return ctx.total_memory; }
inline size_t memory_used()      { ctx.get_memory_info(); return ctx.total_memory - ctx.free_memory; }

// Runtime GPU capability queries
inline int  gpu_sm_version()     { return ctx.gpu_caps.sm_version; }
inline bool gpu_has_bf16()       { return ctx.gpu_caps.has_bf16; }
inline bool gpu_has_tensor_cores() { return ctx.gpu_caps.has_tensor_cores; }
inline int  gpu_sm_count()       { return ctx.gpu_caps.sm_count; }

} // namespace cuda
