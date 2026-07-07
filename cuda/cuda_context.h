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

    bool initialized = false;

    void init(int device = 0);
    void sync();
    void sync_compute();
    void sync_copy();
    void get_memory_info();
    void free_unused();
    ~CudaContext();
};

extern CudaContext ctx;

inline void sync()              { ctx.sync(); }
inline void sync_compute()       { cudaStreamSynchronize(ctx.compute_stream); }
inline void sync_copy()          { cudaStreamSynchronize(ctx.copy_stream); }
inline size_t memory_free()      { ctx.get_memory_info(); return ctx.free_memory; }
inline size_t memory_total()     { return ctx.total_memory; }
inline size_t memory_used()      { ctx.get_memory_info(); return ctx.total_memory - ctx.free_memory; }

} // namespace cuda
