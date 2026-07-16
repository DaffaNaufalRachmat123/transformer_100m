#include "cuda_context.h"
#include "cuda_allocator.h"
#include <cstdio>
#include <cstdlib>

namespace cuda {

CudaContext ctx;
CudaArena arena;

void CudaContext::init(int device) {
    if (initialized) return;

    device_id = device;
    cudaSetDevice(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    total_memory = prop.totalGlobalMem;
    compute_cap_major = prop.major;
    compute_cap_minor = prop.minor;

    printf("[CUDA] Device %d: %s\n", device_id, prop.name);
    printf("[CUDA] Compute: sm_%d%d | VRAM: %.1f GB\n",
           compute_cap_major, compute_cap_minor,
           total_memory / (1024.0 * 1024.0 * 1024.0));

    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&loss_stream);

    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, compute_stream);

    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, compute_stream);

    get_memory_info();
    initialized = true;
}

void CudaContext::sync() {
    cudaDeviceSynchronize();
}

void CudaContext::sync_compute() {
    cudaStreamSynchronize(compute_stream);
}

void CudaContext::sync_copy() {
    cudaStreamSynchronize(copy_stream);
}

void CudaContext::get_memory_info() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    free_memory = free_bytes;
    total_memory = total_bytes;
}

void CudaContext::free_unused() {
    cudaDeviceSynchronize();
    cudaMemPool_t pool;
    cudaDeviceGetMemPool(&pool, device_id);
    cudaMemPoolTrimTo(pool, 0);
}

CudaContext::~CudaContext() {
    if (compute_stream) { cudaStreamDestroy(compute_stream); compute_stream = nullptr; }
    if (copy_stream)    { cudaStreamDestroy(copy_stream);    copy_stream = nullptr; }
    if (loss_stream)    { cudaStreamDestroy(loss_stream);    loss_stream = nullptr; }
    if (cublas_handle)  { cublasDestroy(cublas_handle);     cublas_handle = nullptr; }
    if (cudnn_handle)   { cudnnDestroy(cudnn_handle);       cudnn_handle = nullptr; }
    initialized = false;
}

} // namespace cuda
