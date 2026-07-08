#include "cuda_context.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cuda {

CudaContext ctx;

void CudaContext::init(int device) {
    if (initialized) return;

    device_id = device;
    cudaSetDevice(device_id);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    total_memory = prop.totalGlobalMem;
    compute_cap_major = prop.major;
    compute_cap_minor = prop.minor;

    // Populate GPU capabilities
    strncpy(gpu_caps.name, prop.name, sizeof(gpu_caps.name) - 1);
    gpu_caps.sm_version = prop.major * 10 + prop.minor;
    gpu_caps.sm_count = prop.multiProcessorCount;
    gpu_caps.vram_bytes = prop.totalGlobalMem;
    gpu_caps.max_threads_per_block = prop.maxThreadsPerBlock;
    gpu_caps.warp_size = prop.warpSize;
    gpu_caps.has_bf16 = (gpu_caps.sm_version >= 80);
    gpu_caps.has_fp8 = (gpu_caps.sm_version >= 89);
    gpu_caps.has_tensor_cores = (gpu_caps.sm_version >= 70);

    print_gpu_info();

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

void CudaContext::print_gpu_info() {
    printf("[CUDA] Device %d: %s\n", device_id, gpu_caps.name);
    printf("[CUDA] Compute: sm_%d | SMs: %d | VRAM: %.1f GB\n",
           gpu_caps.sm_version, gpu_caps.sm_count,
           gpu_caps.vram_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("[CUDA] Capabilities: TensorCores=%s BF16=%s FP8=%s\n",
           gpu_caps.has_tensor_cores ? "yes" : "no",
           gpu_caps.has_bf16 ? "yes" : "no",
           gpu_caps.has_fp8 ? "yes" : "no");
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
