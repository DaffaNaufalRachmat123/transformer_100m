#pragma once
// =============================================================================
// cuda_allocator.h — CUDA memory pool + host pinning
// =============================================================================
#include <cuda_runtime.h>
#include <cstddef>

namespace cuda {

void set_memory_limit(size_t bytes);
size_t get_active_bytes();
void clear_cache();

// Host-pinned buffer for async DMA
template<typename T>
class PinnedBuffer {
    T*   ptr  = nullptr;
    size_t len = 0;
public:
    explicit PinnedBuffer(size_t n) : len(n) {
        cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault);
    }
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return len; }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

} // namespace cuda
