#pragma once
// =============================================================================
// cuda_allocator.h — CUDA arena memory pool + host pinning
//
// The arena eliminates ~600+ cudaMalloc/cudaFree calls per training step.
// Pattern: alloc from bump pointer, reset once per step.
// =============================================================================
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdint>

namespace cuda {

// =============================================================================
// CudaArena — O(1) bump-pointer GPU allocator with per-step reset
//
// Usage:
//   cuda::arena.init(capacity_bytes);   // once at startup
//   // ... training loop ...
//   cuda::arena.reset();                // top of each step
//   void* p = cuda::arena.alloc(n);     // O(1), no driver call
//   // no free needed — reset reclaims all
// =============================================================================
class CudaArena {
public:
    void* base_    = nullptr;
    size_t capacity_ = 0;
    size_t offset_  = 0;
    size_t peak_    = 0;
    bool   active_  = false;

    // Initialize pool with given capacity (bytes). Call once before training.
    bool init(size_t bytes) {
        if (base_) return true; // already initialized
        cudaError_t err = cudaMalloc(&base_, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[POOL] cudaMalloc failed for %.1f MB pool: %s\n",
                    bytes / 1e6, cudaGetErrorString(err));
            return false;
        }
        capacity_ = bytes;
        offset_ = 0;
        peak_ = 0;
        active_ = true;
        printf("[POOL] Arena initialized: %.1f MB\n", bytes / 1e6);
        return true;
    }

    // Auto-init: query free VRAM, allocate fraction of it
    bool init_auto(float fraction = 0.75f) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        // Reserve some for cuBLAS workspace and other system use
        size_t pool_bytes = (size_t)(free_bytes * fraction);
        // Align to 1MB
        pool_bytes = pool_bytes & ~((size_t)(1024*1024 - 1));
        printf("[POOL] VRAM free=%.1f MB, allocating pool=%.1f MB (%.0f%%)\n",
               free_bytes / 1e6, pool_bytes / 1e6, fraction * 100.0f);
        return init(pool_bytes);
    }

    // Allocate bytes from the pool. Returns nullptr if not enough space.
    // Alignment: 256 bytes (satisfies all CUDA alignment requirements).
    void* alloc(size_t bytes) {
        if (!active_ || bytes == 0) return nullptr;
        // Align offset to 256 bytes
        size_t aligned_offset = (offset_ + 255) & ~(size_t)255;
        if (aligned_offset + bytes > capacity_) {
            return nullptr; // caller should fallback to cudaMalloc
        }
        void* ptr = (char*)base_ + aligned_offset;
        offset_ = aligned_offset + bytes;
        if (offset_ > peak_) peak_ = offset_;
        return ptr;
    }

    // Reset the arena — reclaims all memory instantly (O(1)).
    // Call at the top of each training step.
    void reset() { offset_ = 0; }

    // Check if a pointer belongs to this arena
    bool owns(const void* ptr) const {
        if (!base_ || !ptr) return false;
        uintptr_t p = (uintptr_t)ptr;
        uintptr_t lo = (uintptr_t)base_;
        uintptr_t hi = lo + capacity_;
        return p >= lo && p < hi;
    }

    // Stats
    size_t used() const { return offset_; }
    size_t free() const { return capacity_ - offset_; }
    size_t peak_used() const { return peak_; }
    float utilization() const { return capacity_ > 0 ? (float)peak_ / capacity_ : 0.0f; }

    void print_stats() const {
        printf("[POOL] used=%.1f MB peak=%.1f MB / %.1f MB (%.1f%% util)\n",
               offset_ / 1e6, peak_ / 1e6, capacity_ / 1e6, utilization() * 100.0f);
    }

    void destroy() {
        if (base_) { cudaFree(base_); base_ = nullptr; }
        capacity_ = 0; offset_ = 0; peak_ = 0; active_ = false;
    }

    ~CudaArena() { destroy(); }
};

// Global arena instance
extern CudaArena arena;

// Legacy stubs (unused)
inline void set_memory_limit(size_t) {}
inline size_t get_active_bytes() { return arena.used(); }
inline void clear_cache() { arena.reset(); }

// =============================================================================
// Host-pinned buffer for async DMA
// =============================================================================
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
