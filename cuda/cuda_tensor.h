#pragma once
// =============================================================================
// cuda_tensor.h — Minimal GPU tensor replacing mx::array
//
// Design: thin wrapper around device memory pointer + metadata.
// No expression templates, no lazy evaluation. All ops are explicit
// free functions in cuda_ops.h, mirroring MLX's functional API style.
//
// Memory model: RAII — owns its device allocation unless moved.
// Copy is disabled (use cuda::copy() explicitly).
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <initializer_list>
#include <utility>

namespace cuda {

enum class DType : uint8_t {
    F32  = 0,
    FP16 = 1,
    BF16 = 2,
    I32  = 3,
    U32  = 4,
};

inline constexpr size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::I32:  return 4;
        case DType::U32:  return 4;
    }
    return 4;
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "F32";
        case DType::FP16: return "FP16";
        case DType::BF16: return "BF16";
        case DType::I32:  return "I32";
        case DType::U32:  return "U32";
    }
    return "?";
}

// =============================================================================
// CudaTensor — lightweight GPU array
// =============================================================================
class CudaTensor {
public:
    void*   data  = nullptr;
    int     ndim  = 0;
    int64_t shape[4] = {0, 0, 0, 0};
    int64_t stride[4] = {0, 0, 0, 0};
    size_t  nbytes = 0;
    DType   dtype = DType::F32;
    bool    owns  = true;

    // ─── Constructors ──────────────────────────────────────────────────────
    CudaTensor() = default;

    explicit CudaTensor(float scalar) {
        cudaMalloc(&data, sizeof(float));
        float h = scalar;
        cudaMemcpy(data, &h, sizeof(float), cudaMemcpyHostToDevice);
        ndim = 0; shape[0] = 1; nbytes = sizeof(float); dtype = DType::F32; owns = true;
    }

    explicit CudaTensor(int scalar) {
        cudaMalloc(&data, sizeof(int));
        cudaMemcpy(data, &scalar, sizeof(int), cudaMemcpyHostToDevice);
        ndim = 0; shape[0] = 1; nbytes = sizeof(int); dtype = DType::I32; owns = true;
    }

    CudaTensor(std::initializer_list<int64_t> dims, DType dt)
        : ndim((int)dims.size()), dtype(dt), owns(true)
    {
        size_t total = 1;
        int i = 0;
        for (auto d : dims) { shape[i++] = d; total *= d; }
        for (; i < 4; i++) shape[i] = 1;
        nbytes = total * dtype_size(dt);
        cudaMalloc(&data, nbytes);
    }

    CudaTensor(const std::vector<int64_t>& dims, DType dt)
        : ndim((int)dims.size()), dtype(dt), owns(true)
    {
        size_t total = 1;
        for (int i = 0; i < ndim && i < 4; i++) {
            shape[i] = dims[i]; total *= dims[i];
        }
        for (int i = ndim; i < 4; i++) shape[i] = 1;
        nbytes = total * dtype_size(dt);
        cudaMalloc(&data, nbytes);
    }

    template<typename T>
    static CudaTensor from_host(const T* host_ptr, const std::vector<int64_t>& dims) {
        CudaTensor t(dims, dtype_of<T>());
        cudaMemcpy(t.data, host_ptr, t.nbytes, cudaMemcpyHostToDevice);
        return t;
    }

    // ─── Move semantics ────────────────────────────────────────────────────
    CudaTensor(CudaTensor&& other) noexcept
        : data(other.data), ndim(other.ndim), nbytes(other.nbytes),
          dtype(other.dtype), owns(other.owns)
    {
        for (int i = 0; i < 4; i++) { shape[i] = other.shape[i]; stride[i] = other.stride[i]; }
        other.data = nullptr; other.owns = false; other.nbytes = 0;
    }

    CudaTensor& operator=(CudaTensor&& other) noexcept {
        if (this != &other) {
            if (owns && data) cudaFree(data);
            data = other.data; ndim = other.ndim; nbytes = other.nbytes;
            dtype = other.dtype; owns = other.owns;
            for (int i = 0; i < 4; i++) { shape[i] = other.shape[i]; stride[i] = other.stride[i]; }
            other.data = nullptr; other.owns = false; other.nbytes = 0;
        }
        return *this;
    }

    CudaTensor(const CudaTensor&) = delete;
    CudaTensor& operator=(const CudaTensor&) = delete;

    ~CudaTensor() {
        if (owns && data) cudaFree(data);
    }

    // ─── Metadata helpers ─────────────────────────────────────────────────
    size_t numel() const {
        size_t n = 1;
        for (int i = 0; i < ndim; i++) n *= shape[i];
        return n;
    }

    std::vector<int64_t> shape_vec() const {
        std::vector<int64_t> v;
        for (int i = 0; i < ndim; i++) v.push_back(shape[i]);
        return v;
    }

    // ─── Host ↔ Device ────────────────────────────────────────────────────
    float item_f32() const {
        float h = 0;
        cudaMemcpy(&h, data, sizeof(float), cudaMemcpyDeviceToHost);
        return h;
    }

    int item_i32() const {
        int h = 0;
        cudaMemcpy(&h, data, sizeof(int), cudaMemcpyDeviceToHost);
        return h;
    }

    template<typename T>
    void copy_to_host(std::vector<T>& out) const {
        out.resize(numel());
        cudaMemcpy(out.data(), data, numel() * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template<typename T>
    void copy_from_host(const T* host, cudaStream_t stream = 0) {
        cudaMemcpyAsync(data, host, nbytes, cudaMemcpyHostToDevice, stream);
    }

    // ─── Fill helpers ─────────────────────────────────────────────────────
    void fill_f32(float value, cudaStream_t stream = 0);
    void fill_zero(cudaStream_t stream = 0);

    void release() {
        if (owns && data) { cudaFree(data); data = nullptr; }
        owns = false; nbytes = 0;
    }

    // ─── Type deduction ────────────────────────────────────────────────────
    template<typename T> static DType dtype_of() {
        if (std::is_same<T, float>::value)  return DType::F32;
        if (std::is_same<T, int>::value)    return DType::I32;
        if (std::is_same<T, uint32_t>::value) return DType::U32;
        return DType::F32;
    }
};

// =============================================================================
// View / reshape helpers (zero-copy)
// =============================================================================
inline CudaTensor view(CudaTensor& src, const std::vector<int64_t>& new_shape) {
    CudaTensor v;
    v.data   = src.data;
    v.ndim   = (int)new_shape.size();
    v.nbytes = src.nbytes;
    v.dtype  = src.dtype;
    v.owns   = false;
    for (int i = 0; i < v.ndim && i < 4; i++) v.shape[i] = new_shape[i];
    for (int i = v.ndim; i < 4; i++) v.shape[i] = 1;
    return v;
}

inline CudaTensor reshape(CudaTensor& src, const std::vector<int64_t>& new_shape) {
    return view(src, new_shape);
}

// =============================================================================
// Allocation helpers
// =============================================================================
CudaTensor zeros(const std::vector<int64_t>& shape, DType dtype);
CudaTensor ones(const std::vector<int64_t>& shape, DType dtype);
CudaTensor randn(const std::vector<int64_t>& shape, float stddev, DType dtype, uint64_t seed = 42);
CudaTensor arange(int n, DType dtype);

// =============================================================================
// Inline fill implementations
// =============================================================================
inline void CudaTensor::fill_zero(cudaStream_t stream) {
    cudaMemsetAsync(data, 0, nbytes, stream);
}

} // namespace cuda
