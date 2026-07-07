cmake_minimum_required(VERSION 3.20)
project(cuda_backend CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Turing (sm_75) — override for Blackwell
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr")

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# cuBLAS
find_library(CUBLAS_LIB cublas
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED)

# cuDNN
find_library(CUDNN_LIB cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu
    REQUIRED)

# NVML
find_library(NVML_LIB nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu)

# ─── Library ────────────────────────────────────────────────────────────────
add_library(cuda_backend STATIC
    cuda_context.cu
    cuda_kernels.cu
    cuda_ops.cu
    cuda_monitor.cu
    cuda_autograd.cu
)

target_include_directories(cuda_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# NVML library (for GPU monitoring)
find_library(NVML_LIB nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu
    PATHS /usr/lib/x86_64-linux-gnu)

target_link_libraries(cuda_backend PUBLIC
    CUDA::cudart
    CUDA::cublas
    ${CUDNN_LIB}
)

if(NVML_LIB)
    target_link_libraries(cuda_backend PUBLIC ${NVML_LIB})
    target_compile_definitions(cuda_backend PUBLIC USE_NVML)
endif()

install(TARGETS cuda_backend DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include/cuda
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
