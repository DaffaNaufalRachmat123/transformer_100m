cmake_minimum_required(VERSION 3.20)
project(transformer_100m_cuda CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════════════════════════════════════════
# GPU TARGET SELECTION
# Uncomment ONE line below to target your GPU, or pass -DGPU_TARGET=xxx
# ═══════════════════════════════════════════════════════════════════════════════
# set(GPU_TARGET "T4")         # Kaggle/Colab — NVIDIA T4 16GB (sm_75, Turing)
# set(GPU_TARGET "V100")       # Cloud — NVIDIA V100 16/32GB (sm_70, Volta)
# set(GPU_TARGET "A100")       # Cloud — NVIDIA A100 40/80GB (sm_80, Ampere)
# set(GPU_TARGET "RTX3090")    # Desktop — RTX 3090 24GB (sm_86, Ampere)
# set(GPU_TARGET "RTX4090")    # Desktop — RTX 4090 24GB (sm_89, Ada Lovelace)
# set(GPU_TARGET "H100")       # HPC — NVIDIA H100 80GB (sm_90, Hopper)
# set(GPU_TARGET "ALL")        # Fat binary (larger, runs on all above)
# ═══════════════════════════════════════════════════════════════════════════════

if(NOT GPU_TARGET)
    set(GPU_TARGET "T4")  # Default: Kaggle T4
endif()

# Map GPU_TARGET to CUDA architectures
if(GPU_TARGET STREQUAL "T4")
    set(CMAKE_CUDA_ARCHITECTURES 75)
elseif(GPU_TARGET STREQUAL "V100")
    set(CMAKE_CUDA_ARCHITECTURES 70)
elseif(GPU_TARGET STREQUAL "A100")
    set(CMAKE_CUDA_ARCHITECTURES 80)
    add_compile_definitions(GPU_HAS_BF16=1)
elseif(GPU_TARGET STREQUAL "RTX3090")
    set(CMAKE_CUDA_ARCHITECTURES 86)
    add_compile_definitions(GPU_HAS_BF16=1)
elseif(GPU_TARGET STREQUAL "RTX4090")
    set(CMAKE_CUDA_ARCHITECTURES 89)
    add_compile_definitions(GPU_HAS_BF16=1)
elseif(GPU_TARGET STREQUAL "H100")
    set(CMAKE_CUDA_ARCHITECTURES 90)
    add_compile_definitions(GPU_HAS_BF16=1)
elseif(GPU_TARGET STREQUAL "ALL")
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    add_compile_definitions(GPU_HAS_BF16=1)
else()
    message(FATAL_ERROR "Unknown GPU_TARGET: ${GPU_TARGET}. Use T4/V100/A100/RTX3090/RTX4090/H100/ALL")
endif()

message(STATUS "[GPU] Target: ${GPU_TARGET} → sm_${CMAKE_CUDA_ARCHITECTURES}")

# Override if user passes -DCMAKE_CUDA_ARCHITECTURES directly
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    message(STATUS "[GPU] User override: sm_${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ─── Build type ─────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr")

# ─── Options ────────────────────────────────────────────────────────────────
option(USE_DDP "Enable NCCL multi-GPU DDP" OFF)

# ─── Find packages ─────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ─── CUDA backend library ──────────────────────────────────────────────────
add_subdirectory(cuda)

# ─── BPE tokenizer library ─────────────────────────────────────────────────
add_library(bpe_tokenizer STATIC
    bpe/bpe_tokenizer.cpp
)
target_include_directories(bpe_tokenizer PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/bpe
)
target_link_libraries(bpe_tokenizer PUBLIC Threads::Threads)

# ─── Transformer core library (portable C++) ───────────────────────────────
add_library(transformer_core STATIC
    transformer/model_config.cpp
    transformer/pretrain_dataset.cpp
)
target_include_directories(transformer_core PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/transformer
)
target_link_libraries(transformer_core PUBLIC
    Threads::Threads
    bpe_tokenizer
)

# ─── Pretrain binary (CUDA) ────────────────────────────────────────────────
add_executable(train_pretrain_cuda
    transformer/train_cuda.cu
)
target_include_directories(train_pretrain_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(train_pretrain_cuda PRIVATE
    cuda_backend
    transformer_core
    bpe_tokenizer
    CUDA::cudart
    stdc++fs
)

# ─── SFT/DPO/Phone binary (CUDA) ──────────────────────────────────────────
add_executable(train_sft_cuda
    transformer/train_cuda_unified.cu
)
target_include_directories(train_sft_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(train_sft_cuda PRIVATE
    cuda_backend
    transformer_core
    bpe_tokenizer
    CUDA::cudart
    stdc++fs
)

# ─── NCCL for DDP ──────────────────────────────────────────────────────────
if(USE_DDP)
    find_library(NCCL_LIB nccl
        HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu)
    if(NCCL_LIB)
        target_compile_definitions(train_pretrain_cuda PRIVATE USE_DDP)
        target_link_libraries(train_pretrain_cuda PRIVATE ${NCCL_LIB})
        target_compile_definitions(train_sft_cuda PRIVATE USE_DDP)
        target_link_libraries(train_sft_cuda PRIVATE ${NCCL_LIB})
        message(STATUS "[DDP] NCCL found: ${NCCL_LIB}")
    else()
        message(WARNING "[DDP] NCCL not found — DDP disabled")
    endif()
endif()

# ─── NVML explicit link (some distros need it on the exe, not just the lib) ──
find_library(NVML_LIB_EXE nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR}
          /usr/lib/x86_64-linux-gnu
          /usr/lib64
          /usr/local/cuda/lib64
          /usr/local/cuda/targets/x86_64-linux/lib/stubs)
if(NVML_LIB_EXE)
    target_link_libraries(train_pretrain_cuda PRIVATE ${NVML_LIB_EXE})
    target_link_libraries(train_sft_cuda PRIVATE ${NVML_LIB_EXE})
    message(STATUS "[NVML] linked: ${NVML_LIB_EXE}")
else()
    # NVML not available: cuda_monitor.cu compiles stubs (see #ifdef USE_NVML).
    # Nothing to link. GPU utilization/temp reporting will show zeros.
    message(STATUS "[NVML] libnvidia-ml not found — building with monitor stubs")
endif()

# ─── Install ───────────────────────────────────────────────────────────────
install(TARGETS train_pretrain_cuda train_sft_cuda DESTINATION bin)
install(TARGETS cuda_backend transformer_core bpe_tokenizer DESTINATION lib)
