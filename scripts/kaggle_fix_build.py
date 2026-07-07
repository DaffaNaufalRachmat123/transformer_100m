#!/usr/bin/env python3
"""
Kaggle in-place build fix — paste this cell into your Kaggle notebook.
Fixes 3 compile errors:
  1. cuda_kernels.cu: atomicAdd(__half*, float) type mismatch
  2. cuda_ops.cu: duplicate reshape() definition
  3. lora.h / lora.cpp: missing transformer.h / adamw.h includes
"""
import re, os

BASE = "/kaggle/working/transformer_100m_cuda"

# ==========================================================================
# FIX 1: cuda/cuda_kernels.cu — atomicAdd type mismatch
# The old kernel uses __half* dgamma; needs float* dgamma_f32 + converter.
# ==========================================================================
path = os.path.join(BASE, "cuda/cuda_kernels.cu")
with open(path, "r") as f:
    src = f.read()

# 1a: Change kernel parameter from __half* dgamma to float* dgamma_f32
src = re.sub(
    r'(__half\*\s*(?:__restrict__\s*)?)dgamma\b(?!\w)',
    r'float* __restrict__ dgamma_f32',
    src,
    count=1  # only the backward kernel signature
)

# 1b: Fix the atomicAdd call — old version: atomicAdd(dgamma + tid, d)
#     New: atomicAdd(dgamma_f32 + i, d)  (loop var might be tid or i)
src = re.sub(
    r'atomicAdd\(dgamma\s*\+\s*(\w+),\s*d\)',
    r'atomicAdd(dgamma_f32 + \1, d)',
    src
)

# 1c: Also fix variant where it's dgamma + i with variable name 'i'
# (already handled by 1b regex, but ensure the comment is updated)
src = src.replace(
    "// fp16 atomicAdd",
    "// FP32 accumulation (no fp16 atomicAdd on sm_75)"
)

# 1d: Add the f32→f16 converter kernel if not present
if "rmsnorm_dgamma_f32_to_f16" not in src:
    # Insert after the backward kernel closing brace
    converter = '''
// tiny converter: fp32 dgamma scratch -> fp16 output
__global__ void rmsnorm_dgamma_f32_to_f16(const float* __restrict__ src,
                                          __half* __restrict__ dst, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) dst[i] = __float2half(src[i]);
}
'''
    # Find the end of rms_norm_backward_kernel (closing brace after atomicAdd section)
    # Insert converter right before the launcher function
    launch_pat = r'(void\s+rms_norm_fused_kernel_backward\s*\()'
    if re.search(launch_pat, src):
        src = re.sub(launch_pat, converter + r'\n\1', src)

# 1e: Update the launcher to allocate float buffer + call converter
# Old launcher passes dgamma directly to kernel; new version uses float scratch.
old_launch_body = re.search(
    r'(void\s+rms_norm_fused_kernel_backward\s*\([^)]*\)\s*\{)(.*?)(^\})',
    src, re.DOTALL | re.MULTILINE
)
if old_launch_body and "dgamma_f32" not in old_launch_body.group(2):
    new_launcher_body = '''
    size_t shared_bytes = 256 * sizeof(float);
    // FP32 scratch for dgamma accumulation
    float* dgamma_f32 = nullptr;
    cudaMallocAsync((void**)&dgamma_f32, (size_t)D * sizeof(float), stream);
    cudaMemsetAsync(dgamma_f32, 0, (size_t)D * sizeof(float), stream);

    dim3 grid(B * S, 2);
    rms_norm_backward_kernel<<<grid, 256, shared_bytes, stream>>>(
        dy, gamma, saved_rsqrt, saved_normed, dx, dgamma_f32, B, S, D, eps);

    int blk = 256, g = (D + blk - 1) / blk;
    rmsnorm_dgamma_f32_to_f16<<<g, blk, 0, stream>>>(dgamma_f32, dgamma, D);
    cudaFreeAsync(dgamma_f32, stream);
'''
    src = src[:old_launch_body.start(2)] + new_launcher_body + "\n" + src[old_launch_body.start(3):]

with open(path, "w") as f:
    f.write(src)
print(f"[FIX 1] Patched: {path}")

# ==========================================================================
# FIX 2: cuda/cuda_ops.cu — remove duplicate reshape() definition
# ==========================================================================
path = os.path.join(BASE, "cuda/cuda_ops.cu")
with open(path, "r") as f:
    src = f.read()

# Remove the reshape function that conflicts with cuda_tensor.h's inline version
# Pattern: CudaTensor reshape(CudaTensor& src, const std::vector<int64_t>& new_shape) { ... }
src = re.sub(
    r'// *(?:─+)? *Reshape.*?\n'  # comment line before reshape
    r'(?:CudaTensor\s+reshape\s*\([^)]*\)\s*\{[^}]*\}\s*)',
    '// reshape() is already defined inline in cuda_tensor.h — no redefinition here\n\n',
    src,
    flags=re.DOTALL
)

# Also handle if reshape is standalone without the comment header
src = re.sub(
    r'\nCudaTensor\s+reshape\s*\(\s*CudaTensor\s*&\s*\w+\s*,\s*const\s+std::vector<int64_t>\s*&\s*\w+\s*\)\s*\{[^}]*\}',
    '\n// reshape() is already defined inline in cuda_tensor.h — no redefinition here',
    src,
    flags=re.DOTALL
)

with open(path, "w") as f:
    f.write(src)
print(f"[FIX 2] Patched: {path}")

# ==========================================================================
# FIX 3: lora.h — replace missing #include "../transformer/transformer.h"
# ==========================================================================
path = os.path.join(BASE, "transformer/lora.h")
if os.path.exists(path):
    with open(path, "r") as f:
        src = f.read()

    # Replace broken include with model_config.h
    src = src.replace(
        '#include "../transformer/transformer.h"',
        '#include "model_config.h"'
    )

    # Add forward declaration for Transformer class after the includes block
    if "class Transformer" not in src:
        # Insert after the last #include and before struct/class definitions
        src = re.sub(
            r'(#include\s+<unordered_map>)',
            r'''\1

// Forward declaration — full Transformer class lives in train_cuda.cu
class Transformer {
public:
    void reset_kv_cache();
    void forward_one_token(std::vector<float>& logits, int token_id, int pos);
};''',
            src
        )

    with open(path, "w") as f:
        f.write(src)
    print(f"[FIX 3a] Patched: {path}")

# ==========================================================================
# FIX 3b: lora.cpp — replace missing #include "../transformer/adamw.h"
# ==========================================================================
path = os.path.join(BASE, "transformer/lora.cpp")
if os.path.exists(path):
    with open(path, "r") as f:
        src = f.read()

    src = src.replace(
        '#include "../transformer/adamw.h"',
        '''
// Minimal CPU AdamW state for LoRA training (not the CUDA version)
struct AdamWState {
    std::vector<float> m, v;
    int t = 0;
    void update_flat(float* param, const float* grad, int n, float lr,
                     float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        if ((int)m.size() != n) { m.resize(n, 0.f); v.resize(n, 0.f); }
        t++;
        float bc1 = 1.f - powf(beta1, (float)t);
        float bc2 = 1.f - powf(beta2, (float)t);
        for (int i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1.f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.f - beta2) * grad[i] * grad[i];
            param[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
        }
    }
};'''
    )

    with open(path, "w") as f:
        f.write(src)
    print(f"[FIX 3b] Patched: {path}")

# ==========================================================================
# FIX 4: CMakeLists.txt — remove lora.cpp if accidentally included
# ==========================================================================
path = os.path.join(BASE, "CMakeLists.txt")
with open(path, "r") as f:
    src = f.read()

if "lora.cpp" in src:
    src = re.sub(r'\s*transformer/lora\.cpp', '', src)
    with open(path, "w") as f:
        f.write(src)
    print(f"[FIX 4] Removed lora.cpp from CMakeLists.txt")
else:
    print(f"[FIX 4] CMakeLists.txt OK (no lora.cpp)")

print("\n✓ All fixes applied. Rebuild with:")
print("  cd build && cmake --build . -j$(nproc)")
