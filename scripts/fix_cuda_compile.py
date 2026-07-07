#!/usr/bin/env python3
"""Fix CUDA compilation errors: lambda kernels + duplicate reshape"""
import os
os.chdir('/kaggle/working/transformer_100m_cuda')

# ============================================================
# Fix 1: cuda_ops.cu - replace lambda kernels
# ============================================================
with open('cuda/cuda_ops.cu', 'r') as f:
    lines = f.readlines()

out = []
skip_until_brace = 0
i = 0

# First pass: add helper kernels before "namespace cuda {"
insert_kernels = """// Helper kernels (must be free functions, not lambdas)
__global__ void fill_fp16_kernel(__half* d, __half v, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = v;
}

__global__ void arange_i32_kernel(int* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = i;
}

"""

already_has_kernels = any('fill_fp16_kernel' in l for l in lines)

while i < len(lines):
    line = lines[i]

    # Insert helper kernels before namespace
    if not already_has_kernels and line.strip() == 'namespace cuda {':
        out.append(insert_kernels)
        out.append(line)
        i += 1
        continue

    # Skip lambda kernel definitions (4 lines: auto kernel = ..., body, body, };)
    if 'auto kernel = []' in line and '__global__' in line:
        # Skip until we find "};"
        while i < len(lines) and '};' not in lines[i]:
            i += 1
        i += 1  # skip the "};" line too
        continue

    # Replace "kernel<<<" calls with proper kernel names
    if 'kernel<<<' in line and 'auto' not in line:
        # Determine which kernel based on context
        # Look back to find what we're in
        context = ''.join(out[-20:])
        if 'ones_like' in context or ('ones(' in context and 'out.data' in line):
            out.append('        fill_fp16_kernel<<<grid, block, 0, ctx.compute_stream>>>(\n')
            out.append('            (__half*)out.data, __float2half(1.0f), out.numel());\n')
        elif 'ones(' in context or ('t.data' in line and '__half' in line):
            out.append('        fill_fp16_kernel<<<grid, block, 0, ctx.compute_stream>>>(\n')
            out.append('            (__half*)t.data, __float2half(1.0f), t.numel());\n')
        elif 'arange' in context:
            out.append('        arange_i32_kernel<<<grid, block, 0, ctx.compute_stream>>>((int*)t.data, n);\n')
        else:
            out.append(line)  # keep as-is if unknown
        i += 1
        continue

    # Remove duplicate reshape definition
    if line.strip().startswith('CudaTensor reshape(CudaTensor& src, const std::vector<int64_t>& new_shape)'):
        # Skip until closing brace
        out.append('// reshape() is already defined inline in cuda_tensor.h\n')
        while i < len(lines) and not (lines[i].strip() == '}' and i > 0):
            i += 1
        i += 1  # skip the "}"
        out.append('\n')
        continue

    out.append(line)
    i += 1

with open('cuda/cuda_ops.cu', 'w') as f:
    f.writelines(out)
print('[OK] cuda_ops.cu fixed')

# ============================================================
# Fix 2: cuda_kernels.cu - replace lambda kernel
# ============================================================
with open('cuda/cuda_kernels.cu', 'r') as f:
    lines = f.readlines()

out = []
i = 0
added_scale_kernel = any('scale_f32_to_fp16_kernel' in l for l in lines)

while i < len(lines):
    line = lines[i]

    # Add scale kernel before cross_entropy_fused_backward
    if not added_scale_kernel and 'void cross_entropy_fused_backward(' in line:
        out.append('// Helper kernel: scale float buffer to fp16\n')
        out.append('__global__ void scale_f32_to_fp16_kernel(const float* src, __half* dst, int n, float s) {\n')
        out.append('    int i = blockIdx.x * blockDim.x + threadIdx.x;\n')
        out.append('    if (i < n) dst[i] = __float2half(src[i] * s);\n')
        out.append('}\n\n')
        added_scale_kernel = True

    # Replace dim3 g(grid) + lambda pattern
    if 'dim3 g(grid)' in line:
        # Skip: dim3 g(grid);, comment, auto kernel, 3 body lines, };, kernel<<<g,...
        j = i
        while j < len(lines) and 'kernel<<<g' not in lines[j]:
            j += 1
        if j < len(lines):
            # Replace entire block with direct call
            out.append('    scale_f32_to_fp16_kernel<<<grid, block, 0, stream>>>(\n')
            out.append('        (const float*)logits, d_logits, total, scale);\n')
            i = j + 1
            continue

    out.append(line)
    i += 1

with open('cuda/cuda_kernels.cu', 'w') as f:
    f.writelines(out)
print('[OK] cuda_kernels.cu fixed')

# ============================================================
# Fix 3: cuda_ops.h - remove reshape declaration
# ============================================================
with open('cuda/cuda_ops.h', 'r') as f:
    src = f.read()

src = src.replace(
    'CudaTensor reshape(CudaTensor& src, const std::vector<int64_t>& new_shape);',
    '// reshape() defined inline in cuda_tensor.h')

with open('cuda/cuda_ops.h', 'w') as f:
    f.write(src)
print('[OK] cuda_ops.h fixed')
print('\n[DONE] All compilation fixes applied!')
