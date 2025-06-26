#include <hip/hip_runtime.h>

using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x32 = __attribute__((__vector_size__(32 * sizeof(float)))) float;
using bfloat16x4 = __attribute__((__vector_size__(4 * sizeof(int16_t)))) int16_t;


__global__ void mfma_f32_32x32x8bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
{
    uint32_t tid = threadIdx.x;
    bfloat16x4 a_frag;
    bfloat16x4 b_frag;
    float32x16 c_frag;
    float32x16 d_frag;
    // load a
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid % 32;
        uint32_t col = tid / 32 * 4 + i;
        a_frag[i] = a[row * 8 + col];
    }
    // load b
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid / 32 * 4 + i;
        uint32_t col = tid % 32;
        b_frag[i] = b[row * 32 + col];
    }
    // load c
    for (uint32_t i = 0; i < 16; i++)
    {
        uint32_t row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
        uint32_t col = tid % 32;
        c_frag[i] = c[row * 32 + col];
    }
    // mfma
    d_frag = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
    // store d
    for (uint32_t i = 0; i < 16; i++)
    {
        uint32_t row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
        uint32_t col = tid % 32;
        d[row * 32 + col] = d_frag[i];
    }
}

__global__ void mfma_f32_32x32x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
{
    uint32_t tid = threadIdx.x;
    bfloat16x4 a_frag;
    bfloat16x4 b_frag;
    float32x32 c_frag;
    float32x32 d_frag;
    // load a
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid % 32;
        uint32_t col = i;
        a_frag[i] = a[row * 4 + col];
    }
    // load b
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = i;
        uint32_t col = tid % 32;
        b_frag[i] = b[row * 32 + col];
    }
    // load c
    for (uint32_t i = 0; i < 32; i++)
    {
        uint32_t row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
        uint32_t col = tid % 32;
        c_frag[i] = c[row * 32 + col];
    }
    // mfma
    d_frag = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
    // store d
    for (uint32_t i = 0; i < 32; i++)
    {
        uint32_t row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
        uint32_t col = tid % 32;
        d[row * 32 + col] = d_frag[i];
    }
}

__global__ void mfma_f32_16x16x16bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
{
    uint32_t tid = threadIdx.x;
    bfloat16x4 a_frag;
    bfloat16x4 b_frag;
    float32x4 c_frag;
    float32x4 d_frag;
    // load a
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid % 16;
        uint32_t col = tid / 16 * 4 + i;
        a_frag[i] = a[row * 16 + col];
    }
    // load b
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid / 16 * 4 + i;
        uint32_t col = tid % 16;
        b_frag[i] = b[row * 16 + col];
    }
    // load c
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid / 16 * 4 + i;
        uint32_t col = tid % 16;
        c_frag[i] = c[row * 16 + col];
    }
    // mfma
    d_frag = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid / 16 * 4 + i;
        uint32_t col = tid % 16;
        d[row * 16 + col] = d_frag[i];
    }
}

__global__ void mfma_f32_16x16x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
{
    uint32_t tid = threadIdx.x;
    bfloat16x4 a_frag;
    bfloat16x4 b_frag;
    float32x16 c_frag;
    float32x16 d_frag;
    // load a
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid % 16;
        uint32_t col = i;
        a_frag[i] = a[row * 4 + col];
    }
    // load b
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = i;
        uint32_t col = tid % 16;
        b_frag[i] = b[row * 16 + col];
    }
    // load c
    for (uint32_t i = 0; i < 16; i++)
    {
        uint32_t row = tid / 16 * 4 + i % 4;
        uint32_t col = tid % 16;
        c_frag[i] = c[row * 16 + col];
    }
    // mfma
    d_frag = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
    // store d
    for (uint32_t i = 0; i < 16; i++)
    {
        uint32_t row = tid / 16 * 4 + i % 4;
        uint32_t col = tid % 16;
        d[row * 16 + col] = d_frag[i];
    }
}

__global__ void mfma_f32_4x4x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
{
    uint32_t tid = threadIdx.x;
    bfloat16x4 a_frag;
    bfloat16x4 b_frag;
    float32x4 c_frag;
    float32x4 d_frag;
    // load a
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = tid % 4;
        uint32_t col = i;
        a_frag[i] = a[row * 4 + col];
    }
    // load b
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = i;
        uint32_t col = tid % 4;
        b_frag[i] = b[row * 4 + col];
    }
    // load c
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = i;
        uint32_t col = tid % 4;
        c_frag[i] = c[row * 4 + col];
    }
    // mfma
    d_frag = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = i;
        uint32_t col = tid % 4;
        d[row * 4 + col] = d_frag[i];
    }
}

extern "C" void mfma_f32_32x32x8bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
{
    dim3 grid(1);
    dim3 block(64);
    mfma_f32_32x32x8bf16_1k_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mfma_f32_32x32x4bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
{
    dim3 grid(1);
    dim3 block(64);
    mfma_f32_32x32x4bf16_1k_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mfma_f32_16x16x16bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
{
    dim3 grid(1);
    dim3 block(64);
    mfma_f32_16x16x16bf16_1k_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mfma_f32_16x16x4bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
{
    dim3 grid(1);
    dim3 block(64);
    mfma_f32_16x16x4bf16_1k_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mfma_f32_4x4x4bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
{
    dim3 grid(1);
    dim3 block(64);
    mfma_f32_4x4x4bf16_1k_kernel<<<grid, block>>>(d, a, b, c);
}
