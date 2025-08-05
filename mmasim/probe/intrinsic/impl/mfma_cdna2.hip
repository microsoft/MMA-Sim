#include <hip/hip_runtime.h>

using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x32 = __attribute__((__vector_size__(32 * sizeof(float)))) float;
using bfloat16x4 = __attribute__((__vector_size__(4 * sizeof(int16_t)))) int16_t;

extern "C" // bf16
{
    __global__ void mfma_f32_32x32x8bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 8;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x4 a_frag;
        bfloat16x4 b_frag;
        float32x16 c_frag;
        float32x16 d_frag;
        // load a
        for (i = 0; i < 4; i++)
        {
            row = tid % 32;
            col = tid / 32 * 4 + i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 4; i++)
        {
            row = tid / 32 * 4 + i;
            col = tid % 32;
            b_frag[i] = b[row * N + col];
        }
        // load c
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_32x32x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x4 a_frag;
        bfloat16x4 b_frag;
        float32x32 c_frag;
        float32x32 d_frag;
        // load a
        for (i = 0; i < 4; i++)
        {
            row = tid % 32;
            col = i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 32;
            b_frag[i] = b[row * N + col];
        }
        // load c
        for (i = 0; i < 32; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 32; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x16bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 16;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x4 a_frag;
        bfloat16x4 b_frag;
        float32x4 c_frag;
        float32x4 d_frag;
        // load a
        for (i = 0; i < 4; i++)
        {
            row = tid % 16;
            col = tid / 16 * 4 + i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            b_frag[i] = b[row * N + col];
        }
        // load c
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x4 a_frag;
        bfloat16x4 b_frag;
        float32x16 c_frag;
        float32x16 d_frag;
        // load a
        for (i = 0; i < 4; i++)
        {
            row = tid % 16;
            col = i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 16;
            b_frag[i] = b[row * N + col];
        }
        // load c
        for (i = 0; i < 16; i++)
        {
            row = tid / 16 * 4 + i % 4;
            col = tid % 16;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 16 * 4 + i % 4;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_4x4x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 4, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x4 a_frag;
        bfloat16x4 b_frag;
        float32x4 c_frag;
        float32x4 d_frag;
        // load a
        for (i = 0; i < 4; i++)
        {
            row = tid % 4;
            col = i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 4;
            b_frag[i] = b[row * N + col];
        }
        // load c
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 4;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 4;
            d[row * N + col] = d_frag[i];
        }
    }

    void mfma_f32_32x32x8bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_32x32x8bf16_1k_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x4bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_32x32x4bf16_1k_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x16bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_16x16x16bf16_1k_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x4bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_16x16x4bf16_1k_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x4bf16_1k(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_4x4x4bf16_1k_kernel<<<1, 64>>>(d, a, b, c);
    }
}
