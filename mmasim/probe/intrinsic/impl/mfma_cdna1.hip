#include <hip/hip_runtime.h>

using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x32 = __attribute__((__vector_size__(32 * sizeof(float)))) float;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using bfloat16x2 = __attribute__((__vector_size__(2 * sizeof(int16_t)))) int16_t;

extern "C" // fp32
{
    __global__ void mfma_f32_32x32x2f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 2;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float a_frag;
        float b_frag;
        float32x16 c_frag;
        float32x16 d_frag;
        // load a
        row = tid % 32;
        col = tid / 32;
        a_frag = a[row * K + col];
        // load b
        row = tid / 32;
        col = tid % 32;
        b_frag = b[row * N + col];
        // load c
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 + i % 4;
            col = tid % 32;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_32x32x2f32(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_32x32x1f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 1;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float a_frag;
        float b_frag;
        float32x32 c_frag;
        float32x32 d_frag;
        // load a
        row = tid % 32;
        col = 0;
        a_frag = a[row * K + col];
        // load b
        row = 0;
        col = tid % 32;
        b_frag = b[row * N + col];
        // load c
        for (i = 0; i < 32; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_32x32x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 32; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x4f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float a_frag;
        float b_frag;
        float32x4 c_frag;
        float32x4 d_frag;
        // load a
        row = tid % 16;
        col = tid / 16;
        a_frag = a[row * K + col];
        // load b
        row = tid / 16;
        col = tid % 16;
        b_frag = b[row * N + col];
        // load c
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4f32(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x1f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 1;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float a_frag;
        float b_frag;
        float32x16 c_frag;
        float32x16 d_frag;
        // load a
        row = tid % 16;
        col = 0;
        a_frag = a[row * K + col];
        // load b
        row = 0;
        col = tid % 16;
        b_frag = b[row * N + col];
        // load c
        for (i = 0; i < 16; i++)
        {
            row = tid / 16 * 4 + i % 4;
            col = tid % 16;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_16x16x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 16 * 4 + i % 4;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_4x4x1f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 4, K = 1;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float a_frag;
        float b_frag;
        float32x4 c_frag;
        float32x4 d_frag;
        // load a
        row = tid % 4;
        col = 0;
        a_frag = a[row * K + col];
        // load b
        row = 0;
        col = tid % 4;
        b_frag = b[row * N + col];
        // load c
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 4;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_4x4x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    void mfma_f32_32x32x2f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_32x32x2f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x1f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_32x32x1f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x4f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_16x16x4f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x1f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_16x16x1f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x1f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_4x4x1f32_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // fp16
{
    __global__ void mfma_f32_32x32x8f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 32, K = 8;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float16x4 a_frag;
        float16x4 b_frag;
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
        d_frag = __builtin_amdgcn_mfma_f32_32x32x8f16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_32x32x4f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float16x4 a_frag;
        float16x4 b_frag;
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
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 32; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x16f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 16, K = 16;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float16x4 a_frag;
        float16x4 b_frag;
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
        d_frag = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x4f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float16x4 a_frag;
        float16x4 b_frag;
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
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 16 * 4 + i % 4;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_4x4x4f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 4, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        float16x4 a_frag;
        float16x4 b_frag;
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
        d_frag = __builtin_amdgcn_mfma_f32_4x4x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 4;
            d[row * N + col] = d_frag[i];
        }
    }

    void mfma_f32_32x32x8f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_32x32x8f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x4f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_32x32x4f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x16f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_16x16x16f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x4f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_16x16x4f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x4f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_4x4x4f16_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // bf16
{
    __global__ void mfma_f32_32x32x4bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x2 a_frag;
        bfloat16x2 b_frag;
        float32x16 c_frag;
        float32x16 d_frag;
        // load a
        for (i = 0; i < 2; i++)
        {
            row = tid % 32;
            col = tid / 32 * 2 + i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 2; i++)
        {
            row = tid / 32 * 2 + i;
            col = tid % 32;
            b_frag[i] = b[row * N + col];
        }
        // load c
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 + i % 4;
            col = tid % 32;
            c_frag[i] = c[row * N + col];
        }
        // mfma
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_32x32x2bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 2;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x2 a_frag;
        bfloat16x2 b_frag;
        float32x32 c_frag;
        float32x32 d_frag;
        // load a
        for (i = 0; i < 2; i++)
        {
            row = tid % 32;
            col = i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 2; i++)
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
        d_frag = __builtin_amdgcn_mfma_f32_32x32x2bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 32; i++)
        {
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4;
            col = tid % 32;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x8bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 8;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x2 a_frag;
        bfloat16x2 b_frag;
        float32x4 c_frag;
        float32x4 d_frag;
        // load a
        for (i = 0; i < 2; i++)
        {
            row = tid % 16;
            col = tid / 16 * 2 + i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 2; i++)
        {
            row = tid / 16 * 2 + i;
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
        d_frag = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = tid / 16 * 4 + i;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_16x16x2bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 2;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x2 a_frag;
        bfloat16x2 b_frag;
        float32x16 c_frag;
        float32x16 d_frag;
        // load a
        for (i = 0; i < 2; i++)
        {
            row = tid % 16;
            col = i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 2; i++)
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
        d_frag = __builtin_amdgcn_mfma_f32_16x16x2bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 16; i++)
        {
            row = tid / 16 * 4 + i % 4;
            col = tid % 16;
            d[row * N + col] = d_frag[i];
        }
    }

    __global__ void mfma_f32_4x4x2bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 4, K = 2;
        uint32_t tid = threadIdx.x;
        uint32_t row, col, i;
        bfloat16x2 a_frag;
        bfloat16x2 b_frag;
        float32x4 c_frag;
        float32x4 d_frag;
        // load a
        for (i = 0; i < 2; i++)
        {
            row = tid % 4;
            col = i;
            a_frag[i] = a[row * K + col];
        }
        // load b
        for (i = 0; i < 2; i++)
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
        d_frag = __builtin_amdgcn_mfma_f32_4x4x2bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        // store d
        for (i = 0; i < 4; i++)
        {
            row = i;
            col = tid % 4;
            d[row * N + col] = d_frag[i];
        }
    }

    void mfma_f32_32x32x4bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_32x32x4bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x2bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_32x32x2bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x8bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_16x16x8bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x2bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_16x16x2bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x2bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_4x4x2bf16_kernel<<<1, 64>>>(d, a, b, c);
    }
}
