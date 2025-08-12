#include "mfma.h"

extern "C" // fp32
{
    __global__ void mfma_f32_32x32x2f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 2;
        uint32_t row, col, i, tid = threadIdx.x;
        float a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K2();
        LOAD_B_N32K2();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x2f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x1f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 1;
        uint32_t row, col, i, tid = threadIdx.x;
        float a_frag, b_frag;
        float32x32 c_frag, d_frag;

        LOAD_A_M32K1_2B();
        LOAD_B_N32K1_2B();
        LOAD_C_M32N32_2B();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32_2B();
    }

    __global__ void mfma_f32_16x16x4f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t row, col, i, tid = threadIdx.x;
        float a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K4();
        LOAD_B_N16K4();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x1f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 1;
        uint32_t row, col, i, tid = threadIdx.x;
        float a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M16K1_4B();
        LOAD_B_N16K1_4B();
        LOAD_C_M16N16_4B();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_4B();
    }

    __global__ void mfma_f32_4x4x1f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 4, K = 1;
        uint32_t row, col, i, tid = threadIdx.x;
        float a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M4K1_16B();
        LOAD_B_N4K1_16B();
        LOAD_C_M4N4_16B();
        d_frag = __builtin_amdgcn_mfma_f32_4x4x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_16B();
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
        uint32_t row, col, i, tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K8();
        LOAD_B_N32K8();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x8f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x4f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t row, col, i, tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x32 c_frag, d_frag;

        LOAD_A_M32K4_2B();
        LOAD_B_N32K4_2B();
        LOAD_C_M32N32_2B();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32_2B();
    }

    __global__ void mfma_f32_16x16x16f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 16, K = 16;
        uint32_t row, col, i, tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K16();
        LOAD_B_N16K16();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x4f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t row, col, i, tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M16K4_4B();
        LOAD_B_N16K4_4B();
        LOAD_C_M16N16_4B();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_4B();
    }

    __global__ void mfma_f32_4x4x4f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 4, K = 4;
        uint32_t row, col, i, tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M4K4_16B();
        LOAD_B_N4K4_16B();
        LOAD_C_M4N4_16B();
        d_frag = __builtin_amdgcn_mfma_f32_4x4x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_16B();
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
        uint32_t row, col, i, tid = threadIdx.x;
        bfloat16x2 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K4();
        LOAD_B_N32K4();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x2bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 2;
        uint32_t row, col, i, tid = threadIdx.x;
        bfloat16x2 a_frag, b_frag;
        float32x32 c_frag, d_frag;

        LOAD_A_M32K2_2B();
        LOAD_B_N32K2_2B();
        LOAD_C_M32N32_2B();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x2bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32_2B();
    }

    __global__ void mfma_f32_16x16x8bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 8;
        uint32_t row, col, i, tid = threadIdx.x;
        bfloat16x2 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K8();
        LOAD_B_N16K8();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x2bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 2;
        uint32_t row, col, i, tid = threadIdx.x;
        bfloat16x2 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M16K2_4B();
        LOAD_B_N16K2_4B();
        LOAD_C_M16N16_4B();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x2bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_4B();
    }

    __global__ void mfma_f32_4x4x2bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 4, K = 2;
        uint32_t row, col, i, tid = threadIdx.x;
        bfloat16x2 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M4K2_16B();
        LOAD_B_N4K2_16B();
        LOAD_C_M4N4_16B();
        d_frag = __builtin_amdgcn_mfma_f32_4x4x2bf16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_16B();
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
