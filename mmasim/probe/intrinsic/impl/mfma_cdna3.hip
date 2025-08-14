#include "mfma.h"

extern "C" // fp32
{
    __global__ void mfma_f32_32x32x2_f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 2;
        uint32_t tid = threadIdx.x;
        float a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K2();
        LOAD_B_N32K2();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x2f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x1_2b_f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 1;
        uint32_t tid = threadIdx.x;
        float a_frag, b_frag;
        float32x32 c_frag, d_frag;

        LOAD_A_M32K1_2B();
        LOAD_B_N32K1_2B();
        LOAD_C_M32N32_2B();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32_2B();
    }

    __global__ void mfma_f32_16x16x4_f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        float a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K4();
        LOAD_B_N16K4();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x1_4b_f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 1;
        uint32_t tid = threadIdx.x;
        float a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M16K1_4B();
        LOAD_B_N16K1_4B();
        LOAD_C_M16N16_4B();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_4B();
    }

    __global__ void mfma_f32_4x4x1_16b_f32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 4, K = 1;
        uint32_t tid = threadIdx.x;
        float a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M4K1_16B();
        LOAD_B_N4K1_16B();
        LOAD_C_M4N4_16B();
        d_frag = __builtin_amdgcn_mfma_f32_4x4x1f32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_16B();
    }

    void mfma_f32_32x32x2_f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_32x32x2_f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x1_2b_f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_32x32x1_2b_f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x4_f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_16x16x4_f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x1_4b_f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_16x16x1_4b_f32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x1_16b_f32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_4x4x1_16b_f32_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // xf32
{
    __global__ void mfma_f32_16x16x8_xf32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 16, K = 8;
        uint32_t tid = threadIdx.x;
        float32x2 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K8();
        LOAD_B_N16K8();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_32x32x4_xf32_kernel(float *d, float *a, float *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t tid = threadIdx.x;
        float32x2 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K4();
        LOAD_B_N32K4();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4_xf32(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    void mfma_f32_16x16x8_xf32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_16x16x8_xf32_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x4_xf32(float *d, float *a, float *b, float *c)
    {
        mfma_f32_32x32x4_xf32_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // f16
{
    __global__ void mfma_f32_32x32x8_f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 32, K = 8;
        uint32_t tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K8();
        LOAD_B_N32K8();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x8f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x4_2b_f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x32 c_frag, d_frag;

        LOAD_A_M32K4_2B();
        LOAD_B_N32K4_2B();
        LOAD_C_M32N32_2B();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32_2B();
    }

    __global__ void mfma_f32_16x16x16_f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 16, K = 16;
        uint32_t tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K16();
        LOAD_B_N16K16();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x4_4b_f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M16K4_4B();
        LOAD_B_N16K4_4B();
        LOAD_C_M16N16_4B();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_4B();
    }

    __global__ void mfma_f32_4x4x4_16b_f16_kernel(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        const uint32_t N = 4, K = 4;
        uint32_t tid = threadIdx.x;
        float16x4 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M4K4_16B();
        LOAD_B_N4K4_16B();
        LOAD_C_M4N4_16B();
        d_frag = __builtin_amdgcn_mfma_f32_4x4x4f16(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_16B();
    }

    void mfma_f32_32x32x8_f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_32x32x8_f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x4_2b_f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_32x32x4_2b_f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x16_f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_16x16x16_f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x4_4b_f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_16x16x4_4b_f16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x4_16b_f16(float *d, _Float16 *a, _Float16 *b, float *c)
    {
        mfma_f32_4x4x4_16b_f16_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // bf16
{
    __global__ void mfma_f32_32x32x8_bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 8;
        uint32_t tid = threadIdx.x;
        bfloat16x4 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K8();
        LOAD_B_N32K8();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x4_2b_bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 32, K = 4;
        uint32_t tid = threadIdx.x;
        bfloat16x4 a_frag, b_frag;
        float32x32 c_frag, d_frag;

        LOAD_A_M32K4_2B();
        LOAD_B_N32K4_2B();
        LOAD_C_M32N32_2B();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M32N32_2B();
    }

    __global__ void mfma_f32_16x16x16_bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 16;
        uint32_t tid = threadIdx.x;
        bfloat16x4 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K16();
        LOAD_B_N16K16();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x4_4b_bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        bfloat16x4 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M16K4_4B();
        LOAD_B_N16K4_4B();
        LOAD_C_M16N16_4B();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_4B();
    }

    __global__ void mfma_f32_4x4x4_16b_bf16_kernel(float *d, int16_t *a, int16_t *b, float *c)
    {
        const uint32_t N = 4, K = 4;
        uint32_t tid = threadIdx.x;
        bfloat16x4 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M4K4_16B();
        LOAD_B_N4K4_16B();
        LOAD_C_M4N4_16B();
        d_frag = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_16B();
    }

    void mfma_f32_32x32x8_bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_32x32x8_bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x4_2b_bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_32x32x4_2b_bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x16_bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_16x16x16_bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x4_4b_bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_16x16x4_4b_bf16_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_4x4x4_16b_bf16(float *d, int16_t *a, int16_t *b, float *c)
    {
        mfma_f32_4x4x4_16b_bf16_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // fp8
{
    __global__ void mfma_f32_16x16x32_fp8_fp8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 16, K = 32;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K32();
        LOAD_B_N16K32();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x32_fp8_bf8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 16, K = 32;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K32();
        LOAD_B_N16K32();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x32_bf8_fp8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 16, K = 32;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K32();
        LOAD_B_N16K32();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_16x16x32_bf8_bf8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 16, K = 32;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x4 c_frag, d_frag;

        LOAD_A_M16K32();
        LOAD_B_N16K32();
        LOAD_C_M16N16();
        d_frag = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M16N16();
    }

    __global__ void mfma_f32_32x32x16_fp8_fp8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 32, K = 16;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K16();
        LOAD_B_N32K16();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x16_fp8_bf8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 32, K = 16;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K16();
        LOAD_B_N32K16();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x16_bf8_fp8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 32, K = 16;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K16();
        LOAD_B_N32K16();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    __global__ void mfma_f32_32x32x16_bf8_bf8_kernel(float *d, int8_t *a, int8_t *b, float *c)
    {
        const uint32_t N = 32, K = 16;
        uint32_t tid = threadIdx.x;
        float8x8 a_frag, b_frag;
        float32x16 c_frag, d_frag;

        LOAD_A_M32K16();
        LOAD_B_N32K16();
        LOAD_C_M32N32();
        d_frag = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a_frag.all, b_frag.all, c_frag, 0, 0, 0);
        STORE_D_M32N32();
    }

    void mfma_f32_16x16x32_fp8_fp8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_16x16x32_fp8_fp8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x32_fp8_bf8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_16x16x32_fp8_bf8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x32_bf8_fp8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_16x16x32_bf8_fp8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_16x16x32_bf8_bf8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_16x16x32_bf8_bf8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x16_fp8_fp8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_32x32x16_fp8_fp8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x16_fp8_bf8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_32x32x16_fp8_bf8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x16_bf8_fp8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_32x32x16_bf8_fp8_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f32_32x32x16_bf8_bf8(float *d, int8_t *a, int8_t *b, float *c)
    {
        mfma_f32_32x32x16_bf8_bf8_kernel<<<1, 64>>>(d, a, b, c);
    }
}
