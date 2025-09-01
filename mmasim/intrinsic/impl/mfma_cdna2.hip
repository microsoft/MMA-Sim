#include "mfma.h"

extern "C" // f64
{
    __global__ void mfma_f64_16x16x4f64_kernel(double *d, double *a, double *b, double *c)
    {
        const uint32_t N = 16, K = 4;
        uint32_t tid = threadIdx.x;
        double a_frag, b_frag;
        float64x4 c_frag, d_frag;

        LOAD_A_M16K4();
        LOAD_B_N16K4();
        LOAD_C_M16N16_F64();
        d_frag = __builtin_amdgcn_mfma_f64_16x16x4f64(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M16N16_F64();
    }

    __global__ void mfma_f64_4x4x4f64_kernel(double *d, double *a, double *b, double *c)
    {
        const uint32_t N = 4, K = 4;
        uint32_t tid = threadIdx.x;
        double a_frag, b_frag;
        double c_frag, d_frag;

        LOAD_A_M4K4_4B();
        LOAD_B_N4K4_4B();
        LOAD_C_M4N4_4B();
        d_frag = __builtin_amdgcn_mfma_f64_4x4x4f64(a_frag, b_frag, c_frag, 0, 0, 0);
        STORE_D_M4N4_4B();
    }

    void mfma_f64_16x16x4f64(double *d, double *a, double *b, double *c)
    {
        mfma_f64_16x16x4f64_kernel<<<1, 64>>>(d, a, b, c);
    }

    void mfma_f64_4x4x4f64(double *d, double *a, double *b, double *c)
    {
        mfma_f64_4x4x4f64_kernel<<<1, 64>>>(d, a, b, c);
    }
}

extern "C" // bf16
{
    __global__ void mfma_f32_32x32x8bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
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

    __global__ void mfma_f32_32x32x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
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

    __global__ void mfma_f32_16x16x16bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
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

    __global__ void mfma_f32_16x16x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
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

    __global__ void mfma_f32_4x4x4bf16_1k_kernel(float *d, int16_t *a, int16_t *b, float *c)
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
