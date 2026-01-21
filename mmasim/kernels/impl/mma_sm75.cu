#include "mma.h"

extern "C" // f16
{
    __global__ void mma_m16n8k8_f32_f16_f16_f32_kernel(
        float *d, uint16_t *a, uint16_t *b, float *c)
    {
        const uint32_t N = 8, K = 8;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K8_F16BF16();
        LOAD_B_N8K8_F16BF16();
        LOAD_C_M16N8_F32();
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, "
                     "{%4, %5}, "
                     "{%6}, "
                     "{%7, %8, %9, %10};"
                     : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
                     : "r"(a_frag[0]), "r"(a_frag[1]),
                       "r"(b_frag[0]),
                       "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
        STORE_D_M16N8_F32();
    }

    __global__ void mma_m16n8k8_f16_f16_f16_f16_kernel(
        uint16_t *d, uint16_t *a, uint16_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 8;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K8_F16BF16();
        LOAD_B_N8K8_F16BF16();
        LOAD_C_M16N8_F16();
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                     "{%0, %1}, "
                     "{%2, %3}, "
                     "{%4}, "
                     "{%5, %6};"
                     : "=r"(d_frag[0]), "=r"(d_frag[1])
                     : "r"(a_frag[0]), "r"(a_frag[1]),
                       "r"(b_frag[0]),
                       "r"(c_frag[0]), "r"(c_frag[1]));
        STORE_D_M16N8_F16();
    }

    void mma_m16n8k8_f32_f16_f16_f32(
        float *d, uint16_t *a, uint16_t *b, float *c)
    {
        mma_m16n8k8_f32_f16_f16_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k8_f16_f16_f16_f16(
        uint16_t *d, uint16_t *a, uint16_t *b, uint16_t *c)
    {
        mma_m16n8k8_f16_f16_f16_f16_kernel<<<1, 32>>>(d, a, b, c);
    }
}