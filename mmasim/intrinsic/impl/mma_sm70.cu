#include "mma.h"

extern "C" // f16
{
    __global__ void mma_m8n8k4_f32_f16_f16_f32_kernel(
        float *d, uint16_t *a, uint16_t *b, float *c)
    {
        const uint32_t N = 8, K = 4;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[2];
        float c_frag[8], d_frag[8];

        LOAD_A_M8K4_X4_F16();
        LOAD_B_N8K4_X4_F16();
        LOAD_C_M8N8_X4_F32();
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%10, %11}, "
                     "{%12, %13, %14, %15, %16, %17, %18, %19};"
                     : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3]),
                       "=f"(d_frag[4]), "=f"(d_frag[5]), "=f"(d_frag[6]), "=f"(d_frag[7])
                     : "r"(a_frag[0]), "r"(a_frag[1]),
                       "r"(b_frag[0]), "r"(b_frag[1]),
                       "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
                       "f"(c_frag[4]), "f"(c_frag[5]), "f"(c_frag[6]), "f"(c_frag[7]));
        STORE_D_M8N8_X4_F32();
    }

    __global__ void mma_m8n8k4_f32_f16_f16_f16_kernel(
        float *d, uint16_t *a, uint16_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 4;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[2], c_frag[4];
        float d_frag[8];

        LOAD_A_M8K4_X4_F16();
        LOAD_B_N8K4_X4_F16();
        LOAD_C_M8N8_X4_F16();
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, "
                     "{%8, %9}, "
                     "{%10, %11}, "
                     "{%12, %13, %14, %15};"
                     : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3]),
                       "=f"(d_frag[4]), "=f"(d_frag[5]), "=f"(d_frag[6]), "=f"(d_frag[7])
                     : "r"(a_frag[0]), "r"(a_frag[1]),
                       "r"(b_frag[0]), "r"(b_frag[1]),
                       "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3]));
        STORE_D_M8N8_X4_F32();
    }

    __global__ void mma_m8n8k4_f16_f16_f16_f16_kernel(
        uint16_t *d, uint16_t *a, uint16_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 4;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[2];
        uint32_t c_frag[4], d_frag[4];

        LOAD_A_M8K4_X4_F16();
        LOAD_B_N8K4_X4_F16();
        LOAD_C_M8N8_X4_F16();
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
                     "{%0, %1, %2, %3}, "
                     "{%4, %5}, "
                     "{%6, %7}, "
                     "{%8, %9, %10, %11};"
                     : "=r"(d_frag[0]), "=r"(d_frag[1]), "=r"(d_frag[2]), "=r"(d_frag[3])
                     : "r"(a_frag[0]), "r"(a_frag[1]),
                       "r"(b_frag[0]), "r"(b_frag[1]),
                       "r"(c_frag[0]), "r"(c_frag[1]), "r"(c_frag[2]), "r"(c_frag[3]));
        STORE_D_M8N8_X4_F16();
    }

    void mma_m8n8k4_f32_f16_f16_f32(
        float *d, uint16_t *a, uint16_t *b, float *c)
    {
        mma_m8n8k4_f32_f16_f16_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m8n8k4_f32_f16_f16_f16(
        float *d, uint16_t *a, uint16_t *b, uint16_t *c)
    {
        mma_m8n8k4_f32_f16_f16_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m8n8k4_f16_f16_f16_f16(
        uint16_t *d, uint16_t *a, uint16_t *b, uint16_t *c)
    {
        mma_m8n8k4_f16_f16_f16_f16_kernel<<<1, 32>>>(d, a, b, c);
    }
}
