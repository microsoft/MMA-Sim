#include "mma.h"

extern "C" // fp8 m16n8k32 f32_output
{
    __global__ void mma_m16n8k32_f32_e5m2_e5m2_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
        STORE_D_M16N8_F32();
    }

    __global__ void mma_m16n8k32_f32_e5m2_e4m3_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
        STORE_D_M16N8_F32();
    }

    __global__ void mma_m16n8k32_f32_e4m3_e5m2_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
        STORE_D_M16N8_F32();
    }

    __global__ void mma_m16n8k32_f32_e4m3_e4m3_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
        STORE_D_M16N8_F32();
    }

    void mma_m16n8k32_f32_e5m2_e5m2_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k32_f32_e5m2_e5m2_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k32_f32_e5m2_e4m3_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k32_f32_e5m2_e4m3_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k32_f32_e4m3_e5m2_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k32_f32_e4m3_e5m2_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k32_f32_e4m3_e4m3_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k32_f32_e4m3_e4m3_f32_kernel<<<1, 32>>>(d, a, b, c);
    }
}

extern "C" // fp8 m16n8k16 f32_output
{
    __global__ void mma_m16n8k16_f32_e5m2_e5m2_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32 "
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

    __global__ void mma_m16n8k16_f32_e5m2_e4m3_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32 "
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

    __global__ void mma_m16n8k16_f32_e4m3_e5m2_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e5m2.f32 "
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

    __global__ void mma_m16n8k16_f32_e4m3_e4m3_f32_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        float c_frag[4], d_frag[4];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F32();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
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

    void mma_m16n8k16_f32_e5m2_e5m2_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k16_f32_e5m2_e5m2_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k16_f32_e5m2_e4m3_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k16_f32_e5m2_e4m3_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k16_f32_e4m3_e5m2_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k16_f32_e4m3_e5m2_f32_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k16_f32_e4m3_e4m3_f32(
        float *d, uint8_t *a, uint8_t *b, float *c)
    {
        mma_m16n8k16_f32_e4m3_e4m3_f32_kernel<<<1, 32>>>(d, a, b, c);
    }
}

extern "C" // fp8 m16n8k32 f16_output
{
    __global__ void mma_m16n8k32_f16_e5m2_e5m2_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            : "=r"(d_frag[0]), "=r"(d_frag[1])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "r"(c_frag[0]), "r"(c_frag[1]));
        STORE_D_M16N8_F16();
    }

    __global__ void mma_m16n8k32_f16_e5m2_e4m3_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            : "=r"(d_frag[0]), "=r"(d_frag[1])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "r"(c_frag[0]), "r"(c_frag[1]));
        STORE_D_M16N8_F16();
    }

    __global__ void mma_m16n8k32_f16_e4m3_e5m2_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            : "=r"(d_frag[0]), "=r"(d_frag[1])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "r"(c_frag[0]), "r"(c_frag[1]));
        STORE_D_M16N8_F16();
    }

    __global__ void mma_m16n8k32_f16_e4m3_e4m3_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
            "{%0, %1}, "
            "{%2, %3, %4, %5}, "
            "{%6, %7}, "
            "{%8, %9};"
            : "=r"(d_frag[0]), "=r"(d_frag[1])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "r"(c_frag[0]), "r"(c_frag[1]));
        STORE_D_M16N8_F16();
    }

    void mma_m16n8k32_f16_e5m2_e5m2_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k32_f16_e5m2_e5m2_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k32_f16_e5m2_e4m3_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k32_f16_e5m2_e4m3_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k32_f16_e4m3_e5m2_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k32_f16_e4m3_e5m2_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k32_f16_e4m3_e4m3_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k32_f16_e4m3_e4m3_f16_kernel<<<1, 32>>>(d, a, b, c);
    }
}

extern "C" // fp8 m16n8k16 f16_output
{
    __global__ void mma_m16n8k16_f16_e5m2_e5m2_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16 "
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

    __global__ void mma_m16n8k16_f16_e5m2_e4m3_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e4m3.f16 "
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

    __global__ void mma_m16n8k16_f16_e4m3_e5m2_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e5m2.f16 "
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

    __global__ void mma_m16n8k16_f16_e4m3_e4m3_f16_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        const uint32_t N = 8, K = 32;
        uint32_t row, col, i, laneid = threadIdx.x;
        uint32_t a_frag[2], b_frag[1];
        uint32_t c_frag[2], d_frag[2];

        LOAD_A_M16K16_FP8();
        LOAD_B_N8K16_FP8();
        LOAD_C_M16N8_F16();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16 "
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

    void mma_m16n8k16_f16_e5m2_e5m2_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k16_f16_e5m2_e5m2_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k16_f16_e5m2_e4m3_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k16_f16_e5m2_e4m3_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k16_f16_e4m3_e5m2_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k16_f16_e4m3_e5m2_f16_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k16_f16_e4m3_e4m3_f16(
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
    {
        mma_m16n8k16_f16_e4m3_e4m3_f16_kernel<<<1, 32>>>(d, a, b, c);
    }
}
