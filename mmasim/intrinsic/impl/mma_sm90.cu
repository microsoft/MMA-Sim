#include "mma.h"

extern "C" // f64
{
    __global__ void mma_m16n8k16_f64_f64_f64_f64_kernel(
        double *d, double *a, double *b, double *c)
    {
        const uint32_t N = 8, K = 16;
        uint32_t laneid = threadIdx.x;
        double a_frag[8], b_frag[4];
        double c_frag[4], d_frag[4];

        LOAD_A_M16K16_F64();
        LOAD_B_N8K16_F64();
        LOAD_C_M16N8_F64();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7, %8, %9, %10, %11}, "
            "{%12, %13, %14, %15}, "
            "{%16, %17, %18, %19};"
            : "=d"(d_frag[0]), "=d"(d_frag[1]), "=d"(d_frag[2]), "=d"(d_frag[3])
            : "d"(a_frag[0]), "d"(a_frag[1]), "d"(a_frag[2]), "d"(a_frag[3]),
              "d"(a_frag[4]), "d"(a_frag[5]), "d"(a_frag[6]), "d"(a_frag[7]),
              "d"(b_frag[0]), "d"(b_frag[1]), "d"(b_frag[2]), "d"(b_frag[3]),
              "d"(c_frag[0]), "d"(c_frag[1]), "d"(c_frag[2]), "d"(c_frag[3]));
        STORE_D_M16N8_F64();
    }

    __global__ void mma_m16n8k8_f64_f64_f64_f64_kernel(
        double *d, double *a, double *b, double *c)
    {
        const uint32_t N = 8, K = 8;
        uint32_t laneid = threadIdx.x;
        double a_frag[4], b_frag[2];
        double c_frag[4], d_frag[4];

        LOAD_A_M16K8_F64();
        LOAD_B_N8K8_F64();
        LOAD_C_M16N8_F64();
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=d"(d_frag[0]), "=d"(d_frag[1]), "=d"(d_frag[2]), "=d"(d_frag[3])
            : "d"(a_frag[0]), "d"(a_frag[1]), "d"(a_frag[2]), "d"(a_frag[3]),
              "d"(b_frag[0]), "d"(b_frag[1]),
              "d"(c_frag[0]), "d"(c_frag[1]), "d"(c_frag[2]), "d"(c_frag[3]));
        STORE_D_M16N8_F64();
    }

    __global__ void mma_m16n8k4_f64_f64_f64_f64_kernel(
        double *d, double *a, double *b, double *c)
    {
        const uint32_t N = 8, K = 4;
        uint32_t laneid = threadIdx.x;
        double a_frag[2], b_frag[1];
        double c_frag[4], d_frag[4];

        LOAD_A_M16K4_F64();
        LOAD_B_N8K4_F64();
        LOAD_C_M16N8_F64();
        asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1, %2, %3}, "
            "{%4, %5}, "
            "{%6}, "
            "{%7, %8, %9, %10};"
            : "=d"(d_frag[0]), "=d"(d_frag[1]), "=d"(d_frag[2]), "=d"(d_frag[3])
            : "d"(a_frag[0]), "d"(a_frag[1]),
              "d"(b_frag[0]),
              "d"(c_frag[0]), "d"(c_frag[1]), "d"(c_frag[2]), "d"(c_frag[3]));
        STORE_D_M16N8_F64();
    }

    void mma_m16n8k16_f64_f64_f64_f64(
        double *d, double *a, double *b, double *c)
    {
        mma_m16n8k16_f64_f64_f64_f64_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k8_f64_f64_f64_f64(
        double *d, double *a, double *b, double *c)
    {
        mma_m16n8k8_f64_f64_f64_f64_kernel<<<1, 32>>>(d, a, b, c);
    }

    void mma_m16n8k4_f64_f64_f64_f64(
        double *d, double *a, double *b, double *c)
    {
        mma_m16n8k4_f64_f64_f64_f64_kernel<<<1, 32>>>(d, a, b, c);
    }
}
