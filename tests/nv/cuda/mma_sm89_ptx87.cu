#if __CUDACC_VER_MAJOR__ > 12 || __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8

#include <stdint.h>

__global__ void mma_m16n8k32_row_col_f16_e5m2_e5m2_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 32;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 16; i += 4)
    {
        uint32_t row = (i % 8 < 4) ? groupid : groupid + 8;
        uint32_t col = (i < 8) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16"
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k32_row_col_f16_e5m2_e4m3_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 32;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 16; i += 4)
    {
        uint32_t row = (i % 8 < 4) ? groupid : groupid + 8;
        uint32_t col = (i < 8) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16"
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k32_row_col_f16_e4m3_e5m2_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 32;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 16; i += 4)
    {
        uint32_t row = (i % 8 < 4) ? groupid : groupid + 8;
        uint32_t col = (i < 8) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16"
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k32_row_col_f16_e4m3_e4m3_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 32;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[4];
    uint32_t b_frag[2];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 16; i += 4)
    {
        uint32_t row = (i % 8 < 4) ? groupid : groupid + 8;
        uint32_t col = (i < 8) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? ((threadID_in_group * 4) + (i & 0x3)) : ((threadID_in_group * 4) + (i & 0x3) + 16);
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k16_row_col_f32_e5m2_e5m2_f32_kernel(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    float c_frag[4];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i] = *(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32"
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void mma_m16n8k16_row_col_f32_e5m2_e4m3_f32_kernel(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    float c_frag[4];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i] = *(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32"
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void mma_m16n8k16_row_col_f32_e4m3_e5m2_f32_kernel(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    float c_frag[4];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i] = *(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e5m2.f32"
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void mma_m16n8k16_row_col_f32_e4m3_e4m3_f32_kernel(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    float c_frag[4];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i] = *(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32"
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};"
        : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void mma_m16n8k16_row_col_f16_e5m2_e5m2_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16"
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  %6};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k16_row_col_f16_e5m2_e4m3_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e4m3.f16"
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  %6};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k16_row_col_f16_e4m3_e5m2_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e5m2.f16"
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  %6};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void mma_m16n8k16_row_col_f16_e4m3_e4m3_f16_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    const uint32_t N = 8, K = 16;
    uint32_t laneid = threadIdx.x % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    uint32_t a_frag[2];
    uint32_t b_frag[1];
    uint32_t c_frag[2];
    // load a
    for (uint32_t i = 0; i < 8; i += 4)
    {
        uint32_t row = (i < 4) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 4) + (i & 0x3);
        a_frag[i / 4] = *(uint32_t *)(a + row * K + col);
    }
    // load b
    for (uint32_t i = 0; i < 4; i += 4)
    {
        uint32_t row = (threadID_in_group * 4) + i;
        uint32_t col = groupid;
        b_frag[i / 4] = *(uint32_t *)(b + col * K + row);
    }
    // load c
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        c_frag[i / 2] = *(uint32_t *)(c + row * N + col);
    }
    // mma
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16"
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  %6};"
        : "=r"(d_frag[0]), "=r"(d_frag[1])
        : "r"(a_frag[0]), "r"(a_frag[1]),
          "r"(b_frag[0]),
          "r"(c_frag[0]), "r"(c_frag[1]));
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = (i < 2) ? groupid : groupid + 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

extern "C" void mma_m16n8k32_row_col_f16_e5m2_e5m2_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k32_row_col_f16_e5m2_e5m2_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k32_row_col_f16_e5m2_e4m3_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k32_row_col_f16_e5m2_e4m3_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k32_row_col_f16_e4m3_e5m2_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k32_row_col_f16_e4m3_e5m2_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k32_row_col_f16_e4m3_e4m3_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k32_row_col_f16_e4m3_e4m3_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f32_e5m2_e5m2_f32(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f32_e5m2_e5m2_f32_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f32_e5m2_e4m3_f32(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f32_e5m2_e4m3_f32_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f32_e4m3_e5m2_f32(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f32_e4m3_e5m2_f32_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f32_e4m3_e4m3_f32(
    float *d, uint8_t *a, uint8_t *b, float *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f32_e4m3_e4m3_f32_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f16_e5m2_e5m2_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f16_e5m2_e5m2_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f16_e5m2_e4m3_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f16_e5m2_e4m3_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f16_e4m3_e5m2_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f16_e4m3_e5m2_f16_kernel<<<grid, block>>>(d, a, b, c);
}

extern "C" void mma_m16n8k16_row_col_f16_e4m3_e4m3_f16(
    uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)
{
    dim3 block(32);
    dim3 grid(1);
    mma_m16n8k16_row_col_f16_e4m3_e4m3_f16_kernel<<<grid, block>>>(d, a, b, c);
}

#else

#warning "FP8 MMA instructions with FP16 accumulator or with m16n8k16 shape (introduced since PTX 8.7) are not compiled because the CUDA version is lower than 12.8"

#endif
