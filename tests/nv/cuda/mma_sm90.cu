#include <stdint.h>

__global__ void wgmma_m64n8k16_row_col_f16_f16_f16_kernel(
    uint16_t *d, uint16_t *a, uint16_t *b)
{
    const uint32_t N = 8, K = 16;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    __shared__ uint16_t a_smem[64 * K];
    __shared__ uint16_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        uint32_t row = i * 8 + tid % 64 / 8;
        uint32_t col = tid % 8 + tid / 64 * 8;
        a_smem[i * 128 + tid] = *(a + row * K + col);
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    uint32_t row = tid % 8 + tid / 64 * 8;
    uint32_t col = tid % 64 / 8;
    b_smem[tid] = *(b + col * K + row);
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i / 2] = *(uint32_t *)(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16"
        "{%0, %1},"
        " %2, %3, 1, 1, 1, 0, 0;"
        : "+r"(d_frag[0]), "+r"(d_frag[1])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void wgmma_m64n8k16_row_col_f32_f16_f16_kernel(
    float *d, uint16_t *a, uint16_t *b)
{
    const uint32_t N = 8, K = 16;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint16_t a_smem[64 * K];
    __shared__ uint16_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        uint32_t row = i * 8 + tid % 64 / 8;
        uint32_t col = tid % 8 + tid / 64 * 8;
        a_smem[i * 128 + tid] = *(a + row * K + col);
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    uint32_t row = tid % 8 + tid / 64 * 8;
    uint32_t col = tid % 64 / 8;
    b_smem[tid] = *(b + col * K + row);
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1, 0, 0;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void wgmma_m64n8k16_row_col_f32_bf16_bf16_kernel(
    float *d, uint16_t *a, uint16_t *b)
{
    const uint32_t N = 8, K = 16;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint16_t a_smem[64 * K];
    __shared__ uint16_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        uint32_t row = i * 8 + tid % 64 / 8;
        uint32_t col = tid % 8 + tid / 64 * 8;
        a_smem[i * 128 + tid] = *(a + row * K + col);
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    uint32_t row = tid % 8 + tid / 64 * 8;
    uint32_t col = tid % 64 / 8;
    b_smem[tid] = *(b + col * K + row);
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1, 0, 0;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void wgmma_m64n8k8_row_col_f32_tf32_tf32_kernel(
    float *d, uint32_t *a, uint32_t *b)
{
    const uint32_t N = 8, K = 8;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint32_t a_smem[64 * K];
    __shared__ uint32_t b_smem[N * K];
    // load a
    if (tid < 64)
    {
        for (uint32_t i = 0; i < 8; i++)
        {
            uint32_t row = i * 8 + tid / 4 - tid / 32 * 8;
            uint32_t col = tid % 4 + tid / 32 * 4;
            a_smem[i * 64 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    if (tid < 64)
    {
        uint32_t row = tid % 4 + tid / 32 * 4;
        uint32_t col = tid / 4 - tid / 32 * 8;
        b_smem[tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void wgmma_m64n8k32_row_col_f16_e5m2_e5m2_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i / 2] = *(uint32_t *)(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e5m2"
        "{%0, %1},"
        " %2, %3, 1, 1, 1;"
        : "+r"(d_frag[0]), "+r"(d_frag[1])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void wgmma_m64n8k32_row_col_f16_e5m2_e4m3_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i / 2] = *(uint32_t *)(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e4m3"
        "{%0, %1},"
        " %2, %3, 1, 1, 1;"
        : "+r"(d_frag[0]), "+r"(d_frag[1])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void wgmma_m64n8k32_row_col_f16_e4m3_e5m2_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i / 2] = *(uint32_t *)(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2"
        "{%0, %1},"
        " %2, %3, 1, 1, 1;"
        : "+r"(d_frag[0]), "+r"(d_frag[1])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void wgmma_m64n8k32_row_col_f16_e4m3_e4m3_kernel(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    uint32_t d_frag[2];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i / 2] = *(uint32_t *)(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3"
        "{%0, %1},"
        " %2, %3, 1, 1, 1;"
        : "+r"(d_frag[0]), "+r"(d_frag[1])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i += 2)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(uint32_t *)(d + row * N + col) = d_frag[i / 2];
    }
}

__global__ void wgmma_m64n8k32_row_col_f32_e5m2_e5m2_kernel(
    float *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void wgmma_m64n8k32_row_col_f32_e5m2_e4m3_kernel(
    float *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void wgmma_m64n8k32_row_col_f32_e4m3_e5m2_kernel(
    float *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e5m2"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

__global__ void wgmma_m64n8k32_row_col_f32_e4m3_e4m3_kernel(
    float *d, uint8_t *a, uint8_t *b)
{
    const uint32_t N = 8, K = 32;
    uint32_t tid = threadIdx.x;
    uint32_t warpid = tid / 32;
    uint32_t laneid = tid % 32;
    uint32_t groupid = laneid >> 2;
    uint32_t threadID_in_group = laneid % 4;
    float d_frag[4];
    __shared__ uint8_t a_smem[64 * K];
    __shared__ uint8_t b_smem[N * K];
    // load a
    for (uint32_t i = 0; i < 8; i++)
    {
        for (uint32_t k = 0; k < 2; k++)
        {
            uint32_t row = i * 8 + tid / 16;
            uint32_t col = k * 16 + tid % 16;
            a_smem[i * 256 + k * 128 + tid] = *(a + row * K + col);
        }
    }
    uint64_t a_desc = (__cvta_generic_to_shared(a_smem) & 0x3ffff) >> 4;
    a_desc |= (128ll & 0x3ffff) >> 4 << 16;
    a_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load b
    for (uint32_t k = 0; k < 2; k++)
    {
        uint32_t row = k * 16 + tid % 16;
        uint32_t col = tid / 16;
        b_smem[k * 128 + tid] = *(b + col * K + row);
    }
    uint64_t b_desc = (__cvta_generic_to_shared(b_smem) & 0x3ffff) >> 4;
    b_desc |= (128ll & 0x3ffff) >> 4 << 16;
    b_desc |= (256ll & 0x3ffff) >> 4 << 32;
    // load d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        d_frag[i] = *(d + row * N + col);
    }
    // mma
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3"
        "{%0, %1, %2, %3},"
        " %4, %5, 1, 1, 1;"
        : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
        : "l"(a_desc), "l"(b_desc));
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    // store d
    for (uint32_t i = 0; i < 4; i++)
    {
        uint32_t row = warpid * 16 + groupid + i / 2 * 8;
        uint32_t col = (threadID_in_group * 2) + (i & 0x1);
        *(d + row * N + col) = d_frag[i];
    }
}

extern "C" void wgmma_m64n8k16_row_col_f16_f16_f16(
    uint16_t *d, uint16_t *a, uint16_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k16_row_col_f16_f16_f16_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k16_row_col_f32_f16_f16(
    float *d, uint16_t *a, uint16_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k16_row_col_f32_f16_f16_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k16_row_col_f32_bf16_bf16(
    float *d, uint16_t *a, uint16_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k16_row_col_f32_bf16_bf16_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k8_row_col_f32_tf32_tf32(
    float *d, uint32_t *a, uint32_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k8_row_col_f32_tf32_tf32_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f16_e5m2_e5m2(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f16_e5m2_e5m2_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f16_e5m2_e4m3(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f16_e5m2_e4m3_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f16_e4m3_e5m2(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f16_e4m3_e5m2_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f16_e4m3_e4m3(
    uint16_t *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f16_e4m3_e4m3_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f32_e5m2_e5m2(
    float *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f32_e5m2_e5m2_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f32_e5m2_e4m3(
    float *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f32_e5m2_e4m3_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f32_e4m3_e5m2(
    float *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f32_e4m3_e5m2_kernel<<<grid, block>>>(d, a, b);
}

extern "C" void wgmma_m64n8k32_row_col_f32_e4m3_e4m3(
    float *d, uint8_t *a, uint8_t *b)
{
    dim3 block(128);
    dim3 grid(1);
    wgmma_m64n8k32_row_col_f32_e4m3_e4m3_kernel<<<grid, block>>>(d, a, b);
}
