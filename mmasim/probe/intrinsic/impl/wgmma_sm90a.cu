#include "wgmma.h"

extern "C" // tf32
{
    __global__ void wgmma_m64n8k8_f32_tf32_tf32_kernel(
        float *d, uint32_t *a, uint32_t *b)
    {
        const uint32_t M = 64, N = 8, K = 8;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint32_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K8_TF32();
        LOAD_B_N8K8_TF32();
        LOAD_D_M64N8_F32();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    void wgmma_m64n8k8_f32_tf32_tf32(
        float *d, uint32_t *a, uint32_t *b)
    {
        wgmma_m64n8k8_f32_tf32_tf32_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // fp16
{
    __global__ void wgmma_m64n8k16_f32_f16_f16_kernel(
        float *d, uint16_t *a, uint16_t *b)
    {
        const uint32_t M = 64, N = 8, K = 16;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint16_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K16_F16BF16();
        LOAD_B_N8K16_F16BF16();
        LOAD_D_M64N8_F32();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    __global__ void wgmma_m64n8k16_f16_f16_f16_kernel(
        uint16_t *d, uint16_t *a, uint16_t *b)
    {
        const uint32_t M = 64, N = 8, K = 16;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint16_t a_smem[M * K], b_smem[N * K];
        uint32_t d_frag[2];

        LOAD_A_M64K16_F16BF16();
        LOAD_B_N8K16_F16BF16();
        LOAD_D_M64N8_F16();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
            "{%0, %1}, %2, %3, 1, 1, 1, 0, 0;"
            : "+r"(d_frag[0]), "+r"(d_frag[1])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F16();
    }

    void wgmma_m64n8k16_f32_f16_f16(
        float *d, uint16_t *a, uint16_t *b)
    {
        wgmma_m64n8k16_f32_f16_f16_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k16_f16_f16_f16(
        uint16_t *d, uint16_t *a, uint16_t *b)
    {
        wgmma_m64n8k16_f16_f16_f16_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // bf16
{
    __global__ void wgmma_m64n8k16_f32_bf16_bf16_kernel(
        float *d, uint16_t *a, uint16_t *b)
    {
        const uint32_t M = 64, N = 8, K = 16;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint16_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K16_F16BF16();
        LOAD_B_N8K16_F16BF16();
        LOAD_D_M64N8_F32();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    void wgmma_m64n8k16_f32_bf16_bf16(
        float *d, uint16_t *a, uint16_t *b)
    {
        wgmma_m64n8k16_f32_bf16_bf16_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // fp8 m64n8k32 f32_output
{

    __global__ void wgmma_m64n8k32_f32_e5m2_e5m2_kernel(
        float *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F32();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    __global__ void wgmma_m64n8k32_f32_e5m2_e4m3_kernel(
        float *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F32();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    __global__ void wgmma_m64n8k32_f32_e4m3_e5m2_kernel(
        float *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F32();
        __syncthreads();
        // mma
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e5m2 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    __global__ void wgmma_m64n8k32_f32_e4m3_e4m3_kernel(
        float *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        float d_frag[4];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F32();
        __syncthreads();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3}, %4, %5, 1, 1, 1;"
            : "+f"(d_frag[0]), "+f"(d_frag[1]), "+f"(d_frag[2]), "+f"(d_frag[3])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F32();
    }

    void wgmma_m64n8k32_f32_e5m2_e5m2(
        float *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f32_e5m2_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k32_f32_e5m2_e4m3(
        float *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f32_e5m2_e4m3_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k32_f32_e4m3_e5m2(
        float *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f32_e4m3_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k32_f32_e4m3_e4m3(
        float *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f32_e4m3_e4m3_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // fp8 m64n8k32 f16_output
{
    __global__ void wgmma_m64n8k32_f16_e5m2_e5m2_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        uint32_t d_frag[2];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F16();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e5m2 "
            "{%0, %1}, %2, %3, 1, 1, 1;"
            : "+r"(d_frag[0]), "+r"(d_frag[1])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F16();
    }

    __global__ void wgmma_m64n8k32_f16_e5m2_e4m3_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        uint32_t d_frag[2];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F16();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e4m3 "
            "{%0, %1}, %2, %3, 1, 1, 1;"
            : "+r"(d_frag[0]), "+r"(d_frag[1])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F16();
    }

    __global__ void wgmma_m64n8k32_f16_e4m3_e5m2_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        uint32_t d_frag[2];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F16();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2 "
            "{%0, %1}, %2, %3, 1, 1, 1;"
            : "+r"(d_frag[0]), "+r"(d_frag[1])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F16();
    }

    __global__ void wgmma_m64n8k32_f16_e4m3_e4m3_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        uint32_t d_frag[2];

        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        LOAD_D_M64N8_F16();
        __syncthreads();
        asm volatile("wgmma.fence.sync.aligned;");
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3 "
            "{%0, %1}, %2, %3, 1, 1, 1;"
            : "+r"(d_frag[0]), "+r"(d_frag[1])
            : "l"(a_desc), "l"(b_desc));
        asm volatile("wgmma.commit_group.sync.aligned;");
        asm volatile("wgmma.wait_group.sync.aligned 0;");
        STORE_D_M64N8_F16();
    }

    void wgmma_m64n8k32_f16_e5m2_e5m2(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f16_e5m2_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k32_f16_e5m2_e4m3(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f16_e5m2_e4m3_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k32_f16_e4m3_e5m2(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f16_e4m3_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void wgmma_m64n8k32_f16_e4m3_e4m3(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        wgmma_m64n8k32_f16_e4m3_e4m3_kernel<<<1, 128>>>(d, a, b);
    }
}
