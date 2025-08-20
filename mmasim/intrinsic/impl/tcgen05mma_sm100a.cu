#include "tcgen05mma.h"

extern "C" // tf32
{
    __global__ void tcgen05mma_tf32_m64n8k8_f32_tf32_tf32_kernel(
        uint32_t *d, uint32_t *a, uint32_t *b)
    {
        const uint32_t M = 64, N = 8, K = 8;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint32_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K8_TF32();
        LOAD_B_N8K8_TF32();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= (1 << 4) | (2 << 7) | (2 << 10); // f32_tf32_tf32
        MMA("tf32");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    void tcgen05mma_tf32_m64n8k8_f32_tf32_tf32(
        uint32_t *d, uint32_t *a, uint32_t *b)
    {
        tcgen05mma_tf32_m64n8k8_f32_tf32_tf32_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // f16 and bf16
{
    __global__ void tcgen05mma_f16_m64n8k16_f32_f16_f16_kernel(
        uint32_t *d, uint16_t *a, uint16_t *b)
    {
        const uint32_t M = 64, N = 8, K = 16;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint16_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K16_F16BF16();
        LOAD_B_N8K16_F16BF16();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= 1 << 4; // f32_f16_f16
        MMA("f16");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_f16_m64n8k16_f32_bf16_bf16_kernel(
        uint32_t *d, uint16_t *a, uint16_t *b)
    {
        const uint32_t M = 64, N = 8, K = 16;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint16_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K16_F16BF16();
        LOAD_B_N8K16_F16BF16();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= (1 << 4) | (1 << 7) | (1 << 10); // f32_bf16_bf16
        MMA("f16");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_f16_m64n8k16_f16_f16_f16_kernel(
        uint16_t *d, uint16_t *a, uint16_t *b)
    {
        const uint32_t M = 64, N = 8, K = 16;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint16_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K16_F16BF16();
        LOAD_B_N8K16_F16BF16();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= 0; // f16_f16_f16
        MMA("f16");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    void tcgen05mma_f16_m64n8k16_f32_f16_f16(
        uint32_t *d, uint16_t *a, uint16_t *b)
    {
        tcgen05mma_f16_m64n8k16_f32_f16_f16_kernel<<<1, 128>>>(d, a, b);
    }

    void tcgen05mma_f16_m64n8k16_f32_bf16_bf16(
        uint32_t *d, uint16_t *a, uint16_t *b)
    {
        tcgen05mma_f16_m64n8k16_f32_bf16_bf16_kernel<<<1, 128>>>(d, a, b);
    }

    void tcgen05mma_f16_m64n8k16_f16_f16_f16(
        uint16_t *d, uint16_t *a, uint16_t *b)
    {
        tcgen05mma_f16_m64n8k16_f16_f16_f16_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // fp8 m64n8k32 f32_output
{
    __global__ void tcgen05mma_f8f6f4_m64n8k32_f32_e5m2_e5m2_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= (1 << 4) | (1 << 7) | (1 << 10); // f32_e5m2_e5m2
        MMA("f8f6f4");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_f8f6f4_m64n8k32_f32_e4m3_e4m3_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= 1 << 4; // f32_e4m3_e4m3
        MMA("f8f6f4");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    void tcgen05mma_f8f6f4_m64n8k32_f32_e5m2_e5m2(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_f8f6f4_m64n8k32_f32_e5m2_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void tcgen05mma_f8f6f4_m64n8k32_f32_e4m3_e4m3(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_f8f6f4_m64n8k32_f32_e4m3_e4m3_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // fp8 m64n8k32 f16_output
{
    __global__ void tcgen05mma_f8f6f4_m64n8k32_f16_e5m2_e5m2_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= (1 << 7) | (1 << 10); // f16_e5m2_e5m2
        MMA("f8f6f4");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_f8f6f4_m64n8k32_f16_e4m3_e4m3_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M64K32_FP8();
        LOAD_B_N8K32_FP8();
        ALLOC_TMEM(d_tmem_addr);
        __syncthreads();
        LOAD_D_M64N8();
        __syncthreads();
        i_desc |= 0; // f16_e4m3_e4m3
        MMA("f8f6f4");
        STORE_D_M64N8();
        DEALLOC_TMEM(d_tmem_addr);
        RELINQUISH_TMEM();
    }

    void tcgen05mma_f8f6f4_m64n8k32_f16_e5m2_e5m2(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_f8f6f4_m64n8k32_f16_e5m2_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void tcgen05mma_f8f6f4_m64n8k32_f16_e4m3_e4m3(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_f8f6f4_m64n8k32_f16_e4m3_e4m3_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // mxfp8
{
    __global__ void tcgen05mma_mxf8f6f4_m128n8k32_block32_f32_e5m2_e5m2_ue8m0_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t M = 128, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 7 << 27), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr, sfa_tmem_addr, sfb_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M128K32_FP8();
        LOAD_B_N8K32_FP8();
        ALLOC_TMEM(d_tmem_addr);
        ALLOC_TMEM(sfa_tmem_addr);
        ALLOC_TMEM(sfb_tmem_addr);
        __syncthreads();
        LOAD_D_M128N8();
        LOAD_SFA_M128SFK1();
        LOAD_SFB_N8SFK1();
        __syncthreads();
        i_desc |= (1 << 7) | (1 << 10) | (1 << 23); // e5m2_e5m2, ue8m0_scale
        MMA_WITH_BLOCK_SCALE("mxf8f6f4", "scale_vec::1X");
        STORE_D_M128N8();
        DEALLOC_TMEM(d_tmem_addr);
        DEALLOC_TMEM(sfa_tmem_addr);
        DEALLOC_TMEM(sfb_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_mxf8f6f4_m128n8k32_block32_f32_e4m3_e4m3_ue8m0_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t M = 128, N = 8, K = 32;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 7 << 27), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K], b_smem[N * K];
        __shared__ uint32_t d_tmem_addr, sfa_tmem_addr, sfb_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M128K32_FP8();
        LOAD_B_N8K32_FP8();
        ALLOC_TMEM(d_tmem_addr);
        ALLOC_TMEM(sfa_tmem_addr);
        ALLOC_TMEM(sfb_tmem_addr);
        __syncthreads();
        LOAD_D_M128N8();
        LOAD_SFA_M128SFK1();
        LOAD_SFB_N8SFK1();
        __syncthreads();
        i_desc |= 1 << 23; // e4m3_e4m3, ue8m0_scale
        MMA_WITH_BLOCK_SCALE("mxf8f6f4", "scale_vec::1X");
        STORE_D_M128N8();
        DEALLOC_TMEM(d_tmem_addr);
        DEALLOC_TMEM(sfa_tmem_addr);
        DEALLOC_TMEM(sfb_tmem_addr);
        RELINQUISH_TMEM();
    }

    void tcgen05mma_mxf8f6f4_m128n8k32_block32_f32_e5m2_e5m2_ue8m0(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        tcgen05mma_mxf8f6f4_m128n8k32_block32_f32_e5m2_e5m2_ue8m0_kernel<<<1, 128>>>(d, a, b, sfa, sfb);
    }

    void tcgen05mma_mxf8f6f4_m128n8k32_block32_f32_e4m3_e4m3_ue8m0(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        tcgen05mma_mxf8f6f4_m128n8k32_block32_f32_e4m3_e4m3_ue8m0_kernel<<<1, 128>>>(d, a, b, sfa, sfb);
    }
}

extern "C" // mxfp4
{
    __global__ void tcgen05mma_mxf4nvf4_m128n8k64_block32_f32_e2m1_e2m1_ue8m0_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t M = 128, N = 8, K = 64;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 7 << 27), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K / 2], b_smem[N * K / 2];
        __shared__ uint32_t d_tmem_addr, sfa_tmem_addr, sfb_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M128K64_FP4();
        LOAD_B_N8K64_FP4();
        ALLOC_TMEM(d_tmem_addr);
        ALLOC_TMEM(sfa_tmem_addr);
        ALLOC_TMEM(sfb_tmem_addr);
        __syncthreads();
        LOAD_D_M128N8();
        LOAD_SFA_M128SFK2();
        LOAD_SFB_N8SFK2();
        __syncthreads();
        i_desc |= (1 << 7) | (1 << 10) | (1 << 23); // e2m1, ue8m0_scale
        MMA_WITH_BLOCK_SCALE("mxf4nvf4", "scale_vec::2X");
        STORE_D_M128N8();
        DEALLOC_TMEM(d_tmem_addr);
        DEALLOC_TMEM(sfa_tmem_addr);
        DEALLOC_TMEM(sfb_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_mxf4nvf4_m128n8k64_block16_f32_e2m1_e2m1_ue8m0_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t M = 128, N = 8, K = 64;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 7 << 27), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K / 2], b_smem[N * K / 2];
        __shared__ uint32_t d_tmem_addr, sfa_tmem_addr, sfb_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M128K64_FP4();
        LOAD_B_N8K64_FP4();
        ALLOC_TMEM(d_tmem_addr);
        ALLOC_TMEM(sfa_tmem_addr);
        ALLOC_TMEM(sfb_tmem_addr);
        __syncthreads();
        LOAD_D_M128N8();
        LOAD_SFA_M128SFK4();
        LOAD_SFB_N8SFK4();
        __syncthreads();
        i_desc |= (1 << 7) | (1 << 10) | (1 << 23); // e2m1, ue8m0_scale
        MMA_WITH_BLOCK_SCALE("mxf4nvf4", "scale_vec::4X");
        STORE_D_M128N8();
        DEALLOC_TMEM(d_tmem_addr);
        DEALLOC_TMEM(sfa_tmem_addr);
        DEALLOC_TMEM(sfb_tmem_addr);
        RELINQUISH_TMEM();
    }

    __global__ void tcgen05mma_mxf4nvf4_m128n8k64_block16_f32_e2m1_e2m1_ue4m3_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t M = 128, N = 8, K = 64;
        uint32_t tid = threadIdx.x, warpid = tid / 32, laneid = tid % 32;
        uint32_t i_desc = (N >> 3 << 17) | (M >> 7 << 27), mma_barrier_phase_bit = 0;
        uint64_t a_desc, b_desc;
        __shared__ uint8_t a_smem[M * K / 2], b_smem[N * K / 2];
        __shared__ uint32_t d_tmem_addr, sfa_tmem_addr, sfb_tmem_addr;
        __shared__ uint64_t mma_barrier;

        INIT_MBARRIER();
        LOAD_A_M128K64_FP4();
        LOAD_B_N8K64_FP4();
        ALLOC_TMEM(d_tmem_addr);
        ALLOC_TMEM(sfa_tmem_addr);
        ALLOC_TMEM(sfb_tmem_addr);
        __syncthreads();
        LOAD_D_M128N8();
        LOAD_SFA_M128SFK4();
        LOAD_SFB_N8SFK4();
        __syncthreads();
        i_desc |= (1 << 7) | (1 << 10); // e2m1, ue4m3_scale
        MMA_WITH_BLOCK_SCALE("mxf4nvf4", "scale_vec::4X");
        STORE_D_M128N8();
        DEALLOC_TMEM(d_tmem_addr);
        DEALLOC_TMEM(sfa_tmem_addr);
        DEALLOC_TMEM(sfb_tmem_addr);
        RELINQUISH_TMEM();
    }

    void tcgen05mma_mxf4nvf4_m128n8k64_block32_f32_e2m1_e2m1_ue8m0(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        tcgen05mma_mxf4nvf4_m128n8k64_block32_f32_e2m1_e2m1_ue8m0_kernel<<<1, 128>>>(d, a, b, sfa, sfb);
    }

    void tcgen05mma_mxf4nvf4_m128n8k64_block16_f32_e2m1_e2m1_ue8m0(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        tcgen05mma_mxf4nvf4_m128n8k64_block16_f32_e2m1_e2m1_ue8m0_kernel<<<1, 128>>>(d, a, b, sfa, sfb);
    }

    void tcgen05mma_mxf4nvf4_m128n8k64_block16_f32_e2m1_e2m1_ue4m3(
        uint32_t *d, uint8_t *a, uint8_t *b, uint8_t *sfa, uint8_t *sfb)
    {
        tcgen05mma_mxf4nvf4_m128n8k64_block16_f32_e2m1_e2m1_ue4m3_kernel<<<1, 128>>>(d, a, b, sfa, sfb);
    }
}
