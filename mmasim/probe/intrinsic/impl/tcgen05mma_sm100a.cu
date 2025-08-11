#include <stdint.h>

#define desc_encode(x) (((x) & 0x3ffff) >> 4)

extern "C" // fp8 m64n8k32 f32_output
{
    __global__ void tcgen05mma_m64n8k32_f32_e5m2_e5m2_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t row, col, i;
        uint32_t tid = threadIdx.x;
        uint32_t warpid = tid / 32;
        uint32_t laneid = tid % 32;
        __shared__ uint8_t a_smem[M * K];
        __shared__ uint8_t b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        uint32_t d_frag[4];
        // load a
        for (i = 0; i < 8; i++)
        {
            for (uint32_t k = 0; k < 2; k++)
            {
                row = i * 8 + tid / 16;
                col = k * 16 + tid % 16;
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col];
            }
        }
        uint64_t a_desc = desc_encode(__cvta_generic_to_shared(a_smem));
        a_desc |= desc_encode(128ll) << 16;
        a_desc |= desc_encode(256ll) << 32;
        a_desc |= 1ll << 46;
        // load b
        for (uint32_t k = 0; k < 2; k++)
        {
            row = k * 16 + tid % 16;
            col = tid / 16;
            b_smem[k * 128 + tid] = b[col * K + row];
        }
        uint64_t b_desc = desc_encode(__cvta_generic_to_shared(b_smem));
        b_desc |= desc_encode(128ll) << 16;
        b_desc |= desc_encode(256ll) << 32;
        b_desc |= 1ll << 46;
        // load d
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&d_tmem_addr)));
        }
        __syncthreads();
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d_frag[i] = d[row * N + col];
        }
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 "
                     "[%0], {%1, %2, %3, %4};"
                     :
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)),
                       "r"(d_frag[0]), "r"(d_frag[1]), "r"(d_frag[2]), "r"(d_frag[3]));
        asm volatile("tcgen05.wait::st.sync.aligned;");
        __syncthreads();
        // mma
        uint32_t i_desc = (1 << 4) | (1 << 7) | (1 << 10);
        __shared__ uint64_t mma_barrier;
        const uint32_t mask[4] = {0, 0, 0, 0};
        i_desc |= N >> 3 << 17;
        i_desc |= M >> 4 << 24;
        if (tid == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :
                         : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        __syncthreads();
        uint32_t mma_barrier_phase_bit = 0;
        if (tid == 0)
        {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, 1;"
                :
                : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1    bra DONE;\n"
            "bra    LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)), "r"(mma_barrier_phase_bit));
        mma_barrier_phase_bit ^= 1;
        // store d
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(d_frag[0]), "=r"(d_frag[1]), "=r"(d_frag[2]), "=r"(d_frag[3])
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d[row * N + col] = d_frag[i];
        }
        __syncthreads();
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(d_tmem_addr));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    }

    __global__ void tcgen05mma_m64n8k32_f32_e4m3_e4m3_kernel(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t row, col, i;
        uint32_t tid = threadIdx.x;
        uint32_t warpid = tid / 32;
        uint32_t laneid = tid % 32;
        __shared__ uint8_t a_smem[M * K];
        __shared__ uint8_t b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        uint32_t d_frag[4];
        // load a
        for (i = 0; i < 8; i++)
        {
            for (uint32_t k = 0; k < 2; k++)
            {
                row = i * 8 + tid / 16;
                col = k * 16 + tid % 16;
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col];
            }
        }
        uint64_t a_desc = desc_encode(__cvta_generic_to_shared(a_smem));
        a_desc |= desc_encode(128ll) << 16;
        a_desc |= desc_encode(256ll) << 32;
        a_desc |= 1ll << 46;
        // load b
        for (uint32_t k = 0; k < 2; k++)
        {
            row = k * 16 + tid % 16;
            col = tid / 16;
            b_smem[k * 128 + tid] = b[col * K + row];
        }
        uint64_t b_desc = desc_encode(__cvta_generic_to_shared(b_smem));
        b_desc |= desc_encode(128ll) << 16;
        b_desc |= desc_encode(256ll) << 32;
        b_desc |= 1ll << 46;
        // load d
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&d_tmem_addr)));
        }
        __syncthreads();
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d_frag[i] = d[row * N + col];
        }
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 "
                     "[%0], {%1, %2, %3, %4};"
                     :
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)),
                       "r"(d_frag[0]), "r"(d_frag[1]), "r"(d_frag[2]), "r"(d_frag[3]));
        asm volatile("tcgen05.wait::st.sync.aligned;");
        __syncthreads();
        // mma
        uint32_t i_desc = 1 << 4;
        __shared__ uint64_t mma_barrier;
        const uint32_t mask[4] = {0, 0, 0, 0};
        i_desc |= N >> 3 << 17;
        i_desc |= M >> 4 << 24;
        if (tid == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :
                         : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        __syncthreads();
        uint32_t mma_barrier_phase_bit = 0;
        if (tid == 0)
        {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, 1;"
                :
                : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1    bra DONE;\n"
            "bra    LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)), "r"(mma_barrier_phase_bit));
        mma_barrier_phase_bit ^= 1;
        // store d
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(d_frag[0]), "=r"(d_frag[1]), "=r"(d_frag[2]), "=r"(d_frag[3])
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d[row * N + col] = d_frag[i];
        }
        __syncthreads();
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(d_tmem_addr));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    }

    void tcgen05mma_m64n8k32_f32_e5m2_e5m2(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_m64n8k32_f32_e5m2_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void tcgen05mma_m64n8k32_f32_e4m3_e4m3(
        uint32_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_m64n8k32_f32_e4m3_e4m3_kernel<<<1, 128>>>(d, a, b);
    }
}

extern "C" // fp8 m64n8k32 f16_output
{
    __global__ void tcgen05mma_m64n8k32_f16_e5m2_e5m2_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t row, col, i;
        uint32_t tid = threadIdx.x;
        uint32_t warpid = tid / 32;
        uint32_t laneid = tid % 32;
        __shared__ uint8_t a_smem[M * K];
        __shared__ uint8_t b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        uint32_t d_frag[4];
        // load a
        for (i = 0; i < 8; i++)
        {
            for (uint32_t k = 0; k < 2; k++)
            {
                row = i * 8 + tid / 16;
                col = k * 16 + tid % 16;
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col];
            }
        }
        uint64_t a_desc = desc_encode(__cvta_generic_to_shared(a_smem));
        a_desc |= desc_encode(128ll) << 16;
        a_desc |= desc_encode(256ll) << 32;
        a_desc |= 1ll << 46;
        // load b
        for (uint32_t k = 0; k < 2; k++)
        {
            row = k * 16 + tid % 16;
            col = tid / 16;
            b_smem[k * 128 + tid] = b[col * K + row];
        }
        uint64_t b_desc = desc_encode(__cvta_generic_to_shared(b_smem));
        b_desc |= desc_encode(128ll) << 16;
        b_desc |= desc_encode(256ll) << 32;
        b_desc |= 1ll << 46;
        // load d
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&d_tmem_addr)));
        }
        __syncthreads();
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d_frag[i] = d[row * N + col];
        }
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 "
                     "[%0], {%1, %2, %3, %4};"
                     :
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)),
                       "r"(d_frag[0]), "r"(d_frag[1]), "r"(d_frag[2]), "r"(d_frag[3]));
        asm volatile("tcgen05.wait::st.sync.aligned;");
        __syncthreads();
        // mma
        uint32_t i_desc = (1 << 7) | (1 << 10);
        __shared__ uint64_t mma_barrier;
        const uint32_t mask[4] = {0, 0, 0, 0};
        i_desc |= N >> 3 << 17;
        i_desc |= M >> 4 << 24;
        if (tid == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :
                         : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        __syncthreads();
        uint32_t mma_barrier_phase_bit = 0;
        if (tid == 0)
        {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, 1;"
                :
                : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1    bra DONE;\n"
            "bra    LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)), "r"(mma_barrier_phase_bit));
        mma_barrier_phase_bit ^= 1;
        // store d
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(d_frag[0]), "=r"(d_frag[1]), "=r"(d_frag[2]), "=r"(d_frag[3])
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d[row * N + col] = d_frag[i];
        }
        __syncthreads();
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(d_tmem_addr));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    }

    __global__ void tcgen05mma_m64n8k32_f16_e4m3_e4m3_kernel(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        const uint32_t M = 64, N = 8, K = 32;
        uint32_t row, col, i;
        uint32_t tid = threadIdx.x;
        uint32_t warpid = tid / 32;
        uint32_t laneid = tid % 32;
        __shared__ uint8_t a_smem[M * K];
        __shared__ uint8_t b_smem[N * K];
        __shared__ uint32_t d_tmem_addr;
        uint32_t d_frag[4];
        // load a
        for (i = 0; i < 8; i++)
        {
            for (uint32_t k = 0; k < 2; k++)
            {
                row = i * 8 + tid / 16;
                col = k * 16 + tid % 16;
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col];
            }
        }
        uint64_t a_desc = desc_encode(__cvta_generic_to_shared(a_smem));
        a_desc |= desc_encode(128ll) << 16;
        a_desc |= desc_encode(256ll) << 32;
        a_desc |= 1ll << 46;
        // load b
        for (uint32_t k = 0; k < 2; k++)
        {
            row = k * 16 + tid % 16;
            col = tid / 16;
            b_smem[k * 128 + tid] = b[col * K + row];
        }
        uint64_t b_desc = desc_encode(__cvta_generic_to_shared(b_smem));
        b_desc |= desc_encode(128ll) << 16;
        b_desc |= desc_encode(256ll) << 32;
        b_desc |= 1ll << 46;
        // load d
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&d_tmem_addr)));
        }
        __syncthreads();
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d_frag[i] = d[row * N + col];
        }
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 "
                     "[%0], {%1, %2, %3, %4};"
                     :
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)),
                       "r"(d_frag[0]), "r"(d_frag[1]), "r"(d_frag[2]), "r"(d_frag[3]));
        asm volatile("tcgen05.wait::st.sync.aligned;");
        __syncthreads();
        // mma
        uint32_t i_desc = 0;
        __shared__ uint64_t mma_barrier;
        const uint32_t mask[4] = {0, 0, 0, 0};
        i_desc |= N >> 3 << 17;
        i_desc |= M >> 4 << 24;
        if (tid == 0)
        {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :
                         : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        __syncthreads();
        uint32_t mma_barrier_phase_bit = 0;
        if (tid == 0)
        {
            asm volatile(
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%4, %5, %6, %7}, 1;"
                :
                : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));
        }
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1    bra DONE;\n"
            "bra    LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)), "r"(mma_barrier_phase_bit));
        mma_barrier_phase_bit ^= 1;
        // store d
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(d_frag[0]), "=r"(d_frag[1]), "=r"(d_frag[2]), "=r"(d_frag[3])
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        for (i = 0; i < 4; i++)
        {
            row = warpid * 16 + i / 2 * 8 + laneid / 4;
            col = laneid % 4 * 2 + i % 2;
            d[row * N + col] = d_frag[i];
        }
        __syncthreads();
        if (warpid == 0)
        {
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" : : "r"(d_tmem_addr));
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        }
    }

    void tcgen05mma_m64n8k32_f16_e5m2_e5m2(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_m64n8k32_f16_e5m2_e5m2_kernel<<<1, 128>>>(d, a, b);
    }

    void tcgen05mma_m64n8k32_f16_e4m3_e4m3(
        uint16_t *d, uint8_t *a, uint8_t *b)
    {
        tcgen05mma_m64n8k32_f16_e4m3_e4m3_kernel<<<1, 128>>>(d, a, b);
    }
}