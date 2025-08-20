#include <stdint.h>

#define desc_encode(x) (((x) & 0x3ffff) >> 4)

#define ALLOC_TMEM(tmem_addr)                                           \
    do                                                                  \
    {                                                                   \
        if (warpid == 0)                                                \
        {                                                               \
            asm volatile(                                               \
                "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;" \
                :                                                       \
                : "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr))); \
        }                                                               \
    } while (0)

#define DEALLOC_TMEM(tmem_addr)                                         \
    do                                                                  \
    {                                                                   \
        if (warpid == 0)                                                \
        {                                                               \
            asm volatile(                                               \
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" \
                :                                                       \
                : "r"(tmem_addr));                                      \
        }                                                               \
    } while (0)

#define RELINQUISH_TMEM()                                                               \
    do                                                                                  \
    {                                                                                   \
        if (warpid == 0)                                                                \
        {                                                                               \
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); \
        }                                                                               \
    } while (0)

#define INIT_MBARRIER()                                                            \
    do                                                                             \
    {                                                                              \
        if (tid == 0)                                                              \
        {                                                                          \
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"                  \
                         :                                                         \
                         : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier))); \
        }                                                                          \
    } while (0)

#define WAIT_MBARRIER()                                                \
    do                                                                 \
    {                                                                  \
        asm volatile(                                                  \
            "{\n"                                                      \
            ".reg .pred P1;\n"                                         \
            "LAB_WAIT:\n"                                              \
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n" \
            "@P1    bra DONE;\n"                                       \
            "bra    LAB_WAIT;\n"                                       \
            "DONE:\n"                                                  \
            "}\n"                                                      \
            :                                                          \
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)),   \
              "r"(mma_barrier_phase_bit));                             \
        mma_barrier_phase_bit ^= 1;                                    \
    } while (0)

// (8x2) tiles, each with (8x4) elements
#define LOAD_A_M64K8_TF32()                                     \
    do                                                          \
    {                                                           \
        for (uint32_t i = 0; i < 8; i += 2)                     \
        {                                                       \
            uint32_t row = i * 8 + warpid / 2 * 8 + laneid / 4; \
            uint32_t col = warpid % 2 * 4 + laneid % 4;         \
            a_smem[i * 64 + tid] = a[row * K + col];            \
        }                                                       \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem)); \
        a_desc |= desc_encode(128ll) << 16;                     \
        a_desc |= desc_encode(256ll) << 32;                     \
        a_desc |= 1ll << 46;                                    \
    } while (0)

// (8x2) tiles, each with (8x8) elements
#define LOAD_A_M64K16_F16BF16()                                 \
    do                                                          \
    {                                                           \
        for (uint32_t i = 0; i < 8; i++)                        \
        {                                                       \
            uint32_t row = i * 8 + tid % 64 / 8;                \
            uint32_t col = tid / 64 * 8 + tid % 8;              \
            a_smem[i * 128 + tid] = a[row * K + col];           \
        }                                                       \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem)); \
        a_desc |= desc_encode(128ll) << 16;                     \
        a_desc |= desc_encode(256ll) << 32;                     \
        a_desc |= 1ll << 46;                                    \
    } while (0)

// (16x2) tiles, each with (8x16) elements
#define LOAD_A_M128K32_FP8()                                        \
    do                                                              \
    {                                                               \
        for (uint32_t i = 0; i < 16; i++)                           \
        {                                                           \
            for (uint32_t k = 0; k < 2; k++)                        \
            {                                                       \
                uint32_t row = i * 8 + tid / 16;                    \
                uint32_t col = k * 16 + tid % 16;                   \
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col]; \
            }                                                       \
        }                                                           \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem));     \
        a_desc |= desc_encode(128ll) << 16;                         \
        a_desc |= desc_encode(256ll) << 32;                         \
        a_desc |= 1ll << 46;                                        \
    } while (0)

// (8x2) tiles, each with (8x16) elements
#define LOAD_A_M64K32_FP8()                                         \
    do                                                              \
    {                                                               \
        for (uint32_t i = 0; i < 8; i++)                            \
        {                                                           \
            for (uint32_t k = 0; k < 2; k++)                        \
            {                                                       \
                uint32_t row = i * 8 + tid / 16;                    \
                uint32_t col = k * 16 + tid % 16;                   \
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col]; \
            }                                                       \
        }                                                           \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem));     \
        a_desc |= desc_encode(128ll) << 16;                         \
        a_desc |= desc_encode(256ll) << 32;                         \
        a_desc |= 1ll << 46;                                        \
    } while (0)

// (16x2) tiles, each with (8x32) elements
#define LOAD_A_M128K64_FP4()                                            \
    do                                                                  \
    {                                                                   \
        for (uint32_t i = 0; i < 16; i++)                               \
        {                                                               \
            for (uint32_t k = 0; k < 2; k++)                            \
            {                                                           \
                uint32_t row = i * 8 + tid / 16;                        \
                uint32_t col = k * 16 + tid % 16;                       \
                a_smem[i * 256 + k * 128 + tid] = a[row * K / 2 + col]; \
            }                                                           \
        }                                                               \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem));         \
        a_desc |= desc_encode(128ll) << 16;                             \
        a_desc |= desc_encode(256ll) << 32;                             \
        a_desc |= 1ll << 46;                                            \
    } while (0)

// (2x1) tiles, each with (4x8) elements
#define LOAD_B_N8K8_TF32()                                      \
    do                                                          \
    {                                                           \
        if (tid < 64)                                           \
        {                                                       \
            uint32_t row = warpid * 4 + laneid % 4;             \
            uint32_t col = laneid / 4;                          \
            b_smem[tid] = b[col * K + row];                     \
        }                                                       \
        b_desc = desc_encode(__cvta_generic_to_shared(b_smem)); \
        b_desc |= desc_encode(128ll) << 16;                     \
        b_desc |= desc_encode(256ll) << 32;                     \
        b_desc |= 1ll << 46;                                    \
    } while (0)

// (2x1) tiles, each with (8x8) elements
#define LOAD_B_N8K16_F16BF16()                                  \
    do                                                          \
    {                                                           \
        uint32_t row = tid / 64 * 8 + tid % 8;                  \
        uint32_t col = tid % 64 / 8;                            \
        b_smem[tid] = b[col * K + row];                         \
        b_desc = desc_encode(__cvta_generic_to_shared(b_smem)); \
        b_desc |= desc_encode(128ll) << 16;                     \
        b_desc |= desc_encode(256ll) << 32;                     \
        b_desc |= 1ll << 46;                                    \
    } while (0)

// (2x1) tiles, each with (16x8) elements
#define LOAD_B_N8K32_FP8()                                      \
    do                                                          \
    {                                                           \
        for (uint32_t k = 0; k < 2; k++)                        \
        {                                                       \
            uint32_t row = k * 16 + tid % 16;                   \
            uint32_t col = tid / 16;                            \
            b_smem[k * 128 + tid] = b[col * K + row];           \
        }                                                       \
        b_desc = desc_encode(__cvta_generic_to_shared(b_smem)); \
        b_desc |= desc_encode(128ll) << 16;                     \
        b_desc |= desc_encode(256ll) << 32;                     \
        b_desc |= 1ll << 46;                                    \
    } while (0)

// (2x1) tiles, each with (32x8) elements
#define LOAD_B_N8K64_FP4()                                      \
    do                                                          \
    {                                                           \
        for (uint32_t k = 0; k < 2; k++)                        \
        {                                                       \
            uint32_t row = k * 16 + tid % 16;                   \
            uint32_t col = tid / 16;                            \
            b_smem[k * 128 + tid] = b[col * K / 2 + row];       \
        }                                                       \
        b_desc = desc_encode(__cvta_generic_to_shared(b_smem)); \
        b_desc |= desc_encode(128ll) << 16;                     \
        b_desc |= desc_encode(256ll) << 32;                     \
        b_desc |= 1ll << 46;                                    \
    } while (0)

#define LOAD_D_M128N8()                                                                 \
    do                                                                                  \
    {                                                                                   \
        uint32_t d_frag[8];                                                             \
        for (uint32_t i = 0; i < 8; i++)                                                \
        {                                                                               \
            uint32_t row = tid;                                                         \
            uint32_t col = i;                                                           \
            d_frag[i] = d[row * N + col];                                               \
        }                                                                               \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 "                           \
                     "[%0], {%1, %2, %3, %4, %5, %6, %7, %8};"                          \
                     :                                                                  \
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)),                        \
                       "r"(d_frag[0]), "r"(d_frag[1]), "r"(d_frag[2]), "r"(d_frag[3]),  \
                       "r"(d_frag[4]), "r"(d_frag[5]), "r"(d_frag[6]), "r"(d_frag[7])); \
        asm volatile("tcgen05.wait::st.sync.aligned;");                                 \
    } while (0)

#define LOAD_D_M64N8()                                           \
    do                                                           \
    {                                                            \
        uint32_t d_frag[4];                                      \
        for (uint32_t i = 0; i < 4; i++)                         \
        {                                                        \
            uint32_t row = warpid * 16 + i / 2 * 8 + laneid / 4; \
            uint32_t col = laneid % 4 * 2 + i % 2;               \
            d_frag[i] = d[row * N + col];                        \
        }                                                        \
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 "   \
                     "[%0], {%1, %2, %3, %4};"                   \
                     :                                           \
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)), \
                       "r"(d_frag[0]), "r"(d_frag[1]),           \
                       "r"(d_frag[2]), "r"(d_frag[3]));          \
        asm volatile("tcgen05.wait::st.sync.aligned;");          \
    } while (0)

#define STORE_D_M128N8()                                                                   \
    do                                                                                     \
    {                                                                                      \
        uint32_t d_frag[8];                                                                \
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "                              \
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"                             \
                     : "=r"(d_frag[0]), "=r"(d_frag[1]), "=r"(d_frag[2]), "=r"(d_frag[3]), \
                       "=r"(d_frag[4]), "=r"(d_frag[5]), "=r"(d_frag[6]), "=r"(d_frag[7])  \
                     : "r"(d_tmem_addr + ((warpid * 32) << 16)));                          \
        asm volatile("tcgen05.wait::ld.sync.aligned;");                                    \
        for (uint32_t i = 0; i < 8; i++)                                                   \
        {                                                                                  \
            uint32_t row = tid;                                                            \
            uint32_t col = i;                                                              \
            d[row * N + col] = d_frag[i];                                                  \
        }                                                                                  \
    } while (0)

#define STORE_D_M64N8()                                           \
    do                                                            \
    {                                                             \
        uint32_t d_frag[4];                                       \
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "    \
                     "{%0, %1, %2, %3}, [%4];"                    \
                     : "=r"(d_frag[0]), "=r"(d_frag[1]),          \
                       "=r"(d_frag[2]), "=r"(d_frag[3])           \
                     : "r"(d_tmem_addr + ((warpid * 32) << 16))); \
        asm volatile("tcgen05.wait::ld.sync.aligned;");           \
        for (uint32_t i = 0; i < 4; i++)                          \
        {                                                         \
            uint32_t row = warpid * 16 + i / 2 * 8 + laneid / 4;  \
            uint32_t col = laneid % 4 * 2 + i % 2;                \
            d[row * N + col] = d_frag[i];                         \
        }                                                         \
        __syncthreads();                                          \
    } while (0)

#define LOAD_SFA_M128SFK1()                                                          \
    do                                                                               \
    {                                                                                \
        uint32_t sfa_frag[4];                                                        \
        for (uint32_t i = 0; i < 4; i++)                                             \
        {                                                                            \
            uint32_t row = i * 32 + laneid;                                          \
            sfa_frag[i] = sfa[row];                                                  \
        }                                                                            \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" \
                     :                                                               \
                     : "r"(sfa_tmem_addr + ((warpid * 32) << 16)),                   \
                       "r"(sfa_frag[0]), "r"(sfa_frag[1]),                           \
                       "r"(sfa_frag[2]), "r"(sfa_frag[3]));                          \
        asm volatile("tcgen05.wait::st.sync.aligned;");                              \
    } while (0)

#define LOAD_SFA_M128SFK2()                                                          \
    do                                                                               \
    {                                                                                \
        uint32_t sfa_frag[4];                                                        \
        for (uint32_t i = 0; i < 4; i++)                                             \
        {                                                                            \
            uint32_t row = i * 32 + laneid;                                          \
            sfa_frag[i] = *(uint16_t *)(sfa + row * 2);                              \
        }                                                                            \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" \
                     :                                                               \
                     : "r"(sfa_tmem_addr + ((warpid * 32) << 16)),                   \
                       "r"(sfa_frag[0]), "r"(sfa_frag[1]),                           \
                       "r"(sfa_frag[2]), "r"(sfa_frag[3]));                          \
        asm volatile("tcgen05.wait::st.sync.aligned;");                              \
    } while (0)

#define LOAD_SFA_M128SFK4()                                                          \
    do                                                                               \
    {                                                                                \
        uint32_t sfa_frag[4];                                                        \
        for (uint32_t i = 0; i < 4; i++)                                             \
        {                                                                            \
            uint32_t row = i * 32 + laneid;                                          \
            sfa_frag[i] = *(uint32_t *)(sfa + row * 4);                              \
        }                                                                            \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};" \
                     :                                                               \
                     : "r"(sfa_tmem_addr + ((warpid * 32) << 16)),                   \
                       "r"(sfa_frag[0]), "r"(sfa_frag[1]),                           \
                       "r"(sfa_frag[2]), "r"(sfa_frag[3]));                          \
        asm volatile("tcgen05.wait::st.sync.aligned;");                              \
    } while (0)

#define LOAD_SFB_N8SFK1()                                                \
    do                                                                   \
    {                                                                    \
        uint32_t sfb_frag[1];                                            \
        uint32_t col = laneid % 8;                                       \
        sfb_frag[0] = sfb[col];                                          \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" \
                     :                                                   \
                     : "r"(sfb_tmem_addr + ((warpid * 32) << 16)),       \
                       "r"(sfb_frag[0]));                                \
        asm volatile("tcgen05.wait::st.sync.aligned;");                  \
    } while (0)

#define LOAD_SFB_N8SFK2()                                                \
    do                                                                   \
    {                                                                    \
        uint32_t sfb_frag[1];                                            \
        uint32_t col = laneid % 8;                                       \
        sfb_frag[0] = *(uint16_t *)(sfb + col * 2);                      \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" \
                     :                                                   \
                     : "r"(sfb_tmem_addr + ((warpid * 32) << 16)),       \
                       "r"(sfb_frag[0]));                                \
        asm volatile("tcgen05.wait::st.sync.aligned;");                  \
    } while (0)

#define LOAD_SFB_N8SFK4()                                                \
    do                                                                   \
    {                                                                    \
        uint32_t sfb_frag[1];                                            \
        uint32_t col = laneid % 8;                                       \
        sfb_frag[0] = *(uint32_t *)(sfb + col * 4);                      \
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" \
                     :                                                   \
                     : "r"(sfb_tmem_addr + ((warpid * 32) << 16)),       \
                       "r"(sfb_frag[0]));                                \
        asm volatile("tcgen05.wait::st.sync.aligned;");                  \
    } while (0)

#define MMA(kind)                                                                                 \
    do                                                                                            \
    {                                                                                             \
        if (tid == 0)                                                                             \
        {                                                                                         \
            const uint32_t mask[4] = {0, 0, 0, 0};                                                \
            asm volatile(                                                                         \
                "tcgen05.mma.cta_group::1.kind::" kind " [%0], %1, %2, %3, {%4, %5, %6, %7}, 1;"  \
                :                                                                                 \
                : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc),                        \
                  "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));                        \
            asm volatile(                                                                         \
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"     \
                :                                                                                 \
                : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));                         \
        }                                                                                         \
        asm volatile(                                                                             \
            "{\n"                                                                                 \
            ".reg .pred P1;\n"                                                                    \
            "LAB_WAIT:\n"                                                                         \
            "       mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"                     \
            "@P1    bra DONE;\n"                                                                  \
            "       bra LAB_WAIT;\n"                                                              \
            "DONE:\n"                                                                             \
            "}\n"                                                                                 \
            :                                                                                     \
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)), "r"(mma_barrier_phase_bit)); \
        mma_barrier_phase_bit ^= 1;                                                               \
    } while (0)

#define MMA_WITH_BLOCK_SCALE(kind, block_size)                                                    \
    do                                                                                            \
    {                                                                                             \
        if (tid == 0)                                                                             \
        {                                                                                         \
            asm volatile(                                                                         \
                "tcgen05.mma.cta_group::1.kind::" kind ".block_scale." block_size " "             \
                "[%0], %1, %2, %3, [%4], [%5], 1;"                                                \
                :                                                                                 \
                : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc),                        \
                  "r"(sfa_tmem_addr), "r"(sfb_tmem_addr));                                        \
            asm volatile(                                                                         \
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"     \
                :                                                                                 \
                : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)));                         \
        }                                                                                         \
        asm volatile(                                                                             \
            "{\n"                                                                                 \
            ".reg .pred P1;\n"                                                                    \
            "LAB_WAIT:\n"                                                                         \
            "       mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"                     \
            "@P1    bra DONE;\n"                                                                  \
            "       bra LAB_WAIT;\n"                                                              \
            "DONE:\n"                                                                             \
            "}\n"                                                                                 \
            :                                                                                     \
            : "r"((uint32_t)__cvta_generic_to_shared(&mma_barrier)), "r"(mma_barrier_phase_bit)); \
        mma_barrier_phase_bit ^= 1;                                                               \
    } while (0)
