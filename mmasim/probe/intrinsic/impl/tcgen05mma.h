#include <stdint.h>

#define desc_encode(x) (((x) & 0x3ffff) >> 4)

#define ALLOC_TMEM()                                                      \
    do                                                                    \
    {                                                                     \
        if (warpid == 0)                                                  \
        {                                                                 \
            asm volatile(                                                 \
                "tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], 32;"   \
                :                                                         \
                : "r"((uint32_t)__cvta_generic_to_shared(&d_tmem_addr))); \
        }                                                                 \
    } while (0)

#define DEALLOC_TMEM()                                                                  \
    do                                                                                  \
    {                                                                                   \
        if (warpid == 0)                                                                \
        {                                                                               \
            asm volatile(                                                               \
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"                 \
                :                                                                       \
                : "r"(d_tmem_addr));                                                    \
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
        for (i = 0; i < 8; i += 2)                              \
        {                                                       \
            row = i * 8 + warpid / 2 * 8 + laneid / 4;          \
            col = warpid % 2 * 4 + laneid % 4;                  \
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
        for (i = 0; i < 8; i++)                                 \
        {                                                       \
            row = i * 8 + tid % 64 / 8;                         \
            col = tid / 64 * 8 + tid % 8;                       \
            a_smem[i * 128 + tid] = a[row * K + col];           \
        }                                                       \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem)); \
        a_desc |= desc_encode(128ll) << 16;                     \
        a_desc |= desc_encode(256ll) << 32;                     \
        a_desc |= 1ll << 46;                                    \
    } while (0)

// (8x2) tiles, each with (8x16) elements
#define LOAD_A_M64K32_F8F6F4()                                      \
    do                                                              \
    {                                                               \
        for (i = 0; i < 8; i++)                                     \
        {                                                           \
            for (uint32_t k = 0; k < 2; k++)                        \
            {                                                       \
                row = i * 8 + tid / 16;                             \
                col = k * 16 + tid % 16;                            \
                a_smem[i * 256 + k * 128 + tid] = a[row * K + col]; \
            }                                                       \
        }                                                           \
        a_desc = desc_encode(__cvta_generic_to_shared(a_smem));     \
        a_desc |= desc_encode(128ll) << 16;                         \
        a_desc |= desc_encode(256ll) << 32;                         \
        a_desc |= 1ll << 46;                                        \
    } while (0)

// (2x1) tiles, each with (4x8) elements
#define LOAD_B_N8K8_TF32()                                      \
    do                                                          \
    {                                                           \
        if (tid < 64)                                           \
        {                                                       \
            row = warpid * 4 + laneid % 4;                      \
            col = laneid / 4;                                   \
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
        row = tid / 64 * 8 + tid % 8;                           \
        col = tid % 64 / 8;                                     \
        b_smem[tid] = b[col * K + row];                         \
        b_desc = desc_encode(__cvta_generic_to_shared(b_smem)); \
        b_desc |= desc_encode(128ll) << 16;                     \
        b_desc |= desc_encode(256ll) << 32;                     \
        b_desc |= 1ll << 46;                                    \
    } while (0)

// (2x1) tiles, each with (16x8) elements
#define LOAD_B_N8K32_F8F6F4()                                   \
    do                                                          \
    {                                                           \
        for (uint32_t k = 0; k < 2; k++)                        \
        {                                                       \
            row = k * 16 + tid % 16;                            \
            col = tid / 16;                                     \
            b_smem[k * 128 + tid] = b[col * K + row];           \
        }                                                       \
        b_desc = desc_encode(__cvta_generic_to_shared(b_smem)); \
        b_desc |= desc_encode(128ll) << 16;                     \
        b_desc |= desc_encode(256ll) << 32;                     \
        b_desc |= 1ll << 46;                                    \
    } while (0)

#define LOAD_D_M64N8()                                           \
    do                                                           \
    {                                                            \
        for (i = 0; i < 4; i++)                                  \
        {                                                        \
            row = warpid * 16 + i / 2 * 8 + laneid / 4;          \
            col = laneid % 4 * 2 + i % 2;                        \
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

#define STORE_D_M64N8()                                           \
    do                                                            \
    {                                                             \
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 "    \
                     "{%0, %1, %2, %3}, [%4];"                    \
                     : "=r"(d_frag[0]), "=r"(d_frag[1]),          \
                       "=r"(d_frag[2]), "=r"(d_frag[3])           \
                     : "r"(d_tmem_addr + ((warpid * 32) << 16))); \
        asm volatile("tcgen05.wait::ld.sync.aligned;");           \
        for (i = 0; i < 4; i++)                                   \
        {                                                         \
            row = warpid * 16 + i / 2 * 8 + laneid / 4;           \
            col = laneid % 4 * 2 + i % 2;                         \
            d[row * N + col] = d_frag[i];                         \
        }                                                         \
        __syncthreads();                                          \
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
