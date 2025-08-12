#include <stdint.h>

#define desc_encode(x) (((x) & 0x3ffff) >> 4)

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
    } while (0)

// (8x2) tiles, each with (8x16) elements
#define LOAD_A_M64K32_FP8()                                         \
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
    } while (0)

// (2x1) tiles, each with (16x8) elements
#define LOAD_B_N8K32_FP8()                                      \
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
    } while (0)

#define LOAD_D_M64N8_F32()                              \
    do                                                  \
    {                                                   \
        for (i = 0; i < 4; i++)                         \
        {                                               \
            row = warpid * 16 + i / 2 * 8 + laneid / 4; \
            col = laneid % 4 * 2 + i % 2;               \
            d_frag[i] = d[row * N + col];               \
        }                                               \
    } while (0)

#define LOAD_D_M64N8_F16()                                    \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = warpid * 16 + i / 2 * 8 + laneid / 4;       \
            col = laneid % 4 * 2;                             \
            d_frag[i / 2] = *(uint32_t *)(d + row * N + col); \
        }                                                     \
    } while (0)

#define STORE_D_M64N8_F32()                             \
    do                                                  \
    {                                                   \
        for (i = 0; i < 4; i++)                         \
        {                                               \
            row = warpid * 16 + i / 2 * 8 + laneid / 4; \
            col = laneid % 4 * 2 + i % 2;               \
            d[row * N + col] = d_frag[i];               \
        }                                               \
    } while (0)

#define STORE_D_M64N8_F16()                                   \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = warpid * 16 + i / 2 * 8 + laneid / 4;       \
            col = laneid % 4 * 2;                             \
            *(uint32_t *)(d + row * N + col) = d_frag[i / 2]; \
        }                                                     \
    } while (0)
