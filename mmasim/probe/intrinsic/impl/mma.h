#include <stdint.h>

#define LOAD_A_M16K8_TF32()                               \
    do                                                    \
    {                                                     \
        for (i = 0; i < 4; i++)                           \
        {                                                 \
            row = i % 2 * 8 + laneid / 4;                 \
            col = i / 2 * 4 + laneid % 4;                 \
            a_frag[i] = *(uint32_t *)(a + row * K + col); \
        }                                                 \
    } while (0)

#define LOAD_A_M16K4_TF32()                               \
    do                                                    \
    {                                                     \
        for (i = 0; i < 2; i++)                           \
        {                                                 \
            row = i * 8 + laneid / 4;                     \
            col = laneid % 4;                             \
            a_frag[i] = *(uint32_t *)(a + row * K + col); \
        }                                                 \
    } while (0)

#define LOAD_A_M16K16_F16BF16()                               \
    do                                                        \
    {                                                         \
        for (i = 0; i < 8; i += 2)                            \
        {                                                     \
            row = i % 4 / 2 * 8 + laneid / 4;                 \
            col = i / 4 * 8 + laneid % 4 * 2;                 \
            a_frag[i / 2] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M16K8_F16BF16()                                \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = i / 2 * 8 + laneid / 4;                     \
            col = laneid % 4 * 2;                             \
            a_frag[i / 2] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M8K4_X4_F16()                                  \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = laneid / 16 * 4 + laneid % 4;               \
            col = i;                                          \
            a_frag[i / 2] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M16K32_FP8()                                   \
    do                                                        \
    {                                                         \
        for (i = 0; i < 16; i += 4)                           \
        {                                                     \
            row = i % 8 / 4 * 8 + laneid / 4;                 \
            col = i / 8 * 16 + laneid % 4 * 4 + i % 4;        \
            a_frag[i / 4] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M16K16_FP8()                                   \
    do                                                        \
    {                                                         \
        for (i = 0; i < 8; i += 4)                            \
        {                                                     \
            row = i / 4 * 8 + laneid / 4;                     \
            col = laneid % 4 * 4 + i % 4;                     \
            a_frag[i / 4] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_B_N8K8_TF32()                                \
    do                                                    \
    {                                                     \
        for (i = 0; i < 2; i++)                           \
        {                                                 \
            row = i * 4 + laneid % 4;                     \
            col = laneid / 4;                             \
            b_frag[i] = *(uint32_t *)(b + col * K + row); \
        }                                                 \
    } while (0)

#define LOAD_B_N8K4_TF32()                            \
    do                                                \
    {                                                 \
        row = laneid % 4;                             \
        col = laneid / 4;                             \
        b_frag[0] = *(uint32_t *)(b + col * K + row); \
    } while (0)

#define LOAD_B_N8K16_F16BF16()                                \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = i / 2 * 8 + laneid % 4 * 2;                 \
            col = laneid / 4;                                 \
            b_frag[i / 2] = *(uint32_t *)(b + col * K + row); \
        }                                                     \
    } while (0)

#define LOAD_B_N8K8_F16BF16()                         \
    do                                                \
    {                                                 \
        row = laneid % 4 * 2;                         \
        col = laneid / 4;                             \
        b_frag[0] = *(uint32_t *)(b + col * K + row); \
    } while (0)

#define LOAD_B_N8K4_X4_F16()                                  \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = i;                                          \
            col = laneid / 16 * 4 + laneid % 4;               \
            b_frag[i / 2] = *(uint32_t *)(b + col * K + row); \
        }                                                     \
    } while (0)

#define LOAD_B_N8K32_FP8()                                    \
    do                                                        \
    {                                                         \
        for (i = 0; i < 8; i += 4)                            \
        {                                                     \
            row = i / 4 * 16 + laneid % 4 * 4 + i % 4;        \
            col = laneid / 4;                                 \
            b_frag[i / 4] = *(uint32_t *)(b + col * K + row); \
        }                                                     \
    } while (0)

#define LOAD_B_N8K16_FP8()                            \
    do                                                \
    {                                                 \
        row = laneid % 4 * 4;                         \
        col = laneid / 4;                             \
        b_frag[0] = *(uint32_t *)(b + col * K + row); \
    } while (0)

#define LOAD_C_M16N8_F32()                \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i / 2 * 8 + laneid / 4; \
            col = laneid % 4 * 2 + i % 2; \
            c_frag[i] = c[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_C_M16N8_F16()                                    \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = i / 2 * 8 + laneid / 4;                     \
            col = laneid % 4 * 2;                             \
            c_frag[i / 2] = *(uint32_t *)(c + row * N + col); \
        }                                                     \
    } while (0)

#define LOAD_C_M8N8_X4_F32()                                \
    do                                                      \
    {                                                       \
        for (i = 0; i < 8; i++)                             \
        {                                                   \
            row = laneid / 16 * 4 + (i & 2) + (laneid & 1); \
            col = (i & 4) + (laneid & 2) + (i & 1);         \
            c_frag[i] = c[row * N + col];                   \
        }                                                   \
    } while (0)

#define LOAD_C_M8N8_X4_F16()                                  \
    do                                                        \
    {                                                         \
        for (i = 0; i < 8; i += 2)                            \
        {                                                     \
            row = laneid / 16 * 4 + laneid % 4;               \
            col = i;                                          \
            c_frag[i / 2] = *(uint32_t *)(c + row * N + col); \
        }                                                     \
    } while (0)

#define STORE_D_M16N8_F32()               \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i / 2 * 8 + laneid / 4; \
            col = laneid % 4 * 2 + i % 2; \
            d[row * N + col] = d_frag[i]; \
        }                                 \
    } while (0)

#define STORE_D_M16N8_F16()                                   \
    do                                                        \
    {                                                         \
        for (i = 0; i < 4; i += 2)                            \
        {                                                     \
            row = i / 2 * 8 + laneid / 4;                     \
            col = laneid % 4 * 2;                             \
            *(uint32_t *)(d + row * N + col) = d_frag[i / 2]; \
        }                                                     \
    } while (0)

#define STORE_D_M8N8_X4_F32()                               \
    do                                                      \
    {                                                       \
        for (i = 0; i < 8; i++)                             \
        {                                                   \
            row = laneid / 16 * 4 + (i & 2) + (laneid & 1); \
            col = (i & 4) + (laneid & 2) + (i & 1);         \
            d[row * N + col] = d_frag[i];                   \
        }                                                   \
    } while (0)

#define STORE_D_M8N8_X4_F16()                                 \
    do                                                        \
    {                                                         \
        for (i = 0; i < 8; i += 2)                            \
        {                                                     \
            row = laneid / 16 * 4 + laneid % 4;               \
            col = i;                                          \
            *(uint32_t *)(d + row * N + col) = d_frag[i / 2]; \
        }                                                     \
    } while (0)
