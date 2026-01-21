#include <stdint.h>

#define LOAD_A_M16K16_F64()                        \
    do                                             \
    {                                              \
        for (uint32_t i = 0; i < 8; i++)           \
        {                                          \
            uint32_t row = i % 2 * 8 + laneid / 4; \
            uint32_t col = i / 2 * 4 + laneid % 4; \
            a_frag[i] = a[row * K + col];          \
        }                                          \
    } while (0)

#define LOAD_A_M16K8_F64()                         \
    do                                             \
    {                                              \
        for (uint32_t i = 0; i < 4; i++)           \
        {                                          \
            uint32_t row = i % 2 * 8 + laneid / 4; \
            uint32_t col = i / 2 * 4 + laneid % 4; \
            a_frag[i] = a[row * K + col];          \
        }                                          \
    } while (0)

#define LOAD_A_M16K4_F64()                     \
    do                                         \
    {                                          \
        for (uint32_t i = 0; i < 2; i++)       \
        {                                      \
            uint32_t row = i * 8 + laneid / 4; \
            uint32_t col = laneid % 4;         \
            a_frag[i] = a[row * K + col];      \
        }                                      \
    } while (0)

#define LOAD_A_M8K4_F64()             \
    do                                \
    {                                 \
        uint32_t row = laneid / 4;    \
        uint32_t col = laneid % 4;    \
        a_frag[0] = a[row * K + col]; \
    } while (0)

#define LOAD_A_M16K8_TF32()                               \
    do                                                    \
    {                                                     \
        for (uint32_t i = 0; i < 4; i++)                  \
        {                                                 \
            uint32_t row = i % 2 * 8 + laneid / 4;        \
            uint32_t col = i / 2 * 4 + laneid % 4;        \
            a_frag[i] = *(uint32_t *)(a + row * K + col); \
        }                                                 \
    } while (0)

#define LOAD_A_M16K4_TF32()                               \
    do                                                    \
    {                                                     \
        for (uint32_t i = 0; i < 2; i++)                  \
        {                                                 \
            uint32_t row = i * 8 + laneid / 4;            \
            uint32_t col = laneid % 4;                    \
            a_frag[i] = *(uint32_t *)(a + row * K + col); \
        }                                                 \
    } while (0)

#define LOAD_A_M16K16_F16BF16()                               \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 8; i += 2)                   \
        {                                                     \
            uint32_t row = i % 4 / 2 * 8 + laneid / 4;        \
            uint32_t col = i / 4 * 8 + laneid % 4 * 2;        \
            a_frag[i / 2] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M16K8_F16BF16()                                \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 4; i += 2)                   \
        {                                                     \
            uint32_t row = i / 2 * 8 + laneid / 4;            \
            uint32_t col = laneid % 4 * 2;                    \
            a_frag[i / 2] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M8K4_X4_F16()                                  \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 4; i += 2)                   \
        {                                                     \
            uint32_t row = laneid / 16 * 4 + laneid % 4;      \
            uint32_t col = i;                                 \
            a_frag[i / 2] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M16K32_FP8()                                     \
    do                                                          \
    {                                                           \
        for (uint32_t i = 0; i < 16; i += 4)                    \
        {                                                       \
            uint32_t row = i % 8 / 4 * 8 + laneid / 4;          \
            uint32_t col = i / 8 * 16 + laneid % 4 * 4 + i % 4; \
            a_frag[i / 4] = *(uint32_t *)(a + row * K + col);   \
        }                                                       \
    } while (0)

#define LOAD_A_M16K16_FP8()                                   \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 8; i += 4)                   \
        {                                                     \
            uint32_t row = i / 4 * 8 + laneid / 4;            \
            uint32_t col = laneid % 4 * 4 + i % 4;            \
            a_frag[i / 4] = *(uint32_t *)(a + row * K + col); \
        }                                                     \
    } while (0)

#define LOAD_A_M16K64_FP4()                                           \
    do                                                                \
    {                                                                 \
        for (uint32_t i = 0; i < 32; i += 8)                          \
        {                                                             \
            uint32_t row = i % 16 / 8 * 8 + laneid / 4;               \
            uint32_t col = i / 16 * 32 + laneid % 4 * 8 + i % 8;      \
            a_frag[i / 8] = *(uint32_t *)(a + row * K / 2 + col / 2); \
        }                                                             \
    } while (0)

#define LOAD_B_N8K16_F64()                     \
    do                                         \
    {                                          \
        for (uint32_t i = 0; i < 4; i++)       \
        {                                      \
            uint32_t row = i * 4 + laneid % 4; \
            uint32_t col = laneid / 4;         \
            b_frag[i] = b[col * K + row];      \
        }                                      \
    } while (0)

#define LOAD_B_N8K8_F64()                      \
    do                                         \
    {                                          \
        for (uint32_t i = 0; i < 2; i++)       \
        {                                      \
            uint32_t row = laneid % 4 * 2 + i; \
            uint32_t col = laneid / 4;         \
            b_frag[i] = b[col * K + row];      \
        }                                      \
    } while (0)

#define LOAD_B_N8K4_F64()             \
    do                                \
    {                                 \
        uint32_t row = laneid % 4;    \
        uint32_t col = laneid / 4;    \
        b_frag[0] = b[col * K + row]; \
    } while (0)

#define LOAD_B_N8K8_TF32()                                \
    do                                                    \
    {                                                     \
        for (uint32_t i = 0; i < 2; i++)                  \
        {                                                 \
            uint32_t row = i * 4 + laneid % 4;            \
            uint32_t col = laneid / 4;                    \
            b_frag[i] = *(uint32_t *)(b + col * K + row); \
        }                                                 \
    } while (0)

#define LOAD_B_N8K4_TF32()                            \
    do                                                \
    {                                                 \
        uint32_t row = laneid % 4;                    \
        uint32_t col = laneid / 4;                    \
        b_frag[0] = *(uint32_t *)(b + col * K + row); \
    } while (0)

#define LOAD_B_N8K16_F16BF16()                                \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 4; i += 2)                   \
        {                                                     \
            uint32_t row = i / 2 * 8 + laneid % 4 * 2;        \
            uint32_t col = laneid / 4;                        \
            b_frag[i / 2] = *(uint32_t *)(b + col * K + row); \
        }                                                     \
    } while (0)

#define LOAD_B_N8K8_F16BF16()                         \
    do                                                \
    {                                                 \
        uint32_t row = laneid % 4 * 2;                \
        uint32_t col = laneid / 4;                    \
        b_frag[0] = *(uint32_t *)(b + col * K + row); \
    } while (0)

#define LOAD_B_N8K4_X4_F16()                                  \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 4; i += 2)                   \
        {                                                     \
            uint32_t row = i;                                 \
            uint32_t col = laneid / 16 * 4 + laneid % 4;      \
            b_frag[i / 2] = *(uint32_t *)(b + col * K + row); \
        }                                                     \
    } while (0)

#define LOAD_B_N8K32_FP8()                                      \
    do                                                          \
    {                                                           \
        for (uint32_t i = 0; i < 8; i += 4)                     \
        {                                                       \
            uint32_t row = i / 4 * 16 + laneid % 4 * 4 + i % 4; \
            uint32_t col = laneid / 4;                          \
            b_frag[i / 4] = *(uint32_t *)(b + col * K + row);   \
        }                                                       \
    } while (0)

#define LOAD_B_N8K16_FP8()                            \
    do                                                \
    {                                                 \
        uint32_t row = laneid % 4 * 4;                \
        uint32_t col = laneid / 4;                    \
        b_frag[0] = *(uint32_t *)(b + col * K + row); \
    } while (0)

#define LOAD_B_N8K64_FP4()                                            \
    do                                                                \
    {                                                                 \
        for (uint32_t i = 0; i < 16; i += 8)                          \
        {                                                             \
            uint32_t row = i / 8 * 32 + laneid % 4 * 8 + i % 8;       \
            uint32_t col = laneid / 4;                                \
            b_frag[i / 8] = *(uint32_t *)(b + col * K / 2 + row / 2); \
        }                                                             \
    } while (0)

#define LOAD_C_M16N8_F64()                         \
    do                                             \
    {                                              \
        for (uint32_t i = 0; i < 4; i++)           \
        {                                          \
            uint32_t row = i / 2 * 8 + laneid / 4; \
            uint32_t col = laneid % 4 * 2 + i;     \
            c_frag[i] = c[row * N + col];          \
        }                                          \
    } while (0)

#define LOAD_C_M8N8_F64()                      \
    do                                         \
    {                                          \
        for (uint32_t i = 0; i < 2; i++)       \
        {                                      \
            uint32_t row = laneid / 4;         \
            uint32_t col = laneid % 4 * 2 + i; \
            c_frag[i] = c[row * N + col];      \
        }                                      \
    } while (0)

#define LOAD_C_M16N8_F32()                         \
    do                                             \
    {                                              \
        for (uint32_t i = 0; i < 4; i++)           \
        {                                          \
            uint32_t row = i / 2 * 8 + laneid / 4; \
            uint32_t col = laneid % 4 * 2 + i % 2; \
            c_frag[i] = c[row * N + col];          \
        }                                          \
    } while (0)

#define LOAD_C_M16N8_F16()                                    \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 4; i += 2)                   \
        {                                                     \
            uint32_t row = i / 2 * 8 + laneid / 4;            \
            uint32_t col = laneid % 4 * 2;                    \
            c_frag[i / 2] = *(uint32_t *)(c + row * N + col); \
        }                                                     \
    } while (0)

#define LOAD_C_M8N8_X4_F32()                                         \
    do                                                               \
    {                                                                \
        for (uint32_t i = 0; i < 8; i++)                             \
        {                                                            \
            uint32_t row = laneid / 16 * 4 + (i & 2) + (laneid & 1); \
            uint32_t col = (i & 4) + (laneid & 2) + (i & 1);         \
            c_frag[i] = c[row * N + col];                            \
        }                                                            \
    } while (0)

#define LOAD_C_M8N8_X4_F16()                                  \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 8; i += 2)                   \
        {                                                     \
            uint32_t row = laneid / 16 * 4 + laneid % 4;      \
            uint32_t col = i;                                 \
            c_frag[i / 2] = *(uint32_t *)(c + row * N + col); \
        }                                                     \
    } while (0)

#define LOAD_SFA_M16SFK1()                          \
    do                                              \
    {                                               \
        uint32_t row = laneid % 2 * 8 + laneid / 4; \
        sfa_frag = sfa[row];                        \
    } while (0)

#define LOAD_SFA_M16SFK2()                          \
    do                                              \
    {                                               \
        uint32_t row = laneid % 2 * 8 + laneid / 4; \
        sfa_frag = *(uint16_t *)(sfa + row * 2);    \
    } while (0)

#define LOAD_SFA_M16SFK4()                          \
    do                                              \
    {                                               \
        uint32_t row = laneid % 2 * 8 + laneid / 4; \
        sfa_frag = *(uint32_t *)(sfa + row * 4);    \
    } while (0)

#define LOAD_SFB_N8SFK1()          \
    do                             \
    {                              \
        uint32_t col = laneid / 4; \
        sfb_frag = sfb[col];       \
    } while (0)

#define LOAD_SFB_N8SFK2()                        \
    do                                           \
    {                                            \
        uint32_t col = laneid / 4;               \
        sfb_frag = *(uint16_t *)(sfb + col * 2); \
    } while (0)

#define LOAD_SFB_N8SFK4()                        \
    do                                           \
    {                                            \
        uint32_t col = laneid / 4;               \
        sfb_frag = *(uint32_t *)(sfb + col * 4); \
    } while (0)

#define STORE_D_M16N8_F64()                        \
    do                                             \
    {                                              \
        for (uint32_t i = 0; i < 4; i++)           \
        {                                          \
            uint32_t row = i / 2 * 8 + laneid / 4; \
            uint32_t col = laneid % 4 * 2 + i;     \
            d[row * N + col] = d_frag[i];          \
        }                                          \
    } while (0)

#define STORE_D_M8N8_F64()                     \
    do                                         \
    {                                          \
        for (uint32_t i = 0; i < 2; i++)       \
        {                                      \
            uint32_t row = laneid / 4;         \
            uint32_t col = laneid % 4 * 2 + i; \
            d[row * N + col] = d_frag[i];      \
        }                                      \
    } while (0)

#define STORE_D_M16N8_F32()                        \
    do                                             \
    {                                              \
        for (uint32_t i = 0; i < 4; i++)           \
        {                                          \
            uint32_t row = i / 2 * 8 + laneid / 4; \
            uint32_t col = laneid % 4 * 2 + i % 2; \
            d[row * N + col] = d_frag[i];          \
        }                                          \
    } while (0)

#define STORE_D_M16N8_F16()                                   \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 4; i += 2)                   \
        {                                                     \
            uint32_t row = i / 2 * 8 + laneid / 4;            \
            uint32_t col = laneid % 4 * 2;                    \
            *(uint32_t *)(d + row * N + col) = d_frag[i / 2]; \
        }                                                     \
    } while (0)

#define STORE_D_M8N8_X4_F32()                                        \
    do                                                               \
    {                                                                \
        for (uint32_t i = 0; i < 8; i++)                             \
        {                                                            \
            uint32_t row = laneid / 16 * 4 + (i & 2) + (laneid & 1); \
            uint32_t col = (i & 4) + (laneid & 2) + (i & 1);         \
            d[row * N + col] = d_frag[i];                            \
        }                                                            \
    } while (0)

#define STORE_D_M8N8_X4_F16()                                 \
    do                                                        \
    {                                                         \
        for (uint32_t i = 0; i < 8; i += 2)                   \
        {                                                     \
            uint32_t row = laneid / 16 * 4 + laneid % 4;      \
            uint32_t col = i;                                 \
            *(uint32_t *)(d + row * N + col) = d_frag[i / 2]; \
        }                                                     \
    } while (0)
