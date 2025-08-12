#include <hip/hip_runtime.h>

using float32x2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x32 = __attribute__((__vector_size__(32 * sizeof(float)))) float;
using float16x4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using bfloat16x2 = __attribute__((__vector_size__(2 * sizeof(int16_t)))) int16_t;
using bfloat16x4 = __attribute__((__vector_size__(4 * sizeof(int16_t)))) int16_t;
union float8x8
{
    long all;
    int8_t part[8];
};

#define LOAD_A_M32K16()                        \
    do                                         \
    {                                          \
        for (i = 0; i < 8; i++)                \
        {                                      \
            row = tid % 32;                    \
            col = tid / 32 * 8 + i;            \
            a_frag.part[i] = a[row * K + col]; \
        }                                      \
    } while (0)

#define LOAD_A_M32K8()                    \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid % 32;               \
            col = tid / 32 * 4 + i;       \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M32K4()                    \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid % 32;               \
            col = tid / 32 * 2 + i;       \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M32K4_2B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid % 32;               \
            col = i;                      \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M32K2()             \
    do                             \
    {                              \
        row = tid % 32;            \
        col = tid / 32;            \
        a_frag = a[row * K + col]; \
    } while (0)

#define LOAD_A_M32K2_2B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid % 32;               \
            col = i;                      \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M32K1_2B()          \
    do                             \
    {                              \
        row = tid % 32;            \
        col = 0;                   \
        a_frag = a[row * K + col]; \
    } while (0)

#define LOAD_A_M16K32()                        \
    do                                         \
    {                                          \
        for (i = 0; i < 8; i++)                \
        {                                      \
            row = tid % 16;                    \
            col = tid / 16 * 8 + i;            \
            a_frag.part[i] = a[row * K + col]; \
        }                                      \
    } while (0)

#define LOAD_A_M16K16()                   \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid % 16;               \
            col = tid / 16 * 4 + i;       \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M16K8()                    \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid % 16;               \
            col = tid / 16 * 2 + i;       \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M16K4()             \
    do                             \
    {                              \
        row = tid % 16;            \
        col = tid / 16;            \
        a_frag = a[row * K + col]; \
    } while (0)

#define LOAD_A_M16K4_4B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid % 16;               \
            col = i;                      \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M16K2_4B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid % 16;               \
            col = i;                      \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M16K1_4B()          \
    do                             \
    {                              \
        row = tid % 16;            \
        col = 0;                   \
        a_frag = a[row * K + col]; \
    } while (0)

#define LOAD_A_M4K4_16B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid % 4;                \
            col = i;                      \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M4K2_16B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid % 4;                \
            col = i;                      \
            a_frag[i] = a[row * K + col]; \
        }                                 \
    } while (0)

#define LOAD_A_M4K1_16B()          \
    do                             \
    {                              \
        row = tid % 4;             \
        col = 0;                   \
        a_frag = a[row * K + col]; \
    } while (0)

#define LOAD_B_N32K16()                        \
    do                                         \
    {                                          \
        for (i = 0; i < 8; i++)                \
        {                                      \
            row = tid / 32 * 8 + i;            \
            col = tid % 32;                    \
            b_frag.part[i] = b[row * N + col]; \
        }                                      \
    } while (0)

#define LOAD_B_N32K8()                    \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid / 32 * 4 + i;       \
            col = tid % 32;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N32K4()                    \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid / 32 * 2 + i;       \
            col = tid % 32;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N32K4_2B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 32;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N32K2()             \
    do                             \
    {                              \
        row = tid / 32;            \
        col = tid % 32;            \
        b_frag = b[row * N + col]; \
    } while (0)

#define LOAD_B_N32K2_2B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 32;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N32K1_2B()          \
    do                             \
    {                              \
        row = 0;                   \
        col = tid % 32;            \
        b_frag = b[row * N + col]; \
    } while (0)

#define LOAD_B_N16K32()                        \
    do                                         \
    {                                          \
        for (i = 0; i < 8; i++)                \
        {                                      \
            row = tid / 16 * 8 + i;            \
            col = tid % 16;                    \
            b_frag.part[i] = b[row * N + col]; \
        }                                      \
    } while (0)

#define LOAD_B_N16K16()                   \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid / 16 * 4 + i;       \
            col = tid % 16;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N16K8()                    \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = tid / 16 * 2 + i;       \
            col = tid % 16;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N16K4()             \
    do                             \
    {                              \
        row = tid / 16;            \
        col = tid % 16;            \
        b_frag = b[row * N + col]; \
    } while (0)

#define LOAD_B_N16K4_4B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 16;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N16K2_4B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 16;               \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N16K1_4B()          \
    do                             \
    {                              \
        row = 0;                   \
        col = tid % 16;            \
        b_frag = b[row * N + col]; \
    } while (0)

#define LOAD_B_N4K4_16B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 4;                \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N4K2_16B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 2; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 4;                \
            b_frag[i] = b[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_B_N4K1_16B()          \
    do                             \
    {                              \
        row = 0;                   \
        col = tid % 4;             \
        b_frag = b[row * N + col]; \
    } while (0)

#define LOAD_C_M32N32()                             \
    do                                              \
    {                                               \
        for (i = 0; i < 16; i++)                    \
        {                                           \
            row = tid / 32 * 4 + i / 4 * 8 + i % 4; \
            col = tid % 32;                         \
            c_frag[i] = c[row * N + col];           \
        }                                           \
    } while (0)

#define LOAD_C_M32N32_2B()                               \
    do                                                   \
    {                                                    \
        for (i = 0; i < 32; i++)                         \
        {                                                \
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4; \
            col = tid % 32;                              \
            c_frag[i] = c[row * N + col];                \
        }                                                \
    } while (0)

#define LOAD_C_M16N16()                   \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid / 16 * 4 + i;       \
            col = tid % 16;               \
            c_frag[i] = c[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_C_M16N16_4B()                \
    do                                    \
    {                                     \
        for (i = 0; i < 16; i++)          \
        {                                 \
            row = tid / 16 * 4 + i % 4;   \
            col = tid % 16;               \
            c_frag[i] = c[row * N + col]; \
        }                                 \
    } while (0)

#define LOAD_C_M4N4_16B()                 \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 4;                \
            c_frag[i] = c[row * N + col]; \
        }                                 \
    } while (0)

#define STORE_D_M32N32()                            \
    do                                              \
    {                                               \
        for (i = 0; i < 16; i++)                    \
        {                                           \
            row = tid / 32 * 4 + i / 4 * 8 + i % 4; \
            col = tid % 32;                         \
            d[row * N + col] = d_frag[i];           \
        }                                           \
    } while (0)

#define STORE_D_M32N32_2B()                              \
    do                                                   \
    {                                                    \
        for (i = 0; i < 32; i++)                         \
        {                                                \
            row = tid / 32 * 4 + i / 4 * 8 % 32 + i % 4; \
            col = tid % 32;                              \
            d[row * N + col] = d_frag[i];                \
        }                                                \
    } while (0)

#define STORE_D_M16N16()                  \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = tid / 16 * 4 + i;       \
            col = tid % 16;               \
            d[row * N + col] = d_frag[i]; \
        }                                 \
    } while (0)

#define STORE_D_M16N16_4B()               \
    do                                    \
    {                                     \
        for (i = 0; i < 16; i++)          \
        {                                 \
            row = tid / 16 * 4 + i % 4;   \
            col = tid % 16;               \
            d[row * N + col] = d_frag[i]; \
        }                                 \
    } while (0)

#define STORE_D_M4N4_16B()                \
    do                                    \
    {                                     \
        for (i = 0; i < 4; i++)           \
        {                                 \
            row = i;                      \
            col = tid % 4;                \
            d[row * N + col] = d_frag[i]; \
        }                                 \
    } while (0)
