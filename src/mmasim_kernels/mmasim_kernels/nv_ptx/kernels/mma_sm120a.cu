#include "mma.h"

extern "C" int mmasim_cuda_version() { return CUDART_VERSION; }

extern "C" // kind::f8f6f4
{
#define DEFINE_MMA_F8F6F4_F32(atype, btype)                                                 \
    __global__ void mma_m16n8k32_kind_f8f6f4_f32_##atype##_##btype##_f32_kernel(            \
        float *d, uint8_t *a, uint8_t *b, float *c)                                         \
    {                                                                                       \
        const uint32_t N = 8, K = 32;                                                       \
        uint32_t laneid = threadIdx.x;                                                      \
        uint32_t a_frag[4], b_frag[2];                                                      \
        float c_frag[4], d_frag[4];                                                         \
        LOAD_A_M16K32_F8F6F4(F8F6F4_MASK_##atype);                                          \
        LOAD_B_N8K32_F8F6F4(F8F6F4_MASK_##btype);                                           \
        LOAD_C_M16N8_F32();                                                                 \
        asm volatile(                                                                       \
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32." #atype "." #btype ".f32 " \
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"           \
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])            \
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),               \
              "r"(b_frag[0]), "r"(b_frag[1]),                                               \
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]));              \
        STORE_D_M16N8_F32();                                                                \
    }                                                                                       \
    void mma_m16n8k32_kind_f8f6f4_f32_##atype##_##btype##_f32(                              \
        float *d, uint8_t *a, uint8_t *b, float *c)                                         \
    {                                                                                       \
        mma_m16n8k32_kind_f8f6f4_f32_##atype##_##btype##_f32_kernel<<<1, 32>>>(d, a, b, c); \
    }

#define DEFINE_MMA_F8F6F4_F16(atype, btype)                                                 \
    __global__ void mma_m16n8k32_kind_f8f6f4_f16_##atype##_##btype##_f16_kernel(            \
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)                                   \
    {                                                                                       \
        const uint32_t N = 8, K = 32;                                                       \
        uint32_t laneid = threadIdx.x;                                                      \
        uint32_t a_frag[4], b_frag[2];                                                      \
        uint32_t c_frag[2], d_frag[2];                                                      \
        LOAD_A_M16K32_F8F6F4(F8F6F4_MASK_##atype);                                          \
        LOAD_B_N8K32_F8F6F4(F8F6F4_MASK_##btype);                                           \
        LOAD_C_M16N8_F16();                                                                 \
        asm volatile(                                                                       \
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16." #atype "." #btype ".f16 " \
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};"                               \
            : "=r"(d_frag[0]), "=r"(d_frag[1])                                              \
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),               \
              "r"(b_frag[0]), "r"(b_frag[1]),                                               \
              "r"(c_frag[0]), "r"(c_frag[1]));                                              \
        STORE_D_M16N8_F16();                                                                \
    }                                                                                       \
    void mma_m16n8k32_kind_f8f6f4_f16_##atype##_##btype##_f16(                              \
        uint16_t *d, uint8_t *a, uint8_t *b, uint16_t *c)                                   \
    {                                                                                       \
        mma_m16n8k32_kind_f8f6f4_f16_##atype##_##btype##_f16_kernel<<<1, 32>>>(d, a, b, c); \
    }

#define DEFINE_MMA_F8F6F4(atype, btype) \
    DEFINE_MMA_F8F6F4_F32(atype, btype) \
    DEFINE_MMA_F8F6F4_F16(atype, btype)

#define DEFINE_MMA_F8F6F4_FOR_A(atype) \
    DEFINE_MMA_F8F6F4(atype, e5m2)     \
    DEFINE_MMA_F8F6F4(atype, e4m3)     \
    DEFINE_MMA_F8F6F4(atype, e3m2)     \
    DEFINE_MMA_F8F6F4(atype, e2m3)     \
    DEFINE_MMA_F8F6F4(atype, e2m1)

    DEFINE_MMA_F8F6F4_FOR_A(e5m2)
    DEFINE_MMA_F8F6F4_FOR_A(e4m3)
    DEFINE_MMA_F8F6F4_FOR_A(e3m2)
    DEFINE_MMA_F8F6F4_FOR_A(e2m3)
    DEFINE_MMA_F8F6F4_FOR_A(e2m1)

#define DEFINE_MMA_MXF8F6F4(atype, btype)                                                                                     \
    __global__ void mma_m16n8k32_kind_mxf8f6f4_block32_f32_##atype##_##btype##_f32_ue8m0_kernel(                              \
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)                                               \
    {                                                                                                                         \
        const uint32_t N = 8, K = 32;                                                                                         \
        const uint16_t selector = 0;                                                                                          \
        uint32_t laneid = threadIdx.x;                                                                                        \
        uint32_t a_frag[4], b_frag[2], sfa_frag, sfb_frag;                                                                    \
        float c_frag[4], d_frag[4];                                                                                           \
        LOAD_A_M16K32_F8F6F4(F8F6F4_MASK_##atype);                                                                            \
        LOAD_B_N8K32_F8F6F4(F8F6F4_MASK_##btype);                                                                             \
        LOAD_C_M16N8_F32();                                                                                                   \
        LOAD_SFA_M16SFK1();                                                                                                   \
        LOAD_SFB_N8SFK1();                                                                                                    \
        asm volatile(                                                                                                         \
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32." #atype "." #btype ".f32.ue8m0 " \
            "{%0, %1, %2, %3}, "                                                                                              \
            "{%4, %5, %6, %7}, "                                                                                              \
            "{%8, %9}, "                                                                                                      \
            "{%10, %11, %12, %13}, "                                                                                          \
            "%14, {%15, %16}, "                                                                                               \
            "%17, {%18, %19};"                                                                                                \
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])                                              \
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),                                                 \
              "r"(b_frag[0]), "r"(b_frag[1]),                                                                                 \
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),                                                 \
              "r"(sfa_frag), "h"(selector), "h"(selector),                                                                    \
              "r"(sfb_frag), "h"(selector), "h"(selector));                                                                   \
        STORE_D_M16N8_F32();                                                                                                  \
    }                                                                                                                         \
    void mma_m16n8k32_kind_mxf8f6f4_block32_f32_##atype##_##btype##_f32_ue8m0(                                                \
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)                                               \
    {                                                                                                                         \
        mma_m16n8k32_kind_mxf8f6f4_block32_f32_##atype##_##btype##_f32_ue8m0_kernel<<<1, 32>>>(                               \
            d, a, b, c, sfa, sfb);                                                                                            \
    }

#define DEFINE_MMA_MXF8F6F4_FOR_A(atype) \
    DEFINE_MMA_MXF8F6F4(atype, e5m2)     \
    DEFINE_MMA_MXF8F6F4(atype, e4m3)     \
    DEFINE_MMA_MXF8F6F4(atype, e3m2)     \
    DEFINE_MMA_MXF8F6F4(atype, e2m3)     \
    DEFINE_MMA_MXF8F6F4(atype, e2m1)

    DEFINE_MMA_MXF8F6F4_FOR_A(e5m2)
    DEFINE_MMA_MXF8F6F4_FOR_A(e4m3)
    DEFINE_MMA_MXF8F6F4_FOR_A(e3m2)
    DEFINE_MMA_MXF8F6F4_FOR_A(e2m3)
    DEFINE_MMA_MXF8F6F4_FOR_A(e2m1)
}

extern "C" // kind::mxf4
{
    __global__ void mma_m16n8k64_kind_mxf4_block32_f32_e2m1_e2m1_f32_ue8m0_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t N = 8, K = 64;
        const uint16_t selector = 0;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2], sfa_frag, sfb_frag;
        float c_frag[4], d_frag[4];

        LOAD_A_M16K64_FP4();
        LOAD_B_N8K64_FP4();
        LOAD_C_M16N8_F32();
        LOAD_SFA_M16SFK2();
        LOAD_SFB_N8SFK2();
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "%14, {%15, %16}, "
            "%17, {%18, %19};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
              "r"(sfa_frag), "h"(selector), "h"(selector),
              "r"(sfb_frag), "h"(selector), "h"(selector));
        STORE_D_M16N8_F32();
    }

    void mma_m16n8k64_kind_mxf4_block32_f32_e2m1_e2m1_f32_ue8m0(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        mma_m16n8k64_kind_mxf4_block32_f32_e2m1_e2m1_f32_ue8m0_kernel<<<1, 32>>>(d, a, b, c, sfa, sfb);
    }
}

extern "C" // kind::mxf4nvf4
{
    __global__ void mma_m16n8k64_kind_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue8m0_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t N = 8, K = 64;
        const uint16_t selector = 0;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2], sfa_frag, sfb_frag;
        float c_frag[4], d_frag[4];

        LOAD_A_M16K64_FP4();
        LOAD_B_N8K64_FP4();
        LOAD_C_M16N8_F32();
        LOAD_SFA_M16SFK2();
        LOAD_SFB_N8SFK2();
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "%14, {%15, %16}, "
            "%17, {%18, %19};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
              "r"(sfa_frag), "h"(selector), "h"(selector),
              "r"(sfb_frag), "h"(selector), "h"(selector));
        STORE_D_M16N8_F32();
    }

    void mma_m16n8k64_kind_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue8m0(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        mma_m16n8k64_kind_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue8m0_kernel<<<1, 32>>>(d, a, b, c, sfa, sfb);
    }

    // invalid
    // __global__ void mma_m16n8k64_kind_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue4m3_kernel(
    //     float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    // {
    //     const uint32_t N = 8, K = 64;
    //     const uint16_t selector = 0;
    //     uint32_t laneid = threadIdx.x;
    //     uint32_t a_frag[4], b_frag[2], sfa_frag, sfb_frag;
    //     float c_frag[4], d_frag[4];

    //     LOAD_A_M16K64_FP4();
    //     LOAD_B_N8K64_FP4();
    //     LOAD_C_M16N8_F32();
    //     LOAD_SFA_M16SFK2();
    //     LOAD_SFB_N8SFK2();
    //     asm volatile(
    //         "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3 "
    //         "{%0, %1, %2, %3}, "
    //         "{%4, %5, %6, %7}, "
    //         "{%8, %9}, "
    //         "{%10, %11, %12, %13}, "
    //         "%14, {%15, %16}, "
    //         "%17, {%18, %19};"
    //         : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
    //         : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
    //           "r"(b_frag[0]), "r"(b_frag[1]),
    //           "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
    //           "r"(sfa_frag), "h"(selector), "h"(selector),
    //           "r"(sfb_frag), "h"(selector), "h"(selector));
    //     STORE_D_M16N8_F32();
    // }

    // void mma_m16n8k64_kind_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue4m3(
    //     float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    // {
    //     mma_m16n8k64_kind_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue4m3_kernel<<<1, 32>>>(d, a, b, c, sfa, sfb);
    // }

    __global__ void mma_m16n8k64_kind_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue4m3_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        const uint32_t N = 8, K = 64;
        const uint16_t selector = 0;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2], sfa_frag, sfb_frag;
        float c_frag[4], d_frag[4];

        LOAD_A_M16K64_FP4();
        LOAD_B_N8K64_FP4();
        LOAD_C_M16N8_F32();
        LOAD_SFA_M16SFK4();
        LOAD_SFB_N8SFK4();
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "%14, {%15, %16}, "
            "%17, {%18, %19};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
              "r"(sfa_frag), "h"(selector), "h"(selector),
              "r"(sfb_frag), "h"(selector), "h"(selector));
        STORE_D_M16N8_F32();
    }

    void mma_m16n8k64_kind_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue4m3(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        mma_m16n8k64_kind_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue4m3_kernel<<<1, 32>>>(d, a, b, c, sfa, sfb);
    }

    __global__ void mma_m16n8k64_kind_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue8m0_kernel(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
#if CUDART_VERSION >= 13010
        const uint32_t N = 8, K = 64;
        const uint16_t selector = 0;
        uint32_t laneid = threadIdx.x;
        uint32_t a_frag[4], b_frag[2], sfa_frag, sfb_frag;
        float c_frag[4], d_frag[4];

        LOAD_A_M16K64_FP4();
        LOAD_B_N8K64_FP4();
        LOAD_C_M16N8_F32();
        LOAD_SFA_M16SFK4();
        LOAD_SFB_N8SFK4();
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "%14, {%15, %16}, "
            "%17, {%18, %19};"
            : "=f"(d_frag[0]), "=f"(d_frag[1]), "=f"(d_frag[2]), "=f"(d_frag[3])
            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
              "r"(b_frag[0]), "r"(b_frag[1]),
              "f"(c_frag[0]), "f"(c_frag[1]), "f"(c_frag[2]), "f"(c_frag[3]),
              "r"(sfa_frag), "h"(selector), "h"(selector),
              "r"(sfb_frag), "h"(selector), "h"(selector));
        STORE_D_M16N8_F32();
#endif
    }

    void mma_m16n8k64_kind_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue8m0(
        float *d, uint8_t *a, uint8_t *b, float *c, uint8_t *sfa, uint8_t *sfb)
    {
        mma_m16n8k64_kind_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue8m0_kernel<<<1, 32>>>(d, a, b, c, sfa, sfb);
    }
}
