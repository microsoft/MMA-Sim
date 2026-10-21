#pragma once
#include <stdint.h>
#include <type_traits>

enum class TCGen05Kind
{
    tf32,
    f16,
    f8f6f4,
    mxf8f6f4,
    mxf4,
    mxf4nvf4
};
enum class TCGen05Type
{
    tf32,
    f16,
    bf16,
    e4m3,
    e5m2,
    e2m3,
    e3m2,
    e2m1
};
enum class TCGen05ScaleType
{
    ue8m0,
    ue4m3
};

template <TCGen05Kind Kind, TCGen05Type Type>
__device__ constexpr uint32_t tcgen05_type_code()
{
    if constexpr (Kind == TCGen05Kind::mxf4 || Kind == TCGen05Kind::mxf4nvf4)
        return 1;
    if constexpr (Type == TCGen05Type::f16 || Type == TCGen05Type::e4m3)
        return 0;
    if constexpr (Type == TCGen05Type::bf16 || Type == TCGen05Type::e5m2)
        return 1;
    if constexpr (Type == TCGen05Type::tf32)
        return 2;
    if constexpr (Type == TCGen05Type::e2m3)
        return 3;
    if constexpr (Type == TCGen05Type::e3m2)
        return 4;
    return 5;
}

__device__ inline uint32_t tcgen05_desc_encode(uint64_t value)
{
    return (value & 0x3ffff) >> 4;
}

__device__ inline uint32_t tcgen05_tmem_columns(uint32_t n)
{
    uint32_t cols = 32;
    while (cols < n)
        cols *= 2;
    return cols;
}

__device__ inline void tcgen05_alloc_tmem(uint32_t &addr, uint32_t cols)
{
    if (threadIdx.x < 32)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(&addr)), "r"(cols));
}

__device__ inline void tcgen05_dealloc_tmem(uint32_t addr, uint32_t cols)
{
    if (threadIdx.x < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(addr), "r"(cols));
}

__device__ inline void tcgen05_relinquish_tmem()
{
    if (threadIdx.x < 32)
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ inline void tcgen05_init_barrier(uint64_t &barrier)
{
    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&barrier)));
}

// Each kernel issues one MMA, so the newly initialized barrier uses phase 0.
__device__ inline void tcgen05_commit_and_wait(uint64_t &barrier)
{
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(&barrier);
    if (threadIdx.x == 0)
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(addr));
    asm volatile(
        "{\n"
        "  .reg .pred ready;\n"
        "wait_mma:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], 0;\n"
        "  @!ready bra wait_mma;\n"
        "}\n" ::"r"(addr) : "memory");
}

#define TCGEN05_MMA_ASM(kind)                                                                                                                  \
    asm volatile(                                                                                                                              \
        "tcgen05.mma.cta_group::1.kind::" kind " [%0], %1, %2, %3, {%4, %5, %6, %7}, 1;" ::"r"(d_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), \
        "r"(0u), "r"(0u), "r"(0u), "r"(0u))

template <TCGen05Kind Kind>
__device__ void tcgen05_mma(uint32_t d_addr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc)
{
    if (threadIdx.x == 0)
    {
        if constexpr (Kind == TCGen05Kind::tf32)
        {
            TCGEN05_MMA_ASM("tf32");
        }
        if constexpr (Kind == TCGen05Kind::f16)
        {
            TCGEN05_MMA_ASM("f16");
        }
        if constexpr (Kind == TCGen05Kind::f8f6f4)
        {
            TCGEN05_MMA_ASM("f8f6f4");
        }
    }
}

#define TCGEN05_MMA_BLOCK_SCALE_ASM(kind, scale_vec)                                    \
    asm volatile(                                                                       \
        "tcgen05.mma.cta_group::1.kind::" kind ".block_scale.scale_vec::" scale_vec " " \
        "[%0], %1, %2, %3, [%4], [%5], 1;" ::"r"(d_addr),                               \
        "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_addr), "r"(sfb_addr))

template <TCGen05Kind Kind, int BlockSize>
__device__ void tcgen05_mma_block_scale(uint32_t d_addr, uint64_t a_desc, uint64_t b_desc,
                                        uint32_t i_desc, uint32_t sfa_addr, uint32_t sfb_addr)
{
    if (threadIdx.x == 0)
    {
        if constexpr (Kind == TCGen05Kind::mxf8f6f4)
        {
            TCGEN05_MMA_BLOCK_SCALE_ASM("mxf8f6f4", "1X");
        }
        if constexpr (Kind == TCGen05Kind::mxf4)
        {
            TCGEN05_MMA_BLOCK_SCALE_ASM("mxf4", "2X");
        }
        if constexpr (Kind == TCGen05Kind::mxf4nvf4 && BlockSize == 32)
        {
            TCGEN05_MMA_BLOCK_SCALE_ASM("mxf4nvf4", "2X");
        }
        if constexpr (Kind == TCGen05Kind::mxf4nvf4 && BlockSize == 16)
        {
            TCGEN05_MMA_BLOCK_SCALE_ASM("mxf4nvf4", "4X");
        }
    }
}

// Descriptor for K-major 8-row x 16-byte tiles, two K tiles per row group.
__device__ inline uint64_t tcgen05_smem_desc(const void *ptr)
{
    return tcgen05_desc_encode(__cvta_generic_to_shared(ptr)) | (uint64_t(tcgen05_desc_encode(128)) << 16) | (uint64_t(tcgen05_desc_encode(256)) << 32) | (1ull << 46);
}

template <TCGen05Kind Kind, TCGen05Type Type>
__device__ void tcgen05_pack_operand(uint8_t *dst, const void *input,
                                     uint32_t rows, uint32_t tid)
{
    const uint8_t *src = static_cast<const uint8_t *>(input);
    for (uint32_t index = tid; index < rows * 32; index += 128)
    {
        uint32_t row = index / 32, tile = index % 32 / 16, byte = index % 16;
        uint8_t value;
        if constexpr ((Kind == TCGen05Kind::f8f6f4 || Kind == TCGen05Kind::mxf8f6f4) && (Type == TCGen05Type::e2m3 || Type == TCGen05Type::e3m2))
        {
            // Sixteen 6-bit values occupy 12 bytes; the last 4 bytes are padding.
            value = 0;
            if (byte < 12)
            {
                uint32_t col = tile * 16 + byte * 8 / 6;
                uint32_t shift = byte * 8 % 6;
                uint32_t bits = src[row * 32 + col] & 0x3f;
                if (col % 16 != 15)
                    bits |= (src[row * 32 + col + 1] & 0x3f) << 6;
                value = bits >> shift;
            }
        }
        else if constexpr ((Kind == TCGen05Kind::f8f6f4 || Kind == TCGen05Kind::mxf8f6f4) && Type == TCGen05Type::e2m1)
        {
            // Sixteen FP4 values occupy 8 bytes; the last 8 bytes are padding.
            value = 0;
            if (byte < 8)
            {
                uint32_t col = tile * 16 + byte * 2;
                value = ((src[row * 32 + col] >> 2) & 15) | (((src[row * 32 + col + 1] >> 2) & 15) << 4);
            }
        }
        else
        {
            // TF32/F16/BF16, FP8, and already packed MXFP4/NVFP4.
            value = src[index];
        }
        dst[(row / 8) * 256 + tile * 128 + (row % 8) * 16 + byte] = value;
    }
}

// PTX requires each D element (including F16) in a separate 32-bit word.
// M=64 uses only 16 datapath lanes in each 32-lane partition (layout F).
template <typename D, bool Store>
__device__ void tcgen05_transfer_d(D *d, uint32_t addr, uint32_t m, uint32_t n,
                                   uint32_t tid)
{
    uint32_t warp = tid / 32, lane = tid % 32;
    uint32_t base = addr + ((warp * 32) << 16);
    if (m == 128)
    {
        for (uint32_t col = 0; col < n; ++col)
        {
            uint32_t word;
            if constexpr (Store)
            {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                             : "=r"(word) : "r"(base + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                d[tid * n + col] = D(word);
            }
            else
            {
                word = uint32_t(d[tid * n + col]);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(base + col), "r"(word));
                asm volatile("tcgen05.wait::st.sync.aligned;");
            }
        }
    }
    else
    {
        for (uint32_t col = 0; col < n; col += 8)
        {
            uint32_t r[4];
            if constexpr (Store)
            {
                asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                             : "r"(base + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
            }
            for (uint32_t i = 0; i < 4; ++i)
            {
                uint32_t row = warp * 16 + i / 2 * 8 + lane / 4;
                uint32_t offset = row * n + col + lane % 4 * 2 + i % 2;
                if constexpr (Store)
                    d[offset] = D(r[i]);
                else
                    r[i] = uint32_t(d[offset]);
            }
            if constexpr (!Store)
            {
                asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};" ::"r"(base + col), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
                asm volatile("tcgen05.wait::st.sync.aligned;");
            }
        }
    }
}

template <int SF>
__device__ void tcgen05_store_scales(uint32_t addr, const uint8_t *src,
                                     uint32_t rows, uint32_t tid)
{
    for (uint32_t col = 0; col < (rows + 31) / 32; ++col)
    {
        uint32_t row = col * 32 + tid % 32;
        uint32_t word = 0;
        if (row < rows)
            for (uint32_t k = 0; k < SF; ++k)
                word |= uint32_t(src[row * SF + k]) << (k * 8);
        uint32_t dst = addr + ((tid / 32 * 32) << 16) + col;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(dst), "r"(word));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;");
}
