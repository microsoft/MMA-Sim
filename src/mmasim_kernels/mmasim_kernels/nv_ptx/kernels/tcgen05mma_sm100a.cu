#include "tcgen05mma.h"

template <TCGen05Kind Kind, TCGen05Type AType, TCGen05Type BType, typename D>
__global__ void tcgen05mma_kernel(D *d, const void *a, const void *b, uint32_t M, uint32_t N)
{
    uint32_t tid = threadIdx.x;
    uint32_t cols = tcgen05_tmem_columns(N);
    __shared__ __align__(16) uint8_t a_smem[128 * 32], b_smem[256 * 32];
    __shared__ uint32_t d_tmem_addr;
    __shared__ uint64_t mma_barrier;

    tcgen05_init_barrier(mma_barrier);
    tcgen05_pack_operand<Kind, AType>(a_smem, a, M, tid);
    tcgen05_pack_operand<Kind, BType>(b_smem, b, N, tid);
    uint64_t a_desc = tcgen05_smem_desc(a_smem);
    uint64_t b_desc = tcgen05_smem_desc(b_smem);
    tcgen05_alloc_tmem(d_tmem_addr, cols);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    tcgen05_transfer_d<D, false>(d, d_tmem_addr, M, N, tid);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    constexpr uint32_t d_code = std::is_same<D, uint16_t>::value ? 0 : 1;
    uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24) | (d_code << 4) | (tcgen05_type_code<Kind, AType>() << 7) | (tcgen05_type_code<Kind, BType>() << 10);
    tcgen05_mma<Kind>(d_tmem_addr, a_desc, b_desc, i_desc);
    tcgen05_commit_and_wait(mma_barrier);

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    tcgen05_transfer_d<D, true>(d, d_tmem_addr, M, N, tid);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    tcgen05_dealloc_tmem(d_tmem_addr, cols);
    tcgen05_relinquish_tmem();
}

template <TCGen05Kind Kind, TCGen05Type AType, TCGen05Type BType,
          int BlockSize, TCGen05ScaleType SType>
__global__ void tcgen05mma_block_scale_kernel(uint32_t *d, const void *a, const void *b,
                                              const uint8_t *sfa, const uint8_t *sfb,
                                              uint32_t M, uint32_t N)
{
    constexpr int scale_vectors = Kind == TCGen05Kind::mxf8f6f4 ? 1 : 64 / BlockSize;
    constexpr uint32_t scale_code = SType == TCGen05ScaleType::ue8m0 ? 1 : 0;
    uint32_t tid = threadIdx.x;
    uint32_t cols = tcgen05_tmem_columns(N);
    __shared__ __align__(16) uint8_t a_smem[128 * 32], b_smem[256 * 32];
    __shared__ uint32_t d_tmem_addr, sfa_tmem_addr, sfb_tmem_addr;
    __shared__ uint64_t mma_barrier;

    tcgen05_init_barrier(mma_barrier);
    tcgen05_pack_operand<Kind, AType>(a_smem, a, M, tid);
    tcgen05_pack_operand<Kind, BType>(b_smem, b, N, tid);
    uint64_t a_desc = tcgen05_smem_desc(a_smem);
    uint64_t b_desc = tcgen05_smem_desc(b_smem);
    tcgen05_alloc_tmem(d_tmem_addr, cols);
    tcgen05_alloc_tmem(sfa_tmem_addr, 32);
    tcgen05_alloc_tmem(sfb_tmem_addr, 32);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    tcgen05_transfer_d<uint32_t, false>(d, d_tmem_addr, M, N, tid);
    tcgen05_store_scales<scale_vectors>(sfa_tmem_addr, sfa, M, tid);
    tcgen05_store_scales<scale_vectors>(sfb_tmem_addr, sfb, N, tid);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    uint32_t i_desc = (N >> 3 << 17) | (M >> 4 << 24) | (scale_code << 23) | (tcgen05_type_code<Kind, AType>() << 7) | (tcgen05_type_code<Kind, BType>() << 10);
    tcgen05_mma_block_scale<Kind, BlockSize>(
        d_tmem_addr, a_desc, b_desc, i_desc, sfa_tmem_addr, sfb_tmem_addr);
    tcgen05_commit_and_wait(mma_barrier);

    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    tcgen05_transfer_d<uint32_t, true>(d, d_tmem_addr, M, N, tid);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    tcgen05_dealloc_tmem(d_tmem_addr, cols);
    tcgen05_dealloc_tmem(sfa_tmem_addr, 32);
    tcgen05_dealloc_tmem(sfb_tmem_addr, 32);
    tcgen05_relinquish_tmem();
}

#define TCGEN05_DTYPE_f32 uint32_t
#define TCGEN05_DTYPE_f16 uint16_t

// One entry point per kind/type combination; Python binds the M/N dimensions.
#define DEFINE_TCGEN05(kind, dtype, atype, btype)                                                           \
    extern "C" void tcgen05mma_kind_##kind##_##dtype##_##atype##_##btype(                                   \
        TCGEN05_DTYPE_##dtype *d, void *a, void *b, uint32_t m, uint32_t n)                                 \
    {                                                                                                       \
        tcgen05mma_kernel<TCGen05Kind::kind, TCGen05Type::atype, TCGen05Type::btype, TCGEN05_DTYPE_##dtype> \
            <<<1, 128>>>(d, a, b, m, n);                                                                    \
    }

#define DEFINE_TCGEN05_BLOCK_SCALE(kind, block, stype, atype, btype)                             \
    extern "C" void tcgen05mma_kind_##kind##_block##block##_f32_##atype##_##btype##_##stype(     \
        uint32_t *d, void *a, void *b, uint8_t *sfa, uint8_t *sfb, uint32_t m, uint32_t n)       \
    {                                                                                            \
        tcgen05mma_block_scale_kernel<TCGen05Kind::kind, TCGen05Type::atype, TCGen05Type::btype, \
                                      block, TCGen05ScaleType::stype>                            \
            <<<1, 128>>>(d, a, b, sfa, sfb, m, n);                                               \
    }

DEFINE_TCGEN05(tf32, f32, tf32, tf32)

DEFINE_TCGEN05(f16, f32, f16, f16)
DEFINE_TCGEN05(f16, f32, f16, bf16)
DEFINE_TCGEN05(f16, f32, bf16, f16)
DEFINE_TCGEN05(f16, f32, bf16, bf16)
DEFINE_TCGEN05(f16, f16, f16, f16)

#define DEFINE_TCGEN05_F8F6F4(atype, btype)   \
    DEFINE_TCGEN05(f8f6f4, f32, atype, btype) \
    DEFINE_TCGEN05(f8f6f4, f16, atype, btype)

#define DEFINE_TCGEN05_F8F6F4_FOR_A(atype) \
    DEFINE_TCGEN05_F8F6F4(atype, e5m2)     \
    DEFINE_TCGEN05_F8F6F4(atype, e4m3)     \
    DEFINE_TCGEN05_F8F6F4(atype, e3m2)     \
    DEFINE_TCGEN05_F8F6F4(atype, e2m3)     \
    DEFINE_TCGEN05_F8F6F4(atype, e2m1)

DEFINE_TCGEN05_F8F6F4_FOR_A(e5m2)
DEFINE_TCGEN05_F8F6F4_FOR_A(e4m3)
DEFINE_TCGEN05_F8F6F4_FOR_A(e3m2)
DEFINE_TCGEN05_F8F6F4_FOR_A(e2m3)
DEFINE_TCGEN05_F8F6F4_FOR_A(e2m1)

#define DEFINE_TCGEN05_MXF8F6F4_FOR_A(atype)                     \
    DEFINE_TCGEN05_BLOCK_SCALE(mxf8f6f4, 32, ue8m0, atype, e5m2) \
    DEFINE_TCGEN05_BLOCK_SCALE(mxf8f6f4, 32, ue8m0, atype, e4m3) \
    DEFINE_TCGEN05_BLOCK_SCALE(mxf8f6f4, 32, ue8m0, atype, e3m2) \
    DEFINE_TCGEN05_BLOCK_SCALE(mxf8f6f4, 32, ue8m0, atype, e2m3) \
    DEFINE_TCGEN05_BLOCK_SCALE(mxf8f6f4, 32, ue8m0, atype, e2m1)

DEFINE_TCGEN05_MXF8F6F4_FOR_A(e5m2)
DEFINE_TCGEN05_MXF8F6F4_FOR_A(e4m3)
DEFINE_TCGEN05_MXF8F6F4_FOR_A(e3m2)
DEFINE_TCGEN05_MXF8F6F4_FOR_A(e2m3)
DEFINE_TCGEN05_MXF8F6F4_FOR_A(e2m1)

DEFINE_TCGEN05_BLOCK_SCALE(mxf4, 32, ue8m0, e2m1, e2m1)

DEFINE_TCGEN05_BLOCK_SCALE(mxf4nvf4, 32, ue8m0, e2m1, e2m1)
DEFINE_TCGEN05_BLOCK_SCALE(mxf4nvf4, 16, ue8m0, e2m1, e2m1)
// DEFINE_TCGEN05_BLOCK_SCALE(mxf4nvf4, 32, ue4m3, e2m1, e2m1)  // requires sm_107
DEFINE_TCGEN05_BLOCK_SCALE(mxf4nvf4, 16, ue4m3, e2m1, e2m1)
