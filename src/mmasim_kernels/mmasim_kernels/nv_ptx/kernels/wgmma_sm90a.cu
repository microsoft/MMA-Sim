#include "wgmma.h"

// One full m64nNkK instruction per kernel, including the complete N-column
// accumulator. The matrix is never decomposed into smaller WGMMA shapes.
template <int N, WGMMAType DType, WGMMAType AType, WGMMAType BType>
__global__ void wgmma_kernel(typename WGMMAAccumulator<DType>::Element *d,
                             const void *a, const void *b)
{
    using Accumulator = WGMMAAccumulator<DType>;
    constexpr int registers = N / 8 * Accumulator::registers_per_8_columns;
    typename Accumulator::Register d_frag[registers];
    uint32_t tid = threadIdx.x;
    __shared__ __align__(16) uint8_t a_smem[64 * 32], b_smem[N * 32];

    uint64_t a_desc = wgmma_pack_operand(a_smem, a, 64, tid);
    uint64_t b_desc = wgmma_pack_operand(b_smem, b, N, tid);
    wgmma_transfer_d<N, false>(d, d_frag, tid);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    WGMMAInstruction<N, DType, AType, BType>::run(d_frag, a_desc, b_desc);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    wgmma_transfer_d<N, true>(d, d_frag, tid);
}

#define DEFINE_WGMMA(n, ksize, dtype, atype, btype) \
extern "C" void wgmma_m64n##n##k##ksize##_##dtype##_##atype##_##btype( \
    WGMMAAccumulator<WGMMAType::dtype>::Element *d, void *a, void *b) \
{ \
    wgmma_kernel<n, WGMMAType::dtype, WGMMAType::atype, WGMMAType::btype> \
        <<<1, 128>>>(d, a, b); \
}

#define DEFINE_WGMMA_SHAPE(n) \
    DEFINE_WGMMA(n, 8, f32, tf32, tf32) \
    DEFINE_WGMMA(n, 16, f32, f16, f16) \
    DEFINE_WGMMA(n, 16, f16, f16, f16) \
    DEFINE_WGMMA(n, 16, f32, bf16, bf16) \
    DEFINE_WGMMA(n, 32, f32, e5m2, e5m2) \
    DEFINE_WGMMA(n, 32, f32, e5m2, e4m3) \
    DEFINE_WGMMA(n, 32, f32, e4m3, e5m2) \
    DEFINE_WGMMA(n, 32, f32, e4m3, e4m3) \
    DEFINE_WGMMA(n, 32, f16, e5m2, e5m2) \
    DEFINE_WGMMA(n, 32, f16, e5m2, e4m3) \
    DEFINE_WGMMA(n, 32, f16, e4m3, e5m2) \
    DEFINE_WGMMA(n, 32, f16, e4m3, e4m3)

DEFINE_WGMMA_SHAPE(8)
DEFINE_WGMMA_SHAPE(16)
DEFINE_WGMMA_SHAPE(24)
DEFINE_WGMMA_SHAPE(32)
DEFINE_WGMMA_SHAPE(40)
DEFINE_WGMMA_SHAPE(48)
DEFINE_WGMMA_SHAPE(56)
DEFINE_WGMMA_SHAPE(64)
DEFINE_WGMMA_SHAPE(72)
DEFINE_WGMMA_SHAPE(80)
DEFINE_WGMMA_SHAPE(88)
DEFINE_WGMMA_SHAPE(96)
DEFINE_WGMMA_SHAPE(104)
DEFINE_WGMMA_SHAPE(112)
DEFINE_WGMMA_SHAPE(120)
DEFINE_WGMMA_SHAPE(128)
DEFINE_WGMMA_SHAPE(136)
DEFINE_WGMMA_SHAPE(144)
DEFINE_WGMMA_SHAPE(152)
DEFINE_WGMMA_SHAPE(160)
DEFINE_WGMMA_SHAPE(168)
DEFINE_WGMMA_SHAPE(176)
DEFINE_WGMMA_SHAPE(184)
DEFINE_WGMMA_SHAPE(192)
DEFINE_WGMMA_SHAPE(200)
DEFINE_WGMMA_SHAPE(208)
DEFINE_WGMMA_SHAPE(216)
DEFINE_WGMMA_SHAPE(224)
DEFINE_WGMMA_SHAPE(232)
DEFINE_WGMMA_SHAPE(240)
DEFINE_WGMMA_SHAPE(248)
DEFINE_WGMMA_SHAPE(256)
