#pragma once
#include <stdint.h>
#include <type_traits>

// All supported floating types use 32 bytes per row. Arrange two K tiles
// of eight rows x 16 bytes, matching the unswizzled K-major descriptor.
__device__ inline uint64_t wgmma_pack_operand(uint8_t *dst, const void *input,
                                            uint32_t rows, uint32_t tid)
{
    const uint8_t *src = static_cast<const uint8_t *>(input);
    for (uint32_t index = tid; index < rows * 32; index += 128)
    {
        uint32_t row = index / 32, tile = index % 32 / 16, byte = index % 16;
        dst[(row / 8) * 256 + tile * 128 + (row % 8) * 16 + byte] = src[index];
    }
    return ((__cvta_generic_to_shared(dst) & 0x3ffffull) >> 4)
         | (8ull << 16) | (16ull << 32);
}

template <int N, bool Store>
__device__ void wgmma_transfer_f32(float *d, float *frag, uint32_t tid)
{
    uint32_t warp = tid / 32, lane = tid % 32;
    #pragma unroll
    for (uint32_t i = 0; i < N / 2; ++i)
    {
        uint32_t row = warp * 16 + (i % 4) / 2 * 8 + lane / 4;
        uint32_t col = i / 4 * 8 + lane % 4 * 2 + i % 2;
        if constexpr (Store) d[row * N + col] = frag[i];
        else frag[i] = d[row * N + col];
    }
}

template <int N, bool Store>
__device__ void wgmma_transfer_f16(uint16_t *d, uint32_t *frag, uint32_t tid)
{
    uint32_t warp = tid / 32, lane = tid % 32;
    #pragma unroll
    for (uint32_t i = 0; i < N / 4; ++i)
    {
        uint32_t row = warp * 16 + i % 2 * 8 + lane / 4;
        uint32_t col = i / 2 * 8 + lane % 4 * 2;
        uint32_t *word = reinterpret_cast<uint32_t *>(d + row * N + col);
        if constexpr (Store) *word = frag[i];
        else frag[i] = *word;
    }
}

// The output element type and the type of one accumulator register.
enum class WGMMAType { f32, f16, bf16, tf32, e4m3, e5m2 };

template <WGMMAType Type> struct WGMMAAccumulator;

template <> struct WGMMAAccumulator<WGMMAType::f32>
{
    using Element = float;
    using Register = float;
    static constexpr int registers_per_8_columns = 4;
};

template <> struct WGMMAAccumulator<WGMMAType::f16>
{
    using Element = uint16_t;
    using Register = uint32_t;
    static constexpr int registers_per_8_columns = 2;
};

template <int N, bool Store, typename Element, typename Register>
__device__ void wgmma_transfer_d(Element *d, Register *frag, uint32_t tid)
{
    if constexpr (std::is_same<Element, float>::value)
        wgmma_transfer_f32<N, Store>(d, frag, tid);
    else
        wgmma_transfer_f16<N, Store>(d, frag, tid);
}

// Inline PTX needs an explicit register list for each N. Keep this mechanical
// operand-list expansion here; wgmma_sm90a.cu uses ordinary C++ templates.
#define WGMMA_REGS_2 "%0, %1"
#define WGMMA_ARGS_2(arg) arg(0), arg(1)
#define WGMMA_REGS_4 WGMMA_REGS_2 ", %2, %3"
#define WGMMA_ARGS_4(arg) WGMMA_ARGS_2(arg), arg(2), arg(3)
#define WGMMA_REGS_6 WGMMA_REGS_4 ", %4, %5"
#define WGMMA_ARGS_6(arg) WGMMA_ARGS_4(arg), arg(4), arg(5)
#define WGMMA_REGS_8 WGMMA_REGS_6 ", %6, %7"
#define WGMMA_ARGS_8(arg) WGMMA_ARGS_6(arg), arg(6), arg(7)
#define WGMMA_REGS_10 WGMMA_REGS_8 ", %8, %9"
#define WGMMA_ARGS_10(arg) WGMMA_ARGS_8(arg), arg(8), arg(9)
#define WGMMA_REGS_12 WGMMA_REGS_10 ", %10, %11"
#define WGMMA_ARGS_12(arg) WGMMA_ARGS_10(arg), arg(10), arg(11)
#define WGMMA_REGS_14 WGMMA_REGS_12 ", %12, %13"
#define WGMMA_ARGS_14(arg) WGMMA_ARGS_12(arg), arg(12), arg(13)
#define WGMMA_REGS_16 WGMMA_REGS_14 ", %14, %15"
#define WGMMA_ARGS_16(arg) WGMMA_ARGS_14(arg), arg(14), arg(15)
#define WGMMA_REGS_18 WGMMA_REGS_16 ", %16, %17"
#define WGMMA_ARGS_18(arg) WGMMA_ARGS_16(arg), arg(16), arg(17)
#define WGMMA_REGS_20 WGMMA_REGS_18 ", %18, %19"
#define WGMMA_ARGS_20(arg) WGMMA_ARGS_18(arg), arg(18), arg(19)
#define WGMMA_REGS_22 WGMMA_REGS_20 ", %20, %21"
#define WGMMA_ARGS_22(arg) WGMMA_ARGS_20(arg), arg(20), arg(21)
#define WGMMA_REGS_24 WGMMA_REGS_22 ", %22, %23"
#define WGMMA_ARGS_24(arg) WGMMA_ARGS_22(arg), arg(22), arg(23)
#define WGMMA_REGS_26 WGMMA_REGS_24 ", %24, %25"
#define WGMMA_ARGS_26(arg) WGMMA_ARGS_24(arg), arg(24), arg(25)
#define WGMMA_REGS_28 WGMMA_REGS_26 ", %26, %27"
#define WGMMA_ARGS_28(arg) WGMMA_ARGS_26(arg), arg(26), arg(27)
#define WGMMA_REGS_30 WGMMA_REGS_28 ", %28, %29"
#define WGMMA_ARGS_30(arg) WGMMA_ARGS_28(arg), arg(28), arg(29)
#define WGMMA_REGS_32 WGMMA_REGS_30 ", %30, %31"
#define WGMMA_ARGS_32(arg) WGMMA_ARGS_30(arg), arg(30), arg(31)
#define WGMMA_REGS_34 WGMMA_REGS_32 ", %32, %33"
#define WGMMA_ARGS_34(arg) WGMMA_ARGS_32(arg), arg(32), arg(33)
#define WGMMA_REGS_36 WGMMA_REGS_34 ", %34, %35"
#define WGMMA_ARGS_36(arg) WGMMA_ARGS_34(arg), arg(34), arg(35)
#define WGMMA_REGS_38 WGMMA_REGS_36 ", %36, %37"
#define WGMMA_ARGS_38(arg) WGMMA_ARGS_36(arg), arg(36), arg(37)
#define WGMMA_REGS_40 WGMMA_REGS_38 ", %38, %39"
#define WGMMA_ARGS_40(arg) WGMMA_ARGS_38(arg), arg(38), arg(39)
#define WGMMA_REGS_42 WGMMA_REGS_40 ", %40, %41"
#define WGMMA_ARGS_42(arg) WGMMA_ARGS_40(arg), arg(40), arg(41)
#define WGMMA_REGS_44 WGMMA_REGS_42 ", %42, %43"
#define WGMMA_ARGS_44(arg) WGMMA_ARGS_42(arg), arg(42), arg(43)
#define WGMMA_REGS_46 WGMMA_REGS_44 ", %44, %45"
#define WGMMA_ARGS_46(arg) WGMMA_ARGS_44(arg), arg(44), arg(45)
#define WGMMA_REGS_48 WGMMA_REGS_46 ", %46, %47"
#define WGMMA_ARGS_48(arg) WGMMA_ARGS_46(arg), arg(46), arg(47)
#define WGMMA_REGS_50 WGMMA_REGS_48 ", %48, %49"
#define WGMMA_ARGS_50(arg) WGMMA_ARGS_48(arg), arg(48), arg(49)
#define WGMMA_REGS_52 WGMMA_REGS_50 ", %50, %51"
#define WGMMA_ARGS_52(arg) WGMMA_ARGS_50(arg), arg(50), arg(51)
#define WGMMA_REGS_54 WGMMA_REGS_52 ", %52, %53"
#define WGMMA_ARGS_54(arg) WGMMA_ARGS_52(arg), arg(52), arg(53)
#define WGMMA_REGS_56 WGMMA_REGS_54 ", %54, %55"
#define WGMMA_ARGS_56(arg) WGMMA_ARGS_54(arg), arg(54), arg(55)
#define WGMMA_REGS_58 WGMMA_REGS_56 ", %56, %57"
#define WGMMA_ARGS_58(arg) WGMMA_ARGS_56(arg), arg(56), arg(57)
#define WGMMA_REGS_60 WGMMA_REGS_58 ", %58, %59"
#define WGMMA_ARGS_60(arg) WGMMA_ARGS_58(arg), arg(58), arg(59)
#define WGMMA_REGS_62 WGMMA_REGS_60 ", %60, %61"
#define WGMMA_ARGS_62(arg) WGMMA_ARGS_60(arg), arg(60), arg(61)
#define WGMMA_REGS_64 WGMMA_REGS_62 ", %62, %63"
#define WGMMA_ARGS_64(arg) WGMMA_ARGS_62(arg), arg(62), arg(63)
#define WGMMA_REGS_66 WGMMA_REGS_64 ", %64, %65"
#define WGMMA_ARGS_66(arg) WGMMA_ARGS_64(arg), arg(64), arg(65)
#define WGMMA_REGS_68 WGMMA_REGS_66 ", %66, %67"
#define WGMMA_ARGS_68(arg) WGMMA_ARGS_66(arg), arg(66), arg(67)
#define WGMMA_REGS_70 WGMMA_REGS_68 ", %68, %69"
#define WGMMA_ARGS_70(arg) WGMMA_ARGS_68(arg), arg(68), arg(69)
#define WGMMA_REGS_72 WGMMA_REGS_70 ", %70, %71"
#define WGMMA_ARGS_72(arg) WGMMA_ARGS_70(arg), arg(70), arg(71)
#define WGMMA_REGS_74 WGMMA_REGS_72 ", %72, %73"
#define WGMMA_ARGS_74(arg) WGMMA_ARGS_72(arg), arg(72), arg(73)
#define WGMMA_REGS_76 WGMMA_REGS_74 ", %74, %75"
#define WGMMA_ARGS_76(arg) WGMMA_ARGS_74(arg), arg(74), arg(75)
#define WGMMA_REGS_78 WGMMA_REGS_76 ", %76, %77"
#define WGMMA_ARGS_78(arg) WGMMA_ARGS_76(arg), arg(76), arg(77)
#define WGMMA_REGS_80 WGMMA_REGS_78 ", %78, %79"
#define WGMMA_ARGS_80(arg) WGMMA_ARGS_78(arg), arg(78), arg(79)
#define WGMMA_REGS_82 WGMMA_REGS_80 ", %80, %81"
#define WGMMA_ARGS_82(arg) WGMMA_ARGS_80(arg), arg(80), arg(81)
#define WGMMA_REGS_84 WGMMA_REGS_82 ", %82, %83"
#define WGMMA_ARGS_84(arg) WGMMA_ARGS_82(arg), arg(82), arg(83)
#define WGMMA_REGS_86 WGMMA_REGS_84 ", %84, %85"
#define WGMMA_ARGS_86(arg) WGMMA_ARGS_84(arg), arg(84), arg(85)
#define WGMMA_REGS_88 WGMMA_REGS_86 ", %86, %87"
#define WGMMA_ARGS_88(arg) WGMMA_ARGS_86(arg), arg(86), arg(87)
#define WGMMA_REGS_90 WGMMA_REGS_88 ", %88, %89"
#define WGMMA_ARGS_90(arg) WGMMA_ARGS_88(arg), arg(88), arg(89)
#define WGMMA_REGS_92 WGMMA_REGS_90 ", %90, %91"
#define WGMMA_ARGS_92(arg) WGMMA_ARGS_90(arg), arg(90), arg(91)
#define WGMMA_REGS_94 WGMMA_REGS_92 ", %92, %93"
#define WGMMA_ARGS_94(arg) WGMMA_ARGS_92(arg), arg(92), arg(93)
#define WGMMA_REGS_96 WGMMA_REGS_94 ", %94, %95"
#define WGMMA_ARGS_96(arg) WGMMA_ARGS_94(arg), arg(94), arg(95)
#define WGMMA_REGS_98 WGMMA_REGS_96 ", %96, %97"
#define WGMMA_ARGS_98(arg) WGMMA_ARGS_96(arg), arg(96), arg(97)
#define WGMMA_REGS_100 WGMMA_REGS_98 ", %98, %99"
#define WGMMA_ARGS_100(arg) WGMMA_ARGS_98(arg), arg(98), arg(99)
#define WGMMA_REGS_102 WGMMA_REGS_100 ", %100, %101"
#define WGMMA_ARGS_102(arg) WGMMA_ARGS_100(arg), arg(100), arg(101)
#define WGMMA_REGS_104 WGMMA_REGS_102 ", %102, %103"
#define WGMMA_ARGS_104(arg) WGMMA_ARGS_102(arg), arg(102), arg(103)
#define WGMMA_REGS_106 WGMMA_REGS_104 ", %104, %105"
#define WGMMA_ARGS_106(arg) WGMMA_ARGS_104(arg), arg(104), arg(105)
#define WGMMA_REGS_108 WGMMA_REGS_106 ", %106, %107"
#define WGMMA_ARGS_108(arg) WGMMA_ARGS_106(arg), arg(106), arg(107)
#define WGMMA_REGS_110 WGMMA_REGS_108 ", %108, %109"
#define WGMMA_ARGS_110(arg) WGMMA_ARGS_108(arg), arg(108), arg(109)
#define WGMMA_REGS_112 WGMMA_REGS_110 ", %110, %111"
#define WGMMA_ARGS_112(arg) WGMMA_ARGS_110(arg), arg(110), arg(111)
#define WGMMA_REGS_114 WGMMA_REGS_112 ", %112, %113"
#define WGMMA_ARGS_114(arg) WGMMA_ARGS_112(arg), arg(112), arg(113)
#define WGMMA_REGS_116 WGMMA_REGS_114 ", %114, %115"
#define WGMMA_ARGS_116(arg) WGMMA_ARGS_114(arg), arg(114), arg(115)
#define WGMMA_REGS_118 WGMMA_REGS_116 ", %116, %117"
#define WGMMA_ARGS_118(arg) WGMMA_ARGS_116(arg), arg(116), arg(117)
#define WGMMA_REGS_120 WGMMA_REGS_118 ", %118, %119"
#define WGMMA_ARGS_120(arg) WGMMA_ARGS_118(arg), arg(118), arg(119)
#define WGMMA_REGS_122 WGMMA_REGS_120 ", %120, %121"
#define WGMMA_ARGS_122(arg) WGMMA_ARGS_120(arg), arg(120), arg(121)
#define WGMMA_REGS_124 WGMMA_REGS_122 ", %122, %123"
#define WGMMA_ARGS_124(arg) WGMMA_ARGS_122(arg), arg(122), arg(123)
#define WGMMA_REGS_126 WGMMA_REGS_124 ", %124, %125"
#define WGMMA_ARGS_126(arg) WGMMA_ARGS_124(arg), arg(124), arg(125)
#define WGMMA_REGS_128 WGMMA_REGS_126 ", %126, %127"
#define WGMMA_ARGS_128(arg) WGMMA_ARGS_126(arg), arg(126), arg(127)
#define WGMMA_INPUTS_2 "%2, %3"
#define WGMMA_INPUTS_4 "%4, %5"
#define WGMMA_INPUTS_6 "%6, %7"
#define WGMMA_INPUTS_8 "%8, %9"
#define WGMMA_INPUTS_10 "%10, %11"
#define WGMMA_INPUTS_12 "%12, %13"
#define WGMMA_INPUTS_14 "%14, %15"
#define WGMMA_INPUTS_16 "%16, %17"
#define WGMMA_INPUTS_18 "%18, %19"
#define WGMMA_INPUTS_20 "%20, %21"
#define WGMMA_INPUTS_22 "%22, %23"
#define WGMMA_INPUTS_24 "%24, %25"
#define WGMMA_INPUTS_26 "%26, %27"
#define WGMMA_INPUTS_28 "%28, %29"
#define WGMMA_INPUTS_30 "%30, %31"
#define WGMMA_INPUTS_32 "%32, %33"
#define WGMMA_INPUTS_34 "%34, %35"
#define WGMMA_INPUTS_36 "%36, %37"
#define WGMMA_INPUTS_38 "%38, %39"
#define WGMMA_INPUTS_40 "%40, %41"
#define WGMMA_INPUTS_42 "%42, %43"
#define WGMMA_INPUTS_44 "%44, %45"
#define WGMMA_INPUTS_46 "%46, %47"
#define WGMMA_INPUTS_48 "%48, %49"
#define WGMMA_INPUTS_50 "%50, %51"
#define WGMMA_INPUTS_52 "%52, %53"
#define WGMMA_INPUTS_54 "%54, %55"
#define WGMMA_INPUTS_56 "%56, %57"
#define WGMMA_INPUTS_58 "%58, %59"
#define WGMMA_INPUTS_60 "%60, %61"
#define WGMMA_INPUTS_62 "%62, %63"
#define WGMMA_INPUTS_64 "%64, %65"
#define WGMMA_INPUTS_66 "%66, %67"
#define WGMMA_INPUTS_68 "%68, %69"
#define WGMMA_INPUTS_70 "%70, %71"
#define WGMMA_INPUTS_72 "%72, %73"
#define WGMMA_INPUTS_74 "%74, %75"
#define WGMMA_INPUTS_76 "%76, %77"
#define WGMMA_INPUTS_78 "%78, %79"
#define WGMMA_INPUTS_80 "%80, %81"
#define WGMMA_INPUTS_82 "%82, %83"
#define WGMMA_INPUTS_84 "%84, %85"
#define WGMMA_INPUTS_86 "%86, %87"
#define WGMMA_INPUTS_88 "%88, %89"
#define WGMMA_INPUTS_90 "%90, %91"
#define WGMMA_INPUTS_92 "%92, %93"
#define WGMMA_INPUTS_94 "%94, %95"
#define WGMMA_INPUTS_96 "%96, %97"
#define WGMMA_INPUTS_98 "%98, %99"
#define WGMMA_INPUTS_100 "%100, %101"
#define WGMMA_INPUTS_102 "%102, %103"
#define WGMMA_INPUTS_104 "%104, %105"
#define WGMMA_INPUTS_106 "%106, %107"
#define WGMMA_INPUTS_108 "%108, %109"
#define WGMMA_INPUTS_110 "%110, %111"
#define WGMMA_INPUTS_112 "%112, %113"
#define WGMMA_INPUTS_114 "%114, %115"
#define WGMMA_INPUTS_116 "%116, %117"
#define WGMMA_INPUTS_118 "%118, %119"
#define WGMMA_INPUTS_120 "%120, %121"
#define WGMMA_INPUTS_122 "%122, %123"
#define WGMMA_INPUTS_124 "%124, %125"
#define WGMMA_INPUTS_126 "%126, %127"
#define WGMMA_INPUTS_128 "%128, %129"

#define WGMMA_ARG_f32(i) "+f"(d_frag[i])
#define WGMMA_ARG_f16(i) "+r"(d_frag[i])

template <int N, WGMMAType DType, WGMMAType AType, WGMMAType BType>
struct WGMMAInstruction;

#define DEFINE_WGMMA_INSTRUCTION(n, ksize, dtype, atype, btype, regs, extra) \
template <> \
struct WGMMAInstruction<n, WGMMAType::dtype, WGMMAType::atype, WGMMAType::btype> \
{ \
    using Register = WGMMAAccumulator<WGMMAType::dtype>::Register; \
    __device__ static void run(Register *d_frag, uint64_t a_desc, uint64_t b_desc) \
    { \
        asm volatile( \
            "wgmma.mma_async.sync.aligned.m64n" #n "k" #ksize "." #dtype "." #atype "." #btype " " \
            "{" WGMMA_REGS_##regs "}, " WGMMA_INPUTS_##regs ", 1, 1, 1" extra ";" \
            : WGMMA_ARGS_##regs(WGMMA_ARG_##dtype) \
            : "l"(a_desc), "l"(b_desc)); \
    } \
};

#define DEFINE_WGMMA_INSTRUCTIONS(n, fregs, hregs) \
    DEFINE_WGMMA_INSTRUCTION(n, 8, f32, tf32, tf32, fregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 16, f32, f16, f16, fregs, ", 0, 0") \
    DEFINE_WGMMA_INSTRUCTION(n, 16, f16, f16, f16, hregs, ", 0, 0") \
    DEFINE_WGMMA_INSTRUCTION(n, 16, f32, bf16, bf16, fregs, ", 0, 0") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f32, e5m2, e5m2, fregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f32, e5m2, e4m3, fregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f32, e4m3, e5m2, fregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f32, e4m3, e4m3, fregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f16, e5m2, e5m2, hregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f16, e5m2, e4m3, hregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f16, e4m3, e5m2, hregs, "") \
    DEFINE_WGMMA_INSTRUCTION(n, 32, f16, e4m3, e4m3, hregs, "")

DEFINE_WGMMA_INSTRUCTIONS(8, 4, 2)
DEFINE_WGMMA_INSTRUCTIONS(16, 8, 4)
DEFINE_WGMMA_INSTRUCTIONS(24, 12, 6)
DEFINE_WGMMA_INSTRUCTIONS(32, 16, 8)
DEFINE_WGMMA_INSTRUCTIONS(40, 20, 10)
DEFINE_WGMMA_INSTRUCTIONS(48, 24, 12)
DEFINE_WGMMA_INSTRUCTIONS(56, 28, 14)
DEFINE_WGMMA_INSTRUCTIONS(64, 32, 16)
DEFINE_WGMMA_INSTRUCTIONS(72, 36, 18)
DEFINE_WGMMA_INSTRUCTIONS(80, 40, 20)
DEFINE_WGMMA_INSTRUCTIONS(88, 44, 22)
DEFINE_WGMMA_INSTRUCTIONS(96, 48, 24)
DEFINE_WGMMA_INSTRUCTIONS(104, 52, 26)
DEFINE_WGMMA_INSTRUCTIONS(112, 56, 28)
DEFINE_WGMMA_INSTRUCTIONS(120, 60, 30)
DEFINE_WGMMA_INSTRUCTIONS(128, 64, 32)
DEFINE_WGMMA_INSTRUCTIONS(136, 68, 34)
DEFINE_WGMMA_INSTRUCTIONS(144, 72, 36)
DEFINE_WGMMA_INSTRUCTIONS(152, 76, 38)
DEFINE_WGMMA_INSTRUCTIONS(160, 80, 40)
DEFINE_WGMMA_INSTRUCTIONS(168, 84, 42)
DEFINE_WGMMA_INSTRUCTIONS(176, 88, 44)
DEFINE_WGMMA_INSTRUCTIONS(184, 92, 46)
DEFINE_WGMMA_INSTRUCTIONS(192, 96, 48)
DEFINE_WGMMA_INSTRUCTIONS(200, 100, 50)
DEFINE_WGMMA_INSTRUCTIONS(208, 104, 52)
DEFINE_WGMMA_INSTRUCTIONS(216, 108, 54)
DEFINE_WGMMA_INSTRUCTIONS(224, 112, 56)
DEFINE_WGMMA_INSTRUCTIONS(232, 116, 58)
DEFINE_WGMMA_INSTRUCTIONS(240, 120, 60)
DEFINE_WGMMA_INSTRUCTIONS(248, 124, 62)
DEFINE_WGMMA_INSTRUCTIONS(256, 128, 64)
