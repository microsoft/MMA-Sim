import ctypes


from behavior_test_utils import (
    DotAdd,
    test_tf32_input_rounding,
    test_fused_dot_add,
    test_f32_output_rounding,
    test_f16_output_rounding,
)

lib = ctypes.CDLL("./hip/cdna2.so")
# cdna1 instructions
lib.mfma_f32_32x32x2f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x8f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x2bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x8bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x2bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x2bf16.argtypes = [ctypes.c_void_p] * 4
# cdna2 instructions
lib.mfma_f32_32x32x8bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4bf16_1k.argtypes = [ctypes.c_void_p] * 4


def test_cdna2_mma_f16_behavior():
    # sm_75 instructions
    dotadd = DotAdd(lib.mma_m16n8k8_row_col_f16_f16_f16_f16, "m16n8k8.f16.f16.f16.f16")
    test_fused_dot_add(dotadd, n_fraction_bits=24)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(lib.mma_m16n8k8_row_col_f32_f16_f16_f32, "m16n8k8.f32.f16.f16.f32")
    test_fused_dot_add(dotadd, n_fraction_bits=24)
    test_f32_output_rounding(dotadd)

    # sm_80 instructions
    dotadd = DotAdd(
        lib.mma_m16n8k16_row_col_f16_f16_f16_f16, "m16n8k16.f16.f16.f16.f16"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=24, split_k=True)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.mma_m16n8k16_row_col_f32_f16_f16_f32, "m16n8k16.f32.f16.f16.f32"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=24, split_k=True)
    test_f32_output_rounding(dotadd)


def test_cdna2_mma_bf16_behavior():
    # sm_80 instructions
    dotadd = DotAdd(
        lib.mma_m16n8k8_row_col_f32_bf16_bf16_f32, "m16n8k8.f32.bf16.bf16.f32"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=24)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.mma_m16n8k16_row_col_f32_bf16_bf16_f32, "m16n8k16.f32.bf16.bf16.f32"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=24, split_k=True)
    test_f32_output_rounding(dotadd)


def test_cdna2_mma_f32_behavior():
    # sm_80 instructions
    dotadd = DotAdd(
        lib.mma_m16n8k4_row_col_f32_tf32_tf32_f32, "m16n8k4.f32.tf32.tf32.f32"
    )
    test_tf32_input_rounding(dotadd)
    test_fused_dot_add(dotadd, n_fraction_bits=24)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.mma_m16n8k8_row_col_f32_tf32_tf32_f32, "m16n8k8.f32.tf32.tf32.f32"
    )
    test_tf32_input_rounding(dotadd)
    test_fused_dot_add(dotadd, n_fraction_bits=24, split_k=True)
    test_f32_output_rounding(dotadd)


if __name__ == "__main__":
    test_cdna2_mma_f16_behavior()
    test_cdna2_mma_bf16_behavior()
    test_cdna2_mma_f32_behavior()
    print("Test passed!")
