import ctypes


from behavior_test_utils import (
    DotAdd,
    test_tf32_input_rounding,
    test_fused_dot_add,
    test_f32_output_rounding,
    test_f16_output_rounding,
)

lib = ctypes.CDLL("./cuda/ampere.so")
# sm_75 instructions
lib.mma_m16n8k8_row_col_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_row_col_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
# sm_80 instructions
lib.mma_m16n8k16_row_col_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_row_col_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_row_col_f32_bf16_bf16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_row_col_f32_bf16_bf16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k4_row_col_f32_tf32_tf32_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_row_col_f32_tf32_tf32_f32.argtypes = [ctypes.c_void_p] * 4


def test_ampere_mma_f16_behavior():
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


def test_ampere_mma_bf16_behavior():
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


def test_ampere_mma_tf32_behavior():
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
    test_ampere_mma_f16_behavior()
    test_ampere_mma_bf16_behavior()
    test_ampere_mma_tf32_behavior()
    print("Test passed!")
