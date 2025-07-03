import ctypes


from behavior_test_utils import (
    DotAdd,
    test_fused_dot_add,
    test_f16_output_rounding,
    test_f32_output_rounding,
)


lib = ctypes.CDLL("./cuda/turing.so")
lib.mma_m16n8k8_row_col_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_row_col_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4


def test_turing_mma_f16_behavior():
    dotadd = DotAdd(lib.mma_m16n8k8_row_col_f16_f16_f16_f16, "m16n8k8.f16.f16.f16.f16")
    test_fused_dot_add(dotadd, n_fraction_bits=24)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(lib.mma_m16n8k8_row_col_f32_f16_f16_f32, "m16n8k8.f32.f16.f16.f32")
    test_fused_dot_add(dotadd, n_fraction_bits=24)
    test_f32_output_rounding(dotadd)


if __name__ == "__main__":
    test_turing_mma_f16_behavior()
    print("Test passed!")
