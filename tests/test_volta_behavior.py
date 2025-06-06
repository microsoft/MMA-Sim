import ctypes


from behavior_test_utils import (
    DotAdd,
    test_fused_dot_add,
    test_f16_output_rounding,
    test_f32_output_rounding,
)


lib = ctypes.CDLL("./cuda/volta.so")
lib.mma_m8n8k4_row_col_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m8n8k4_row_col_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m8n8k4_row_col_f32_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4


def test_volta_mma_f16_behavior():
    dotadd = DotAdd(lib.mma_m8n8k4_row_col_f16_f16_f16_f16, "m8n8k4.f16.f16.f16.f16")
    test_fused_dot_add(dotadd, n_fraction_bits=23)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(lib.mma_m8n8k4_row_col_f32_f16_f16_f16, "m8n8k4.f32.f16.f16.f16")
    test_fused_dot_add(dotadd, n_fraction_bits=23)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(lib.mma_m8n8k4_row_col_f32_f16_f16_f32, "m8n8k4.f32.f16.f16.f32")
    test_fused_dot_add(dotadd, n_fraction_bits=23)
    test_f32_output_rounding(dotadd)


if __name__ == "__main__":
    test_volta_mma_f16_behavior()
    print("Test passed!")
