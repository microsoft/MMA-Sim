import ctypes


from behavior_test_utils import (
    DotAdd,
    test_tf32_input_rounding,
    test_fused_dot_add,
    test_f32_output_rounding,
    test_f16_output_rounding,
)

lib = ctypes.CDLL("./cuda/hopper.so")
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
# sm_90a instructions
lib.wgmma_m64n8k16_row_col_f16_f16_f16.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k16_row_col_f32_f16_f16.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k16_row_col_f32_bf16_bf16.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k8_row_col_f32_tf32_tf32.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f32_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f32_e5m2_e4m3.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f32_e4m3_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f32_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f16_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f16_e5m2_e4m3.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f16_e4m3_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_row_col_f16_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3


def test_hopper_mma_f16_behavior():
    # sm_75 instructions
    dotadd = DotAdd(lib.mma_m16n8k8_row_col_f16_f16_f16_f16, "m16n8k8.f16.f16.f16.f16")
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(lib.mma_m16n8k8_row_col_f32_f16_f16_f32, "m16n8k8.f32.f16.f16.f32")
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)

    # sm_80 instructions
    dotadd = DotAdd(
        lib.mma_m16n8k16_row_col_f16_f16_f16_f16, "m16n8k16.f16.f16.f16.f16"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.mma_m16n8k16_row_col_f32_f16_f16_f32, "m16n8k16.f32.f16.f16.f32"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)


def test_hopper_wgmma_f16_behavior():
    # sm_90a instructions
    dotadd = DotAdd(
        lib.wgmma_m64n8k16_row_col_f16_f16_f16, "m64n8k16.f16.f16.f16", is_wgmma=True
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k16_row_col_f32_f16_f16, "m64n8k16.f32.f16.f16", is_wgmma=True
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)


def test_hopper_mma_bf16_behavior():
    # sm_80 instructions
    dotadd = DotAdd(
        lib.mma_m16n8k8_row_col_f32_bf16_bf16_f32, "m16n8k8.f32.bf16.bf16.f32"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.mma_m16n8k16_row_col_f32_bf16_bf16_f32, "m16n8k16.f32.bf16.bf16.f32"
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)


def test_hopper_wgmma_bf16_behavior():
    # sm_90a instructions
    dotadd = DotAdd(
        lib.wgmma_m64n8k16_row_col_f32_bf16_bf16,
        "m64n8k16.f32.bf16.bf16",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)


def test_hopper_mma_tf32_behavior():
    # sm_80 instructions
    dotadd = DotAdd(
        lib.mma_m16n8k4_row_col_f32_tf32_tf32_f32, "m16n8k4.f32.tf32.tf32.f32"
    )
    test_tf32_input_rounding(dotadd)
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.mma_m16n8k8_row_col_f32_tf32_tf32_f32, "m16n8k8.f32.tf32.tf32.f32"
    )
    test_tf32_input_rounding(dotadd)
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)


def test_hopper_wgmma_tf32_behavior():
    # sm_90a instructions
    dotadd = DotAdd(
        lib.wgmma_m64n8k8_row_col_f32_tf32_tf32, "m64n8k8.f32.tf32.tf32", is_wgmma=True
    )
    test_tf32_input_rounding(dotadd)
    test_fused_dot_add(dotadd, n_fraction_bits=25)
    test_f32_output_rounding(dotadd)


def test_hopper_wgmma_fp8_behavior():
    # sm_90a instructions
    # d_type = f32
    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f32_e5m2_e5m2,
        "m64n8k32.f32.e5m2.e5m2",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f32_e5m2_e4m3,
        "m64n8k32.f32.e5m2.e4m3",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f32_e4m3_e5m2,
        "m64n8k32.f32.e4m3.e5m2",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f32_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f32_e4m3_e4m3,
        "m64n8k32.f32.e4m3.e4m3",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f32_output_rounding(dotadd)

    # d_type = f16
    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f16_e5m2_e5m2,
        "m64n8k32.f16.e5m2.e5m2",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f16_e5m2_e4m3,
        "m64n8k32.f16.e5m2.e4m3",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f16_e4m3_e5m2,
        "m64n8k32.f16.e4m3.e5m2",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f16_output_rounding(dotadd)

    dotadd = DotAdd(
        lib.wgmma_m64n8k32_row_col_f16_e4m3_e4m3,
        "m64n8k32.f16.e4m3.e4m3",
        is_wgmma=True,
    )
    test_fused_dot_add(dotadd, n_fraction_bits=13)
    test_f16_output_rounding(dotadd)


if __name__ == "__main__":
    test_hopper_mma_f16_behavior()
    test_hopper_wgmma_f16_behavior()
    test_hopper_mma_bf16_behavior()
    test_hopper_wgmma_bf16_behavior()
    test_hopper_mma_tf32_behavior()
    test_hopper_wgmma_tf32_behavior()
    test_hopper_wgmma_fp8_behavior()
    print("Test passed!")
