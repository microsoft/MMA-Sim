import argparse
import ctypes

import mmasim
import random_test_utils


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


def test_hopper_mma_f16_random(trials: int):
    # sm_75 instructions
    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k8.f16.f16.f16.f16")
    mma = lib.mma_m16n8k8_row_col_f16_f16_f16_f16
    print(f"Running {trials} trials of test_mma_m16n8k8_f16_f16_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k8.f32.f16.f16.f32")
    mma = lib.mma_m16n8k8_row_col_f32_f16_f16_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_f16_f16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)

    # sm_80 instructions
    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k16.f16.f16.f16.f16")
    mma = lib.mma_m16n8k16_row_col_f16_f16_f16_f16
    print(f"Running {trials} trials of test_mma_m16n8k16_f16_f16_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k16.f32.f16.f16.f32")
    mma = lib.mma_m16n8k16_row_col_f32_f16_f16_f32
    print(f"Running {trials} trials of test_mma_m16n8k16_f32_f16_f16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


def test_hopper_wgmma_f16_random(trials: int):
    # sm_90a instructions
    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k16.f16.f16.f16")
    mma = lib.wgmma_m64n8k16_row_col_f16_f16_f16
    print(f"Running {trials} trials of test_wgmma_m64n8k16_f16_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k16.f32.f16.f16")
    mma = lib.wgmma_m64n8k16_row_col_f32_f16_f16
    print(f"Running {trials} trials of test_wgmma_m64n8k16_f32_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)


def test_hopper_mma_bf16_random(trials: int):
    # sm_80 instructions
    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k8.f32.bf16.bf16.f32")
    mma = lib.mma_m16n8k8_row_col_f32_bf16_bf16_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_bf16_bf16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k16.f32.bf16.bf16.f32")
    mma = lib.mma_m16n8k16_row_col_f32_bf16_bf16_f32
    print(f"Running {trials} trials of test_mma_m16n8k16_f32_bf16_bf16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


def test_hopper_wgmma_bf16_random(trials: int):
    # sm_90a instructions
    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k16.f32.bf16.bf16")
    mma = lib.wgmma_m64n8k16_row_col_f32_bf16_bf16
    print(f"Running {trials} trials of test_wgmma_m64n8k16_f32_bf16_bf16")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)


def test_hopper_mma_tf32_random(trials: int):
    # sm_80 instructions
    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k4.f32.tf32.tf32.f32")
    mma = lib.mma_m16n8k4_row_col_f32_tf32_tf32_f32
    print(f"Running {trials} trials of test_mma_m16n8k4_f32_tf32_tf32_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Hopper", "m16n8k8.f32.tf32.tf32.f32")
    mma = lib.mma_m16n8k8_row_col_f32_tf32_tf32_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_tf32_tf32_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


def test_hopper_wgmma_tf32_random(trials: int):
    # sm_90a instructions
    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k8.f32.tf32.tf32")
    mma = lib.wgmma_m64n8k8_row_col_f32_tf32_tf32
    print(f"Running {trials} trials of test_wgmma_m64n8k8_f32_tf32_tf32")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)


def test_hopper_wgmma_fp8_random(trials: int):
    # sm_90a instructions
    # dtype = f32
    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f32.e5m2.e5m2")
    mma = lib.wgmma_m64n8k32_row_col_f32_e5m2_e5m2
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f32_e5m2_e5m2")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f32.e5m2.e4m3")
    mma = lib.wgmma_m64n8k32_row_col_f32_e5m2_e4m3
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f32_e5m2_e4m3")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f32.e4m3.e5m2")
    mma = lib.wgmma_m64n8k32_row_col_f32_e4m3_e5m2
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f32_e4m3_e5m2")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f32.e4m3.e4m3")
    mma = lib.wgmma_m64n8k32_row_col_f32_e4m3_e4m3
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f32_e4m3_e4m3")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    # dtype = f16
    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f16.e5m2.e5m2")
    mma = lib.wgmma_m64n8k32_row_col_f16_e5m2_e5m2
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f16_e5m2_e5m2")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f16.e5m2.e4m3")
    mma = lib.wgmma_m64n8k32_row_col_f16_e5m2_e4m3
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f16_e5m2_e4m3")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f16.e4m3.e5m2")
    mma = lib.wgmma_m64n8k32_row_col_f16_e4m3_e5m2
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f16_e4m3_e5m2")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)

    mma_sim = mmasim.nv.WGMMA("Hopper", "m64n8k32.f16.e4m3.e4m3")
    mma = lib.wgmma_m64n8k32_row_col_f16_e4m3_e4m3
    print(f"Running {trials} trials of test_wgmma_m64n8k32_f16_e4m3_e4m3")
    random_test_utils.test_mma(mma_sim, mma, trials, is_wgmma=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=int, default=1000, nargs="?")
    args = parser.parse_args()
    test_hopper_mma_f16_random(args.trials)
    test_hopper_wgmma_f16_random(args.trials)
    test_hopper_mma_bf16_random(args.trials)
    test_hopper_wgmma_bf16_random(args.trials)
    test_hopper_mma_tf32_random(args.trials)
    test_hopper_wgmma_tf32_random(args.trials)
    test_hopper_wgmma_fp8_random(args.trials)
    print("Tests passed!")
