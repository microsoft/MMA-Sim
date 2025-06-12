import argparse
import ctypes

import mmasim
import random_test_utils


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


def test_ampere_mma_f16_random(trials: int):
    # sm_75 instructions
    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k8.f16.f16.f16.f16")
    mma = lib.mma_m16n8k8_row_col_f16_f16_f16_f16
    print(f"Running {trials} trials of test_mma_m16n8k8_f16_f16_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k8.f32.f16.f16.f32")
    mma = lib.mma_m16n8k8_row_col_f32_f16_f16_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_f16_f16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)

    # sm_80 instructions
    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k16.f16.f16.f16.f16")
    mma = lib.mma_m16n8k16_row_col_f16_f16_f16_f16
    print(f"Running {trials} trials of test_mma_m16n8k16_f16_f16_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k16.f32.f16.f16.f32")
    mma = lib.mma_m16n8k16_row_col_f32_f16_f16_f32
    print(f"Running {trials} trials of test_mma_m16n8k16_f32_f16_f16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


def test_ampere_mma_bf16_random(trials: int):
    # sm_80 instructions
    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k8.f32.bf16.bf16.f32")
    mma = lib.mma_m16n8k8_row_col_f32_bf16_bf16_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_bf16_bf16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k16.f32.bf16.bf16.f32")
    mma = lib.mma_m16n8k16_row_col_f32_bf16_bf16_f32
    print(f"Running {trials} trials of test_mma_m16n8k16_f32_bf16_bf16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


def test_ampere_mma_tf32_random(trials: int):
    # sm_80 instructions
    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k4.f32.tf32.tf32.f32")
    mma = lib.mma_m16n8k4_row_col_f32_tf32_tf32_f32
    print(f"Running {trials} trials of test_mma_m16n8k4_f32_tf32_tf32_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Ampere", "m16n8k8.f32.tf32.tf32.f32")
    mma = lib.mma_m16n8k8_row_col_f32_tf32_tf32_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_tf32_tf32_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=int, default=1000, nargs="?")
    args = parser.parse_args()
    test_ampere_mma_f16_random(args.trials)
    test_ampere_mma_bf16_random(args.trials)
    test_ampere_mma_tf32_random(args.trials)
    print("Tests passed!")
