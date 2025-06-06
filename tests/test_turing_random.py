import argparse
import ctypes

import mmasim
import random_test_utils

lib = ctypes.CDLL("./cuda/turing.so")
lib.mma_m16n8k8_row_col_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_row_col_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4


def test_turing_mma_f16_random(trials: int):
    mma_sim = mmasim.nv.MMA("Turing", "m16n8k8.f16.f16.f16.f16")
    mma = lib.mma_m16n8k8_row_col_f16_f16_f16_f16
    print(f"Running {trials} trials of test_mma_m16n8k8_f16_f16_f16_f16")
    random_test_utils.test_mma(mma_sim, mma, trials)

    mma_sim = mmasim.nv.MMA("Turing", "m16n8k8.f32.f16.f16.f32")
    mma = lib.mma_m16n8k8_row_col_f32_f16_f16_f32
    print(f"Running {trials} trials of test_mma_m16n8k8_f32_f16_f16_f32")
    random_test_utils.test_mma(mma_sim, mma, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=int, default=1000, nargs="?")
    args = parser.parse_args()
    test_turing_mma_f16_random(args.trials)
    print("Tests passed!")
