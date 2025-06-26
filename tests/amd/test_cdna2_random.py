import argparse
import ctypes

import mmasim
import random_test_utils


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


def test_cdna2_mfmaf16_random(trials: int):
    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x8f16")
    mfma = lib.mfma_f32_32x32x8f16
    print(f"Running {trials} trials of test_mfma_f32_32x32x8f16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x4f16")
    mfma = lib.mfma_f32_32x32x4f16
    print(f"Running {trials} trials of test_mfma_f32_32x32x4f16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x16f16")
    mfma = lib.mfma_f32_16x16x16f16
    print(f"Running {trials} trials of test_mfma_f32_16x16x16f16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x4f16")
    mfma = lib.mfma_f32_16x16x4f16
    print(f"Running {trials} trials of test_mfma_f32_16x16x4f16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_4x4x4f16")
    mfma = lib.mfma_f32_4x4x4f16
    print(f"Running {trials} trials of test_mfma_f32_4x4x4f16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

def test_cdna2_mfmabf16_random(trials: int):
    # CDNA1 instructions
    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x4bf16")
    mfma = lib.mfma_f32_32x32x4bf16
    print(f"Running {trials} trials of test_mfma_f32_32x32x4bf16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x2bf16")
    mfma = lib.mfma_f32_32x32x2bf16
    print(f"Running {trials} trials of test_mfma_f32_32x32x2bf16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x8bf16")
    mfma = lib.mfma_f32_16x16x8bf16
    print(f"Running {trials} trials of test_mfma_f32_16x16x8bf16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x2bf16")
    mfma = lib.mfma_f32_16x16x2bf16
    print(f"Running {trials} trials of test_mfma_f32_16x16x2bf16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_4x4x2bf16")
    mfma = lib.mfma_f32_4x4x2bf16
    print(f"Running {trials} trials of test_mfma_f32_4x4x2bf16")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    # CDNA2 instructions
    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x8bf16_1k")
    mfma = lib.mfma_f32_32x32x8bf16_1k
    print(f"Running {trials} trials of test_mfma_f32_32x32x8bf16_1k")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x4bf16_1k")
    mfma = lib.mfma_f32_32x32x4bf16_1k
    print(f"Running {trials} trials of test_mfma_f32_32x32x4bf16_1k")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x16bf16_1k")
    mfma = lib.mfma_f32_16x16x16bf16_1k
    print(f"Running {trials} trials of test_mfma_f32_16x16x16_bf1k")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x4bf16_1k")
    mfma = lib.mfma_f32_16x16x4bf16_1k
    print(f"Running {trials} trials of test_mfma_f32_16x16x4bf16_1k")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_4x4x4bf16_1k")
    mfma = lib.mfma_f32_4x4x4bf16_1k
    print(f"Running {trials} trials of test_mfma_f32_4x4x4bf16_1k")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)


def test_cdna2_mfma_f32_random(trials: int):
    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x2f32")
    mfma = lib.mfma_f32_32x32x2f32
    print(f"Running {trials} trials of test_mfma_f32_32x32x2f32")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_32x32x1f32")
    mfma = lib.mfma_f32_32x32x1f32
    print(f"Running {trials} trials of test_mfma_f32_32x32x1f32")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x4f32")
    mfma = lib.mfma_f32_16x16x4f32
    print(f"Running {trials} trials of test_mfma_f32_16x16x4f32")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_16x16x1f32")
    mfma = lib.mfma_f32_16x16x1f32
    print(f"Running {trials} trials of test_mfma_f32_16x16x1f32")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)

    mfma_sim = mmasim.amd.MFMA("CDNA2", "f32_4x4x1f32")
    mfma = lib.mfma_f32_4x4x1f32
    print(f"Running {trials} trials of test_mfma_f32_4x4x1f32")
    random_test_utils.test_mfma(mfma_sim, mfma, trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=int, default=1000, nargs="?")
    args = parser.parse_args()
    test_cdna2_mfmaf16_random(args.trials)
    test_cdna2_mfmabf16_random(args.trials)
    test_cdna2_mfma_f32_random(args.trials)
    print("Tests passed!")
