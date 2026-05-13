from mmasim_kernels.amd.cdna1 import mfma_kernels
from mmasim.amd import MFMA

from . import random_test


def test_amd_cdna1(trials: int = 100):
    for suffix, kernel in mfma_kernels.items():
        print(f"Testing CDNA1 instruction mfma_{suffix}")
        random_test(
            MFMA("CDNA1", suffix),
            kernel,
            allow_different_nan=True,
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_amd_cdna1()
