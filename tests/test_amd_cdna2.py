from mmasim.kernels.amd_cdna2 import mfma_kernels
from mmasim.simulator.amd import mfma

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mfma_kernels.items():
        print(f"Testing CDNA2 instruction mfma_{qualifier}")
        random_test(
            mfma("CDNA2", qualifier),
            intrinsic,
            allow_different_nan=True,
            trials=100,
        )
    print("Tests passed!")
