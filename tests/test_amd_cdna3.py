from mmasim.kernels.amd_cdna3 import mfma_kernels
from mmasim.simulator.amd import mfma

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mfma_kernels.items():
        print(f"Testing CDNA3 instruction mfma_{qualifier}")
        random_test(
            mfma("CDNA3", qualifier),
            intrinsic,
            allow_different_nan=True,
            trials=100,
        )
    print("Tests passed!")
