from mmasim.intrinsic.amd_cdna3 import mfma_intrinsics
from mmasim.simulator.amd import MFMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mfma_intrinsics.items():
        print(f"Testing CDNA3 instruction mfma_{qualifier}")
        random_test(
            MFMASim("CDNA3", qualifier),
            intrinsic,
            allow_different_nan=True,
            trials=100,
        )
    print("Tests passed!")
