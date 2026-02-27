from mmasim.kernels.amd_cdna1 import mfma_kernels
from mmasim.simulator.amd import mfma

from .random_test import random_test

if __name__ == "__main__":
    for qualifier, kernel in mfma_kernels.items():
        print(f"Testing CDNA1 instruction mfma_{qualifier}")
        random_test(
            mfma("CDNA1", qualifier),
            kernel,
            allow_different_nan=True,
            trials=100,
        )
    print("Tests passed!")
