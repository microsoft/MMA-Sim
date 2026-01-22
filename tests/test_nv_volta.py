from mmasim.kernels.nv_volta import mma_kernels
from mmasim.simulator.nv_ptx import mma

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_kernels.items():
        print(f"Testing Volta instruction mma.{qualifier}")
        random_test(
            mma("Volta", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
