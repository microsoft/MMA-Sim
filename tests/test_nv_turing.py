from mmasim.kernels.nv_turing import mma_kernels
from mmasim.simulator.nv import MMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_kernels.items():
        print(f"Testing Turing instruction mma.{qualifier}")
        random_test(
            MMASim("Turing", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
