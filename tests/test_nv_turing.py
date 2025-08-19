from mmasim.intrinsic.nv_turing import mma_intrinsics
from mmasim.simulator.nv import MMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing Turing instruction mma.{qualifier}")
        random_test(
            MMASim("Turing", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
