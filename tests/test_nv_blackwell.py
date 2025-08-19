from mmasim.intrinsic.nv_blackwell import mma_intrinsics, tcgen05mma_intrinsics
from mmasim.simulator.nv import MMASim, TCGen05MMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing Blackwell instruction mma.{qualifier}")
        random_test(
            MMASim("Blackwell", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    for qualifier, intrinsic in tcgen05mma_intrinsics.items():
        print(f"Testing Hopper instruction tcgen05.mma.{qualifier}")
        random_test(
            TCGen05MMASim("Hopper", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
