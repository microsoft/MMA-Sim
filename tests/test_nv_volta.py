from mmasim.intrinsic.nv_volta import mma_intrinsics
from mmasim.simulator.nv import MMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing Volta instruction mma.{qualifier}")
        random_test(
            MMASim("Volta", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
