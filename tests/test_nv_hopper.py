from mmasim.probe.intrinsic.nv_hopper import mma_intrinsics, wgmma_intrinsics
from mmasim.simulator.nv import MMA, WGMMA

from utils import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing Hopper instruction mma.{qualifier}")
        random_test(
            MMA("Hopper", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    for qualifier, intrinsic in wgmma_intrinsics.items():
        print(f"Testing Hopper instruction wgmma.{qualifier}")
        random_test(
            WGMMA("Hopper", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
