from mmasim.probe.intrinsic.nv_ampere import mma_intrinsics
from mmasim.simulator.nv import MMA

from utils import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing Ampere instruction mma.{qualifier}")
        random_test(
            MMA("Ampere", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
