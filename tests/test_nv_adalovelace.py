from mmasim.intrinsic.nv_adalovelace import mma_intrinsics
from mmasim.simulator.nv import MMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing Ada Lovelace instruction mma.{qualifier}")
        random_test(
            MMASim("Ada Lovelace", qualifier),
            intrinsic,
            allow_different_nan=qualifier.endswith("f64"),
            trials=100,
        )
    print("Tests passed!")
