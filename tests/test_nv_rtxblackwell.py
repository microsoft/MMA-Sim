from mmasim.intrinsic.nv_rtxblackwell import mma_intrinsics
from mmasim.simulator.nv import MMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing RTX Blackwell instruction mma.{qualifier}")
        random_test(
            MMASim("RTX Blackwell", qualifier),
            intrinsic,
            allow_different_nan=qualifier.endswith("f64"),
            trials=100,
        )
    print("Tests passed!")
