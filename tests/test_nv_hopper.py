from mmasim.kernels.nv_hopper import mma_kernels, wgmma_kernels
from mmasim.simulator.nv import MMASim, WGMMASim

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_kernels.items():
        print(f"Testing Hopper instruction mma.{qualifier}")
        random_test(
            MMASim("Hopper", qualifier),
            intrinsic,
            allow_different_nan=qualifier.endswith("f64"),
            trials=100,
        )
    for qualifier, intrinsic in wgmma_kernels.items():
        print(f"Testing Hopper instruction wgmma.{qualifier}")
        random_test(
            WGMMASim("Hopper", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
