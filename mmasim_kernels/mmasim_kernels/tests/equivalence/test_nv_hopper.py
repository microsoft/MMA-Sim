from mmasim_kernels.nv_ptx.hopper import mma_kernels, wgmma_kernels
from mmasim.nv_ptx import MMA, WGMMA

from . import random_test


def test_nv_hopper(trials: int = 100):
    for shape_and_type, kernel in mma_kernels.items():
        print(f"Testing Hopper instruction mma.{shape_and_type}")
        random_test(
            MMA("Hopper", shape_and_type),
            kernel,
            allow_different_nan=shape_and_type.endswith("f64"),
            trials=trials,
        )
    for shape_and_type, kernel in wgmma_kernels.items():
        print(f"Testing Hopper instruction wgmma.{shape_and_type}")
        random_test(
            WGMMA("Hopper", shape_and_type),
            kernel,
            allow_different_nan=False,
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_nv_hopper()
