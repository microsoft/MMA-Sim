from mmasim_kernels.nv_ptx.blackwell import (
    mma_kernels,
    tcgen05mma_kernels,
    tcgen05mma_block_scale_kernels,
)
from mmasim.nv_ptx import MMA, TCGen05MMA, TCGen05MMABlockScale

from . import random_test


def test_nv_blackwell(trials: int = 100):
    for shape_and_type, kernel in mma_kernels.items():
        print(f"Testing Blackwell instruction mma.{shape_and_type}")
        random_test(
            MMA("Blackwell", shape_and_type),
            kernel,
            allow_different_nan=shape_and_type.endswith("f64"),
            trials=trials,
        )
    for shape_and_type, kernel in tcgen05mma_kernels.items():
        print(f"Testing Blackwell instruction tcgen05.mma.{shape_and_type}")
        random_test(
            TCGen05MMA("Blackwell", shape_and_type),
            kernel,
            allow_different_nan=False,
            trials=trials,
        )
    for shape_and_type, kernel in tcgen05mma_block_scale_kernels.items():
        print(f"Testing Blackwell instruction tcgen05.mma.{shape_and_type}")
        random_test(
            TCGen05MMABlockScale("Blackwell", shape_and_type),
            kernel,
            allow_different_nan=False,
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_nv_blackwell()
