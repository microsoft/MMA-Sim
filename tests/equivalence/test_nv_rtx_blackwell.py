from mmasim_kernels.nv_ptx.rtx_blackwell import mma_kernels, mma_block_scale_kernels
from mmasim.nv_ptx import MMA, MMABlockScale

from helper import random_test


def test_nv_rtx_blackwell(trials: int = 100):
    for shape_and_type, kernel in mma_kernels.items():
        print(f"Testing RTX Blackwell instruction mma.{shape_and_type}")
        random_test(
            MMA("RTX Blackwell", shape_and_type),
            kernel,
            allow_different_nan=shape_and_type.endswith("f64"),
            trials=trials,
        )
    for shape_and_type, kernel in mma_block_scale_kernels.items():
        print(f"Testing RTX Blackwell instruction mma.{shape_and_type}")
        random_test(
            MMABlockScale("RTX Blackwell", shape_and_type),
            kernel,
            allow_different_nan=False,
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_nv_rtx_blackwell()
