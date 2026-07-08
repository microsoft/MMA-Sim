from mmasim_kernels.nv_ptx.turing import mma_kernels
from mmasim.nv_ptx import MMA

from helper import random_test


def test_nv_turing(trials: int = 100):
    for shape_and_type, kernel in mma_kernels.items():
        print(f"Testing Turing instruction mma.{shape_and_type}")
        random_test(
            MMA("Turing", shape_and_type),
            kernel,
            allow_different_nan=False,
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_nv_turing()
