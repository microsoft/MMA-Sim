from mmasim_kernels.nv_ptx.volta import mma_kernels
from mmasim.nv_ptx import MMA

from helper import random_test


def test_nv_volta(trials: int = 100):
    for shape_and_type, kernel in mma_kernels.items():
        print(f"Testing Volta instruction mma.{shape_and_type}")
        random_test(
            MMA("Volta", shape_and_type),
            kernel,
            allow_different_nan=False,
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_nv_volta()
