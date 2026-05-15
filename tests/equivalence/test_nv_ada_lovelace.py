from mmasim_kernels.nv_ptx.ada_lovelace import mma_kernels
from mmasim.nv_ptx import MMA

from helper import random_test


def test_nv_ada_lovelace(trials: int = 100):
    for shape_and_type, kernel in mma_kernels.items():
        print(f"Testing Ada Lovelace instruction mma.{shape_and_type}")
        random_test(
            MMA("Ada Lovelace", shape_and_type),
            kernel,
            allow_different_nan=shape_and_type.endswith("f64"),
            trials=trials,
        )
    print("Tests passed!")


if __name__ == "__main__":
    test_nv_ada_lovelace()
