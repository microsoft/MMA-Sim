from mmasim.kernels.nv_adalovelace import mma_kernels
from mmasim.simulator.nv_ptx import mma

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_kernels.items():
        print(f"Testing Ada Lovelace instruction mma.{qualifier}")
        random_test(
            mma("Ada Lovelace", qualifier),
            intrinsic,
            allow_different_nan=qualifier.endswith("f64"),
            trials=100,
        )
    print("Tests passed!")
