from mmasim.kernels.nv_blackwell import mma_kernels, tcgen05mma_kernels
from mmasim.simulator.nv_ptx import mma, tcgen05mma

from random_test import random_test

if __name__ == "__main__":
    for qualifier, intrinsic in mma_kernels.items():
        print(f"Testing Blackwell instruction mma.{qualifier}")
        random_test(
            mma("Blackwell", qualifier),
            intrinsic,
            allow_different_nan=qualifier.endswith("f64"),
            trials=100,
        )
    for qualifier, intrinsic in tcgen05mma_kernels.items():
        print(f"Testing Blackwell instruction tcgen05.mma.{qualifier}")
        random_test(
            tcgen05mma("Blackwell", qualifier),
            intrinsic,
            allow_different_nan=False,
            trials=100,
        )
    print("Tests passed!")
