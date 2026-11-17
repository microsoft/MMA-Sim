"""Run on a TPU v7 VM with TorchTPU installed."""

import torch

from mmasim.tpu import MMA
from mmasim_kernels.tpu import MMA as HardwareMMA
from helper import random_test


def test_tpu_v7(trials: int = 100) -> None:
    torch.manual_seed(0)
    for k in [128, 256]:
        kernel = HardwareMMA("TPU-v7", k)
        sim = MMA(
            kernel.arch,
            kernel.instruction,
            weight_load=kernel.weight_load,
            shape=(16, k, k),
        )
        print(f"Testing {kernel.arch} K{k}", flush=True)
        random_test(
            sim,
            kernel,
            allow_different_nan=False,
            trials=trials,
            device="cpu",
        )
    print("Tests passed!", flush=True)


if __name__ == "__main__":
    test_tpu_v7()
