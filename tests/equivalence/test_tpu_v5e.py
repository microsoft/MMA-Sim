"""Run on a TPU v5e VM with TorchTPU installed."""

import torch

from mmasim.tpu import MMA
from mmasim_kernels.tpu import MMA as HardwareMMA
from helper import random_test


def test_tpu_v5e(trials: int = 100) -> None:
    torch.manual_seed(0)
    for k in [128]:
        kernel = HardwareMMA("TPU-v5e", k)
        sim = MMA(
            kernel.arch,
            kernel.instruction,
            weight_load=kernel.weight_load,
            shape=(16, k, k),
        )
        for distribution in ("bits", "normal", "exponents"):
            print(f"Testing {kernel.arch} K{k}: {distribution}", flush=True)
            random_test(
                sim,
                kernel,
                allow_different_nan=False,
                trials=trials,
                device="cpu",
                distribution=distribution,
            )
    print("Tests passed!", flush=True)


if __name__ == "__main__":
    test_tpu_v5e()
