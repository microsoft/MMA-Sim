"""Run on the matching TPU with audited bundles built by tpu/export.py.

Set MMA_SIM_TPU_BUNDLES to the parent directory containing ARCH-FORMAT-kK
bundles, e.g. TPU-v7-bf16-k128. No runtime compilation or API fallback.
"""

import os
from pathlib import Path

import torch

from mmasim.tpu import MMA
from mmasim_kernels.tpu import MMA as HardwareMMA
from helper import random_test


def test_tpu_v7(trials: int = 100) -> None:
    torch.manual_seed(0)
    root = Path(os.environ["MMA_SIM_TPU_BUNDLES"])
    for fmt, k in [("bf16", 128), ("bf16", 256), ("e4m3fn", 256), ("e5m2", 256)]:
        bundle = root / f"TPU-v7-{fmt}-k{k}"
        kernel = HardwareMMA("TPU-v7", k, bundle=bundle, format=fmt)
        sim = MMA(
            kernel.arch,
            kernel.instruction,
            weight_load=kernel.weight_load,
            shape=(kernel.m, kernel.k, kernel.n),
        )
        print(
            f"Testing {kernel.arch} {kernel.instruction} {kernel.weight_load} K{k}",
            flush=True,
        )
        random_test(sim, kernel, allow_different_nan=False, trials=trials, device="cpu")
    print("Tests passed!", flush=True)


if __name__ == "__main__":
    test_tpu_v7()
