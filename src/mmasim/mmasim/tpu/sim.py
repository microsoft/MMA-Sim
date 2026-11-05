"""Numerical simulation of prepared TPU MXU instruction groups."""

import torch

from .. import DPOperation
from ..arithmetic.fdp import MMA_FTZ_FDP, MMA_T_FDP, add_fp32
from . import isa_vmatmul


class MMA(isa_vmatmul.MMA):
    """Prepared logical matrix product, including MUBR preparation when used.

    vmatpush3 exposes two independent streams unless an explicit narrow shape
    selects one. An optional FP32 C is added after the complete MXU result using
    external FP32 addition with input/output FTZ. It is not an initial PE carry.
    dp(a, b) exposes the underlying two-input operation; dpa(a, b, c) adds C.
    """

    def __init__(
        self,
        arch: str,
        instruction: str,
        *,
        weight_load: str,
        shape: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__(arch, instruction, weight_load=weight_load, shape=shape)
        self.model: DPOperation
        if self.format != "bf16":
            self.model = MMA_T_FDP()
        else:
            v7 = self.arch == "TPU-v7"
            self.model = MMA_FTZ_FDP(
                fraction=27 if v7 else 26,
                partial=8 if v7 and self.k == 256 else 4,
                merge=4 if self.arch == "TPU-v6e" and self.k == 256 else 2,
                sticky=not v7,
                cross=self.k == 256,
            )

    def validate(self, x: torch.Tensor) -> None:
        if not x.is_floating_point():
            raise ValueError("Floating-point operands required")
        if x.requires_grad:
            raise NotImplementedError(
                "Pass detached tensors; this simulator has no autograd"
            )
        # FP32 carriers are accepted only when they exactly encode the input format.
        value = x.double()
        converted = x.to(self.a_type).double()
        representable = (converted == value) | (
            torch.isnan(converted) & torch.isnan(value)
        )
        if not representable.all():
            raise ValueError(f"Inputs must be exactly representable in {self.a_type}")

    def dp(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != self.k or b.shape[-1] != self.k:
            raise ValueError(f"Expected exactly {self.k} terms per dot")
        if a.device != b.device:
            raise ValueError("Operands must be on the same device")
        self.validate(a)
        self.validate(b)
        return self.model.dp(a, b)

    @staticmethod
    def add_accumulator(result: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply the external FP32 merge to a completed MXU result."""
        if c.dtype != torch.float32 or c.shape != result.shape:
            raise ValueError(f"C must be FP32 with shape {tuple(result.shape)}")
        if c.device != result.device:
            raise ValueError("C must be on the same device as the operands")
        if c.requires_grad:
            raise NotImplementedError(
                "Pass detached tensors; this simulator has no autograd"
            )
        return add_fp32(result, c)

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """dp(a, b), followed by a separate FP32 addition of c."""
        return self.add_accumulator(self.dp(a, b), c)

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tuple(A.shape) != self.a_shape or tuple(B.shape) != self.b_shape:
            raise ValueError(f"Expected operand shapes {self.a_shape}, {self.b_shape}")
        result = self.dp(A.unsqueeze(-2), B.transpose(-1, -2).unsqueeze(-3))
        return result if C is None else self.add_accumulator(result, C)
