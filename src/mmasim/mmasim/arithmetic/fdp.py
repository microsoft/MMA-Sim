"""Tensor implementations of FTZ-FDP BF16 and T-FDP FP8 arithmetic.

All leading dimensions are independent dots. Only the carried accumulation is
sequential; products, alignment, and partial reductions run as tensor operations.
"""

import torch

from .. import DPOperation
from .ftz_mul_add import flush_subnormal, ftz_add


def add_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RNE FP32 addition with sign-preserving input/output FTZ and canonical NaN."""
    result = ftz_add(
        flush_subnormal(a, keep_sign=True), flush_subnormal(b, keep_sign=True)
    )
    return torch.where(
        torch.isnan(result), torch.full_like(result, float("nan")), result
    )


def align_products(
    product: torch.Tensor, fraction: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align exact products to Emax-fraction with truncation toward zero."""
    _, exponent = torch.frexp(product)
    exponent = torch.where(product != 0, exponent - 1, -1024)
    step = exponent.amax(dim=-1) - fraction
    step = torch.where((product != 0).any(dim=-1), step, 0)
    scaled = torch.ldexp(product, -step.unsqueeze(-1))
    aligned = torch.trunc(scaled)
    return aligned, scaled, step


def partial_bf16(
    a: torch.Tensor, b: torch.Tensor, fraction: int, sticky: bool
) -> torch.Tensor:
    """One fused BF16 partial, including product FTZ/overflow and tail handling."""
    product = flush_subnormal(a.float()).double() * flush_subnormal(b.float()).double()
    invalid = torch.isnan(product).any(dim=-1)
    positive_inf = (product >= 2.0**128).any(dim=-1)
    negative_inf = (product <= -(2.0**128)).any(dim=-1)
    product = torch.where(
        (product.abs() < 2.0**-126)
        | ~torch.isfinite(product)
        | (product.abs() >= 2.0**128),
        0.0,
        product,
    )
    aligned, scaled, step = align_products(product, fraction)
    total = aligned.sum(dim=-1)
    if sticky:
        discarded = (scaled != aligned) & (aligned != 0)
        tails = (discarded & (scaled > 0)).any(dim=-1).to(torch.int32) - (
            discarded & (scaled < 0)
        ).any(dim=-1).to(torch.int32)
        large = total.abs() >= 2.0**24
        corrected = torch.where(
            large,
            total + tails.double() * 0.5,
            torch.where(total * tails < 0, total - total.sign(), total),
        )
        total = torch.where(total == 0, total, corrected)
    result = flush_subnormal(torch.ldexp(total, step).float())
    result = torch.where(positive_inf, float("inf"), result)
    result = torch.where(negative_inf, -float("inf"), result)
    return torch.where(invalid | (positive_inf & negative_inf), float("nan"), result)


def quantize_p9(value: torch.Tensor, round_nearest: bool) -> torch.Tensor:
    """Nine significant bits carried in FP32, with RTZ or ties-to-even RNE."""
    finite = torch.isfinite(value)
    mantissa, exponent = torch.frexp(torch.where(finite, value.double(), 0.0))
    scaled = mantissa * 512.0
    rounded = torch.round(scaled) if round_nearest else torch.trunc(scaled)
    result = torch.ldexp(rounded / 512.0, exponent).float()
    result = torch.where(result == 0, torch.zeros_like(result), result)
    result = torch.where(torch.isinf(value), value.float(), result)
    return torch.where(torch.isnan(value), float("nan"), result)


def cross_halves(value: torch.Tensor) -> torch.Tensor:
    """Arrange K256 as consecutive (k[4i:4i+4], k[128+4i:132+4i]) groups."""
    if value.shape[-1] != 256:
        raise ValueError("Cross-half reduction requires K=256")
    return (
        value.reshape(*value.shape[:-1], 2, 32, 4)
        .transpose(-3, -2)
        .reshape(*value.shape[:-1], 256)
    )


class MMA_FTZ_FDP(DPOperation):
    """FTZ-FDP BF16: fused partials, pairwise FP32 merges, and a serial carry.

    fraction is the alignment precision below the leading product bit. partial
    is the fused dot width; merge is the number of FP32 partials per carry update.
    """

    def __init__(
        self, fraction: int, partial: int, merge: int, sticky: bool, cross: bool = False
    ) -> None:
        if fraction not in (26, 27) or partial not in (4, 8) or merge not in (2, 4):
            raise ValueError("Unsupported FTZ-FDP configuration")
        self.fraction: int = fraction
        self.partial: int = partial
        self.merge: int = merge
        self.sticky: bool = sticky
        self.cross: bool = cross

    def dp(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-1] != b.shape[-1] or a.shape[-1] % (self.partial * self.merge):
            raise ValueError("Dot length must be a multiple of partial * merge")
        if self.cross:
            a, b = cross_halves(a), cross_halves(b)
        a = a.reshape(*a.shape[:-1], -1, self.partial)
        b = b.reshape(*b.shape[:-1], -1, self.partial)
        parts = partial_bf16(a, b, self.fraction, self.sticky)
        parts = add_fp32(parts[..., 0::2], parts[..., 1::2])
        if self.merge == 4:
            parts = add_fp32(parts[..., 0::2], parts[..., 1::2])
        acc = torch.zeros_like(parts[..., 0])
        for index in range(parts.shape[-1]):
            acc = add_fp32(acc, parts[..., index])
        return acc

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.dp(A.unsqueeze(-2), B.transpose(-1, -2).unsqueeze(-3))


class MMA_T_FDP(DPOperation):
    """T-FDP FP8: cross-half DP8 RTZ partials and serial p9 RNE accumulation."""

    def dp(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b = cross_halves(a), cross_halves(b)
        a = a.reshape(*a.shape[:-1], 32, 8)
        b = b.reshape(*b.shape[:-1], 32, 8)
        product = a.double() * b.double()
        invalid = torch.isnan(product).any(dim=-1)
        positive_inf = torch.isposinf(product).any(dim=-1)
        negative_inf = torch.isneginf(product).any(dim=-1)
        aligned, _, step = align_products(
            torch.where(torch.isfinite(product), product, 0.0), 8
        )
        parts = quantize_p9(torch.ldexp(aligned.sum(dim=-1), step), False)
        parts = torch.where(positive_inf, float("inf"), parts)
        parts = torch.where(negative_inf, -float("inf"), parts)
        parts = torch.where(
            invalid | (positive_inf & negative_inf), float("nan"), parts
        )
        acc = torch.zeros_like(parts[..., 0])
        for index in range(32):
            acc = quantize_p9(acc.double() + parts[..., index].double(), True)
        return acc

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.dp(A.unsqueeze(-2), B.transpose(-1, -2).unsqueeze(-3))
