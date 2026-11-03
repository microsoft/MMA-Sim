"""Finite-input reference for prepared TPU BF16 and FP8 MXU groups."""

import torch

from . import isa_vmatmul
from ._validation import validate_words as _validate_words
from .arithmetic import bf16_dot, fp8_dot


class MMA(isa_vmatmul.MMA):
    """MMA(arch, instruction, weight_load=...)(A, B, C=None).

    Includes the corresponding MUBR preparation and already loaded RHS state.
    Inputs are logical matrices, not packed VREGs. vmatpush3 returns both
    independent products unless shape selects a single stream. Wide mode uses
    push1+push2 on v6e and push1 on v7.
    C must be zero: arbitrary accumulator initialization has not been decoded.
    An external merge belongs to the caller, not to this instruction group.
    """

    def dot_bits(self, a, b, c=0):
        """One output dot (one stream for push3), using FP32 operand encodings."""
        if len(a) != self.k or len(b) != self.k:
            raise ValueError(f"Expected exactly {self.k} terms per output dot")
        aa = _validate_words(a, self.format)
        bb = _validate_words(b, self.format)
        cc = _validate_words([c], "fp32")[0]
        if cc & 0x7FFFFFFF:
            raise NotImplementedError(
                "Only zero C is supported; merge FP32 results explicitly"
            )
        # Reuse the measured numerical core; no shape-to-compiler-profile lookup.
        return self._dot(aa, bb)

    def _dot(self, a, b):
        if self.format != "bf16":
            return fp8_dot(a, b)
        return bf16_dot(a, b, self.arch)

    @staticmethod
    def _words(x, shape, fmt):
        if not isinstance(x, torch.Tensor) or x.device.type != "cpu":
            raise ValueError("CPU PyTorch tensors required")
        if tuple(x.shape) != shape or not x.is_floating_point():
            raise ValueError(f"Expected floating-point tensor with shape {shape}")
        if x.requires_grad:
            raise NotImplementedError(
                "Reference simulator has no autograd; pass detached tensors"
            )
        if x.dtype == torch.float64 and not torch.equal(x.float().double(), x):
            raise ValueError("FP64 values must be exactly representable in FP32")
        words = (
            x.float()
            .contiguous()
            .view(torch.int32)
            .to(torch.int64)
            .bitwise_and(0xFFFFFFFF)
        )
        return _validate_words(words.flatten().tolist(), fmt)

    def __call__(self, A, B, C=None):
        aa = self._words(A, self.a_shape, self.format)
        bb = self._words(B, self.b_shape, self.format)
        if C is not None:
            if not isinstance(C, torch.Tensor) or C.dtype != torch.float32:
                raise ValueError("C must be an FP32 tensor")
            cc = self._words(C, self.d_shape, "fp32")
            if any(x & 0x7FFFFFFF for x in cc):
                raise NotImplementedError(
                    "Only zero C is supported; merge FP32 results explicitly"
                )
        out = []
        for group in range(self.groups):
            a0, b0 = group * self.m * self.k, group * self.k * self.n
            rows = [aa[a0 + i * self.k : a0 + (i + 1) * self.k] for i in range(self.m)]
            cols = [bb[b0 + j : b0 + self.k * self.n : self.n] for j in range(self.n)]
            # Identical rows/columns have identical arithmetic. Memoization keeps
            # sparse diagnostic matrices practical without changing reduction order.
            cache = {}
            for row in rows:
                for col in cols:
                    key = (tuple(row), tuple(col))
                    if key not in cache:
                        cache[key] = self._dot(row, col)
                    out.append(cache[key])
        return (
            torch.tensor(out, dtype=torch.int64)
            .to(torch.int32)
            .view(torch.float32)
            .reshape(self.d_shape)
        )

    def dpa(self, a, b, c):
        """Evaluate one logical output dot; not a physical DPA8 instruction."""
        aa = self._words(a, (self.k,), self.format)
        bb = self._words(b, (self.k,), self.format)
        cc = self._words(c, (), "fp32")[0]
        word = self.dot_bits(aa, bb, cc)
        return torch.tensor(word, dtype=torch.int64).to(torch.int32).view(torch.float32)
