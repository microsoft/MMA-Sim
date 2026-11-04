"""Descriptors for captured TPU MXU instruction groups.

The mnemonic is real LLO text. Shapes describe logical operands after unpacking
VMEM/register layouts, not a public TPU assembly ABI. Weight-loading state is
required because vmatmul alone does not specify its arithmetic mode.
"""

import torch
from .. import DPOperation

BF16_INSTRUCTIONS: dict[str, list[str]] = {
    "TPU-v5e": ["vmatmul.bf16.gmra", "vmatmul.bf16.vlgmr.msra.gmra"],
    "TPU-v6e": ["vmatmul.mubr.bf16.gmra", "vmatmul.mubr.bf16.vlgmr.msra.gmra"],
    "TPU-v7": [
        "vmatmul.mubr.bf16.gmra",
        "vmatmul.mubr.bf16.vlgmr.msra.gmra",
        "vmatmul.mubr.bf16.vlgmr.msra.gmra.mrb[0]",
        "vmatmul.mubr.bf16.vlgmr2.msra.gmra.mrb",
        "vmatmul.mubr.bf16.msra.gmra.mrb",
    ],
}

FP8_INSTRUCTIONS: dict[str, list[str]] = {
    "e4m3fn": [
        "vmatmul.mubr.f8e4m3fn.gmra",
        "vmatmul.mubr.f8e4m3fn.vlgmr.msra.gmra.mrb[0]",
    ],
    "e5m2": [
        "vmatmul.mubr.f8e5m2.gmra",
        "vmatmul.mubr.f8e5m2.vlgmr.msra.gmra.mrb[0]",
    ],
}


class MMA(DPOperation):
    """Captured instruction descriptor, with logical operand shapes in M,K,N order."""

    arch: str
    instruction: str
    weight_load: str
    format: str
    mxu: str | None
    m: int
    k: int
    n: int
    groups: int
    shape: tuple[int, int, int]
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]
    d_shape: tuple[int, ...]

    def __init__(
        self,
        arch: str,
        instruction: str,
        *,
        weight_load: str,
        shape: tuple[int, int, int] | None = None,
    ) -> None:
        if arch not in ("TPU-v5e", "TPU-v6e", "TPU-v7"):
            raise NotImplementedError("Unsupported TPU architecture")
        base = instruction
        self.mxu: str | None = None
        if base.endswith((".mxu0", ".mxu1", ".mxu2", ".mxu3")):
            base, self.mxu = base.rsplit(".", 1)
        if (arch == "TPU-v5e" and self.mxu not in (None, "mxu0")) or self.mxu in (
            "mxu2",
            "mxu3",
        ):
            raise NotImplementedError("Uncaptured MXU identifier")
        self.format = "bf16"
        fp8_names = {
            name: fmt for fmt, names in FP8_INSTRUCTIONS.items() for name in names
        }
        if arch == "TPU-v7" and base in fp8_names:
            if weight_load != "vmatpush1":
                raise ValueError("FP8 K256 requires vmatpush1 prepared weights")
            if shape is not None and tuple(shape) != (32, 256, 256):
                raise ValueError(
                    "FP8 group requires M,K,N=(32,256,256); pad inactive rows"
                )
            self.format = fp8_names[base]
            self.arch, self.instruction, self.weight_load = (
                arch,
                instruction,
                weight_load,
            )
            self.m, self.k, self.n, self.groups = 32, 256, 256, 1
            self.shape = (self.m, self.k, self.n)
            self.a_shape, self.b_shape, self.d_shape = (32, 256), (256, 256), (32, 256)
            return
        if base not in BF16_INSTRUCTIONS[arch]:
            raise NotImplementedError("Unsupported captured BF16 vmatmul instruction")
        loads = {
            "TPU-v5e": ("vmatpush",),
            "TPU-v6e": ("vmatpush3", "vmatpush1+vmatpush2"),
            "TPU-v7": ("vmatpush3", "vmatpush1"),
        }
        if weight_load not in loads[arch]:
            raise ValueError(f"{arch} weight_load must be one of {loads[arch]}")
        self.arch, self.instruction, self.weight_load = arch, instruction, weight_load
        narrow = arch == "TPU-v5e" or weight_load == "vmatpush3"
        expected = (16, 128, 128) if narrow else (16, 256, 256)
        if shape is not None and tuple(shape) != expected:
            raise ValueError(
                f"This architecture/load configuration requires M,K,N={expected}"
            )
        # Existing push3 calls expose both streams. An explicit narrow shape
        # selects one logical stream, with the other stream inactive.
        self.groups = 2 if weight_load == "vmatpush3" and shape is None else 1
        self.m, self.k, self.n = expected
        self.shape = expected  # M,K,N, unlike GPU mnemonic M,N,K suffixes.
        self.a_shape = ((2,) if self.groups == 2 else ()) + (self.m, self.k)
        self.b_shape = ((2,) if self.groups == 2 else ()) + (self.k, self.n)
        self.d_shape = ((2,) if self.groups == 2 else ()) + (self.m, self.n)

    @property
    def a_type(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "e4m3fn": torch.float8_e4m3fn,
            "e5m2": torch.float8_e5m2,
        }[self.format]

    @property
    def b_type(self) -> torch.dtype:
        return self.a_type

    d_type: torch.dtype = torch.float32
