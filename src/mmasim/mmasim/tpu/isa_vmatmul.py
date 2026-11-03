"""Descriptors for captured TPU MXU instruction groups.

The mnemonic is real LLO text. Shapes describe logical operands after unpacking
VMEM/register layouts, not a public TPU assembly ABI. Weight-loading state is
required because vmatmul alone does not specify its arithmetic mode.
"""

import torch
from .. import MMAOperation


class MMA(MMAOperation):
    def __init__(self, arch: str, instruction: str, *, weight_load: str, shape=None):
        if arch not in ("TPU-v5e", "TPU-v6e", "TPU-v7"):
            raise NotImplementedError("Unsupported TPU architecture")
        base = instruction
        self.mxu = None
        if base.endswith((".mxu0", ".mxu1", ".mxu2", ".mxu3")):
            base, self.mxu = base.rsplit(".", 1)
        if (arch == "TPU-v5e" and self.mxu not in (None, "mxu0")) or self.mxu in (
            "mxu2",
            "mxu3",
        ):
            raise NotImplementedError("Uncaptured MXU identifier")
        self.format = "bf16"
        fp8_names = {
            f"vmatmul.mubr.{kind}.{suffix}": fmt
            for kind, fmt in [("f8e4m3fn", "e4m3fn"), ("f8e5m2", "e5m2")]
            for suffix in ["gmra", "vlgmr.msra.gmra.mrb[0]"]
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
        names = {
            "TPU-v5e": ("vmatmul.bf16.gmra", "vmatmul.bf16.vlgmr.msra.gmra"),
            "TPU-v6e": ("vmatmul.mubr.bf16.gmra", "vmatmul.mubr.bf16.vlgmr.msra.gmra"),
            "TPU-v7": (
                "vmatmul.mubr.bf16.gmra",
                "vmatmul.mubr.bf16.vlgmr.msra.gmra",
                "vmatmul.mubr.bf16.vlgmr.msra.gmra.mrb[0]",
                "vmatmul.mubr.bf16.vlgmr2.msra.gmra.mrb",
                "vmatmul.mubr.bf16.msra.gmra.mrb",
            ),
        }
        if base not in names[arch]:
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
    def a_type(self):
        return {
            "bf16": torch.bfloat16,
            "e4m3fn": torch.float8_e4m3fn,
            "e5m2": torch.float8_e5m2,
        }[self.format]

    @property
    def b_type(self):
        return self.a_type

    c_type = torch.float32
    d_type = torch.float32

    @property
    def reduction_groups(self):
        """Local dot-product term indices per update in the numerical model, before row packing.

        For dual mode this applies separately to each independent stream.
        V6 wide mode merges four rounded DP4 outputs in cross-half pairs.
        V7 wide mode merges two rounded fused DP8 outputs. The merged FP32
        value then updates the running sum.
        This is a numerical mapping, not a physical lane assignment.
        """
        if self.format != "bf16":
            return tuple(
                (tuple(range(j, j + 4)) + tuple(range(j + 128, j + 132)),)
                for j in range(0, 128, 4)
            )
        if self.k == 128:
            return tuple(
                (tuple(range(i, i + 4)), tuple(range(i + 4, i + 8)))
                for i in range(0, 128, 8)
            )
        if self.arch == "TPU-v7":
            return tuple(
                (
                    tuple(range(i, i + 4)) + tuple(range(i + 128, i + 132)),
                    tuple(range(i + 4, i + 8)) + tuple(range(i + 132, i + 136)),
                )
                for i in range(0, 128, 8)
            )
        return tuple(
            (
                tuple(range(i, i + 4)),
                tuple(range(i + 128, i + 132)),
                tuple(range(i + 4, i + 8)),
                tuple(range(i + 132, i + 136)),
            )
            for i in range(0, 128, 8)
        )
