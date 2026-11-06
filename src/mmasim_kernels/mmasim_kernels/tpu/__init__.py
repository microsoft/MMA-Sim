"""TorchTPU hardware kernels for validated BF16 instruction-group footprints."""

import torch

from mmasim.tpu.isa_vmatmul import MMA as Instruction
from .. import MMAKernel


class MMA(MMAKernel):
    """Compiler-generated BF16 MXU product followed by an external FP32 C add.

    These logical footprints were checked against native LLO. This wrapper does
    not assemble an opcode or select a physical MXU; TorchTPU compiles the GEMM.
    """

    def __init__(self, arch: str, k: int = 128) -> None:
        if arch not in ("TPU-v5e", "TPU-v6e") or k not in (128, 256):
            raise ValueError("Supported footprints are v5e K128 and v6e K128/K256")
        if arch == "TPU-v5e" and k != 128:
            raise ValueError("The v5e instruction footprint has K128")
        import torch_tpu  # noqa: F401; register the optional TPU backend

        self.arch: str = arch
        self.instruction: str = (
            "vmatmul.bf16.vlgmr.msra.gmra"
            if arch == "TPU-v5e"
            else "vmatmul.mubr.bf16.vlgmr.msra.gmra"
        )
        self.weight_load: str = (
            "vmatpush"
            if arch == "TPU-v5e"
            else "vmatpush3" if k == 128 else "vmatpush1+vmatpush2"
        )
        descriptor = Instruction(
            arch, self.instruction, weight_load=self.weight_load, shape=(16, k, k)
        )
        self.m, self.n, self.k = descriptor.m, descriptor.n, descriptor.k
        self.a_type, self.b_type = descriptor.a_type, descriptor.b_type
        self.c_type, self.d_type = descriptor.c_type, descriptor.d_type

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        A, B, C = A.to("tpu"), B.to("tpu"), C.to("tpu")
        product = torch.mm(A, B, out_dtype=torch.float32)
        # Materialize the MXU output before C is added: C is not an initial PE carry.
        product.cpu()
        return (product + C).cpu()

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        A = torch.zeros((self.m, self.k), dtype=self.a_type)
        B = torch.zeros((self.k, self.n), dtype=self.b_type)
        C = torch.zeros((self.m, self.n), dtype=self.c_type)
        A[0, : a.numel()] = a.cpu()
        B[: b.numel(), 0] = b.cpu()
        C[0, 0] = c.cpu()
        return self(A, B, C)[0, 0]
