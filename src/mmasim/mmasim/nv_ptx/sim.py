import torch

from . import isa_mma, isa_wgmma, isa_tcgen05mma
from ..arithmetic import fma, fdpa


class MMA(isa_mma.MMA):
    def __init__(self, arch: str, shape_and_type: str):
        super().__init__(arch, shape_and_type)
        if self.a_type == torch.float64:
            self.arithmetic_op = fma.MMA_FMA()
        else:
            if self.arch == "Ada Lovelace" and self.a_type in [
                torch.float8_e5m2,
                torch.float8_e4m3fn,
            ]:
                F = 13
                rho = "RNE-FP16" if self.d_type == torch.float16 else "RZ-E8M13"
            else:
                F_table = {
                    "Volta": 23,
                    "Turing": 24,
                    "Ampere": 24,
                    "Ada Lovelace": 24,
                    "Hopper": 25,
                    "Blackwell": 25,
                    "RTX Blackwell": 25,
                }
                F = F_table[arch]
                rho = "RNE-FP16" if self.d_type == torch.float16 else "RZ-FP32"
            L_max_table = {
                "Volta": 4 * 2,
                "Turing": 8 * 2,
                "Ampere": 8 * 2,
                "Ada Lovelace": 16,
                "Hopper": 32,
                "Blackwell": 32,
                "RTX Blackwell": 32,
            }
            L_max = L_max_table[arch] // self.a_type.itemsize
            self.arithmetic_op = fdpa.MMA_T_FDPA(F, rho, L_max)

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.arithmetic_op.dpa(a, b, c)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op(A, B, C)


class MMABlockScale(isa_mma.MMABlockScale):
    def __init__(self, arch: str, shape_and_type: str):
        super().__init__(arch, shape_and_type)
        if self.k == 64:
            self.arithmetic_op = fdpa.MMA_GST_FDPA(G=16, F=35, rho="RZ-FP32", L_max=64)
        else:
            self.arithmetic_op = fdpa.MMA_ST_FDPA(F=25, rho="RZ-FP32", L_max=32)

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op.dpa(a, b, c, alpha, beta)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op(A, B, C, alpha, beta)


class WGMMA(isa_wgmma.WGMMA):
    def __init__(self, arch: str, shape_and_type: str):
        super().__init__(arch, shape_and_type)
        assert self.arch == "Hopper"
        if self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            F = 13
            rho = "RNE-FP16" if self.d_type == torch.float16 else "RZ-E8M13"
        else:
            F = 25
            rho = "RNE-FP16" if self.d_type == torch.float16 else "RZ-FP32"
        L_max = 32 // self.a_type.itemsize
        self.arithmetic_op = fdpa.MMA_T_FDPA(F, rho, L_max)

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.arithmetic_op.dpa(a, b, c)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op(A, B, C)


class TCGen05MMA(isa_tcgen05mma.TCGen05MMA):
    def __init__(self, arch: str, shape_and_type: str):
        super().__init__(arch, shape_and_type)
        F = 25
        rho = "RNE-FP16" if self.d_type == torch.float16 else "RZ-FP32"
        L_max = 32 // self.a_type.itemsize
        self.arithmetic_op = fdpa.MMA_T_FDPA(F, rho, L_max)

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.arithmetic_op.dpa(a, b, c)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op(A, B, C)


class TCGen05MMABlockScale(isa_tcgen05mma.TCGen05MMABlockScale):
    def __init__(self, arch: str, shape_and_type: str):
        super().__init__(arch, shape_and_type)
        if self.k == 64:
            self.arithmetic_op = fdpa.MMA_GST_FDPA(G=16, F=35, rho="RZ-FP32", L_max=64)
        else:
            self.arithmetic_op = fdpa.MMA_ST_FDPA(F=25, rho="RZ-FP32", L_max=32)

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op.dpa(a, b, c, alpha, beta)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op(A, B, C, alpha, beta)
