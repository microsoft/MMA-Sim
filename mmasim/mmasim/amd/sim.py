import torch

from . import isa_mfma
from ..arithmetic import fma, fdpa, ftz_mul_add


class MFMA(isa_mfma.MFMA):
    def __init__(self, arch: str, shape_and_type: str):
        super().__init__(arch, shape_and_type)
        if self.a_type in [torch.float64, torch.float32] and not self.suffix.endswith(
            "xf32"
        ):
            self.arithmetic_op = fma.MMA_FMA()
        elif self.a_type in [torch.float8_e5m2fnuz, torch.float8_e4m3fnuz]:
            self.arithmetic_op = fdpa.MMA_GTR_FDPA(
                F=24, F2=31, rho="RNE-FP32", L_max=16
            )
        else:  # tf32/bf16/fp16
            if self.arch == "CDNA1":
                L_max = 4 if self.a_type == torch.float16 else 2
                self.arithmetic_op = fdpa.MMA_E_FDPA(L_max)
            elif self.arch == "CDNA2":
                if self.a_type == torch.bfloat16 and not self.suffix.endswith("_1k"):
                    P = 2
                else:
                    P = 4
                self.arithmetic_op = ftz_mul_add.MMA_FTZ_MUL_ADD(P)
            else:  # CDNA3
                L_max = 4 if self.a_type == torch.float32 else 8
                self.arithmetic_op = fdpa.MMA_TR_FDPA(
                    F=24, F2=31, rho="RNE-FP32", L_max=L_max
                )

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.arithmetic_op.dpa(a, b, c)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        return self.arithmetic_op(A, B, C)
