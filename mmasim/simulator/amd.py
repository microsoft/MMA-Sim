import torch

from ..isa import MFMA
from .functions import (
    fma,
    truncate_to_tf32,
    flush_denormal,
    pairwise_dot,
    amd_fused_dot_add,
)


class MFMASim(MFMA):
    def __init__(self, arch: str, qualifier: str):
        MFMA.__init__(self, arch, qualifier)
        self.flush_denormal = False
        self.is_xf32 = False
        self.is_two_stage = False
        if self.a_type == torch.float32:
            if qualifier.endswith("xf32"):
                self.is_xf32 = True
                self.operation_type = "fused_dot_add"
                self.group_size = 4
            else:
                self.operation_type = "standard"
                self.group_size = 1
        elif self.a_type == torch.float16:
            if self.arch == "CDNA3":
                self.operation_type = "fused_dot_add"
                self.group_size = min(8, self.k)
                self.is_two_stage = True
            else:  # CDNA1 or CDNA2
                self.operation_type = "standard"
                self.group_size = 4
                self.flush_denormal = self.arch == "CDNA2"
        elif self.a_type == torch.bfloat16:
            if self.arch == "CDNA3":
                self.operation_type = "fused_dot_add"
                self.group_size = min(8, self.k)
                self.is_two_stage = True
            else:  # CDNA1 or CDNA2
                self.operation_type = "standard"
                self.group_size = 4 if qualifier.endswith("_1k") else 2
                self.flush_denormal = self.arch == "CDNA2"
        else:  # fp8
            self.operation_type = "fused_dot_add"
            self.group_size = 16
            self.is_two_stage = True

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        m, n, k = self.m, self.n, self.k
        A = A.cpu()
        B = B.cpu()
        C = C.cpu()
        D = torch.zeros((m, n), dtype=self.d_type)
        if self.flush_denormal:
            A = flush_denormal(A)
            B = flush_denormal(B)
            C = flush_denormal(C)
        if self.is_xf32:
            A = truncate_to_tf32(A)
            B = truncate_to_tf32(B)
        for i in range(m):
            for j in range(n):
                sum = C[i, j].to(dtype=self.d_type)
                if self.operation_type == "standard":
                    if self.group_size == 1:  # f32
                        for l in range(k):
                            sum = fma(A[i, l], B[l, j], sum)
                    else:
                        l = 0
                        while l < k:
                            group_sum = pairwise_dot(
                                A[i, l : l + self.group_size],
                                B[l : l + self.group_size, j],
                                self.flush_denormal,
                            )
                            sum = sum + group_sum
                            if self.flush_denormal:
                                sum = flush_denormal(sum, keep_sign=True)
                            l += self.group_size
                else:  # self.operation_type == "fused_dot_add"
                    l = 0
                    while l < k:
                        fused_sum = amd_fused_dot_add(
                            A[i, l : l + self.group_size],
                            B[l : l + self.group_size, j],
                            sum,
                            n_fraction_bits=24,
                            is_fp8=self.qualifier.endswith("8"),
                        )
                        sum = torch.tensor(fused_sum, dtype=self.d_type)  # RNE
                        l += self.group_size
                D[i, j] = sum
        return D
