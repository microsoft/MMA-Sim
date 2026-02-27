import math

import torch

from ..isa import amd
from .arithmetic import (
    fma,
    extract_significand_exponent,
    truncate_to_tf32,
    flush_denormal,
    pairwise_dot,
    amd_fused_dot_rd_add,
)


def chain_of_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    for i in range(a.numel()):
        c = fma(a[i], b[i], c)
    return c


def exact_fused_dot_product_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, L: int
) -> torch.Tensor:
    acc = c
    for i in range(0, a.numel(), L):
        p = a[i : i + L].double() * b[i : i + L].double()
        sum = acc + p.sum()
        if sum.isinf() or sum.isnan():
            acc = sum.float()
            continue
        s, e = extract_significand_exponent(acc)
        emin = e
        sum = int(s * 2**23)
        for j in range(L):
            sj, ej = extract_significand_exponent(p[j])
            if ej < emin:
                sum <<= emin - ej
                emin = ej
            sum += int(sj * 2**23) << (ej - emin)
        sign = 1 if sum >= 0 else -1
        sum = abs(sum)
        t = int(math.log2(sum))
        if t > 25:
            sticky = bool(sum & ((1 << (t - 25)) - 1))
            sum = (sum >> (t - 25) << 1) | sticky
            acc = torch.tensor(
                sign * sum * 2.0 ** (emin - 23 + t - 26), dtype=torch.float32
            )
        else:
            acc = torch.tensor(sign * sum * 2.0 ** (emin - 23), dtype=torch.float32)
    return acc


class mfma(amd.mfma):
    def __init__(self, arch: str, qualifier: str):
        super().__init__(arch, qualifier)
        self.flush_denormal = False
        self.is_xf32 = False
        self.is_two_stage = False
        if self.a_type == torch.float64:
            self.operation_type = "fma"
            self.group_size = 1
        elif self.a_type == torch.float32:
            if qualifier.endswith("xf32"):
                self.is_xf32 = True
                self.operation_type = "fused_dot_rd_add"
                self.group_size = 4
            else:
                self.operation_type = "fma"
                self.group_size = 1
        elif self.a_type == torch.float16:
            if self.arch == "CDNA3":
                self.operation_type = "fused_dot_rd_add"
                self.group_size = min(8, self.k)
                self.is_two_stage = True
            else:  # CDNA1 or CDNA2
                self.operation_type = "pairwise"
                self.group_size = 4
                self.flush_denormal = self.arch == "CDNA2"
        elif self.a_type == torch.bfloat16:
            if self.arch == "CDNA3":
                self.operation_type = "fused_dot_rd_add"
                self.group_size = min(8, self.k)
                self.is_two_stage = True
            else:  # CDNA1 or CDNA2
                self.operation_type = "pairwise"
                self.group_size = 4 if qualifier.endswith("_1k") else 2
                self.flush_denormal = self.arch == "CDNA2"
        else:  # fp8
            self.operation_type = "fused_dot_rd_add"
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
                sum = C[i, j]
                if self.arch == "CDNA1" and self.a_type == torch.bfloat16:
                    sum = exact_fused_dot_product_add(A[i, :], B[:, j], C[i, j], 2)
                elif self.arch == "CDNA1" and self.a_type == torch.float16:
                    sum = exact_fused_dot_product_add(A[i, :], B[:, j], C[i, j], 4)
                elif self.operation_type == "fma":
                    for l in range(k):
                        sum = fma(A[i, l], B[l, j], sum)
                elif self.operation_type == "pairwise":
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
                else:  # self.operation_type == "fused_dot_rd_add"
                    l = 0
                    while l < k:
                        fused_sum = amd_fused_dot_rd_add(
                            A[i, l : l + self.group_size],
                            B[l : l + self.group_size, j],
                            sum,
                            n_fractional_bits=24,
                            is_fp8=self.qualifier.endswith("8"),
                        )
                        sum = torch.tensor(fused_sum, dtype=self.d_type)  # RNE
                        l += self.group_size
                D[i, j] = sum
        return D
