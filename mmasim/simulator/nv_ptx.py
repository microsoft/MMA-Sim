import torch

from ..isa import nv_ptx
from .arithmetic import (
    fma,
    truncate_to_tf32,
    unpack_fp4_tensor,
    nv_fused_dot_add,
    nv_fused_dot_add_with_block_scale,
)


class mma(nv_ptx.mma):
    def __init__(self, arch: str, qualifier: str):
        super().__init__(arch, qualifier)
        # set n_accum_fractional_bits and output_type
        arch_accum_fractional_bits = {
            "Volta": 23,
            "Turing": 24,
            "Ampere": 24,
            "Ada Lovelace": 24,
            "Hopper": 25,
            "Blackwell": 25,
            "RTX Blackwell": 25,
        }
        self.n_accum_fractional_bits = arch_accum_fractional_bits[arch]
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"
        if self.arch == "Ada Lovelace" and self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            self.n_accum_fractional_bits = 13
            if self.output_type == "f32":
                self.output_type = "f32_e8m13"
        # check if split-k
        self.is_split_k = False
        if arch in ["Ampere", "Ada Lovelace"]:
            if self.k == 8 and self.a_type == torch.float32:
                self.is_split_k = True
            if self.k == 16 and self.a_type in [
                torch.float16,
                torch.bfloat16,
            ]:
                self.is_split_k = True
            if self.k == 32 and self.a_type in [
                torch.float8_e5m2,
                torch.float8_e4m3fn,
            ]:
                self.is_split_k = True

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        A = A.cpu()
        B = B.cpu()
        C = C.cpu()
        m, n, k = self.m, self.n, self.k
        D = torch.zeros((m, n), dtype=self.d_type)
        if self.a_type == torch.float32:  # tf32
            A = truncate_to_tf32(A)
            B = truncate_to_tf32(B)
        for i in range(m):
            for j in range(n):
                sum = C[i, j].to(dtype=self.d_type)
                if self.a_type == torch.float64:
                    for l in range(k):
                        sum = fma(A[i, l], B[l, j], sum)
                else:
                    if self.is_split_k:
                        sum = nv_fused_dot_add(
                            A[i, : k // 2],
                            B[: k // 2, j],
                            sum,
                            n_fractional_bits=self.n_accum_fractional_bits,
                            output_type=self.output_type,
                        )
                        sum = nv_fused_dot_add(
                            A[i, k // 2 : k],
                            B[k // 2 : k, j],
                            sum,
                            n_fractional_bits=self.n_accum_fractional_bits,
                            output_type=self.output_type,
                        )
                    else:
                        sum = nv_fused_dot_add(
                            A[i, :],
                            B[:, j],
                            sum,
                            n_fractional_bits=self.n_accum_fractional_bits,
                            output_type=self.output_type,
                        )
                D[i][j] = sum
        return D


class mma_block_scale(nv_ptx.mma_block_scale):
    def __init__(self, arch: str, qualifier: str):
        super().__init__(arch, qualifier)
        self.n_accum_fractional_bits = 35 if self.k == 64 else 25
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
    ) -> torch.Tensor:
        self.check_input(A, B, C, scale_A, scale_B)
        A = A.cpu()
        B = B.cpu()
        C = C.cpu()
        m, n, k = self.m, self.n, self.k
        D = torch.zeros((m, n), dtype=self.d_type)
        for i in range(m):
            for j in range(n):
                if k == 32:  # mxf8f6f4
                    sum = nv_fused_dot_add(
                        A[i, :],
                        B[:, j],
                        C[i, j],
                        self.n_accum_fractional_bits,
                        self.output_type,
                        scale_A[i, 0],
                        scale_B[0, j],
                    )
                else:  # mxf4nvf4
                    a = unpack_fp4_tensor(A[i, :])
                    b = unpack_fp4_tensor(B[:, j])
                    sum = nv_fused_dot_add_with_block_scale(
                        a,
                        b,
                        C[i, j],
                        scale_A[i, :],
                        scale_B[:, j],
                        self.n_accum_fractional_bits,
                    )
                D[i][j] = sum
        return D


class wgmma(nv_ptx.wgmma):
    def __init__(self, arch: str, qualifier: str):
        super().__init__(arch, qualifier)
        # set n_accum_fractional_bits and output_type
        self.n_accum_fractional_bits = 25
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"
        if self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            self.n_accum_fractional_bits = 13
            if self.output_type == "f32":
                self.output_type = "f32_e8m13"

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        m, n, k = self.m, self.n, self.k
        A = A.cpu()
        B = B.cpu()
        C = C.cpu()
        D = torch.zeros((m, n), dtype=self.d_type)
        if self.a_type == torch.float32:  # tf32
            A = truncate_to_tf32(A)
            B = truncate_to_tf32(B)
        for i in range(m):
            for j in range(n):
                sum = nv_fused_dot_add(
                    A[i, :],
                    B[:, j],
                    C[i, j],
                    n_fractional_bits=self.n_accum_fractional_bits,
                    output_type=self.output_type,
                )
                D[i][j] = sum
        return D


class tcgen05mma(nv_ptx.tcgen05mma):
    def __init__(self, arch: str, qualifier: str):
        super().__init__(arch, qualifier)
        self.n_accum_fractional_bits = 25
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        m, n, k = self.m, self.n, self.k
        A = A.cpu()
        B = B.cpu()
        C = C.cpu()
        D = torch.zeros((m, n), dtype=self.d_type)
        if self.a_type == torch.float32:  # tf32
            A = truncate_to_tf32(A)
            B = truncate_to_tf32(B)
        for i in range(m):
            for j in range(n):
                sum = nv_fused_dot_add(
                    A[i, :],
                    B[:, j],
                    C[i, j],
                    n_fractional_bits=self.n_accum_fractional_bits,
                    output_type=self.output_type,
                )
                D[i][j] = sum
        return D


class tcgen05mma_block_scale(nv_ptx.tcgen05mma_block_scale):
    def __init__(self, arch: str, qualifier: str):
        super().__init__(arch, qualifier)
        self.n_accum_fractional_bits = 35 if self.k == 64 else 25
        self.output_type = "f32"

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
    ) -> torch.Tensor:
        self.check_input(A, B, C, scale_A, scale_B)
        m, n, k = self.m, self.n, self.k
        A = A.cpu()
        B = B.cpu()
        C = C.cpu()
        D = torch.zeros((m, n), dtype=self.d_type)
        for i in range(m):
            for j in range(n):
                if k == 32:  # mxf8f6f4
                    sum = nv_fused_dot_add(
                        A[i, :],
                        B[:, j],
                        C[i, j],
                        self.n_accum_fractional_bits,
                        self.output_type,
                        scale_A[i, 0],
                        scale_B[0, j],
                    )
                else:  # mxf4nvf4
                    a = unpack_fp4_tensor(A[i, :])
                    b = unpack_fp4_tensor(B[:, j])
                    sum = nv_fused_dot_add_with_block_scale(
                        a,
                        b,
                        C[i, j],
                        scale_A[i, :],
                        scale_B[:, j],
                        self.n_accum_fractional_bits,
                    )
                D[i][j] = sum
        return D
