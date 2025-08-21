import torch

from ..isa import MMA, WGMMA, TCGen05MMA
from .functions import (
    truncate_to_tf32,
    unpack_fp4_tensor,
    nv_fused_dot_add,
    nv_fused_dot_add_with_block_scale,
)


arch_accum_fraction_bits = {
    "Volta": 23,
    "Turing": 24,
    "Ampere": 24,
    "Ada Lovelace": 24,
    "Hopper": 25,
    "Blackwell": 25,
    "RTX Blackwell": 25,
}


def is_split_k(arch: str, qualifier: str) -> bool:
    return arch in [
        "Ampere",
        "Ada Lovelace",
    ] and qualifier in [
        # tf32 k8
        "m16n8k8.f32.tf32.tf32.f32",
        # f16 k16
        "m16n8k16.f32.f16.f16.f32",
        "m16n8k16.f16.f16.f16.f16",
        # bf16 k16
        "m16n8k16.f32.bf16.bf16.f32",
        # fp8 k32
        "m16n8k32.f32.e5m2.e5m2.f32",
        "m16n8k32.f32.e5m2.e4m3.f32",
        "m16n8k32.f32.e4m3.e5m2.f32",
        "m16n8k32.f32.e4m3.e4m3.f32",
        "m16n8k32.f16.e5m2.e5m2.f16",
        "m16n8k32.f16.e5m2.e4m3.f16",
        "m16n8k32.f16.e4m3.e5m2.f16",
        "m16n8k32.f16.e4m3.e4m3.f16",
    ]


class MMASim(MMA):
    def __init__(self, arch: str, qualifier: str):
        MMA.__init__(self, arch, qualifier)
        self.n_accum_fraction_bits = arch_accum_fraction_bits[arch]
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"
        # special cases
        if self.block_size > 0 and self.kind.startswith("mxf4"):
            self.n_accum_fraction_bits = 35
        if self.arch == "Ada Lovelace" and self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            self.n_accum_fraction_bits = 13
            if self.output_type == "f32":
                self.output_type = "f32_e8m13"
        self.is_split_k = is_split_k(arch, qualifier)

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor | None = None,
        scale_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_input(A, B, C, scale_A, scale_B)
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
                sum = C[i, j]
                if self.block_size > 0:
                    assert scale_A is not None and scale_B is not None
                    if self.kind.startswith("mxf8"):
                        sum = nv_fused_dot_add(
                            A[i, :],
                            B[:, j],
                            sum,
                            self.n_accum_fraction_bits,
                            self.output_type,
                            scale_A[i, 0],
                            scale_B[0, j],
                        )
                    else:  # mxf4
                        a = unpack_fp4_tensor(A[i, :])
                        b = unpack_fp4_tensor(B[:, j])
                        sum = nv_fused_dot_add_with_block_scale(
                            a,
                            b,
                            sum,
                            scale_A[i, :],
                            scale_B[:, j],
                            self.n_accum_fraction_bits,
                        )
                else:  # without block scale
                    if self.is_split_k:
                        sum = nv_fused_dot_add(
                            A[i, : k // 2],
                            B[: k // 2, j],
                            sum,
                            n_fraction_bits=self.n_accum_fraction_bits,
                            output_type=self.output_type,
                        )
                        sum = nv_fused_dot_add(
                            A[i, k // 2 : k],
                            B[k // 2 : k, j],
                            sum,
                            n_fraction_bits=self.n_accum_fraction_bits,
                            output_type=self.output_type,
                        )
                    else:
                        sum = nv_fused_dot_add(
                            A[i, :],
                            B[:, j],
                            sum,
                            n_fraction_bits=self.n_accum_fraction_bits,
                            output_type=self.output_type,
                        )
                D[i][j] = sum
        return D


class WGMMASim(WGMMA):
    def __init__(self, arch: str, qualifier: str):
        WGMMA.__init__(self, arch, qualifier)
        self.n_accum_fraction_bits = 25
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"
        if self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            self.n_accum_fraction_bits = 13
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
                sum = C[i, j]
                sum = nv_fused_dot_add(
                    A[i, :],
                    B[:, j],
                    sum,
                    n_fraction_bits=self.n_accum_fraction_bits,
                    output_type=self.output_type,
                )
                D[i][j] = sum
        return D


class TCGen05MMASim(TCGen05MMA):
    def __init__(self, arch: str, qualifier: str):
        TCGen05MMA.__init__(self, arch, qualifier)
        self.n_accum_fraction_bits = 35 if self.kind.startswith("mxf4") else 25
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor | None = None,
        scale_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_input(A, B, C, scale_A, scale_B)
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
                sum = C[i, j]
                if self.block_size > 0:
                    assert scale_A is not None and scale_B is not None
                    if self.kind.startswith("mxf8"):
                        sum = nv_fused_dot_add(
                            A[i, :],
                            B[:, j],
                            sum,
                            self.n_accum_fraction_bits,
                            self.output_type,
                            scale_A[i, 0],
                            scale_B[0, j],
                        )
                    else:  # mxf4
                        a = unpack_fp4_tensor(A[i, :])
                        b = unpack_fp4_tensor(B[:, j])
                        sum = nv_fused_dot_add_with_block_scale(
                            a,
                            b,
                            sum,
                            scale_A[i, :],
                            scale_B[:, j],
                            self.n_accum_fraction_bits,
                        )
                else:  # without block scale
                    sum = nv_fused_dot_add(
                        A[i, :],
                        B[:, j],
                        sum,
                        n_fraction_bits=self.n_accum_fraction_bits,
                        output_type=self.output_type,
                    )
                D[i][j] = sum
        return D
