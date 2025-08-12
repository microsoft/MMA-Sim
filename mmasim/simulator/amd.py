import torch

from .utils import (
    fma,
    truncate_to_tf32,
    flush_denormal,
    pairwise_dot,
    amd_fused_dot_add,
)
from ..isa import AMD_MFMABase

cdna1_mfma_qualifiers = [
    # f32
    "f32_32x32x2f32",
    "f32_32x32x1f32",
    "f32_16x16x4f32",
    "f32_16x16x1f32",
    "f32_4x4x1f32",
    # f16
    "f32_32x32x8f16",
    "f32_32x32x4f16",
    "f32_16x16x16f16",
    "f32_16x16x4f16",
    "f32_4x4x4f16",
    # bf16
    "f32_32x32x4bf16",
    "f32_32x32x2bf16",
    "f32_16x16x8bf16",
    "f32_16x16x2bf16",
    "f32_4x4x2bf16",
]
cdna2_mfma_qualifiers = [
    # cdna2 bf16
    "f32_32x32x8bf16_1k",
    "f32_32x32x4bf16_1k",
    "f32_16x16x16bf16_1k",
    "f32_16x16x4bf16_1k",
    "f32_4x4x4bf16_1k",
]
cdna3_mfma_qualifiers = [
    # f32
    "f32_32x32x1_2b_f32",
    "f32_16x16x1_4b_f32",
    "f32_4x4x1_16b_f32",
    "f32_32x32x2_f32",
    "f32_16x16x4_f32",
    # xf32
    "f32_16x16x8_xf32",
    "f32_32x32x4_xf32",
    # f16
    "f32_32x32x4_2b_f16",
    "f32_16x16x4_4b_f16",
    "f32_4x4x4_16b_f16",
    "f32_32x32x8_f16",
    "f32_16x16x16_f16",
    # bf16
    "f32_32x32x4_2b_bf16",
    "f32_16x16x4_4b_bf16",
    "f32_4x4x4_16b_bf16",
    "f32_32x32x8_bf16",
    "f32_16x16x16_bf16",
    # fp8
    "f32_16x16x32_bf8_bf8",
    "f32_16x16x32_bf8_fp8",
    "f32_16x16x32_fp8_bf8",
    "f32_16x16x32_fp8_fp8",
    "f32_32x32x16_bf8_bf8",
    "f32_32x32x16_bf8_fp8",
    "f32_32x32x16_fp8_bf8",
    "f32_32x32x16_fp8_fp8",
]

arch_mfma_qualifiers = {
    "CDNA1": cdna1_mfma_qualifiers,
    "CDNA2": cdna2_mfma_qualifiers + cdna1_mfma_qualifiers,
    "CDNA3": cdna3_mfma_qualifiers,
}


class MFMA(AMD_MFMABase):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mfma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for MFMA.\n"
            f"Supported architectures: {list(arch_mfma_qualifiers.keys())}"
        )
        self.arch = arch
        supported_qualifiers = arch_mfma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported MFMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.qualifier = qualifier
        AMD_MFMABase.__init__(self, qualifier)

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
                self.group_size = 2 if qualifier in cdna1_mfma_qualifiers else 4
                self.flush_denormal = self.arch == "CDNA2"
        else:  # fp8
            self.operation_type = "fused_dot_add"
            self.group_size = 16
            self.is_two_stage = True

    @torch.inference_mode()
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert c.shape == (m, n)
        assert a.dtype == self.a_type
        assert b.dtype == self.b_type
        assert c.dtype == self.c_type
        a = a.cpu()
        b = b.cpu()
        c = c.cpu()
        d = torch.zeros((m, n), dtype=self.d_type)
        if self.flush_denormal:
            a = flush_denormal(a)
            b = flush_denormal(b)
            c = flush_denormal(c)
        if self.is_xf32:
            a = truncate_to_tf32(a)
            b = truncate_to_tf32(b)
        for i in range(m):
            for j in range(n):
                sum = c[i, j].to(dtype=self.d_type)
                if self.operation_type == "standard":
                    if self.group_size == 1:  # f32
                        for l in range(k):
                            sum = fma(a[i, l], b[l, j], sum)
                    else:
                        l = 0
                        while l < k:
                            group_sum = pairwise_dot(
                                a[i, l : l + self.group_size],
                                b[l : l + self.group_size, j],
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
                            a[i, l : l + self.group_size],
                            b[l : l + self.group_size, j],
                            sum,
                            n_fraction_bits=24,
                            is_fp8=self.qualifier.endswith("8"),
                        )
                        sum = torch.tensor(fused_sum, dtype=self.d_type)  # RNE
                        l += self.group_size
                d[i, j] = sum
        return d
