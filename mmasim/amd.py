import ctypes

import torch

libm = ctypes.CDLL("libm.so.6")
libm.fmaf.argtypes = [ctypes.c_float] * 3
libm.fmaf.restype = ctypes.c_float

cdna1_mfma_qualifiers = [
    # cdna1 instructions
    "f32_32x32x2f32",
    "f32_32x32x1f32",
    "f32_16x16x4f32",
    "f32_16x16x1f32",
    "f32_4x4x1f32",
    "f32_32x32x8f16",
    "f32_32x32x4f16",
    "f32_16x16x16f16",
    "f32_16x16x4f16",
    "f32_4x4x4f16",
    "f32_32x32x4bf16",
    "f32_32x32x2bf16",
    "f32_16x16x8bf16",
    "f32_16x16x2bf16",
    "f32_4x4x2bf16",
]
cdna2_mfma_qualifiers = cdna1_mfma_qualifiers + [
    # cdna2 instructions
    "f32_32x32x8bf16_1k",
    "f32_32x32x4bf16_1k",
    "f32_16x16x16bf16_1k",
    "f32_16x16x4bf16_1k",
    "f32_4x4x4bf16_1k",
]
cdna3_mfma_qualifiers = [
    "f32_16x16x8_xf32"
    "f32_32x32x4_xf32"
    "f32_32x32x1_2b_f32"
    "f32_16x16x1_4b_f32"
    "f32_4x4x1_16b_f32"
    "f32_32x32x2_f32"
    "f32_16x16x4_f32"
    "f32_32x32x4_2b_f16"
    "f32_16x16x4_4b_f16"
    "f32_4x4x4_16b_f16"
    "f32_32x32x8_f16"
    "f32_16x16x16_f16"
    "f32_32x32x4_2b_bf16"
    "f32_16x16x4_4b_bf16"
    "f32_4x4x4_16b_bf16"
    "f32_32x32x8_bf16"
    "f32_16x16x16_bf16"
    "f32_16x16x32_bf8_bf8"
    "f32_16x16x32_bf8_fp8"
    "f32_16x16x32_fp8_bf8"
    "f32_16x16x32_fp8_fp8"
    "f32_32x32x16_bf8_bf8"
    "f32_32x32x16_bf8_fp8"
    "f32_32x32x16_fp8_bf8"
    "f32_32x32x16_fp8_fp8"
]

arch_mfma_qualifiers = {
    "CDNA1": cdna1_mfma_qualifiers,
    "CDNA2": cdna2_mfma_qualifiers,
    "CDNA3": cdna3_mfma_qualifiers,
}


def mfma_qualifier_to_shape_and_types(qualifier: str) -> tuple[str]:
    qualifiers = qualifier.split("_")
    if len(qualifiers) == 2:
        # CDNA1 instructions
        d_type, shape_and_type = qualifiers
        c_type = d_type
        if shape_and_type.endswith("f32"):
            a_type = b_type = "f32"
            shape = shape_and_type[:-3]
        elif shape_and_type.endswith("bf16"):
            a_type = b_type = "bf16"
            shape = shape_and_type[:-4]
        elif shape_and_type.endswith("f16"):
            a_type = b_type = "f16"
            shape = shape_and_type[:-3]
    elif len(qualifiers) == 3:
        if qualifiers[-1] == "1k":
            # CDNA2 instructions
            d_type, shape_and_type, _ = qualifiers
            c_type = d_type
            b_type = a_type = "bf16"
            shape = shape_and_type[:-4]
    return shape, d_type, a_type, b_type, c_type


def shape_to_mnk(shape: str) -> tuple[int, int, int]:
    mnk = shape.split("x")
    return int(mnk[0]), int(mnk[1]), int(mnk[2])


torch_dtype = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
}

min_exponent = {
    "f16": -14,
    "f32": -126,
    "bf16": -126,
}


@torch.inference_mode()
def pairwise_dot(
    a: torch.Tensor, b: torch.Tensor, flush_denormal: bool = False
) -> float:
    n = a.numel()
    if n == 1:
        sum = libm.fmaf(a.item(), b.item(), 0.0)
    else:
        m = n // 2
        sum_l = pairwise_dot(a[:m], b[:m], flush_denormal)
        sum_r = pairwise_dot(a[m:], b[m:], flush_denormal)
        sum = libm.fmaf(sum_l, 1.0, sum_r)
    if flush_denormal and abs(sum) < 2.0**-126:
        sum *= 0.0
    return sum


class MatrixCoreInstruction:
    arch: str
    qualifier: str
    shape: str
    a_type: str
    b_type: str
    c_type: str
    d_type: str


class MFMA(MatrixCoreInstruction):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mfma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for MFMA.\n"
            f"Supported architectures: {list(arch_mfma_qualifiers.keys())}"
        )
        self.arch = arch
        supported_qualifiers = arch_mfma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported MMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.qualifier = qualifier
        self.shape, self.d_type, self.a_type, self.b_type, self.c_type = (
            mfma_qualifier_to_shape_and_types(qualifier)
        )

        self.flush_denormal = self.a_type in ["f16", "bf16"] and self.arch == "CDNA2"
        if self.a_type == "f32":
            self.summation_group_size = 1
        elif self.a_type == "f16":
            self.summation_group_size = 4
        elif self.a_type == "bf16":
            self.summation_group_size = 2 if qualifier in cdna1_mfma_qualifiers else 4

    @torch.inference_mode()
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = shape_to_mnk(self.shape)
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert c.shape == (m, n)
        assert a.dtype == torch_dtype[self.a_type]
        assert b.dtype == torch_dtype[self.b_type]
        assert c.dtype == torch_dtype[self.c_type]
        assert a.is_cpu and b.is_cpu and c.is_cpu
        d = torch.empty((m, n), dtype=torch_dtype[self.d_type])
        if self.flush_denormal:
            a[a.abs() < 2.0 ** min_exponent[self.a_type]] = 0.0
            b[b.abs() < 2.0 ** min_exponent[self.b_type]] = 0.0
            c[c.abs() < 2.0 ** min_exponent[self.c_type]] = 0.0
        for i in range(m):
            for j in range(n):
                sum = c[i, j].item()
                if self.summation_group_size == 1:  # f32
                    for l in range(k):
                        sum = libm.fmaf(a[i, l].item(), b[l, j].item(), sum)
                else:
                    l = 0
                    while l < k:
                        partial_sum = pairwise_dot(
                            a[i, l : l + self.summation_group_size],
                            b[l : l + self.summation_group_size, j],
                            self.flush_denormal,
                        )
                        sum = libm.fmaf(partial_sum, 1.0, sum)
                        if self.flush_denormal and abs(sum) < 2.0**-126:
                            sum *= 0.0
                        l += self.summation_group_size
                d[i][j] = sum
        return d
