__all__ = ["mfma"]

from .common import MatrixMultiplyAdd, amd_torch_dtype

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
    # cdna2 f64
    "f64_16x16x4f64",
    "f64_4x4x4f64",
    # cdna2 bf16
    "f32_32x32x8bf16_1k",
    "f32_32x32x4bf16_1k",
    "f32_16x16x16bf16_1k",
    "f32_16x16x4bf16_1k",
    "f32_4x4x4bf16_1k",
]
cdna3_mfma_qualifiers = [
    # f64
    "f64_16x16x4_f64",
    "f64_4x4x4_4b_f64",
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


class mfma(MatrixMultiplyAdd):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mfma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for mfma.\n"
            f"Supported architectures: {list(arch_mfma_qualifiers.keys())}"
        )
        supported_qualifiers = arch_mfma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported qualifier {qualifier} for mfma on {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        qualifiers = qualifier.split("_")
        if len(qualifiers) == 2:
            # CDNA1 instructions
            d_type, shape_and_type = qualifiers
            if shape_and_type.endswith("bf16"):
                a_type = b_type = "bf16"
                shape = shape_and_type[:-4]
            else:
                a_type = b_type = shape_and_type[-3:]
                shape = shape_and_type[:-3]
        elif len(qualifiers) == 3:
            if qualifiers[-1] == "1k":
                # CDNA2 instructions
                d_type, shape_and_type, _ = qualifiers
                b_type = a_type = "bf16"
                shape = shape_and_type[:-4]
            else:
                # CDNA3 instructions
                d_type, shape, a_type = qualifiers
                b_type = a_type
        else:  # len(qualifiers) == 4:
            # CDNA3 instructions
            if qualifiers[-1] in ["fp8", "bf8"]:
                # fp8 instructions
                d_type, shape, a_type, b_type = qualifiers
            else:
                # f64/f32/f16/bf16 instructions
                d_type, shape, _, a_type = qualifiers
                b_type = a_type
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = map(int, shape.split("x"))
        self.a_type = amd_torch_dtype[a_type]
        self.b_type = amd_torch_dtype[b_type]
        self.c_type = amd_torch_dtype[d_type]
        self.d_type = amd_torch_dtype[d_type]
