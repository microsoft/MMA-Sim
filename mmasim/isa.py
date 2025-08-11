import torch


class MMAInstructionBase:
    m: int
    n: int
    k: int
    a_type: torch.dtype
    b_type: torch.dtype
    c_type: torch.dtype
    d_type: torch.dtype

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor: ...


def nv_shape_to_mnk(shape: str) -> tuple[int, int, int]:
    mnk = shape.split("m")[1]
    m, nk = mnk.split("n")
    n, k = nk.split("k")
    return int(m), int(n), int(k)


nv_torch_dtype = {
    "f32": torch.float32,
    "tf32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


class NV_MMABase(MMAInstructionBase):
    def __init__(self, qualifier: str):
        shape, d_type, a_type, b_type, c_type = qualifier.split(".")
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]


class NV_WGMMABase(MMAInstructionBase):
    def __init__(self, qualifier: str):
        shape, d_type, a_type, b_type = qualifier.split(".")
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]


class NV_TCGen05MMABase(MMAInstructionBase):
    def __init__(self, qualifier: str, block_scale: bool = False):
        shape, d_type, a_type, b_type = qualifier.split(".")
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]
        if block_scale:
            raise NotImplementedError


amd_torch_dtype = {
    "f32": torch.float32,
    "xf32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fnuz,
    "bf8": torch.float8_e5m2fnuz,
}


def amd_parse_qualifier(
    qualifier: str,
) -> tuple[str, str, str, str, str]:
    qualifiers = qualifier.split("_")
    if len(qualifiers) == 2:
        # CDNA1 instructions
        d_type, shape_and_type = qualifiers
        if shape_and_type.endswith("f32"):
            a_type = b_type = "f32"
            shape = shape_and_type[:-3]
        elif shape_and_type.endswith("bf16"):
            a_type = b_type = "bf16"
            shape = shape_and_type[:-4]
        else:  # shape_and_type.endswith("f16"):
            a_type = b_type = "f16"
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
            # f32/f16/bf16 instructions
            d_type, shape, _, a_type = qualifiers
            b_type = a_type
    c_type = d_type
    return shape, d_type, a_type, b_type, c_type


class AMD_MFMABase(MMAInstructionBase):
    def __init__(self, qualifier: str):
        shape, d_type, a_type, b_type, c_type = amd_parse_qualifier(qualifier)
        self.m, self.n, self.k = map(int, shape.split("x"))
        self.a_type = amd_torch_dtype[a_type]
        self.b_type = amd_torch_dtype[b_type]
        self.c_type = amd_torch_dtype[c_type]
        self.d_type = amd_torch_dtype[d_type]
