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


def encode_fp4(x: float) -> int:
    if x.hex() == "nan":
        return 0b1000
    encoding = {
        0.0: 0b0000,
        0.5: 0b0001,
        1.0: 0b0010,
        1.5: 0b0011,
        2.0: 0b0100,
        3.0: 0b0101,
        4.0: 0b0110,
        6.0: 0b0111,
    }
    if abs(x) in encoding:
        return encoding[abs(x)] | (0b1000 if x < 0 else 0)
    else:
        raise ValueError(f"Unsupported value for fp4 encoding: {x}")


nv_torch_dtype = {
    "f32": torch.float32,
    "tf32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    "ue8m0": torch.float8_e8m0fnu,
    "ue4m3": torch.float8_e4m3fn,
    "e2m1": torch.uint8,  # torch.float4_e2m1fn_x2 is not well-implemented
}


def nv_shape_to_mnk(shape: str) -> tuple[int, int, int]:
    mnk = shape.split("m")[1]
    m, nk = mnk.split("n")
    n, k = nk.split("k")
    return int(m), int(n), int(k)


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
    def __init__(self, qualifier: str):
        qualifiers = qualifier.split(".")
        if len(qualifiers) == 5:  # mma
            kind, shape, d_type, a_type, b_type = qualifiers
        else:  # block scale mma
            kind, shape, block_size, d_type, a_type, b_type, s_type = qualifiers
            self.block_size = int(block_size[-2:])
            self.s_type = nv_torch_dtype[s_type]
        self.kind = kind
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.packing = 2 if kind.startswith("mxf4") else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]


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
