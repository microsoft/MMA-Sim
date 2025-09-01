import torch


class MatrixMultiplyAdd:
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

    def dotadd(self, a: list[float], b: list[float], c: float) -> float:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        for i in range(len(a)):
            A[0, i] = a[i]
            B_T[0, i] = b[i]
        C[0, 0] = c
        D = self(A, B_T.T, C)
        return D[0, 0].item()


class MatrixMultiplyAddWithBlockScale:
    m: int
    n: int
    k: int
    block_size: int
    packing: int
    a_type: torch.dtype
    b_type: torch.dtype
    c_type: torch.dtype
    d_type: torch.dtype
    s_type: torch.dtype

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
    ) -> torch.Tensor: ...

    def dotadd_with_block_scale(
        self,
        a: list[float],
        b: list[float],
        c: float,
        scale_a: list[float],
        scale_b: list[float],
    ) -> float:
        m, n, k = self.m, self.n, self.k
        packing, block_size = self.packing, self.block_size
        A = torch.zeros([m, k // packing], dtype=self.a_type)
        B_T = torch.zeros([n, k // packing], dtype=self.b_type)
        C = torch.zeros([m, n], dtype=self.c_type)
        for i in range(len(a)):
            if packing == 1:
                A[0, i] = a[i]
                B_T[0, i] = b[i]
            else:
                A[0, i // packing] |= encode_fp4(a[i]) << (i % 2 * 4)
                B_T[0, i // packing] |= encode_fp4(b[i]) << (i % 2 * 4)
        C[0, 0] = c
        scale_A = torch.ones([m, k // block_size], dtype=self.s_type)
        scale_B_T = torch.ones([n, k // block_size], dtype=self.s_type)
        for i in range(len(scale_a)):
            scale_A[0, i] = scale_a[i]
            scale_B_T[0, i] = scale_b[i]
        D = self(A, B_T.T, C, scale_A, scale_B_T.T)
        return D[0, 0].item()


def encode_fp4(x: float) -> int:
    if x.hex() == "-0x0.0p+0":  # -0.0
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
    "f64": torch.float64,
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


amd_torch_dtype = {
    "f64": torch.float64,
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
