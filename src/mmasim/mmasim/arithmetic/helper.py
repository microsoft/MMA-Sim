import torch

dtype_subnormal_exponent = {
    torch.float64: -1022,
    torch.float32: -126,
    torch.float16: -14,
    torch.bfloat16: -126,
    torch.float8_e8m0fnu: -127,
    torch.float8_e5m2: -14,
    torch.float8_e4m3fn: -6,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fnuz: -7,
}
fp6_fp4_subnormal_exponent = {"e3m2": -2, "e2m3": 0, "e2m1": 0}
fp6_fp4_bits = {"e3m2": (3, 2), "e2m3": (2, 3), "e2m1": (2, 1)}  # (E, M)


fp4_value_table = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float16,
)


def truncate_fp32_to_tf32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    x = x.view(torch.int32)  # uint32 operations are not supported by pytorch
    x = x >> 13 << 13  # truncate to tf32
    return x.view(torch.float32)


def truncate_e4m3_to_ue4m3(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float8_e4m3fn
    x = x.view(torch.uint8) & 0x7F
    return x.view(torch.float8_e4m3fn)


def unpack_uint8_to_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    high = x >> 4
    low = x & 0x0F
    unpacked = torch.stack([high, low], dim=-1).view(x.shape[:-1] + (-1,))
    return fp4_value_table.to(x.device)[unpacked.int()]


def decode_uint8_to_fp6_fp4(x: torch.Tensor, x_type: str) -> torch.Tensor:
    assert x.dtype == torch.uint8
    E, M = fp6_fp4_bits[x_type]
    x = (x.int() & 0x3F) >> (5 - E - M)  # fp6 sits in bits 5:0, fp4 in bits 5:2
    bias = 2 ** (E - 1) - 1
    e = (x >> M) & (2**E - 1)
    s = (x & (2**M - 1)) + (e > 0) * 2**M
    res = (s * torch.exp2(e.clamp(min=1) - bias - M)).to(torch.float16)
    return torch.where(x >> (E + M) == 1, -res, res)
