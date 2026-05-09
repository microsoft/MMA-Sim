import torch

dtype_min_exponent = {
    torch.float64: -1022,
    torch.float32: -126,
    torch.float16: -14,
    torch.bfloat16: -126,
    # torch.float8_e8m0fnu: -127,
    torch.float8_e5m2: -14,
    torch.float8_e4m3fn: -6,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fnuz: -7,
}


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
    dtype=torch.float8_e4m3fn,
)


def truncate_fp32_to_tf32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    x = x.view(torch.int32)  # uint32 operations are not supported by pytorch
    x = x >> 13 << 13  # truncate to tf32
    return x.view(torch.float32)


def unpack_uint8_to_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    high = x >> 4
    low = x & 0x0F
    unpacked = torch.stack([high, low], dim=-1).view(x.shape[:-1] + (-1,))
    return fp4_value_table[unpacked]
