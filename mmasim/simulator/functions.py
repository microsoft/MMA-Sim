import ctypes
import math

import torch

libm = ctypes.CDLL("libm.so.6")
libm.fmaf.argtypes = [ctypes.c_float] * 3
libm.fmaf.restype = ctypes.c_float


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == c.dtype == torch.float32
    res = libm.fmaf(a.item(), b.item(), c.item())
    return torch.tensor(res, dtype=torch.float32)


def truncate_to_tf32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    x = x.view(torch.int32)  # uint32 operations are not supported by pytorch
    x = x >> 13 << 13  # truncate to tf32
    return x.view(torch.float32)


def unpack_fp4_tensor(packed: torch.Tensor) -> torch.Tensor:
    n = packed.numel()
    low = packed & 0x0F
    high = packed >> 4
    unpacked = torch.zeros(n * 2, dtype=torch.float32)
    decoding = {
        0b0000: 0.0,
        0b0001: 0.5,
        0b0010: 1.0,
        0b0011: 1.5,
        0b0100: 2.0,
        0b0101: 3.0,
        0b0110: 4.0,
        0b0111: 6.0,
    }
    decoding |= {x + 0b1000: -y for x, y in decoding.items()}
    for i in range(n):
        unpacked[i * 2] = decoding[int(low[i].item())]
        unpacked[i * 2 + 1] = decoding[int(high[i].item())]
    return unpacked


dtype_min_exponent = {
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


def flush_denormal(x: torch.Tensor, keep_sign: bool = False) -> torch.Tensor:
    min_exponent = dtype_min_exponent[x.dtype]
    if keep_sign:
        x[x.abs() < 2.0**min_exponent] *= 0.0
    else:
        x[x.abs() < 2.0**min_exponent] = 0.0
    return x


def extract_significand_exponent(
    x: float | torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[float, int]:
    if isinstance(x, torch.Tensor):
        assert x.numel() == 1
        if dtype is None:
            dtype = x.dtype
        x = x.item()
    assert dtype in dtype_min_exponent, f"Unsupported dtype: {dtype}"
    significand, exponent = math.frexp(x)
    significand *= 2  # 1 <= |significand| < 2
    exponent -= 1
    # handle subnormal
    min_exponent = dtype_min_exponent[dtype]
    if exponent < min_exponent:
        significand *= 2.0 ** (exponent - min_exponent)
        exponent = min_exponent
    if significand == 0.0:
        exponent = -999
    return significand, exponent


def pairwise_dot(
    a: torch.Tensor, b: torch.Tensor, flush_denormal: bool = False
) -> float:
    assert a.dtype == b.dtype == torch.float32
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


def fused_sum(
    significands: list[float], exponents: list[int], n_fraction_bits: int
) -> tuple[float, int]:
    max_exponent = max(exponents)
    significand_sum = 0.0
    for i in range(len(significands)):
        rounded = (
            math.trunc(
                significands[i] * 2.0 ** (n_fraction_bits + exponents[i] - max_exponent)
            )
            * 2.0**-n_fraction_bits
        )
        significand_sum += rounded
    return significand_sum, max_exponent


def fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fraction_bits: int,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> tuple[float, int]:
    # check nan or inf
    products = a.double() * b.double() * scale_a * scale_b
    fp64_sum = products.sum() + c.double()
    if torch.isnan(fp64_sum).any() or torch.isinf(fp64_sum).any():
        return fp64_sum.item(), 0

    sc, ec = extract_significand_exponent(c)
    _, esa = extract_significand_exponent(scale_a)
    _, esb = extract_significand_exponent(scale_b)
    significands = [sc]
    exponents = [ec]
    for i in range(products.numel()):
        sa, ea = extract_significand_exponent(a[i])
        sb, eb = extract_significand_exponent(b[i])
        significands.append(sa * sb)
        exponents.append(ea + eb + esa + esb)
    return fused_sum(significands, exponents, n_fraction_bits)


def nv_fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fraction_bits: int,
    output_type: str,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale_a is None or scale_b is None:
        scale_a = torch.tensor(1.0)
        scale_b = torch.tensor(1.0)
    s, e = fused_dot_add(a, b, c, n_fraction_bits, scale_a, scale_b)
    if s != s:  # nan
        if output_type == "f16":
            return torch.tensor(0x7FFF, dtype=torch.uint16).view(torch.float16)
        else:
            return torch.tensor(0x7FFF_FFFF, dtype=torch.uint32).view(torch.float32)
    if s + 1 == s:  # inf
        return torch.tensor(
            s, dtype=torch.float16 if output_type == "f16" else torch.float32
        )

    # normalize output
    if output_type == "f16":
        # note that direcctly converting f64 to f16 can be incorrect
        # as PyTorch-CPU computes f64 -> f32 -> f16 internally
        s, e = extract_significand_exponent(s * 2.0**e, torch.float16)
        s = round(s * 2.0**10) * 2.0**-10  # RNE
        return torch.tensor(s * 2.0**e, dtype=torch.float16)
    else:  # "f32" or "f32_e8m13"
        s, e = extract_significand_exponent(s * 2.0**e, torch.float32)
        n_fraction_bits = 13 if output_type == "f32_e8m13" else 23
        s = math.trunc(s * 2.0**n_fraction_bits) * 2.0**-n_fraction_bits  # RZ
        return torch.tensor(s * 2.0**e, dtype=torch.float32)


def nv_fused_dot_add_with_block_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    n_fraction_bits: int,
) -> torch.Tensor:
    if torch.isnan(scale_a).any() or torch.isnan(scale_b).any() or torch.isnan(c).any():
        return torch.tensor(0x7FFF_FFFF, dtype=torch.uint32).view(torch.float32)
    if scale_a.dtype == torch.float8_e4m3fn:  # ue4m3
        scale_a = scale_a.abs()
        scale_b = scale_b.abs()
    n = a.numel()
    block_size = n // scale_a.numel()

    sc, ec = extract_significand_exponent(c)
    significands = [sc]
    exponents = [ec]
    for k in range(0, a.numel(), 16):
        s = (a[k : k + 16] * b[k : k + 16]).sum().item()
        s0, e0 = extract_significand_exponent(scale_a[k // block_size])
        s1, e1 = extract_significand_exponent(scale_b[k // block_size])
        significands.append(s * s0 * s1)
        exponents.append(e0 + e1)
    s, e = fused_sum(significands, exponents, n_fraction_bits)
    s, e = extract_significand_exponent(s * 2.0**e, torch.float32)
    s = math.trunc(s * 2.0**23) * 2.0**-23  # RZ
    return torch.tensor(s * 2.0**e, dtype=torch.float32)


def amd_fused_dot_add(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_fraction_bits: int,
    is_fp8: bool = False,
) -> float:
    products = a.double() * b.double()
    fp64_sum = products.sum() + c.double()
    # TODO
    if torch.isnan(fp64_sum).any() or torch.isinf(fp64_sum).any():
        return fp64_sum.item()

    significands = []
    exponents = []
    p_inf = n_inf = False
    for i in range(products.numel()):
        if products[i] >= 2.0**128:
            p_inf = True
        elif products[i] <= -(2.0**128):
            n_inf = True
        else:
            sa, ea = extract_significand_exponent(a[i])
            sb, eb = extract_significand_exponent(b[i])
            significands.append(sa * sb)
            exponents.append(ea + eb)
    if p_inf or n_inf:
        if p_inf and n_inf:
            return float("nan")
        elif p_inf:
            return float("inf")
        else:
            return float("-inf")
    sc, ec = extract_significand_exponent(c)
    if is_fp8:
        s0, e0 = fused_sum(significands[0::2], exponents[0::2], n_fraction_bits)
        s1, e1 = fused_sum(significands[1::2], exponents[1::2], n_fraction_bits)
        max_e = max(e0, e1)
        s0 = (
            math.floor(s0 * 2.0 ** (n_fraction_bits + e0 - max_e))
            * 2.0**-n_fraction_bits
        )
        s1 = (
            math.floor(s1 * 2.0 ** (n_fraction_bits + e1 - max_e))
            * 2.0**-n_fraction_bits
        )
        s, e = s0 + s1, max_e
        max_e = max(e, ec)
        s = math.floor(s * 2.0 ** (31 + e - max_e)) * 2.0**-31
        if ec >= max_e - 25:
            sc = math.floor(sc * 2.0 ** (24 + ec - max_e)) * 2.0**-24
        else:
            sc = 0.0
    else:
        s, e = fused_sum(significands, exponents, n_fraction_bits)
        max_e = max(e, ec)
        s = math.floor(s * 2.0 ** (31 + e - max_e)) * 2.0**-31
        sc = math.floor(sc * 2.0 ** (24 + ec - max_e)) * 2.0**-24
    return (s + sc) * 2.0**max_e
