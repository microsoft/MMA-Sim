import torch
from typing import Annotated
from .. import MMAOperation, MMABlockScaleOperation
from .helper import (
    dtype_subnormal_exponent,
    truncate_fp32_to_tf32,
    truncate_e4m3_to_ue4m3,
    unpack_uint8_to_fp4,
)

DoubleTensor = Annotated[torch.Tensor, "float64"]
FloatTensor = Annotated[torch.Tensor, "float32"]
IntTensor = Annotated[torch.Tensor, "int32"]


def pow2(e: IntTensor) -> FloatTensor | DoubleTensor:
    # torch.ldexp and torch.pow can be inaccurate on gpu
    # torch.exp2(-127) can be inaccurate on gpu for e.dtype == torch.int32
    # so, use DoubleTensor if e < -126 or e > 127
    if (e < -126).any() or (e > 127).any():
        e = e.double()
    return torch.exp2(e)


def frexp_and_normalize(
    x: torch.Tensor, e_subnormal: int | None = None
) -> tuple[FloatTensor | DoubleTensor, IntTensor]:
    assert x.dtype in dtype_subnormal_exponent, f"Unsupported dtype: {x.dtype}"
    if x.dtype == torch.float64:
        s, e = torch.frexp(x)
    else:
        s, e = torch.frexp(x.float())
    s, e = s * 2, e - 1  # let 1 <= |s| < 2
    # handle subnormal
    if e_subnormal is None:
        e_subnormal = dtype_subnormal_exponent[x.dtype]
    subnormals = e < e_subnormal
    s[subnormals] *= pow2(e[subnormals] - e_subnormal)
    e[subnormals] = e_subnormal
    # handle zero
    e[s == 0.0] = e_subnormal
    return s, e


def ldexp_and_normalize(s: DoubleTensor, e: IntTensor, rho: str) -> torch.Tensor:
    if rho == "RNE-FP16":
        # note that direcctly converting FP64 to FP16 can be incorrect
        # as PyTorch-CPU computes FP64 -> FP32 -> FP16 internally
        s, e = frexp_and_normalize(s * pow2(e), e_subnormal=-14)
        s = torch.round(s * 2.0**10) * 2.0**-10  # RNE
        res = (s * pow2(e)).to(torch.float16)
    elif rho == "RNE-FP32":
        res = (s * pow2(e)).to(torch.float32)
    else:  # RZ
        s, e = frexp_and_normalize(s * pow2(e), e_subnormal=-126)
        if rho == "RZ-E8M13":
            s = torch.trunc(s * 2.0**13) * 2.0**-13  # RZ
        else:  # "RZ-FP32"
            s = torch.trunc(s * 2.0**23) * 2.0**-23  # RZ
        res = (s * pow2(e)).to(torch.float32)
    nans = res.isnan()
    if res.dtype == torch.float16:
        res = res.view(torch.int16)
        res[nans] = 0x7FFF
        res = res.view(torch.float16)
    else:  # torch.float32
        res = res.view(torch.int32)
        res[nans] = 0x7FFFFFFF
        res = res.view(torch.float32)
    return res


def truncated_fused_sum(
    s: torch.Tensor, e: IntTensor, F: int
) -> tuple[DoubleTensor, IntTensor]:
    e_max = e.max(dim=-1).values
    delta_e = e - e_max.unsqueeze(-1)
    s = torch.trunc(s.double() * pow2(delta_e + F)) * 2.0**-F
    sum = s.sum(dim=-1)
    return sum, e_max


def t_fdpa(
    a: torch.Tensor,  # [..., K]
    b: torch.Tensor,  # [..., K]
    c: torch.Tensor,  # [...]
    F: int,
    rho: str,
    e_zero: int,
) -> torch.Tensor:  # [...]
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s_c, e_c = frexp_and_normalize(c)
    s = torch.cat([s_a * s_b, s_c.unsqueeze(-1)], dim=-1)
    e = torch.cat([e_a + e_b, e_c.unsqueeze(-1)], dim=-1)
    e[s == 0.0] = e_zero  # handle zero
    sum, e_max = truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_T_FDPA(MMAOperation):
    def __init__(self, F: int, rho: str, L_max: int, e_zero: int):
        self.F = F
        self.rho = rho
        self.L_max = L_max
        self.e_zero = e_zero

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if a.dtype == torch.float32:  # tf32
            a = truncate_fp32_to_tf32(a)
            b = truncate_fp32_to_tf32(b)
        K = a.shape[-1]
        L = min(K, self.L_max)
        for i in range(0, K, L):
            c = t_fdpa(
                a[..., i : i + L], b[..., i : i + L], c, self.F, self.rho, self.e_zero
            )
        return c

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        m, n = C.shape
        a_vectors = A[:, None, :].expand(-1, n, -1)
        b_vectors = B.T[None, :, :].expand(m, -1, -1)
        return self.dpa(a_vectors, b_vectors, C)


def st_fdpa(
    a: torch.Tensor,  # [..., K]
    b: torch.Tensor,  # [..., K]
    c: torch.Tensor,  # [...]
    alpha: torch.Tensor,  # [..., 1]
    beta: torch.Tensor,  # [..., 1]
    F: int,
    rho: str,
    e_zero: int,
) -> torch.Tensor:  # [...]
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s_c, e_c = frexp_and_normalize(c)
    s_alpha, e_alpha = frexp_and_normalize(alpha)
    s_beta, e_beta = frexp_and_normalize(beta)
    # s_alpha, s_beta can be 1.0 or nan
    s = torch.cat([s_a * s_b * s_alpha * s_beta, s_c.unsqueeze(-1)], dim=-1)
    e = torch.cat([e_a + e_b + e_alpha + e_beta, e_c.unsqueeze(-1)], dim=-1)
    e[s == 0.0] = e_zero  # handle zero
    sum, e_max = truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_ST_FDPA(MMABlockScaleOperation):
    def __init__(self, F: int, rho: str, L_max: int, e_zero: int):
        self.F = F
        self.rho = rho
        self.L_max = L_max
        self.e_zero = e_zero

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        K = a.shape[-1]
        L = min(K, self.L_max)
        for i in range(0, K, L):
            c = st_fdpa(
                a[..., i : i + L],
                b[..., i : i + L],
                c,
                alpha,
                beta,
                self.F,
                self.rho,
                self.e_zero,
            )
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        m, n = C.shape
        a_vectors = A[:, None, :].expand(-1, n, -1)
        b_vectors = B.T[None, :, :].expand(m, -1, -1)
        alpha_vectors = alpha[:, None, :].expand(-1, n, -1)
        beta_vectors = beta.T[None, :, :].expand(m, -1, -1)
        return self.dpa(a_vectors, b_vectors, C, alpha_vectors, beta_vectors)


def gst_fdpa(
    a: torch.Tensor,  # [..., K]
    b: torch.Tensor,  # [..., K]
    c: torch.Tensor,  # [...]
    alpha: torch.Tensor,  # [..., K//K_block]
    beta: torch.Tensor,  # [..., K//K_block]
    G: int,
    F: int,
    rho: str,
    e_zero: int,
) -> torch.Tensor:  # [...]
    p = a.float() * b.float()
    p = p.view(*p.shape[:-1], -1, G)  # [..., K//G, G]
    apply_e_zero = (p == 0.0).all(dim=-1)  # [..., K//G]
    p = p.sum(dim=-1)  # zeros resulted from summation do not apply e_zero
    K_block = a.shape[-1] // alpha.shape[-1]
    alpha = torch.repeat_interleave(alpha, K_block // G, dim=-1)  # [..., K//G]
    beta = torch.repeat_interleave(beta, K_block // G, dim=-1)  # [..., K//G]
    s_alpha, e_alpha = frexp_and_normalize(alpha)
    s_beta, e_beta = frexp_and_normalize(beta)
    s_scale, e_scale = s_alpha * s_beta, e_alpha + e_beta
    apply_e_zero |= s_scale == 0.0
    e_scale[apply_e_zero] = e_zero
    s_c, e_c = frexp_and_normalize(c)
    e_c[s_c == 0.0] = e_zero
    s = torch.cat([p * s_scale, s_c.unsqueeze(-1)], dim=-1)
    e = torch.cat([e_scale, e_c.unsqueeze(-1)], dim=-1)
    sum, e_max = truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_GST_FDPA(MMABlockScaleOperation):
    def __init__(self, G: int, F: int, rho: str, L_max: int, e_zero: int):
        self.G = G
        self.F = F
        self.rho = rho
        self.L_max = L_max
        self.e_zero = e_zero

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if a.dtype == torch.uint8:  # unpacked
            a = unpack_uint8_to_fp4(a)
        if b.dtype == torch.uint8:
            b = unpack_uint8_to_fp4(b)
        if alpha.dtype == torch.float8_e4m3fn:
            alpha = truncate_e4m3_to_ue4m3(alpha)
            beta = truncate_e4m3_to_ue4m3(beta)
        K = a.shape[-1]
        L = min(K, self.L_max)
        K_block = K // alpha.shape[-1]
        for i in range(0, K, L):
            c = gst_fdpa(
                a[..., i : i + L],
                b[..., i : i + L],
                c,
                alpha[..., i // K_block : (i + L) // K_block],
                beta[..., i // K_block : (i + L) // K_block],
                self.G,
                self.F,
                self.rho,
                self.e_zero,
            )
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        a_vectors = A[:, None, :].expand(-1, n, -1)
        b_vectors = B.T[None, :, :].expand(m, -1, -1)
        alpha_vectors = alpha[:, None, :].expand(-1, n, -1)
        beta_vectors = beta.T[None, :, :].expand(m, -1, -1)
        return self.dpa(a_vectors, b_vectors, C, alpha_vectors, beta_vectors)


def tr_fdpa(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, F2: int, rho: str
) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s, e = s_a * s_b, e_a + e_b
    overflow = ((s.abs() >= 2.0) + e) >= 128
    s[overflow] *= float("inf")
    e[s == 0.0] = -999  # handle zero
    s_dot, e_dot = truncated_fused_sum(s, e, F)

    s_c, e_c = frexp_and_normalize(c)
    e_max = torch.max(e_dot, e_c)
    s_dot = torch.floor(s_dot * pow2(e_dot - e_max + (F2 + 1))) * 2.0 ** -(F2 + 1)
    s_c = torch.floor(s_c * pow2(e_c - e_max + F)) * 2.0**-F
    sum, e_max = frexp_and_normalize((s_dot + s_c) * pow2(e_max), e_subnormal=-126)
    sum = torch.floor(sum * 2.0**F2) * 2.0**-F2
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_TR_FDPA(MMAOperation):
    def __init__(self, F: int, F2: int, rho: str, L_max: int):
        self.F = F
        self.F2 = F2
        self.rho = rho
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if a.dtype == torch.float32:  # tf32
            a = truncate_fp32_to_tf32(a)
            b = truncate_fp32_to_tf32(b)
        K = a.shape[-1]
        L = min(K, self.L_max)
        for i in range(0, K, L):
            c = tr_fdpa(
                a[..., i : i + L], b[..., i : i + L], c, self.F, self.F2, self.rho
            )
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        a_vectors = A[:, None, :].expand(-1, n, -1)
        b_vectors = B.T[None, :, :].expand(m, -1, -1)
        return self.dpa(a_vectors, b_vectors, C)


def gtr_fdpa(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    F: int,
    F2: int,
    rho: str,
) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s = s_a * s_b
    e = e_a + e_b
    e[s == 0.0] = -999  # handle zero
    s_even, e_even = truncated_fused_sum(s[..., ::2], e[..., ::2], F)
    s_odd, e_odd = truncated_fused_sum(s[..., 1::2], e[..., 1::2], F)

    e_dot = torch.max(e_even, e_odd)
    s_dot = torch.floor(s_even * pow2(e_even - e_dot + F)) * 2.0**-F
    s_dot += torch.floor(s_odd * pow2(e_odd - e_dot + F)) * 2.0**-F

    s_c, e_c = frexp_and_normalize(c)  # -126 <= e_c <= 127
    e_max = torch.max(e_dot, e_c)
    s_dot = torch.floor(s_dot * pow2(e_dot - e_max + (F2 + 1))) * 2.0 ** -(F2 + 1)
    s_c = torch.floor(s_c * pow2(e_c - e_max + F)) * 2.0**-F
    s_c[e_c < e_max - F - 1] = 0.0
    s_dot, e_max = frexp_and_normalize((s_dot + s_c) * pow2(e_max), e_subnormal=-126)
    s_dot = torch.floor(s_dot * 2.0**F2) * 2.0**-F2
    return ldexp_and_normalize(s_dot, e_max, rho)


class MMA_GTR_FDPA(MMAOperation):
    def __init__(self, F: int, F2: int, rho: str, L_max: int):
        self.F = F
        self.F2 = F2
        self.rho = rho
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        K = a.shape[-1]
        L = min(K, self.L_max)
        for i in range(0, K, L):
            c = gtr_fdpa(
                a[..., i : i + L], b[..., i : i + L], c, self.F, self.F2, self.rho
            )
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        a_vectors = A[:, None, :].expand(-1, n, -1)
        b_vectors = B.T[None, :, :].expand(m, -1, -1)
        return self.dpa(a_vectors, b_vectors, C)


def two_sum(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sum = a + b
    a_virtual = sum - b
    b_virtual = sum - a_virtual
    delta_a = a - a_virtual
    delta_b = b - b_virtual
    err = delta_a + delta_b
    return sum, err


def exact_sum(summands: torch.Tensor) -> torch.Tensor:
    parts = torch.zeros_like(summands)
    for i in range(summands.shape[-1]):
        x = summands[..., i]
        for j in range(i):
            y = parts[..., j]
            sum, err = two_sum(x, y)
            parts[..., j] = err
            x = sum
        parts[..., i] = x
    return parts


def normalize_parts_to_fp32(parts: torch.Tensor) -> torch.Tensor:
    while True:
        prev = parts
        p = list(parts.unbind(-1))
        for k in range(len(p) - 1):
            p[k + 1], p[k] = two_sum(p[k + 1], p[k])
        parts = torch.stack(p, -1)
        if torch.equal(parts.nan_to_num(), prev.nan_to_num()):
            break

    abs_p = parts.abs()
    i = abs_p.argmax(-1, keepdim=True)
    xy = parts.gather(-1, i).squeeze(-1)
    abs_p = abs_p.scatter(-1, i, torch.zeros_like(xy.unsqueeze(-1)))
    z = parts.gather(-1, abs_p.argmax(-1, keepdim=True)).squeeze(-1)

    x = xy.float()
    y = xy - x

    _, e = frexp_and_normalize(x)
    ulp = pow2(e - 23)
    is_pow2 = x.abs() == pow2(e)
    toward_zero = (y * x) < 0
    ulp = torch.where(is_pow2 & toward_zero, 0.5 * ulp, ulp)
    is_tie = y.abs() == 0.5 * ulp
    correction = 0.25 * ulp * torch.sign(z)
    y = torch.where(is_tie, y + correction, y)
    return (x + y).float()


def e_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    prod = a.double() * b.double()
    summands = torch.cat([prod, c.double().unsqueeze(-1)], dim=-1)
    special_vals = summands.sum(dim=-1).float()
    parts = exact_sum(summands)
    sum = normalize_parts_to_fp32(parts)
    return torch.where(special_vals.isnan() | special_vals.isinf(), special_vals, sum)


class MMA_E_FDPA(MMAOperation):
    def __init__(self, L_max: int):
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        K = a.shape[-1]
        L = min(K, self.L_max)
        for i in range(0, K, L):
            c = e_fdpa(a[..., i : i + L], b[..., i : i + L], c)
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        a_vectors = A[:, None, :].expand(-1, n, -1)
        b_vectors = B.T[None, :, :].expand(m, -1, -1)
        return self.dpa(a_vectors, b_vectors, C)
