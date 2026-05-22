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


def pow2(e: IntTensor | DoubleTensor) -> FloatTensor | DoubleTensor:
    # torch.ldexp and torch.pow can be inaccurate on gpu
    # torch.exp2(-127) can be inaccurate on gpu for e.dtype == torch.int32
    # so, use DoubleTensor if e < -126 or e > 127
    return torch.exp2(e)


def frexp_and_normalize(
    x: torch.Tensor, e_subnormal: int | None = None
) -> tuple[FloatTensor | DoubleTensor, IntTensor]:
    assert x.dtype in dtype_subnormal_exponent, f"Unsupported dtype: {x.dtype}"
    if x.dtype == torch.float64:
        s, e = torch.frexp(x)
    else:
        s, e = torch.frexp(x.float())
    s *= 2  # let 1 <= |s| < 2
    e -= 1
    # handle subnormal
    if e_subnormal is None:
        e_subnormal = dtype_subnormal_exponent[x.dtype]
    subnormals = e < e_subnormal
    s[subnormals] *= pow2(e[subnormals] - e_subnormal)  # -52 <= delta_e <= 0
    e[subnormals] = e_subnormal
    # handle zero
    e[s == 0.0] = e_subnormal
    return s, e


def ldexp_and_normalize(s: DoubleTensor, e: IntTensor, rho: str) -> torch.Tensor:
    if torch.isnan(s):
        if rho == "RNE-FP16":
            return torch.tensor(0x7FFF, dtype=torch.int16).view(torch.float16)
        else:
            return torch.tensor(0x7FFFFFFF, dtype=torch.int32).view(torch.float32)
    elif torch.isinf(s):
        return s.to(torch.float16 if rho == "RNE-FP16" else torch.float32)

    if rho == "RNE-FP16":
        # -14 <= e < 15*2
        # note that direcctly converting FP64 to FP16 can be incorrect
        # as PyTorch-CPU computes FP64 -> FP32 -> FP16 internally
        s, e = frexp_and_normalize(s * pow2(e), e_subnormal=-14)
        s = torch.round(s * 2.0**10) * 2.0**-10  # RNE
        return (s * pow2(e)).to(torch.float16)
    elif rho == "RNE-FP32":
        # -126 <= e <= 127*2
        return (s * pow2(e.double())).to(torch.float32)
    else:  # RZ
        # -126 <= e <= 127*2
        s, e = frexp_and_normalize(s * pow2(e.double()), e_subnormal=-126)
        if rho == "RZ-E8M13":
            s = torch.trunc(s * 2.0**13) * 2.0**-13  # RZ
        else:  # "RZ-FP32"
            s = torch.trunc(s * 2.0**23) * 2.0**-23  # RZ
        return (s * pow2(e.double())).to(torch.float32)


def truncated_fused_sum(
    s: torch.Tensor, e: IntTensor, F: int
) -> tuple[DoubleTensor, IntTensor]:
    # -126*2 <= e <= 127*2
    e[s == 0.0] = -126  # TODO: handle zero
    e_max = max(e)
    s = torch.trunc(s * pow2(e.double() - e_max + F)) * 2.0**-F
    sum = s.sum()
    return sum, e_max


def t_fdpa(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, rho: str
) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s_c, e_c = frexp_and_normalize(c)
    s = torch.cat([s_a * s_b, s_c.unsqueeze(0)])
    e = torch.cat([e_a + e_b, e_c.unsqueeze(0)])
    sum, e_max = truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_T_FDPA(MMAOperation):
    def __init__(self, F: int, rho: str, L_max: int):
        self.F = F
        self.rho = rho
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if a.dtype == torch.float32:  # tf32
            a = truncate_fp32_to_tf32(a)
            b = truncate_fp32_to_tf32(b)
        L = min(len(a), self.L_max)
        for i in range(0, len(a), L):
            c = t_fdpa(a[i : i + L], b[i : i + L], c, self.F, self.rho)
        return c

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        m, n = C.shape
        dtype = torch.float16 if self.rho.endswith("FP16") else torch.float32
        D = torch.zeros((m, n), dtype=dtype)
        for i in range(m):
            for j in range(n):
                D[i, j] = self.dpa(A[i, :], B[:, j], C[i, j])
        return D


def st_fdpa(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    F: int,
    rho: str,
) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s_c, e_c = frexp_and_normalize(c)
    s_alpha, e_alpha = frexp_and_normalize(alpha)
    s_beta, e_beta = frexp_and_normalize(beta)
    # s_alpha, s_beta can be 1.0 or nan
    s = torch.cat([s_a * s_b * s_alpha * s_beta, s_c.unsqueeze(0)])
    e = torch.cat([e_a + e_b + e_alpha + e_beta, e_c.unsqueeze(0)])
    sum, e_max = truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_ST_FDPA(MMABlockScaleOperation):
    def __init__(self, F: int, rho: str, L_max: int):
        self.F = F
        self.rho = rho
        self.L_max = L_max

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        L = min(len(a), self.L_max)
        for i in range(0, len(a), L):
            c = st_fdpa(
                a[i : i + L],
                b[i : i + L],
                c,
                alpha,
                beta,
                self.F,
                self.rho,
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
        dtype = torch.float16 if self.rho.endswith("FP16") else torch.float32
        D = torch.zeros((m, n), dtype=dtype)
        for i in range(m):
            for j in range(n):
                D[i, j] = self.dpa(
                    A[i, :],
                    B[:, j],
                    C[i, j],
                    alpha[i, :],
                    beta[:, j],
                )
        return D


def gst_fdpa(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    G: int,
    F: int,
    rho: str,
) -> torch.Tensor:
    p = a.float() * b.float()
    p = p.view(-1, G).sum(dim=1).flatten()
    K_block = len(a) // len(alpha)
    alpha = torch.repeat_interleave(alpha, K_block // G)
    beta = torch.repeat_interleave(beta, K_block // G)
    s_alpha, e_alpha = frexp_and_normalize(alpha)
    s_beta, e_beta = frexp_and_normalize(beta)
    s_c, e_c = frexp_and_normalize(c)
    s = torch.cat([p * s_alpha * s_beta, s_c.unsqueeze(0)])
    e = torch.cat([e_alpha + e_beta, e_c.unsqueeze(0)])
    sum, e_max = truncated_fused_sum(s, e, F)
    return ldexp_and_normalize(sum, e_max, rho)


class MMA_GST_FDPA(MMABlockScaleOperation):
    def __init__(self, G: int, F: int, rho: str, L_max: int):
        self.G = G
        self.F = F
        self.rho = rho
        self.L_max = L_max

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
        L = min(len(a), self.L_max)
        K_block = len(a) // len(alpha)
        for i in range(0, len(a), L):
            c = gst_fdpa(
                a[i : i + L],
                b[i : i + L],
                c,
                alpha[i // K_block : (i + L) // K_block],
                beta[i // K_block : (i + L) // K_block],
                self.G,
                self.F,
                self.rho,
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
        dtype = torch.float16 if self.rho.endswith("FP16") else torch.float32
        D = torch.zeros((m, n), dtype=dtype)
        for i in range(m):
            for j in range(n):
                D[i, j] = self.dpa(
                    A[i, :],
                    B[:, j],
                    C[i, j],
                    alpha[i, :],
                    beta[:, j],
                )
        return D


def e_fdpa(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s_c, e_c = frexp_and_normalize(c)
    s = torch.cat([s_a * s_b, s_c.unsqueeze(0)])
    e = torch.cat([e_a + e_b, e_c.unsqueeze(0)])
    has_inf_nan = s.sum()
    if has_inf_nan.isinf() or has_inf_nan.isnan():
        return has_inf_nan
    min_e = int(e.min().item())
    sum = 0
    for i in range(len(s)):
        si = s[i].item()
        ei = int(e[i].item())
        sum += int(si * 2**23) << (ei - min_e)
    sign = 1 if sum >= 0 else -1
    sum = abs(sum)
    t = sum.bit_length()
    if t > 25:
        sticky = bool(sum & ((1 << (t - 25)) - 1))
        sum = (sum >> (t - 25) << 1) | sticky
        return torch.tensor(
            sign * sum * 2.0 ** (min_e - 23 + t - 26),
            dtype=torch.float32,
            device=a.device,
        )
    else:
        return torch.tensor(
            sign * sum * 2.0 ** (min_e - 23), dtype=torch.float32, device=a.device
        )


class MMA_E_FDPA(MMAOperation):
    def __init__(self, L_max: int):
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        L = min(len(a), self.L_max)
        for i in range(0, len(a), L):
            c = e_fdpa(a[i : i + L], b[i : i + L], c)
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        D = torch.zeros((m, n), dtype=torch.float32)
        for i in range(m):
            for j in range(n):
                D[i, j] = self.dpa(A[i, :], B[:, j], C[i, j])
        return D


def tr_fdpa(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, F2: int, rho: str
) -> torch.Tensor:
    s_a, e_a = frexp_and_normalize(a)
    s_b, e_b = frexp_and_normalize(b)
    s, e = s_a * s_b, e_a + e_b
    overflow = ((s.abs() >= 2.0) + e) >= 128
    s[overflow] *= float("inf")
    s_dot, e_dot = truncated_fused_sum(s, e, F)  # -126*2 <= e_dot <= 127*2
    s_c, e_c = frexp_and_normalize(c)  # -126 <= e_c <= 127
    e_max = torch.max(e_dot, e_c)
    s_dot = torch.floor(s_dot * pow2(e_dot.double() - e_max + 52)) * 2.0**-52  # 52 > F2
    s_c = torch.floor(s_c * pow2(e_c.double() - e_max + F)) * 2.0**-F
    sum, e_max = frexp_and_normalize((s_dot + s_c) * pow2(e_max.double()))
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
        L = min(len(a), self.L_max)
        for i in range(0, len(a), L):
            c = tr_fdpa(a[i : i + L], b[i : i + L], c, self.F, self.F2, self.rho)
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        dtype = torch.float16 if self.rho.endswith("FP16") else torch.float32
        D = torch.zeros((m, n), dtype=dtype)
        for i in range(m):
            for j in range(n):
                D[i, j] = self.dpa(A[i, :], B[:, j], C[i, j])
        return D


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
    s_even, e_even = truncated_fused_sum(s_a[::2] * s_b[::2], e_a[::2] + e_b[::2], F)
    s_odd, e_odd = truncated_fused_sum(s_a[1::2] * s_b[1::2], e_a[1::2] + e_b[1::2], F)
    e_dot = torch.max(e_even, e_odd)  # -15*2 <= e_even, e_odd <= 15*2
    s_dot = torch.floor(s_even * pow2(e_even - e_dot + F)) * 2.0**-F
    s_dot += torch.floor(s_odd * pow2(e_odd - e_dot + F)) * 2.0**-F
    s_c, e_c = frexp_and_normalize(c)  # -126 <= e_c <= 127
    e_max = torch.max(e_dot, e_c)
    s_dot = torch.floor(s_dot * pow2(e_dot.double() - e_max + 52)) * 2.0**-52  # 52 > F2
    s_c = torch.floor(s_c * pow2(e_c.double() - e_max + F)) * 2.0**-F
    s_c[e_c < e_max - F - 1] = 0.0
    s_dot, e_max = frexp_and_normalize((s_dot + s_c) * pow2(e_max))
    s_dot = torch.floor(s_dot * 2.0**F2) * 2.0**-F2
    return ldexp_and_normalize(s_dot, e_max, rho)


class MMA_GTR_FDPA(MMAOperation):
    def __init__(self, F: int, F2: int, rho: str, L_max: int):
        self.F = F
        self.F2 = F2
        self.rho = rho
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        L = min(len(a), self.L_max)
        for i in range(0, len(a), L):
            c = gtr_fdpa(a[i : i + L], b[i : i + L], c, self.F, self.F2, self.rho)
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        dtype = torch.float16 if self.rho.endswith("FP16") else torch.float32
        D = torch.zeros((m, n), dtype=dtype)
        for i in range(m):
            for j in range(n):
                D[i, j] = self.dpa(A[i, :], B[:, j], C[i, j])
        return D
