import torch
from typing import Annotated
from .. import MMAOperation, MMABlockScaleOperation
from .helper import (
    dtype_min_exponent,
    truncate_fp32_to_tf32,
    truncate_e4m3_to_ue4m3,
    unpack_uint8_to_fp4,
)

DoubleTensor = Annotated[torch.Tensor, torch.float64]
FloatTensor = Annotated[torch.Tensor, torch.float32]
IntTensor = Annotated[torch.Tensor, torch.int32]


def fp_to_sig_exp(x: torch.Tensor) -> tuple[FloatTensor | DoubleTensor, IntTensor]:
    assert x.dtype in dtype_min_exponent, f"Unsupported dtype: {x.dtype}"
    if x.dtype == torch.float64:
        s, e = torch.frexp(x)
    else:
        s, e = torch.frexp(x.float())
    s *= 2  # let 1 <= |s| < 2
    e -= 1
    # handle subnormal
    min_e = dtype_min_exponent[x.dtype]
    s[e < min_e] *= 2.0 ** (e[e < min_e] - min_e)
    e[e < min_e] = min_e
    # handle zero
    e[s == 0.0] = -min_e
    return s, e


def sig_exp_to_fp(s: DoubleTensor, e: IntTensor, rho: str) -> torch.Tensor:
    if torch.isnan(s):
        if rho == "RNE-FP16":
            return torch.tensor(0x7FFF, dtype=torch.int16).view(torch.float16)
        else:
            return torch.tensor(0x7FFFFFFF, dtype=torch.int32).view(torch.float32)
    elif torch.isinf(s):
        return s.to(torch.float16 if rho == "RNE-FP16" else torch.float32)

    if rho == "RNE-FP16":
        # note that direcctly converting FP64 to FP16 can be incorrect
        # as PyTorch-CPU computes FP64 -> FP32 -> FP16 internally
        s, e = fp_to_sig_exp(torch.ldexp(s, e))  # renormalize
        if e < -14:  # subnormal
            s *= 2.0 ** (e + 14)
            e.fill_(-14)
        s = torch.round(s * 2.0**10) * 2.0**-10  # RNE
        return (torch.ldexp(s, e)).to(torch.float16)
    elif rho == "RNE-FP32":
        return (torch.ldexp(s, e)).to(torch.float32)
    else:  # RZ
        s, e = fp_to_sig_exp(torch.ldexp(s, e))  # renormalize
        if e < -126:  # subnormal
            s *= 2.0 ** (e + 126)
            e.fill_(-126)
        if rho == "RZ-E8M13":
            s = torch.trunc(s * 2.0**13) * 2.0**-13  # RZ
        else:  # "RZ-FP32"
            s = torch.trunc(s * 2.0**23) * 2.0**-23  # RZ
        return (torch.ldexp(s, e)).to(torch.float32)


def truncated_fused_sum(
    s: torch.Tensor, e: torch.Tensor, F: int
) -> tuple[DoubleTensor, IntTensor]:
    e[s == 0.0] = -126  # TODO: handle zero
    max_e = max(e)
    s *= torch.pow(torch.tensor(2.0).double(), e - max_e + F)
    s = torch.trunc(s) * 2.0**-F
    sum = s.double().sum()
    return sum, max_e


def t_fdpa(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, F: int, rho: str
) -> torch.Tensor:
    sa, ea = fp_to_sig_exp(a)
    sb, eb = fp_to_sig_exp(b)
    sc, ec = fp_to_sig_exp(c)
    s = torch.cat([sa * sb, sc.unsqueeze(0)])
    e = torch.cat([ea + eb, ec.unsqueeze(0)])
    sum, max_e = truncated_fused_sum(s, e, F)
    return sig_exp_to_fp(sum, max_e, rho)


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
            c = t_fdpa(
                a[i : i + L], b[i : i + L], c, self.F, self.rho
            )
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
    sa, ea = fp_to_sig_exp(a)
    sb, eb = fp_to_sig_exp(b)
    sc, ec = fp_to_sig_exp(c)
    s_alpha, e_alpha = fp_to_sig_exp(alpha)
    s_beta, e_beta = fp_to_sig_exp(beta)
    if torch.isnan(s_alpha).any() or torch.isnan(s_beta).any():
        return sig_exp_to_fp(torch.tensor(float("nan")).double(), torch.tensor(0), rho)
    s = torch.cat([sa * sb, sc.unsqueeze(0)])
    e = torch.cat([ea + eb + e_alpha + e_beta, ec.unsqueeze(0)])
    sum, max_e = truncated_fused_sum(s, e, F)
    return sig_exp_to_fp(sum, max_e, rho)


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
    s_alpha, e_alpha = fp_to_sig_exp(alpha)
    s_beta, e_beta = fp_to_sig_exp(beta)
    sc, ec = fp_to_sig_exp(c)
    s = torch.cat([p * s_alpha * s_beta, sc.unsqueeze(0)])
    e = torch.cat([e_alpha + e_beta, ec.unsqueeze(0)])
    sum, max_e = truncated_fused_sum(s, e, F)
    return sig_exp_to_fp(sum, max_e, rho)


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
    sa, ea = fp_to_sig_exp(a)
    sb, eb = fp_to_sig_exp(b)
    sc, ec = fp_to_sig_exp(c)
    s = torch.cat([sa * sb, sc.unsqueeze(0)])
    e = torch.cat([ea + eb, ec.unsqueeze(0)])
    has_inf_nan = s.sum()
    if has_inf_nan.isinf() or has_inf_nan.isnan():
        return has_inf_nan
    emin = int(e.min().item())
    sum = 0
    for i in range(len(s)):
        si = s[i].item()
        ei = int(e[i].item())
        sum += int(si * 2**23) << (ei - emin)
    sign = 1 if sum >= 0 else -1
    sum = abs(sum)
    t = sum.bit_length()
    if t > 25:
        sticky = bool(sum & ((1 << (t - 25)) - 1))
        sum = (sum >> (t - 25) << 1) | sticky
        return torch.tensor(
            sign * sum * 2.0 ** (emin - 23 + t - 26), dtype=torch.float32
        )
    else:
        return torch.tensor(sign * sum * 2.0 ** (emin - 23), dtype=torch.float32)


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
    sa, ea = fp_to_sig_exp(a)
    sb, eb = fp_to_sig_exp(b)
    sum, max_e = truncated_fused_sum(sa * sb, ea + eb, F)
    sc, ec = fp_to_sig_exp(c)
    E = torch.max(max_e, ec)
    sum = torch.floor(sum * torch.pow(2.0, max_e - E + F2)) * 2.0**-F2
    sum += torch.floor(sc * torch.pow(2.0, ec - E + F)) * 2.0**-F
    return sig_exp_to_fp(sum, max_e, rho)


class MMA_TR_FDPA(MMAOperation):
    def __init__(self, F: int, F2: int, rho: str, L_max: int):
        self.F = F
        self.F2 = F2
        self.rho = rho
        self.L_max = L_max

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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
    sa, ea = fp_to_sig_exp(a)
    sb, eb = fp_to_sig_exp(b)
    s0, e0 = truncated_fused_sum(sa[::2] * sb[::2], ea[::2] + eb[::2], F)
    s1, e1 = truncated_fused_sum(sa[1::2] * sb[1::2], ea[1::2] + eb[1::2], F)
    e_max = torch.max(e0, e1)
    sum = torch.trunc(s0 * torch.pow(2.0, e0 - e_max + F)) * 2.0**-F
    sum += torch.trunc(s1 * torch.pow(2.0, e1 - e_max + F)) * 2.0**-F
    sc, ec = fp_to_sig_exp(c)
    E = torch.max(e_max, ec)
    sum = torch.trunc(sum * torch.pow(2.0, e_max - E + F2)) * 2.0**-F2
    sum += torch.trunc(sc * torch.pow(2.0, ec - E + F)) * 2.0**-F
    return sig_exp_to_fp(sum, E, rho)


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
