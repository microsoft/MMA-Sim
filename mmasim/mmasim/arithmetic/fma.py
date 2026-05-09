import ctypes

import torch

libm = ctypes.CDLL("libm.so.6")
libm.fmaf.argtypes = [ctypes.c_float] * 3
libm.fmaf.restype = ctypes.c_float
libm.fma.argtypes = [ctypes.c_double] * 3
libm.fma.restype = ctypes.c_double


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == c.dtype
    assert a.dtype in [torch.float32, torch.float64]
    if a.dtype == torch.float32:
        res = libm.fmaf(a.item(), b.item(), c.item())
        return torch.tensor(res, dtype=torch.float32)
    else:  # a.dtype == torch.float64:
        res = libm.fma(a.item(), b.item(), c.item())
        return torch.tensor(res, dtype=torch.float64)


def dpa_on_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    for i in range(len(a)):
        c = fma(a[i], b[i], c)
    return c
