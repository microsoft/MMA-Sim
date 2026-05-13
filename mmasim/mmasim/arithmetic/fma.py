import ctypes

import torch

from .. import MMAOperation

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


class MMA_FMA(MMAOperation):
    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for i in range(len(a)):
            c = fma(a[i], b[i], c)
        return c

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        m, n = C.shape
        D = torch.zeros((m, n), dtype=C.dtype)
        for i in range(m):
            for j in range(n):
                D[i][j] = self.dpa(A[i, :], B[:, j], C[i, j])
        return D
