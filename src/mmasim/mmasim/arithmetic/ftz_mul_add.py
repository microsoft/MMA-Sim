import torch

from .. import MMAOperation
from .helper import dtype_subnormal_exponent


def flush_subnormal(x: torch.Tensor, keep_sign: bool = False) -> torch.Tensor:
    min_exponent = dtype_subnormal_exponent[x.dtype]
    zeros = x * 0.0 if keep_sign else torch.zeros_like(x)
    return torch.where(x.abs() < 2.0**min_exponent, zeros, x)


def ftz_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.dtype == y.dtype
    assert x.dtype in [torch.float16, torch.bfloat16]
    # Flush subnormals to +0.0
    x = flush_subnormal(x, keep_sign=False)
    y = flush_subnormal(y, keep_sign=False)
    return flush_subnormal(x.float() * y.float(), keep_sign=True)


def ftz_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.dtype == y.dtype == torch.float32
    return flush_subnormal(x.float() + y.float(), keep_sign=True)


class MMA_FTZ_MUL_ADD(MMAOperation):
    def __init__(self, P: int):
        super().__init__()
        assert P in [2, 4]
        self.P = P

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        product = ftz_mul(a, b)
        c = flush_subnormal(c, keep_sign=False)
        for i in range(0, a.shape[-1], self.P):
            s = ftz_add(product[..., i], product[..., i + 1])
            if self.P == 4:
                s2 = ftz_add(product[..., i + 2], product[..., i + 3])
                s = ftz_add(s, s2)
            c = ftz_add(c, s)
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
        D = self.dpa(a_vectors, b_vectors, C)
        return D
