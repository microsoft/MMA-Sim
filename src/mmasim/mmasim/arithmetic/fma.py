import ctypes
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .. import MMAOperation


@triton.jit
def fma_gpu_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    tl.store(d_ptr + offs, libdevice.fma(a, b, c), mask=mask)


def fma_gpu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    fma_gpu_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        a, b, c, out, n, BLOCK=1024  # type: ignore
    )
    return out


fma_cpu_kernel_path = load(
    "fma_cpu",
    [str(Path(__file__).with_name("fma_cpu.cpp"))],
    extra_cflags=["-O3", "-fopenmp"],
    # build_directory=str(dir),
    is_python_module=False,
)
fma_cpu_kernel = ctypes.CDLL(fma_cpu_kernel_path)
for kernel in (fma_cpu_kernel.fma_f32, fma_cpu_kernel.fma_f64):
    kernel.restype = None
    kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]


def fma_cpu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    d = torch.empty_like(a)
    kernel = (
        fma_cpu_kernel.fma_f32 if a.dtype == torch.float32 else fma_cpu_kernel.fma_f64
    )
    kernel(a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr(), a.numel())
    return d


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == c.dtype
    assert a.dtype in [torch.float32, torch.float64]
    return fma_gpu(a, b, c) if a.is_cuda else fma_cpu(a, b, c)


class MMA_FMA(MMAOperation):
    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        K = a.shape[-1]
        for i in range(K):
            c = fma(a[..., i], b[..., i], c)
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
