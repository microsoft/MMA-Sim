import torch
from torch.utils.cpp_extension import load_inline

import triton
import triton.language as tl
from triton.language.extra import libdevice

from .. import MMAOperation


@triton.jit
def fma_kernel(a_ptr, b_ptr, c_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    c = tl.load(c_ptr + offs, mask=m)
    tl.store(o_ptr + offs, libdevice.fma(a, b, c), mask=m)  # → fma.rn.f64


def fma_cuda(a, b, c):
    a, b, c = torch.broadcast_tensors(a, b, c)
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    fma_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](
        a, b, c, out, n, BLOCK=1024
    )
    return out


from torch.utils.cpp_extension import load_inline

fma_cpu = load_inline(
    name="fma_cpu",
    cpp_sources=[r"""
#include <torch/extension.h>
#include <cmath>
at::Tensor fma_cpu(at::Tensor a, at::Tensor b, at::Tensor c)
{
    auto t = at::broadcast_tensors({a, b, c});
    a = t[0].contiguous();
    b = t[1].contiguous();
    c = t[2].contiguous();
    auto out = at::empty_like(a);
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "fma_cpu", [&]
                               {
    auto pa=a.data_ptr<scalar_t>(),pb=b.data_ptr<scalar_t>(),
            pc=c.data_ptr<scalar_t>(),po=out.data_ptr<scalar_t>();
    int64_t n=a.numel();
    at::parallel_for(0,n,2048,[&](int64_t s,int64_t e){
        for(int64_t i=s;i<e;++i) po[i]=std::fma(pa[i],pb[i],pc[i]);}); });
    return out;
}"""],
    functions=["fma_cpu"],
    extra_cflags=["-O3"],
)


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    assert a.dtype == b.dtype == c.dtype
    assert a.dtype in [torch.float32, torch.float64]
    return fma_cuda(a, b, c) if a.is_cuda else fma_cpu.fma_cpu(a, b, c)


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
