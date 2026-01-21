from typing import Callable

import torch

from ..isa import MMA, WGMMA, TCGen05MMA, MFMA


class MMAKernel(MMA):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        MMA.__init__(self, arch, qualifier)
        self.kernel = kernel

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor | None = None,
        scale_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_input(A, B, C, scale_A, scale_B)
        if not A.is_contiguous():
            A = A.contiguous()  # Make A row-major
        if not B.T.is_contiguous():
            if self.packing == 2:
                raise ValueError("B must be column-major for fp4")
            B = B.T.contiguous().T  # Make B column-major
        if not C.is_contiguous():
            C = C.contiguous()  # Make C row-major
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()
        D = torch.empty((self.m, self.n), dtype=self.d_type, device="cuda")
        if self.block_size > 0:
            assert scale_A is not None and scale_B is not None
            if not scale_A.is_contiguous():
                scale_A = scale_A.contiguous()  # Make scale_A row-major
            if not scale_B.T.is_contiguous():
                scale_B = scale_B.T.contiguous().T  # Make scale_B column-major
            scale_A = scale_A.cuda()
            scale_B = scale_B.cuda()
            self.kernel(
                D.data_ptr(),
                A.data_ptr(),
                B.data_ptr(),
                C.data_ptr(),
                scale_A.data_ptr(),
                scale_B.data_ptr(),
            )
        else:
            self.kernel(D.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr())
        return D


class WGMMAKernel(WGMMA):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        WGMMA.__init__(self, arch, qualifier)
        self.kernel = kernel

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        if not A.is_contiguous():
            A = A.contiguous()  # Make A row-major
        if not B.T.is_contiguous():
            B = B.T.contiguous().T  # Make B column-major
        if not C.is_contiguous():
            C = C.contiguous()  # Make C row-major
        A = A.cuda()
        B = B.cuda()
        D = C.cuda()
        self.kernel(D.data_ptr(), A.data_ptr(), B.data_ptr())
        return D


class TCGen05MMAKernel(TCGen05MMA):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        TCGen05MMA.__init__(self, arch, qualifier)
        self.kernel = kernel

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor | None = None,
        scale_B: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_input(A, B, C, scale_A, scale_B)
        if not A.is_contiguous():
            A = A.contiguous()  # Make A row-major
        if not B.T.is_contiguous():
            if self.packing == 2:
                raise ValueError("B must be column-major for fp4")
            B = B.T.contiguous().T  # Make B column-major
        if not C.is_contiguous():
            C = C.contiguous()  # Make C row-major
        A = A.cuda()
        B = B.cuda()
        D = C.cuda()
        if self.block_size > 0:
            assert scale_A is not None and scale_B is not None
            if not scale_A.is_contiguous():
                scale_A = scale_A.contiguous()  # Make scale_A row-major
            if not scale_B.T.is_contiguous():
                scale_B = scale_B.T.contiguous().T  # Make scale_B column-major
            scale_A = scale_A.cuda()
            scale_B = scale_B.cuda()
            self.kernel(
                D.data_ptr(),
                A.data_ptr(),
                B.data_ptr(),
                scale_A.data_ptr(),
                scale_B.data_ptr(),
            )
        else:
            self.kernel(D.data_ptr(), A.data_ptr(), B.data_ptr())
        return D


class MFMAKernel(MFMA):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        MFMA.__init__(self, arch, qualifier)
        self.kernel = kernel

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        self.check_input(A, B, C)
        if not A.is_contiguous():
            A = A.contiguous()  # Make A row-major
        if not B.T.is_contiguous():
            B = B.T.contiguous().T  # Make B column-major
        if not C.is_contiguous():
            C = C.contiguous()  # Make C row-major
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()
        D = torch.empty((self.m, self.n), dtype=self.d_type, device="cuda")
        self.kernel(D.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr())
        return D
