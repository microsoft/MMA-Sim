from typing import Callable

import torch

from ..isa import nv_ptx, amd


class mma_kernel(nv_ptx.mma):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        super().__init__(arch, qualifier)
        self.kernel = kernel

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
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


class mma_block_scale_kernel(nv_ptx.mma_block_scale):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        super().__init__(arch, qualifier)
        self.kernel = kernel

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
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
        return D


class wgmma_kernel(nv_ptx.wgmma):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        super().__init__(arch, qualifier)
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


class tcgen05mma_kernel(nv_ptx.tcgen05mma):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        super().__init__(arch, qualifier)
        self.kernel = kernel

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
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


class tcgen05mma_block_scale_kernel(nv_ptx.tcgen05mma_block_scale):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        super().__init__(arch, qualifier)
        self.kernel = kernel

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor,
        scale_B: torch.Tensor,
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
        return D


class mfma_kernel(amd.mfma):
    def __init__(self, arch: str, qualifier: str, kernel: Callable):
        super().__init__(arch, qualifier)
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
