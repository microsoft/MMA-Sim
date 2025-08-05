from typing import Callable

import torch

from ...isa import MMAInstructionBase, NV_MMABase, NV_WGMMABase, AMD_MFMABase


class Intrinsic(MMAInstructionBase):
    intrinsic: Callable

    def dotadd(self, a: list[float], b: list[float], c: float = 0.0) -> float: ...


class NV_MMA(Intrinsic, NV_MMABase):
    def __init__(self, qualifier: str, intrinsic: Callable):
        NV_MMABase.__init__(self, qualifier)
        self.intrinsic = intrinsic

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert A.shape == (m, k)
        assert B.shape == (k, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type
        assert A.is_contiguous() and C.is_contiguous()
        if not B.T.is_contiguous():
            # Ensure B is column-major
            B = B.T.contiguous().T
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()
        D = torch.empty((m, n), dtype=self.d_type, device="cuda")
        self.intrinsic(D.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr())
        return D

    def dotadd(self, a: list[float], b: list[float], c: float = 0.0) -> float:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        for i in range(len(a)):
            A[0, i] = a[i]
            B_T[0, i] = b[i]
        C[0, 0] = c
        D = self(A, B_T.T, C)
        return D[0, 0].item()


class NV_WGMMA(Intrinsic, NV_WGMMABase):
    def __init__(self, qualifier: str, intrinsic: Callable):
        NV_WGMMABase.__init__(self, qualifier)
        self.intrinsic = intrinsic

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert A.shape == (m, k)
        assert B.shape == (k, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type
        assert A.is_contiguous() and C.is_contiguous()
        if not B.T.is_contiguous():
            # Ensure B is column-major
            B = B.T.contiguous().T
        A = A.cuda()
        B = B.cuda()
        D = C.cuda()
        self.intrinsic(D.data_ptr(), A.data_ptr(), B.data_ptr())
        return D

    def dotadd(self, a: list[float], b: list[float], c: float = 0.0) -> float:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        for i in range(len(a)):
            A[0, i] = a[i]
            B_T[0, i] = b[i]
        C[0, 0] = c
        D = self(A, B_T.T, C)
        return D[0, 0].item()


class AMD_MFMA(Intrinsic, AMD_MFMABase):
    def __init__(self, qualifier: str, intrinsic: Callable):
        AMD_MFMABase.__init__(self, qualifier)
        self.intrinsic = intrinsic

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert A.shape == (m, k)
        assert B.shape == (k, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type
        assert (
            A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
        )  # row-major
        A = A.cuda()
        B = B.cuda()
        C = C.cuda()
        D = torch.empty((m, n), dtype=self.d_type, device="cuda")
        self.intrinsic(D.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr())
        return D

    def dotadd(self, a: list[float], b: list[float], c: float = 0.0) -> float:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B = torch.zeros([self.k, self.n], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        for i in range(len(a)):
            A[0, i] = a[i]
            B[i, 0] = b[i]
        C[0, 0] = c
        D = self(A, B, C)
        return D[0, 0].item()
