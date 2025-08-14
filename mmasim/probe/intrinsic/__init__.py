from typing import Callable

import torch

from ...isa import MMAInstructionBase
from ...isa import NV_MMABase, NV_WGMMABase, NV_TCGen05MMABase
from ...isa import AMD_MFMABase


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


class NV_TCGen05MMA(Intrinsic, NV_TCGen05MMABase):
    def __init__(self, qualifier: str, intrinsic: Callable):
        NV_TCGen05MMABase.__init__(self, qualifier)
        self.intrinsic = intrinsic

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor | None = None,
        scale_B: torch.Tensor | None = None,
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
        if self.kind.startswith("mx"):
            assert scale_A is not None and scale_B is not None
            assert scale_A.shape == (m, k // self.block_size)
            assert scale_B.shape == (k // self.block_size, n)
            assert scale_A.element_size() == 1
            assert scale_B.element_size() == 1
            assert scale_A.is_contiguous()
            if not scale_B.T.is_contiguous():
                # Ensure scale_B is column-major
                scale_B = scale_B.T.contiguous().T
            scale_A = scale_A.cuda()
            scale_B = scale_B.cuda()
            self.intrinsic(
                D.data_ptr(),
                A.data_ptr(),
                B.data_ptr(),
                scale_A.data_ptr(),
                scale_B.data_ptr(),
            )
        else:
            assert scale_A is None and scale_B is None
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

    def dotadd_with_block_scale(
        self,
        a: list[float],
        b: list[float],
        c: float,
        scale_a: list[int],
        scale_b: list[int],
    ) -> float:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        for i in range(len(a)):
            A[0, i] = a[i]
            B_T[0, i] = b[i]
        C[0, 0] = c
        scale_A = torch.full(
            [self.m, self.k // self.block_size], 127, dtype=torch.uint8
        )
        scale_B_T = torch.full(
            [self.n, self.k // self.block_size], 127, dtype=torch.uint8
        )
        for i in range(self.k // self.block_size):
            scale_A[0, i] = scale_a[i] + 127
            scale_B_T[0, i] = scale_b[i] + 127
        D = self(A, B_T.T, C, scale_A, scale_B_T.T)
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
