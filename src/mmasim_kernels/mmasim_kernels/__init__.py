from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ctypes import _NamedFuncPointer
else:
    _NamedFuncPointer = object

import torch
import mmasim


class MMAKernel(mmasim.MMAOperation):
    m: int
    n: int
    k: int
    a_type: torch.dtype
    b_type: torch.dtype
    c_type: torch.dtype
    d_type: torch.dtype
    kernel: _NamedFuncPointer

    def check_input(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ):
        m, n, k = self.m, self.n, self.k
        assert A.shape == (m, k)
        assert B.shape == (k, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type

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
        arg_count = len(self.kernel.argtypes)
        if arg_count == 4:
            self.kernel(D.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr())
        else:
            assert arg_count == 3
            D.copy_(C)
            self.kernel(D.data_ptr(), A.data_ptr(), B.data_ptr())
        return D

    def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        A[0, : len(a)] = a
        B_T[0, : len(b)] = b
        C[0, 0] = c
        D = self(A, B_T.T, C)
        return D[0, 0]


class MMABlockScaleKernel(mmasim.MMABlockScaleOperation):
    m: int
    n: int
    k: int
    block_size: int
    packing: int
    a_type: torch.dtype
    b_type: torch.dtype
    c_type: torch.dtype
    d_type: torch.dtype
    s_type: torch.dtype
    kernel: _NamedFuncPointer

    def check_input(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ):
        m, n, k, packing = self.m, self.n, self.k, self.packing
        assert A.shape == (m, k // packing)
        assert B.shape == (k // packing, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type
        assert alpha.shape == (m, k // self.block_size)
        assert beta.shape == (k // self.block_size, n)
        assert alpha.dtype == beta.dtype == self.s_type

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
        arg_count = len(self.kernel.argtypes)
        if arg_count == 6:
            self.kernel(
                D.data_ptr(),
                A.data_ptr(),
                B.data_ptr(),
                C.data_ptr(),
                scale_A.data_ptr(),
                scale_B.data_ptr(),
            )
        else:
            assert arg_count == 5
            D.copy_(C)
            self.kernel(
                D.data_ptr(),
                A.data_ptr(),
                B.data_ptr(),
                scale_A.data_ptr(),
                scale_B.data_ptr(),
            )
        return D

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        packing, block_size = self.packing, self.block_size
        A = torch.zeros([m, k // packing], dtype=self.a_type)
        B_T = torch.zeros([n, k // packing], dtype=self.b_type)
        C = torch.zeros([m, n], dtype=self.c_type)
        scale_A = torch.ones([m, k // block_size], dtype=self.s_type)
        scale_B = torch.ones([n, k // block_size], dtype=self.s_type)
        A[0, : len(a)] = a
        B_T[0, : len(b)] = b
        C[0, 0] = c
        scale_A[0, : len(alpha)] = alpha
        scale_B[0, : len(beta)] = beta
        D = self(A, B_T.T, C, scale_A, scale_B.T)
        return D[0, 0]
