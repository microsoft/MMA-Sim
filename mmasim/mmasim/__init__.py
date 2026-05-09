import torch


# Matrix Multiply-Accumulate (MMA) Operation
class MMAOperation:
    # m: int
    # n: int
    # k: int
    # a_type: torch.dtype
    # b_type: torch.dtype
    # c_type: torch.dtype
    # d_type: torch.dtype

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor: ...

    # dot product and accumulate
    def dpa(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...


    # # dot product and accumulate
    # def dpa(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    #     A = torch.zeros([self.m, self.k], dtype=self.a_type)
    #     B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
    #     C = torch.zeros([self.m, self.n], dtype=self.c_type)
    #     A[0, : len(a)] = a
    #     B_T[0, : len(b)] = b
    #     C[0, 0] = c
    #     D = self(A, B_T.T, C)
    #     return D[0, 0]


class MMABlockScaleOperation:
    # m: int
    # n: int
    # k: int
    # block_size: int
    # packing: int
    # a_type: torch.dtype
    # b_type: torch.dtype
    # c_type: torch.dtype
    # d_type: torch.dtype
    # s_type: torch.dtype

    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor: ...

    # dot product and accumulate
    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor: ...

    # def check_input(
    #     self,
    #     A: torch.Tensor,
    #     B: torch.Tensor,
    #     C: torch.Tensor,
    #     alpha: torch.Tensor,
    #     beta: torch.Tensor,
    # ):
    #     m, n, k, packing = self.m, self.n, self.k, self.packing
    #     assert A.shape == (m, k // packing)
    #     assert B.shape == (k // packing, n)
    #     assert C.shape == (m, n)
    #     assert A.dtype == self.a_type
    #     assert B.dtype == self.b_type
    #     assert C.dtype == self.c_type
    #     assert alpha.shape == (m, k // self.block_size)
    #     assert beta.shape == (k // self.block_size, n)
    #     assert alpha.dtype == beta.dtype == self.s_type

    # def dpa_block_scale(
    #     self,
    #     a: torch.Tensor,
    #     b: torch.Tensor,
    #     c: torch.Tensor,
    #     alpha0: torch.Tensor,
    #     beta0: torch.Tensor,
    # ) -> torch.Tensor:
    #     m, n, k = self.m, self.n, self.k
    #     packing, block_size = self.packing, self.block_size
    #     A = torch.zeros([m, k // packing], dtype=self.a_type)
    #     B_T = torch.zeros([n, k // packing], dtype=self.b_type)
    #     C = torch.zeros([m, n], dtype=self.c_type)
    #     alpha = torch.ones([m, k // block_size], dtype=self.s_type)
    #     beta = torch.ones([n, k // block_size], dtype=self.s_type)
    #     A[0, : len(a)] = a
    #     B_T[0, : len(b)] = b
    #     C[0, 0] = c
    #     alpha[0, : len(alpha0)] = alpha0
    #     beta[0, : len(beta0)] = beta0
    #     D = self(A, B_T.T, C, alpha, beta.T)
    #     return D[0, 0]
