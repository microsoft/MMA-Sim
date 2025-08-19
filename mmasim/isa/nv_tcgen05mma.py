import torch

from .common import (
    MatrixMultiplyAdd,
    MatrixMultiplyAddWithBlockScale,
    nv_shape_to_mnk,
    nv_torch_dtype,
)


class TCGen05MMA(MatrixMultiplyAdd, MatrixMultiplyAddWithBlockScale):
    def __init__(self, arch: str, qualifier: str):
        assert arch == "Blackwell"
        qualifiers = qualifier.split(".")
        if len(qualifiers) == 5:  # without block scale
            kind, shape, d_type, a_type, b_type = qualifiers
            m, n, k = nv_shape_to_mnk(shape)
            assert (
                (m in [64, 128] and n in list(range(8, 256 + 1, 8)))
                or (m in [128, 256] and n in list(range(16, 256 + 1, 16)))
                or (m in [32, 64, 128] and n in [64, 128, 256])
            )
            if kind == "tf32":
                assert k == 8
                assert a_type == b_type == "tf32"
                assert d_type == "f32"
            elif kind == "f16":
                assert k == 16
                assert a_type in ["f16", "bf16"]
                assert b_type in ["f16", "bf16"]
                assert (d_type == "f32") or (a_type == b_type == d_type == "f16")
            else:
                assert kind == "f8f6f4"
                assert k == 32
                assert a_type in ["e5m2", "e4m3"]  # TODO: support e3m2, e2m3, and e2m1
                assert b_type in ["e5m2", "e4m3"]
                assert d_type in ["f32", "f16"]
        else:  # with block scale
            assert len(qualifiers) == 7
            kind, shape, block_size, d_type, a_type, b_type, s_type = qualifiers
            m, n, k = nv_shape_to_mnk(shape)
            assert (m == 128 and n in list(range(8, 256 + 1, 8))) or (
                m in [128, 256] and n in list(range(16, 256 + 1, 16))
            )
            if kind == "mxf8f6f4":
                assert k == 32
                assert a_type in ["e5m2", "e4m3"]  # TODO: support e3m2, e2m3, and e2m1
                assert b_type in ["e5m2", "e4m3"]
                assert d_type == "f32"
                assert s_type == "ue8m0"
                assert block_size == "block32"
            elif kind == "mxf4":
                assert k == 64
                assert a_type == b_type == "e2m1"
                assert d_type == "f32"
                assert s_type == "ue8m0"
                assert block_size == "block32"
            else:
                assert kind == "mxf4nvf4"
                assert k == 64
                assert a_type == b_type == "e2m1"
                assert d_type == "f32"
                assert s_type in ["ue8m0", "ue4m3"]
                assert (block_size == "block16") or (
                    s_type == "ue8m0" and block_size == "block32"
                )
            self.block_size = int(block_size[-2:])
            self.s_type = nv_torch_dtype[s_type]
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = m, n, k
        self.kind = kind
        self.packing = 2 if kind.startswith("mxf4") else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]

    def check_input(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        scale_A: torch.Tensor | None = None,
        scale_B: torch.Tensor | None = None,
    ):
        m, n, k, packing = self.m, self.n, self.k, self.packing
        assert A.shape == (m, k // packing)
        assert B.shape == (k // packing, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type
        if self.kind.startswith("mx"):
            assert scale_A is not None and scale_B is not None
            assert scale_A.shape == (m, k // self.block_size)
            assert scale_B.shape == (k // self.block_size, n)
            assert scale_A.dtype == scale_B.dtype == self.s_type
        else:
            assert scale_A is None and scale_B is None
