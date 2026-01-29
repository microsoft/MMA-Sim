from ..common import (
    MatrixMultiplyAdd,
    MatrixMultiplyAddWithBlockScale,
    nv_shape_to_mnk,
    nv_torch_dtype,
)

# TODO: support tcgen05mma with .ws and .cta_group::2


class tcgen05mma(MatrixMultiplyAdd):
    def __init__(self, arch: str, qualifier: str):
        assert arch == "Blackwell"
        qualifiers = qualifier.split(".")
        assert len(qualifiers) == 4
        shape, d_type, a_type, b_type = qualifiers
        m, n, k = nv_shape_to_mnk(shape)
        assert m in [64, 128]
        assert n in list(range(8, 256 + 1, 8))
        assert k in [8, 16, 32]
        if k == 8:  # kind::tf32
            assert a_type == b_type == "tf32"
            assert d_type == "f32"
        elif k == 16:  # kind::f16
            assert d_type in ["f32", "f16"]
            if d_type == "f32":
                assert a_type in ["f16", "bf16"]
                assert b_type in ["f16", "bf16"]
            else:  # d_type == "f16"
                assert a_type == b_type == "f16"
        else:  # k == 32, kind::f8f6f4
            assert a_type in ["e5m2", "e4m3", "e2m1"]  # TODO: support e3m2 and e2m3
            assert b_type in ["e5m2", "e4m3", "e2m1"]
            assert d_type in ["f32", "f16"]
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = m, n, k
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]


class tcgen05mma_block_scale(MatrixMultiplyAddWithBlockScale):
    def __init__(self, arch: str, qualifier: str):
        assert arch == "Blackwell"
        qualifiers = qualifier.split(".")
        assert len(qualifiers) == 6
        shape, block_size, d_type, a_type, b_type, s_type = qualifiers
        m, n, k = nv_shape_to_mnk(shape)
        assert m == 128
        assert n in list(range(8, 256 + 1, 8))
        assert k in [32, 64]
        assert d_type == "f32"
        if k == 32:  # kind::mxf8f6f4
            assert a_type in ["e5m2", "e4m3", "e2m1"]  # TODO: support e3m2 and e2m3
            assert b_type in ["e5m2", "e4m3", "e2m1"]
            assert s_type == "ue8m0"
            assert block_size == "block32"
        else:  # k == 64, kind::mxf4nvf4
            assert a_type == b_type == "e2m1"
            assert d_type == "f32"
            assert s_type in ["ue8m0", "ue4m3"]
            if s_type == "ue8m0":
                assert block_size in ["block32", "block16"]
            else:  # s_type == "ue4m3"
                assert block_size == "block16"
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = m, n, k
        self.block_size = int(block_size[-2:])
        self.packing = 2 if k == 64 else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]
        self.s_type = nv_torch_dtype[s_type]
