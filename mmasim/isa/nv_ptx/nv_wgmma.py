from ..common import MatrixMultiplyAdd, nv_shape_to_mnk, nv_torch_dtype


class wgmma(MatrixMultiplyAdd):
    def __init__(self, arch: str, qualifier: str):
        assert arch == "Hopper"
        qualifiers = qualifier.split(".")
        assert len(qualifiers) == 4
        shape, d_type, a_type, b_type = qualifiers
        m, n, k = nv_shape_to_mnk(shape)
        assert m == 64
        assert n in list(range(8, 256 + 1, 8))
        assert k in [8, 16, 32]
        if k == 8:
            assert a_type == b_type == "tf32"
            assert d_type == "f32"
        elif k == 16:
            assert a_type in ["f16", "bf16"]
            if a_type == "f16":
                assert b_type == "f16"
                assert d_type in ["f32", "f16"]
            else:  # a_type == "bf16"
                assert b_type == "bf16"
                assert d_type == "f32"
        else:  # k == 32
            assert a_type in ["e5m2", "e4m3"]
            assert b_type in ["e5m2", "e4m3"]
            assert d_type in ["f32", "f16"]
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = m, n, k
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]
