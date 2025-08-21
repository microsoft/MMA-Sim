import torch

from .common import (
    MatrixMultiplyAdd,
    MatrixMultiplyAddWithBlockScale,
    nv_shape_to_mnk,
    nv_torch_dtype,
)

volta_mma_qualifiers = [
    # sm_70 f16
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f32.f16.f16.f32",
    "m8n8k4.f16.f16.f16.f16",
]
turing_mma_qualifiers = [
    # sm_75 f16
    "m16n8k8.f32.f16.f16.f32",
    "m16n8k8.f16.f16.f16.f16",
]
ampere_mma_qualifiers = [
    # sm_80 tf32
    "m16n8k4.f32.tf32.tf32.f32",
    "m16n8k8.f32.tf32.tf32.f32",
    # sm_80 f16
    "m16n8k16.f16.f16.f16.f16",
    "m16n8k16.f32.f16.f16.f32",
    # sm_80 bf16
    "m16n8k8.f32.bf16.bf16.f32",
    "m16n8k16.f32.bf16.bf16.f32",
]
adalovelace_mma_qualifiers = [
    # sm_89 fp8 m16n8k32 f32_output
    "m16n8k32.f32.e5m2.e5m2.f32",
    "m16n8k32.f32.e5m2.e4m3.f32",
    "m16n8k32.f32.e4m3.e5m2.f32",
    "m16n8k32.f32.e4m3.e4m3.f32",
    # sm_89 fp8 m16n8k16 f32_output
    "m16n8k16.f32.e5m2.e5m2.f32",
    "m16n8k16.f32.e5m2.e4m3.f32",
    "m16n8k16.f32.e4m3.e5m2.f32",
    "m16n8k16.f32.e4m3.e4m3.f32",
    # sm_89 fp8 m16n8k32 f16_output
    "m16n8k32.f16.e5m2.e5m2.f16",
    "m16n8k32.f16.e5m2.e4m3.f16",
    "m16n8k32.f16.e4m3.e5m2.f16",
    "m16n8k32.f16.e4m3.e4m3.f16",
    # sm_89 fp8 m16n8k16 f16_output
    "m16n8k16.f16.e5m2.e5m2.f16",
    "m16n8k16.f16.e5m2.e4m3.f16",
    "m16n8k16.f16.e4m3.e5m2.f16",
    "m16n8k16.f16.e4m3.e4m3.f16",
]

arch_mma_qualifiers = {
    "Volta": volta_mma_qualifiers,
    "Turing": turing_mma_qualifiers,
    "Ampere": ampere_mma_qualifiers + turing_mma_qualifiers,
    "Ada Lovelace": adalovelace_mma_qualifiers
    + ampere_mma_qualifiers
    + turing_mma_qualifiers,
    "Hopper": ampere_mma_qualifiers + turing_mma_qualifiers,
    "Blackwell": ampere_mma_qualifiers + turing_mma_qualifiers,
    "RTX Blackwell": adalovelace_mma_qualifiers
    + ampere_mma_qualifiers
    + turing_mma_qualifiers,
}


class MMA(MatrixMultiplyAdd, MatrixMultiplyAddWithBlockScale):
    def __init__(self, arch: str, qualifier: str):
        qualifiers = qualifier.split(".")
        if len(qualifiers) == 5:  # pre-sm120a mma
            assert arch in arch_mma_qualifiers.keys(), (
                f"Unsupported architecture {arch} for MMA.\n"
                f"Supported architectures: {list(arch_mma_qualifiers.keys())}"
            )
            supported_qualifiers = arch_mma_qualifiers[arch]
            assert qualifier in supported_qualifiers, (
                f"Unsupported MMA qualifier {qualifier} for {arch} architecture.\n"
                f"Supported qualifiers: {supported_qualifiers}"
            )
            shape, d_type, a_type, b_type, c_type = qualifiers
            self.block_size = 0
            self.packing = 1
        else:  # sm120a mma
            assert arch == "RTX Blackwell"
            if len(qualifiers) == 6:  # f8f6f4
                shape, kind, d_type, a_type, b_type, c_type = qualifiers
                assert shape == "m16n8k32"
                assert kind == "f8f6f4"
                assert d_type == c_type == "f32" or d_type == c_type == "f16"
                assert a_type in ["e5m2", "e4m3"]  # TODO: support e3m2, e2m3, and e2m1
                assert b_type in ["e5m2", "e4m3"]
                self.block_size = 0
                self.packing = 1
            else:  # mxfp
                assert len(qualifiers) == 8
                shape, kind, block_size, d_type, a_type, b_type, c_type, s_type = (
                    qualifiers
                )
                assert d_type == c_type == "f32"
                if kind == "mxf8f6f4":
                    assert shape == "m16n8k32"
                    assert block_size == "block32"
                    assert a_type in [
                        "e5m2",
                        "e4m3",
                    ]  # TODO: support e3m2, e2m3, and e2m1
                    assert b_type in ["e5m2", "e4m3"]
                    assert s_type == "ue8m0"
                else:  # mxfp4
                    assert shape == "m16n8k64"
                    assert a_type == b_type == "e2m1"
                    if kind == "mxf4":
                        assert block_size == "block32"
                        assert s_type == "ue8m0"
                    else:
                        assert kind == "mxf4nvf4"
                        if s_type == "ue8m0":
                            assert block_size == "block32"
                        else:
                            assert s_type == "ue4m3"
                            assert block_size == "block16"
                self.kind = kind
                self.block_size = int(block_size[-2:])
                self.packing = 2 if kind.startswith("mxf4") else 1
                self.s_type = nv_torch_dtype[s_type]
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
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
        if self.block_size > 0:
            assert scale_A is not None and scale_B is not None
            assert scale_A.shape == (m, k // self.block_size)
            assert scale_B.shape == (k // self.block_size, n)
            assert scale_A.dtype == scale_B.dtype == self.s_type
        else:
            assert scale_A is None and scale_B is None
