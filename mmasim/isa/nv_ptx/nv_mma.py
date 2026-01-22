from ..common import (
    MatrixMultiplyAdd,
    MatrixMultiplyAddWithBlockScale,
    nv_shape_to_mnk,
    nv_torch_dtype,
)

volta_mma_qualifiers = [
    # sm_70 f16
    "m8n8k4.f32.f16.f16.f32",
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f16.f16.f16.f16",
]
turing_mma_qualifiers = [
    # sm_75 f16
    "m16n8k8.f32.f16.f16.f32",
    "m16n8k8.f16.f16.f16.f16",
]
ampere_mma_qualifiers = [
    # sm_80 f64
    "m8n8k4.f64.f64.f64.f64",
    # sm_80 tf32
    "m16n8k8.f32.tf32.tf32.f32",
    "m16n8k4.f32.tf32.tf32.f32",
    # sm_80 f16
    "m16n8k16.f32.f16.f16.f32",
    "m16n8k16.f16.f16.f16.f16",
    # sm_80 bf16
    "m16n8k16.f32.bf16.bf16.f32",
    "m16n8k8.f32.bf16.bf16.f32",
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
hopper_mma_qualifiers = [
    # sm_90a f64
    "m16n8k16.f64.f64.f64.f64",
    "m16n8k8.f64.f64.f64.f64",
    "m16n8k4.f64.f64.f64.f64",
]
rtx_blackwell_mma_qualifiers = [
    # sm_120a f8f6f4 f32_output
    # TODO: support e3m2 and e2m3
    "m16n8k32.f32.e5m2.e5m2.f32",
    "m16n8k32.f32.e5m2.e4m3.f32",
    "m16n8k32.f32.e5m2.e2m1.f32",
    "m16n8k32.f32.e4m3.e5m2.f32",
    "m16n8k32.f32.e4m3.e4m3.f32",
    "m16n8k32.f32.e4m3.e2m1.f32",
    "m16n8k32.f32.e2m1.e5m2.f32",
    "m16n8k32.f32.e2m1.e4m3.f32",
    "m16n8k32.f32.e2m1.e2m1.f32",
    # sm_120a f8f6f4 f16_output
    "m16n8k32.f16.e5m2.e5m2.f16",
    "m16n8k32.f16.e5m2.e4m3.f16",
    "m16n8k32.f16.e5m2.e2m1.f16",
    "m16n8k32.f16.e4m3.e5m2.f16",
    "m16n8k32.f16.e4m3.e4m3.f16",
    "m16n8k32.f16.e4m3.e2m1.f16",
    "m16n8k32.f16.e2m1.e5m2.f16",
    "m16n8k32.f16.e2m1.e4m3.f16",
    "m16n8k32.f16.e2m1.e2m1.f16",
]
rtx_blackwell_mma_block_scale_qualifiers = [
    # sm_120a mxf8f6f4
    # TODO: support e3m2 and e2m3
    "m16n8k32.block32.f32.e5m2.e5m2.f32.ue8m0",
    "m16n8k32.block32.f32.e5m2.e4m3.f32.ue8m0",
    "m16n8k32.block32.f32.e5m2.e2m1.f32.ue8m0",
    "m16n8k32.block32.f32.e4m3.e5m2.f32.ue8m0",
    "m16n8k32.block32.f32.e4m3.e4m3.f32.ue8m0",
    "m16n8k32.block32.f32.e4m3.e2m1.f32.ue8m0",
    "m16n8k32.block32.f32.e2m1.e5m2.f32.ue8m0",
    "m16n8k32.block32.f32.e2m1.e4m3.f32.ue8m0",
    "m16n8k32.block32.f32.e2m1.e2m1.f32.ue8m0",
    # sm_120a mxf4nvf4
    "m16n8k64.block32.f32.e2m1.e2m1.f32.ue8m0",
    "m16n8k64.block16.f32.e2m1.e2m1.f32.ue8m0",
    "m16n8k64.block32.f32.e2m1.e2m1.f32.ue4m3",
    "m16n8k64.block16.f32.e2m1.e2m1.f32.ue4m3",
]

arch_mma_qualifiers = {
    "Volta": volta_mma_qualifiers,
    "Turing": turing_mma_qualifiers + volta_mma_qualifiers,
    "Ampere": ampere_mma_qualifiers + turing_mma_qualifiers,
    "Ada Lovelace": adalovelace_mma_qualifiers
    + ampere_mma_qualifiers
    + turing_mma_qualifiers,
    "Hopper": hopper_mma_qualifiers + ampere_mma_qualifiers + turing_mma_qualifiers,
    "Blackwell": ampere_mma_qualifiers + turing_mma_qualifiers,
    "RTX Blackwell": rtx_blackwell_mma_qualifiers
    + adalovelace_mma_qualifiers
    + ampere_mma_qualifiers
    + turing_mma_qualifiers,
}
arch_mma_block_scale_qualifiers = {
    "RTX Blackwell": rtx_blackwell_mma_block_scale_qualifiers,
}


class mma(MatrixMultiplyAdd):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for mma.\n"
            f"Supported architectures: {list(arch_mma_qualifiers.keys())}"
        )
        supported_qualifiers = arch_mma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported qualifier {qualifier} for mma on {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        shape, d_type, a_type, b_type, c_type = qualifier.split(".")
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]


class mma_block_scale(MatrixMultiplyAddWithBlockScale):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mma_block_scale_qualifiers.keys(), (
            f"Unsupported architecture {arch} for mma.block_scale.\n"
            f"Supported architectures: {list(arch_mma_block_scale_qualifiers.keys())}"
        )
        supported_qualifiers = arch_mma_block_scale_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported qualifier {qualifier} for mma.block_scale on {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        shape, block_size, d_type, a_type, b_type, c_type, s_type = qualifier.split(".")
        self.arch = arch
        self.qualifier = qualifier
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.block_size = int(block_size[-2:])
        self.packing = 2 if self.k == 64 else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]
        self.s_type = nv_torch_dtype[s_type]
