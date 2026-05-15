from .. import (
    MMAOperation,
    MMABlockScaleOperation,
)
from .isa_common import nv_torch_dtype, nv_shape_to_mnk

volta_mma_shape_and_type = [
    # sm_70 f16
    "m8n8k4.f32.f16.f16.f32",
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f16.f16.f16.f16",
]
turing_mma_shape_and_type = [
    # sm_75 f16
    "m16n8k8.f32.f16.f16.f32",
    "m16n8k8.f16.f16.f16.f16",
]
ampere_mma_shape_and_type = [
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
adalovelace_mma_shape_and_type = [
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
hopper_mma_shape_and_type = [
    # sm_90a f64
    "m16n8k16.f64.f64.f64.f64",
    "m16n8k8.f64.f64.f64.f64",
    "m16n8k4.f64.f64.f64.f64",
]
rtx_blackwell_mma_shape_and_type = [
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
rtx_blackwell_mma_block_scale_shape_and_type = [
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

arch_mma_shape_and_type = {
    "Volta": volta_mma_shape_and_type,
    "Turing": turing_mma_shape_and_type + volta_mma_shape_and_type,
    "Ampere": ampere_mma_shape_and_type + turing_mma_shape_and_type,
    "Ada Lovelace": adalovelace_mma_shape_and_type
    + ampere_mma_shape_and_type
    + turing_mma_shape_and_type,
    "Hopper": hopper_mma_shape_and_type
    + ampere_mma_shape_and_type
    + turing_mma_shape_and_type,
    "Blackwell": ampere_mma_shape_and_type + turing_mma_shape_and_type,
    "RTX Blackwell": rtx_blackwell_mma_shape_and_type
    + adalovelace_mma_shape_and_type
    + ampere_mma_shape_and_type
    + turing_mma_shape_and_type,
}
arch_mma_block_scale_shape_and_type = {
    "RTX Blackwell": rtx_blackwell_mma_block_scale_shape_and_type,
}


class MMA(MMAOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in arch_mma_shape_and_type.keys(), (
            f"Unsupported architecture {arch} for mma.\n"
            f"Supported architectures: {list(arch_mma_shape_and_type.keys())}"
        )
        supported_shape_and_type = arch_mma_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for mma on {arch} architecture.\n"
            f"Supported shape_and_type: {supported_shape_and_type}"
        )
        shape, d_type, a_type, b_type, c_type = shape_and_type.split(".")
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]


class MMABlockScale(MMABlockScaleOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in arch_mma_block_scale_shape_and_type.keys(), (
            f"Unsupported architecture {arch} for mma.block_scale.\n"
            f"Supported architectures: {list(arch_mma_block_scale_shape_and_type.keys())}"
        )
        supported_shape_and_type = arch_mma_block_scale_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for mma.block_scale on {arch} architecture.\n"
            f"Supported shape_and_type: {supported_shape_and_type}"
        )
        shape, block_size, d_type, a_type, b_type, c_type, s_type = (
            shape_and_type.split(".")
        )
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.block_size = int(block_size[-2:])
        self.packing = 2 if self.k == 64 else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]
        self.s_type = nv_torch_dtype[s_type]
