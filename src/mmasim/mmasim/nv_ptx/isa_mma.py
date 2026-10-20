from .. import (
    MMAOperation,
    MMABlockScaleOperation,
)
from .isa_common import (
    nv_torch_dtype,
    f8f6f4_types,
    nv_shape_to_mnk,
)

mma_shape_and_type_from_sm_70 = [
    # fp16
    "m8n8k4.f32.f16.f16.f32",
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f16.f16.f16.f16",
]
mma_shape_and_type_from_sm_75 = [
    # fp16
    "m16n8k8.f32.f16.f16.f32",
    "m16n8k8.f16.f16.f16.f16",
]
mma_shape_and_type_from_sm_80 = [
    # fp64
    "m8n8k4.f64.f64.f64.f64",
    # tf32
    "m16n8k8.f32.tf32.tf32.f32",
    "m16n8k4.f32.tf32.tf32.f32",
    # fp16
    "m16n8k16.f32.f16.f16.f32",
    "m16n8k16.f16.f16.f16.f16",
    # bf16
    "m16n8k16.f32.bf16.bf16.f32",
    "m16n8k8.f32.bf16.bf16.f32",
]
mma_shape_and_type_from_sm_89 = [
    # fp8
    f"m16n8k{k}.{d_type}.{a_type}.{b_type}.{d_type}"
    for k in [32, 16]
    for d_type in ["f32", "f16"]
    for a_type in ["e5m2", "e4m3"]
    for b_type in ["e5m2", "e4m3"]
]
mma_shape_and_type_from_sm_90 = [
    # fp64
    "m16n8k16.f64.f64.f64.f64",
    "m16n8k8.f64.f64.f64.f64",
    "m16n8k4.f64.f64.f64.f64",
]
mma_shape_and_type_from_sm_120 = [
    f"m16n8k32.kind::f8f6f4.{d_type}.{a_type}.{b_type}.{d_type}"
    for d_type in ["f32", "f16"]
    for a_type in f8f6f4_types
    for b_type in f8f6f4_types
]
volta_mma_shape_and_type = mma_shape_and_type_from_sm_70
turing_mma_shape_and_type = (
    mma_shape_and_type_from_sm_75 + mma_shape_and_type_from_sm_70
)
ampere_mma_shape_and_type = (
    mma_shape_and_type_from_sm_80 + mma_shape_and_type_from_sm_75
)
ada_lovelace_mma_shape_and_type = (
    mma_shape_and_type_from_sm_89
    + mma_shape_and_type_from_sm_80
    + mma_shape_and_type_from_sm_75
)
hopper_mma_shape_and_type = (
    mma_shape_and_type_from_sm_90
    + mma_shape_and_type_from_sm_80
    + mma_shape_and_type_from_sm_75
)
blackwell_mma_shape_and_type = (
    mma_shape_and_type_from_sm_80 + mma_shape_and_type_from_sm_75
)
rtx_blackwell_mma_shape_and_type = (
    mma_shape_and_type_from_sm_120
    + mma_shape_and_type_from_sm_89
    + mma_shape_and_type_from_sm_80
    + mma_shape_and_type_from_sm_75
)
mma_shape_and_type = {
    "Volta": volta_mma_shape_and_type,
    "Turing": turing_mma_shape_and_type,
    "Ampere": ampere_mma_shape_and_type,
    "Ada Lovelace": ada_lovelace_mma_shape_and_type,
    "Hopper": hopper_mma_shape_and_type,
    "Blackwell": blackwell_mma_shape_and_type,
    "RTX Blackwell": rtx_blackwell_mma_shape_and_type,
}

mma_block_scale_shape_and_type_from_sm_120 = (
    [
        f"m16n8k32.kind::mxf8f6f4.block32.f32.{a_type}.{b_type}.f32.ue8m0"
        for a_type in f8f6f4_types
        for b_type in f8f6f4_types
    ]
    + [
        "m16n8k64.kind::mxf4.block32.f32.e2m1.e2m1.f32.ue8m0",
    ]
    + [
        "m16n8k64.kind::mxf4nvf4.block32.f32.e2m1.e2m1.f32.ue8m0",
        # "m16n8k64.kind::mxf4nvf4.block32.f32.e2m1.e2m1.f32.ue4m3",  # invalid
        "m16n8k64.kind::mxf4nvf4.block16.f32.e2m1.e2m1.f32.ue8m0",
        "m16n8k64.kind::mxf4nvf4.block16.f32.e2m1.e2m1.f32.ue4m3",
    ]
)
rtx_blackwell_mma_block_scale_shape_and_type = (
    mma_block_scale_shape_and_type_from_sm_120
)
mma_block_scale_shape_and_type = {
    "RTX Blackwell": rtx_blackwell_mma_block_scale_shape_and_type,
}


class MMA(MMAOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in mma_shape_and_type.keys(), (
            f"Unsupported architecture {arch} for mma.\n"
            f"Supported architectures: {list(mma_shape_and_type.keys())}"
        )
        supported_shape_and_type = mma_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for mma on {arch} architecture.\n"
            f"Supported shape_and_type: {supported_shape_and_type}"
        )
        qualifiers = shape_and_type.split(".")
        self.kind = None
        if len(qualifiers) == 6:
            shape, kind, d_type, a_type, b_type, c_type = qualifiers
            self.kind = kind.removeprefix("kind::")
        else:
            shape, d_type, a_type, b_type, c_type = qualifiers
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.a_type_ptx = a_type
        self.b_type_ptx = b_type
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]


class MMABlockScale(MMABlockScaleOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in mma_block_scale_shape_and_type.keys(), (
            f"Unsupported architecture {arch} for block scaled mma.\n"
            f"Supported architectures: {list(mma_block_scale_shape_and_type.keys())}"
        )
        supported_shape_and_type = mma_block_scale_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for block scaled mma on {arch} architecture.\n"
            f"Supported shape_and_type: {supported_shape_and_type}"
        )
        shape, kind, block_size, d_type, a_type, b_type, c_type, s_type = (
            shape_and_type.split(".")
        )
        kind = kind.removeprefix("kind::")
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.kind = kind
        self.block_size = int(block_size.removeprefix("block"))
        self.packing = 2 if kind in ("mxf4", "mxf4nvf4") else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.a_type_ptx = a_type
        self.b_type_ptx = b_type
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]
        self.s_type = nv_torch_dtype[s_type]
