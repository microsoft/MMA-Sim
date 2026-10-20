from .. import (
    MMAOperation,
    MMABlockScaleOperation,
)
from .isa_common import (
    nv_torch_dtype,
    f8f6f4_types,
    nv_shape_to_mnk,
)

# TODO: support tcgen05mma with .ws and .cta_group::2
# TODO: support sm_103/sm_107

_tcgen05_mn = [(m, n) for m in (64, 128) for n in range(8, 257, 8)]
blackwell_tcgen05mma_shape_and_type = (
    [f"m{m}n{n}k8.kind::tf32.f32.tf32.tf32" for m, n in _tcgen05_mn]
    + [
        f"m{m}n{n}k16.kind::f16.{d_type}.{a_type}.{b_type}"
        for m, n in _tcgen05_mn
        for d_type in ("f32", "f16")
        for a_type in ("f16", "bf16")
        for b_type in ("f16", "bf16")
        if d_type == "f32" or a_type == b_type == "f16"
    ]
    + [
        f"m{m}n{n}k32.kind::f8f6f4.{d_type}.{a_type}.{b_type}"
        for m, n in _tcgen05_mn
        for d_type in ("f32", "f16")
        for a_type in f8f6f4_types
        for b_type in f8f6f4_types
    ]
)
tcgen05mma_shape_and_type = {
    "Blackwell": blackwell_tcgen05mma_shape_and_type,
}

_tcgen05_block_scale_n = list(range(8, 257, 8))
blackwell_tcgen05mma_block_scale_shape_and_type = (
    [
        f"m128n{n}k32.kind::mxf8f6f4.block32.f32.{a_type}.{b_type}.ue8m0"
        for n in _tcgen05_block_scale_n
        for a_type in f8f6f4_types
        for b_type in f8f6f4_types
    ]
    + [
        f"m128n{n}k64.kind::mxf4.block32.f32.e2m1.e2m1.ue8m0"
        for n in _tcgen05_block_scale_n
    ]
    + [
        f"m128n{n}k64.kind::mxf4nvf4.block32.f32.e2m1.e2m1.f32.ue8m0"
        for n in _tcgen05_block_scale_n
    ]
    + [
        f"m128n{n}k64.kind::mxf4nvf4.block16.f32.e2m1.e2m1.f32.{s_type}"
        for n in _tcgen05_block_scale_n
        for s_type in ["ue8m0", "ue4m3"]
    ]
)
tcgen05mma_block_scale_shape_and_type = {
    "Blackwell": blackwell_tcgen05mma_block_scale_shape_and_type,
}


class TCGen05MMA(MMAOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in tcgen05mma_shape_and_type.keys(), (
            f"Unsupported architecture {arch} for tcgen05.mma.\n"
            f"Supported architectures: {list(tcgen05mma_shape_and_type.keys())}"
        )
        supported_shape_and_type = tcgen05mma_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for tcgen05.mma on {arch} architecture.\n"
            "Supported shape_and_type format: m{m}n{n}k{k}.kind::{kind}.{d_type}.{a_type}.{b_type}\n"
            "Refer to the PTX documentation for valid combinations."
        )
        qualifiers = shape_and_type.split(".")
        shape, kind, d_type, a_type, b_type = qualifiers
        m, n, k = nv_shape_to_mnk(shape)
        kind = kind.removeprefix("kind::")
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = m, n, k
        self.kind = kind
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.a_type_ptx = a_type
        self.b_type_ptx = b_type
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]


class TCGen05MMABlockScale(MMABlockScaleOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in tcgen05mma_block_scale_shape_and_type.keys(), (
            f"Unsupported architecture {arch} for block scaled tcgen05.mma.\n"
            f"Supported architectures: {list(tcgen05mma_block_scale_shape_and_type.keys())}"
        )
        supported_shape_and_type = tcgen05mma_block_scale_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for block scaled tcgen05.mma on {arch} architecture.\n"
            "Supported shape_and_type format: m{m}n{n}k{k}.kind::{kind}.block{block_size}.{d_type}.{a_type}.{b_type}.{s_type}\n"
            "Refer to the PTX documentation for valid combinations."
        )
        qualifiers = shape_and_type.split(".")
        shape, kind, block_size, d_type, a_type, b_type, s_type = qualifiers
        m, n, k = nv_shape_to_mnk(shape)
        kind = kind.removeprefix("kind::")
        block = int(block_size.removeprefix("block"))
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = m, n, k
        self.kind = kind
        self.block_size = block
        self.packing = 2 if kind in ("mxf4", "mxf4nvf4") else 1
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.a_type_ptx = a_type
        self.b_type_ptx = b_type
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]
        self.s_type = nv_torch_dtype[s_type]
