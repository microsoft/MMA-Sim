from .. import MMAOperation
from .isa_common import nv_torch_dtype, nv_shape_to_mnk

_wgmma_n = list(range(8, 257, 8))
hopper_wgmma_shape_and_type = (
    [f"m64n{n}k8.f32.tf32.tf32" for n in _wgmma_n]
    + [f"m64n{n}k16.f32.f16.f16" for n in _wgmma_n]
    + [f"m64n{n}k16.f16.f16.f16" for n in _wgmma_n]
    + [f"m64n{n}k16.f32.bf16.bf16" for n in _wgmma_n]
    + [
        f"m64n{n}k32.{d_type}.{a_type}.{b_type}"
        for n in _wgmma_n
        for d_type in ("f32", "f16")
        for a_type in ("e5m2", "e4m3")
        for b_type in ("e5m2", "e4m3")
    ]
)
wgmma_shape_and_type = {
    "Hopper": hopper_wgmma_shape_and_type,
}


class WGMMA(MMAOperation):
    def __init__(self, arch: str, shape_and_type: str):
        assert arch in wgmma_shape_and_type, (
            f"Unsupported architecture {arch} for wgmma.\n"
            f"Supported architectures: {list(wgmma_shape_and_type.keys())}"
        )
        supported_shape_and_type = wgmma_shape_and_type[arch]
        assert shape_and_type in supported_shape_and_type, (
            f"Unsupported shape_and_type {shape_and_type} for wgmma on {arch} architecture.\n"
            "Supported shape_and_type format: m{m}n{n}k{k}.{d_type}.{a_type}.{b_type}\n"
            "Refer to the PTX documentation for valid combinations."
        )
        shape, d_type, a_type, b_type = shape_and_type.split(".")
        m, n, k = nv_shape_to_mnk(shape)
        self.arch = arch
        self.shape_and_type = shape_and_type
        self.m, self.n, self.k = m, n, k
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.a_type_ptx = a_type
        self.b_type_ptx = b_type
        self.c_type = nv_torch_dtype[d_type]
        self.d_type = nv_torch_dtype[d_type]
