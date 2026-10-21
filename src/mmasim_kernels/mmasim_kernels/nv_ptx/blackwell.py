import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from mmasim.nv_ptx.isa_common import nv_shape_to_mnk
from mmasim.nv_ptx.isa_mma import blackwell_mma_shape_and_type
from mmasim.nv_ptx.isa_tcgen05mma import (
    blackwell_tcgen05mma_shape_and_type,
    blackwell_tcgen05mma_block_scale_shape_and_type,
)
from . import MMA, TCGen05MMA, TCGen05MMABlockScale

dir = Path(__file__).with_name("kernels")
files = [dir / f for f in ["tcgen05mma_sm100a.cu", "mma_sm80.cu", "mma_sm75.cu"]]
kernel_path = load(
    "mmasim_kernels_nv_blackwell",
    [str(f) for f in files],
    extra_cuda_cflags=["-arch=compute_100a", "-code=sm_100a"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)

mma_kernels = {}
for shape_and_type in blackwell_mma_shape_and_type:
    kernel = lib.__getattr__(
        f"mma_{shape_and_type.replace('.', '_').replace('::', '_')}"
    )
    kernel.argtypes = [ctypes.c_void_p] * 4
    mma_kernels[shape_and_type] = MMA("Blackwell", shape_and_type, kernel)


def _bind_shape(kernel, m, n, pointer_count):
    """Bind runtime descriptor dimensions while keeping the common kernel ABI."""
    kernel.argtypes = [ctypes.c_void_p] * pointer_count + [ctypes.c_uint32] * 2
    kernel.restype = None

    def bound(*pointers):
        return kernel(*pointers, m, n)

    bound.argtypes = kernel.argtypes[:pointer_count]
    return bound


tcgen05mma_kernels = {}
for shape_and_type in blackwell_tcgen05mma_shape_and_type:
    shape, type = shape_and_type.split(".", 1)
    m, n, _ = nv_shape_to_mnk(shape)
    kernel = lib.__getattr__(f"tcgen05mma_{type.replace('.', '_').replace('::', '_')}")
    kernel = _bind_shape(kernel, m, n, 3)
    tcgen05mma_kernels[shape_and_type] = TCGen05MMA("Blackwell", shape_and_type, kernel)

tcgen05mma_block_scale_kernels = {}
for shape_and_type in blackwell_tcgen05mma_block_scale_shape_and_type:
    shape, type = shape_and_type.split(".", 1)
    m, n, _ = nv_shape_to_mnk(shape)
    kernel = lib.__getattr__(f"tcgen05mma_{type.replace('.', '_').replace('::', '_')}")
    kernel = _bind_shape(kernel, m, n, 5)
    tcgen05mma_block_scale_kernels[shape_and_type] = TCGen05MMABlockScale(
        "Blackwell", shape_and_type, kernel
    )
