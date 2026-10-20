import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from mmasim.nv_ptx.isa_mma import (
    rtx_blackwell_mma_shape_and_type,
    rtx_blackwell_mma_block_scale_shape_and_type,
)
from . import MMA, MMABlockScale

dir = Path(__file__).with_name("kernels")
files = [
    dir / f for f in ["mma_sm120a.cu", "mma_sm89.cu", "mma_sm80.cu", "mma_sm75.cu"]
]
kernel_path = load(
    "mmasim_kernels_nv_rtx_blackwell",
    [str(f) for f in files],
    extra_cuda_cflags=["-arch=compute_120a", "-code=sm_120a"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)
lib.mmasim_cuda_version.argtypes = []
lib.mmasim_cuda_version.restype = ctypes.c_int

mma_kernels = {}
for shape_and_type in rtx_blackwell_mma_shape_and_type:
    kernel = lib.__getattr__(
        f"mma_{shape_and_type.replace('.', '_').replace('::', '_')}"
    )
    kernel.argtypes = [ctypes.c_void_p] * 4
    mma_kernels[shape_and_type] = MMA("RTX Blackwell", shape_and_type, kernel)

mma_block_scale_kernels = {}
unavailable_mma_block_scale_kernels = {}
for shape_and_type in rtx_blackwell_mma_block_scale_shape_and_type:
    # PTX ISA 9.1 introduced UE8M0 with scale_vec::4X for this instruction.
    if (
        shape_and_type == "m16n8k64.kind::mxf4nvf4.block16.f32.e2m1.e2m1.f32.ue8m0"
        and lib.mmasim_cuda_version() < 13010
    ):
        unavailable_mma_block_scale_kernels[shape_and_type] = (
            "requires PTX ISA 9.1 / CUDA 13.1+"
        )
        continue
    kernel = lib.__getattr__(
        f"mma_{shape_and_type.replace('.', '_').replace('::', '_')}"
    )
    kernel.argtypes = [ctypes.c_void_p] * 6
    mma_block_scale_kernels[shape_and_type] = MMABlockScale(
        "RTX Blackwell", shape_and_type, kernel
    )
