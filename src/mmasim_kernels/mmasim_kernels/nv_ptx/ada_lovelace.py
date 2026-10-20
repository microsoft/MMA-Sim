import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from mmasim.nv_ptx.isa_mma import ada_lovelace_mma_shape_and_type
from . import MMA

dir = Path(__file__).with_name("kernels")
files = [dir / f for f in ["mma_sm89.cu", "mma_sm80.cu", "mma_sm75.cu"]]
kernel_path = load(
    "mmasim_kernels_nv_ada_lovelace",
    [str(f) for f in files],
    extra_cuda_cflags=["-arch=compute_89", "-code=sm_89"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)

mma_kernels = {}
for shape_and_type in ada_lovelace_mma_shape_and_type:
    kernel = lib.__getattr__(f"mma_{shape_and_type.replace('.', '_')}")
    kernel.argtypes = [ctypes.c_void_p] * 4
    mma_kernels[shape_and_type] = MMA("Ada Lovelace", shape_and_type, kernel)
