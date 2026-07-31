import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from . import MMA

dir = Path(__file__).with_name("kernels")
files = [dir / f for f in ["mma_sm75.cu"]]
kernel_path = load(
    "mmasim_kernels_nv_turing",
    [str(f) for f in files],
    extra_cuda_cflags=["-arch=compute_75", "-code=sm_75"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)

# f16
lib.mma_m16n8k8_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4

mma_cuda_kernels = {
    # f16
    "m16n8k8.f32.f16.f16.f32": lib.mma_m16n8k8_f32_f16_f16_f32,
    "m16n8k8.f16.f16.f16.f16": lib.mma_m16n8k8_f16_f16_f16_f16,
}
mma_kernels = {
    shape_and_type: MMA("Turing", shape_and_type, mma_cuda_kernels[shape_and_type])
    for shape_and_type in mma_cuda_kernels
}
