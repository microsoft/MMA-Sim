import ctypes
import pathlib

from . import MMA

path = pathlib.Path(__file__).parent / "kernels/turing.so"
lib = ctypes.CDLL(str(path))

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
