import ctypes
import pathlib

from . import MMA


path = pathlib.Path(__file__).parent / "impl/nv_volta.so"
lib = ctypes.CDLL(str(path))

# f16
lib.mma_m8n8k4_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m8n8k4_f32_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m8n8k4_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4

mma_kernel_impls = {
    # f16
    "m8n8k4.f32.f16.f16.f32": lib.mma_m8n8k4_f32_f16_f16_f32,
    "m8n8k4.f32.f16.f16.f16": lib.mma_m8n8k4_f32_f16_f16_f16,
    "m8n8k4.f16.f16.f16.f16": lib.mma_m8n8k4_f16_f16_f16_f16,
}
mma_kernels = {
    shape_and_type: MMA("Volta", shape_and_type, mma_kernel_impls[shape_and_type])
    for shape_and_type in mma_kernel_impls
}
