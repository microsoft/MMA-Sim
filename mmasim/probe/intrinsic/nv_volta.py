import ctypes
import pathlib

from . import NV_MMA


path = pathlib.Path(__file__).parent / "impl/nv_volta.so"
lib = ctypes.CDLL(str(path))

# f16
lib.mma_m8n8k4_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m8n8k4_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m8n8k4_f32_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4

mma_intrinsic_impls = {
    # f16
    "m8n8k4.f16.f16.f16.f16": lib.mma_m8n8k4_f16_f16_f16_f16,
    "m8n8k4.f32.f16.f16.f32": lib.mma_m8n8k4_f32_f16_f16_f32,
    "m8n8k4.f32.f16.f16.f16": lib.mma_m8n8k4_f32_f16_f16_f16,
}
mma_intrinsics = {
    qualifier: NV_MMA(qualifier, mma_intrinsic_impls[qualifier])
    for qualifier in mma_intrinsic_impls
}
