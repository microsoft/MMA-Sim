import ctypes
import pathlib

from . import MFMA

path = pathlib.Path(__file__).parent / "kernels/cdna1.so"
lib = ctypes.CDLL(str(path))

# f32
lib.mfma_f32_32x32x2f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x1f32.argtypes = [ctypes.c_void_p] * 4
# f16
lib.mfma_f32_32x32x8f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4f16.argtypes = [ctypes.c_void_p] * 4
# bf16
lib.mfma_f32_32x32x4bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x2bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x8bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x2bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x2bf16.argtypes = [ctypes.c_void_p] * 4

mfma_hip_kernels = {
    # f32
    "f32_32x32x2f32": lib.mfma_f32_32x32x2f32,
    "f32_32x32x1f32": lib.mfma_f32_32x32x1f32,
    "f32_16x16x4f32": lib.mfma_f32_16x16x4f32,
    "f32_16x16x1f32": lib.mfma_f32_16x16x1f32,
    "f32_4x4x1f32": lib.mfma_f32_4x4x1f32,
    # f16
    "f32_32x32x8f16": lib.mfma_f32_32x32x8f16,
    "f32_32x32x4f16": lib.mfma_f32_32x32x4f16,
    "f32_16x16x16f16": lib.mfma_f32_16x16x16f16,
    "f32_16x16x4f16": lib.mfma_f32_16x16x4f16,
    "f32_4x4x4f16": lib.mfma_f32_4x4x4f16,
    # bf16
    "f32_32x32x4bf16": lib.mfma_f32_32x32x4bf16,
    "f32_32x32x2bf16": lib.mfma_f32_32x32x2bf16,
    "f32_16x16x8bf16": lib.mfma_f32_16x16x8bf16,
    "f32_16x16x2bf16": lib.mfma_f32_16x16x2bf16,
    "f32_4x4x2bf16": lib.mfma_f32_4x4x2bf16,
}
mfma_kernels = {
    suffix: MFMA("CDNA1", suffix, mfma_hip_kernels[suffix])
    for suffix in mfma_hip_kernels
}
