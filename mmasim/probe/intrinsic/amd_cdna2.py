import ctypes
import pathlib

from . import AMD_MFMA


path = pathlib.Path(__file__).parent / "impl/amd_cdna2.so"
lib = ctypes.CDLL(str(path))

# cdna1 f32
lib.mfma_f32_32x32x2f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x1f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x1f32.argtypes = [ctypes.c_void_p] * 4
# cdna1 f16
lib.mfma_f32_32x32x8f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4f16.argtypes = [ctypes.c_void_p] * 4
# cdna1 bf16
lib.mfma_f32_32x32x4bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x2bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x8bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x2bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x2bf16.argtypes = [ctypes.c_void_p] * 4
# cdna2 bf16
lib.mfma_f32_32x32x8bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4bf16_1k.argtypes = [ctypes.c_void_p] * 4

mfma_intrinsic_impls = {
    # cdna1 f32
    "f32_32x32x2f32": lib.mfma_f32_32x32x2f32,
    "f32_32x32x1f32": lib.mfma_f32_32x32x1f32,
    "f32_16x16x4f32": lib.mfma_f32_16x16x4f32,
    "f32_16x16x1f32": lib.mfma_f32_16x16x1f32,
    "f32_4x4x1f32": lib.mfma_f32_4x4x1f32,
    # cdna1 f16
    "f32_32x32x8f16": lib.mfma_f32_32x32x8f16,
    "f32_32x32x4f16": lib.mfma_f32_32x32x4f16,
    "f32_16x16x16f16": lib.mfma_f32_16x16x16f16,
    "f32_16x16x4f16": lib.mfma_f32_16x16x4f16,
    "f32_4x4x4f16": lib.mfma_f32_4x4x4f16,
    # cdna1 bf16
    "f32_32x32x4bf16": lib.mfma_f32_32x32x4bf16,
    "f32_32x32x2bf16": lib.mfma_f32_32x32x2bf16,
    "f32_16x16x8bf16": lib.mfma_f32_16x16x8bf16,
    "f32_16x16x2bf16": lib.mfma_f32_16x16x2bf16,
    "f32_4x4x2bf16": lib.mfma_f32_4x4x2bf16,
    # cdna2 bf16
    "f32_32x32x8bf16_1k": lib.mfma_f32_32x32x8bf16_1k,
    "f32_32x32x4bf16_1k": lib.mfma_f32_32x32x4bf16_1k,
    "f32_16x16x16bf16_1k": lib.mfma_f32_16x16x16bf16_1k,
    "f32_16x16x4bf16_1k": lib.mfma_f32_16x16x4bf16_1k,
    "f32_4x4x4bf16_1k": lib.mfma_f32_4x4x4bf16_1k,
}
mfma_intrinsics = {
    qualifier: AMD_MFMA(qualifier, mfma_intrinsic_impls[qualifier])
    for qualifier in mfma_intrinsic_impls
}
