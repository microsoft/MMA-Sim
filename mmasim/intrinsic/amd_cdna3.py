import ctypes
import pathlib

from . import MFMAIntrinsic


path = pathlib.Path(__file__).parent / "impl/amd_cdna3.so"
lib = ctypes.CDLL(str(path))

# f32
lib.mfma_f32_32x32x2_f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x1_2b_f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4_f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x1_4b_f32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x1_16b_f32.argtypes = [ctypes.c_void_p] * 4
# xf32
lib.mfma_f32_16x16x8_xf32.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4_xf32.argtypes = [ctypes.c_void_p] * 4
# f16
lib.mfma_f32_32x32x8_f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4_2b_f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16_f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4_4b_f16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4_16b_f16.argtypes = [ctypes.c_void_p] * 4
# bf16
lib.mfma_f32_32x32x8_bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4_2b_bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16_bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4_4b_bf16.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4_16b_bf16.argtypes = [ctypes.c_void_p] * 4
# fp8
lib.mfma_f32_16x16x32_fp8_fp8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x32_fp8_bf8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x32_bf8_fp8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x32_bf8_bf8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x16_fp8_fp8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x16_fp8_bf8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x16_bf8_fp8.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x16_bf8_bf8.argtypes = [ctypes.c_void_p] * 4

mfma_intrinsic_impls = {
    # f32
    "f32_32x32x2_f32": lib.mfma_f32_32x32x2_f32,
    "f32_32x32x1_2b_f32": lib.mfma_f32_32x32x1_2b_f32,
    "f32_16x16x4_f32": lib.mfma_f32_16x16x4_f32,
    "f32_16x16x1_4b_f32": lib.mfma_f32_16x16x1_4b_f32,
    "f32_4x4x1_16b_f32": lib.mfma_f32_4x4x1_16b_f32,
    # xf32
    "f32_16x16x8_xf32": lib.mfma_f32_16x16x8_xf32,
    "f32_32x32x4_xf32": lib.mfma_f32_32x32x4_xf32,
    # f16
    "f32_32x32x8_f16": lib.mfma_f32_32x32x8_f16,
    "f32_32x32x4_2b_f16": lib.mfma_f32_32x32x4_2b_f16,
    "f32_16x16x16_f16": lib.mfma_f32_16x16x16_f16,
    "f32_16x16x4_4b_f16": lib.mfma_f32_16x16x4_4b_f16,
    "f32_4x4x4_16b_f16": lib.mfma_f32_4x4x4_16b_f16,
    # bf16
    "f32_32x32x8_bf16": lib.mfma_f32_32x32x8_bf16,
    "f32_32x32x4_2b_bf16": lib.mfma_f32_32x32x4_2b_bf16,
    "f32_16x16x16_bf16": lib.mfma_f32_16x16x16_bf16,
    "f32_16x16x4_4b_bf16": lib.mfma_f32_16x16x4_4b_bf16,
    "f32_4x4x4_16b_bf16": lib.mfma_f32_4x4x4_16b_bf16,
    # fp8
    "f32_16x16x32_fp8_fp8": lib.mfma_f32_16x16x32_fp8_fp8,
    "f32_16x16x32_fp8_bf8": lib.mfma_f32_16x16x32_fp8_bf8,
    "f32_16x16x32_bf8_fp8": lib.mfma_f32_16x16x32_bf8_fp8,
    "f32_16x16x32_bf8_bf8": lib.mfma_f32_16x16x32_bf8_bf8,
    "f32_32x32x16_fp8_fp8": lib.mfma_f32_32x32x16_fp8_fp8,
    "f32_32x32x16_fp8_bf8": lib.mfma_f32_32x32x16_fp8_bf8,
    "f32_32x32x16_bf8_fp8": lib.mfma_f32_32x32x16_bf8_fp8,
    "f32_32x32x16_bf8_bf8": lib.mfma_f32_32x32x16_bf8_bf8,
}
mfma_intrinsics = {
    qualifier: MFMAIntrinsic("CDNA3", qualifier, mfma_intrinsic_impls[qualifier])
    for qualifier in mfma_intrinsic_impls
}
