import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from . import MFMA

dir = Path(__file__).with_name("kernels")
files = [dir / f for f in ["mfma_cdna3.hip"]]
kernel_path = load(
    "mmasim_kernels_amd_cdna3",
    [str(f) for f in files],
    extra_cflags=["--offload-arch=gfx942"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)

# f64
lib.mfma_f64_16x16x4_f64.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f64_4x4x4_4b_f64.argtypes = [ctypes.c_void_p] * 4
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

mfma_hip_kernels = {
    # f64
    "f64_16x16x4_f64": lib.mfma_f64_16x16x4_f64,
    "f64_4x4x4_4b_f64": lib.mfma_f64_4x4x4_4b_f64,
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
mfma_kernels = {
    suffix: MFMA("CDNA3", suffix, mfma_hip_kernels[suffix])
    for suffix in mfma_hip_kernels
}
