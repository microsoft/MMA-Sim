import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from . import MFMA

dir = Path(__file__).with_name("kernels")
files = [dir / f for f in ["mfma_cdna2.hip", "mfma_cdna1.hip"]]
kernel_path = load(
    "mmasim_kernels_amd_cdna2",
    [str(f) for f in files],
    extra_cflags=["--offload-arch=gfx90a"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)

# cdna2 f64
lib.mfma_f64_16x16x4f64.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f64_4x4x4f64.argtypes = [ctypes.c_void_p] * 4
# cdna2 bf16
lib.mfma_f32_32x32x8bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_32x32x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x16bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_16x16x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
lib.mfma_f32_4x4x4bf16_1k.argtypes = [ctypes.c_void_p] * 4
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

mfma_hip_kernels = {
    # cdna2 f64
    "f64_16x16x4f64": lib.mfma_f64_16x16x4f64,
    "f64_4x4x4f64": lib.mfma_f64_4x4x4f64,
    # cdna2 bf16
    "f32_32x32x8bf16_1k": lib.mfma_f32_32x32x8bf16_1k,
    "f32_32x32x4bf16_1k": lib.mfma_f32_32x32x4bf16_1k,
    "f32_16x16x16bf16_1k": lib.mfma_f32_16x16x16bf16_1k,
    "f32_16x16x4bf16_1k": lib.mfma_f32_16x16x4bf16_1k,
    "f32_4x4x4bf16_1k": lib.mfma_f32_4x4x4bf16_1k,
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
}
mfma_kernels = {
    suffix: MFMA("CDNA2", suffix, mfma_hip_kernels[suffix])
    for suffix in mfma_hip_kernels
}
