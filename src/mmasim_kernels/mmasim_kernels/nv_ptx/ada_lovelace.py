import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
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

# sm_89 fp8 m16n8k32 f32_output
lib.mma_m16n8k32_f32_e5m2_e5m2_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f32_e5m2_e4m3_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f32_e4m3_e5m2_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f32_e4m3_e4m3_f32.argtypes = [ctypes.c_void_p] * 4
# sm_89 fp8 m16n8k16 f32_output
lib.mma_m16n8k16_f32_e5m2_e5m2_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f32_e5m2_e4m3_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f32_e4m3_e5m2_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f32_e4m3_e4m3_f32.argtypes = [ctypes.c_void_p] * 4
# sm_89 fp8 m16n8k32 f16_output
lib.mma_m16n8k32_f16_e5m2_e5m2_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f16_e5m2_e4m3_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f16_e4m3_e5m2_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f16_e4m3_e4m3_f16.argtypes = [ctypes.c_void_p] * 4
# sm_89 fp8 m16n8k16 f16_output
lib.mma_m16n8k16_f16_e5m2_e5m2_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f16_e5m2_e4m3_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f16_e4m3_e5m2_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f16_e4m3_e4m3_f16.argtypes = [ctypes.c_void_p] * 4
# sm_80 f64
lib.mma_m8n8k4_f64_f64_f64_f64.argtypes = [ctypes.c_void_p] * 4
# sm_80 tf32
lib.mma_m16n8k8_f32_tf32_tf32_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k4_f32_tf32_tf32_f32.argtypes = [ctypes.c_void_p] * 4
# sm_80 f16
lib.mma_m16n8k16_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k16_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4
# sm_80 bf16
lib.mma_m16n8k16_f32_bf16_bf16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_f32_bf16_bf16_f32.argtypes = [ctypes.c_void_p] * 4
# sm_75 f16
lib.mma_m16n8k8_f32_f16_f16_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_f16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 4

mma_cuda_kernels = {
    # sm_89 fp8 m16n8k32 f32_output
    "m16n8k32.f32.e5m2.e5m2.f32": lib.mma_m16n8k32_f32_e5m2_e5m2_f32,
    "m16n8k32.f32.e5m2.e4m3.f32": lib.mma_m16n8k32_f32_e5m2_e4m3_f32,
    "m16n8k32.f32.e4m3.e5m2.f32": lib.mma_m16n8k32_f32_e4m3_e5m2_f32,
    "m16n8k32.f32.e4m3.e4m3.f32": lib.mma_m16n8k32_f32_e4m3_e4m3_f32,
    # sm_89 fp8 m16n8k16 f32_output
    "m16n8k16.f32.e5m2.e5m2.f32": lib.mma_m16n8k16_f32_e5m2_e5m2_f32,
    "m16n8k16.f32.e5m2.e4m3.f32": lib.mma_m16n8k16_f32_e5m2_e4m3_f32,
    "m16n8k16.f32.e4m3.e5m2.f32": lib.mma_m16n8k16_f32_e4m3_e5m2_f32,
    "m16n8k16.f32.e4m3.e4m3.f32": lib.mma_m16n8k16_f32_e4m3_e4m3_f32,
    # sm_89 fp8 m16n8k32 f16_output
    "m16n8k32.f16.e5m2.e5m2.f16": lib.mma_m16n8k32_f16_e5m2_e5m2_f16,
    "m16n8k32.f16.e5m2.e4m3.f16": lib.mma_m16n8k32_f16_e5m2_e4m3_f16,
    "m16n8k32.f16.e4m3.e5m2.f16": lib.mma_m16n8k32_f16_e4m3_e5m2_f16,
    "m16n8k32.f16.e4m3.e4m3.f16": lib.mma_m16n8k32_f16_e4m3_e4m3_f16,
    # sm_89 fp8 m16n8k16 f16_output
    "m16n8k16.f16.e5m2.e5m2.f16": lib.mma_m16n8k16_f16_e5m2_e5m2_f16,
    "m16n8k16.f16.e5m2.e4m3.f16": lib.mma_m16n8k16_f16_e5m2_e4m3_f16,
    "m16n8k16.f16.e4m3.e5m2.f16": lib.mma_m16n8k16_f16_e4m3_e5m2_f16,
    "m16n8k16.f16.e4m3.e4m3.f16": lib.mma_m16n8k16_f16_e4m3_e4m3_f16,
    # sm_80 f64
    "m8n8k4.f64.f64.f64.f64": lib.mma_m8n8k4_f64_f64_f64_f64,
    # sm_80 tf32
    "m16n8k8.f32.tf32.tf32.f32": lib.mma_m16n8k8_f32_tf32_tf32_f32,
    "m16n8k4.f32.tf32.tf32.f32": lib.mma_m16n8k4_f32_tf32_tf32_f32,
    # sm_80 f16
    "m16n8k16.f32.f16.f16.f32": lib.mma_m16n8k16_f32_f16_f16_f32,
    "m16n8k16.f16.f16.f16.f16": lib.mma_m16n8k16_f16_f16_f16_f16,
    # sm_80 bf16
    "m16n8k16.f32.bf16.bf16.f32": lib.mma_m16n8k16_f32_bf16_bf16_f32,
    "m16n8k8.f32.bf16.bf16.f32": lib.mma_m16n8k8_f32_bf16_bf16_f32,
    # sm_75 f16
    "m16n8k8.f32.f16.f16.f32": lib.mma_m16n8k8_f32_f16_f16_f32,
    "m16n8k8.f16.f16.f16.f16": lib.mma_m16n8k8_f16_f16_f16_f16,
}

mma_kernels = {
    shape_and_type: MMA(
        "Ada Lovelace", shape_and_type, mma_cuda_kernels[shape_and_type]
    )
    for shape_and_type in mma_cuda_kernels
}
