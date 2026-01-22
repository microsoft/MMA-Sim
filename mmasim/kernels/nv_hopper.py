import ctypes
import pathlib

from . import mma_kernel, wgmma_kernel


path = pathlib.Path(__file__).parent / "impl/nv_hopper.so"
lib = ctypes.CDLL(str(path))

# sm_90a tf32
lib.wgmma_m64n8k8_f32_tf32_tf32.argtypes = [ctypes.c_void_p] * 3
# sm_90a f16
lib.wgmma_m64n8k16_f32_f16_f16.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 3
# sm_90a bf16
lib.wgmma_m64n8k16_f32_bf16_bf16.argtypes = [ctypes.c_void_p] * 3
# sm_90a fp8 f32_output
lib.wgmma_m64n8k32_f32_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_f32_e5m2_e4m3.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_f32_e4m3_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_f32_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3
# sm_90a fp8 f16_output
lib.wgmma_m64n8k32_f16_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_f16_e5m2_e4m3.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_f16_e4m3_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.wgmma_m64n8k32_f16_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3
# sm_90 f64
lib.mma_m16n8k16_f64_f64_f64_f64.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k8_f64_f64_f64_f64.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k4_f64_f64_f64_f64.argtypes = [ctypes.c_void_p] * 4
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

wgmma_kernel_impls = {
    # sm_90a tf32
    "m64n8k8.f32.tf32.tf32": lib.wgmma_m64n8k8_f32_tf32_tf32,
    # sm_90a f16
    "m64n8k16.f32.f16.f16": lib.wgmma_m64n8k16_f32_f16_f16,
    "m64n8k16.f16.f16.f16": lib.wgmma_m64n8k16_f16_f16_f16,
    # sm_90a bf16
    "m64n8k16.f32.bf16.bf16": lib.wgmma_m64n8k16_f32_bf16_bf16,
    # sm_90a fp8 f32_output
    "m64n8k32.f32.e5m2.e5m2": lib.wgmma_m64n8k32_f32_e5m2_e5m2,
    "m64n8k32.f32.e5m2.e4m3": lib.wgmma_m64n8k32_f32_e5m2_e4m3,
    "m64n8k32.f32.e4m3.e5m2": lib.wgmma_m64n8k32_f32_e4m3_e5m2,
    "m64n8k32.f32.e4m3.e4m3": lib.wgmma_m64n8k32_f32_e4m3_e4m3,
    # sm_90a fp8 f16_output
    "m64n8k32.f16.e5m2.e5m2": lib.wgmma_m64n8k32_f16_e5m2_e5m2,
    "m64n8k32.f16.e5m2.e4m3": lib.wgmma_m64n8k32_f16_e5m2_e4m3,
    "m64n8k32.f16.e4m3.e5m2": lib.wgmma_m64n8k32_f16_e4m3_e5m2,
    "m64n8k32.f16.e4m3.e4m3": lib.wgmma_m64n8k32_f16_e4m3_e4m3,
}
mma_kernel_impls = {
    # sm_90 f64
    "m16n8k16.f64.f64.f64.f64": lib.mma_m16n8k16_f64_f64_f64_f64,
    "m16n8k8.f64.f64.f64.f64": lib.mma_m16n8k8_f64_f64_f64_f64,
    "m16n8k4.f64.f64.f64.f64": lib.mma_m16n8k4_f64_f64_f64_f64,
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
wgmma_kernels = {
    qualifier: wgmma_kernel("Hopper", qualifier, wgmma_kernel_impls[qualifier])
    for qualifier in wgmma_kernel_impls
}
mma_kernels = {
    qualifier: mma_kernel("Hopper", qualifier, mma_kernel_impls[qualifier])
    for qualifier in mma_kernel_impls
}
