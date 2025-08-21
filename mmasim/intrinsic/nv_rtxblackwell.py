import ctypes
import pathlib

from . import MMAIntrinsic


path = pathlib.Path(__file__).parent / "impl/nv_rtxblackwell.so"
lib = ctypes.CDLL(str(path))

# sm_120a f8f6f4
lib.mma_m16n8k32_f8f6f4_f32_e5m2_e5m2_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f8f6f4_f32_e4m3_e4m3_f32.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f8f6f4_f16_e5m2_e5m2_f16.argtypes = [ctypes.c_void_p] * 4
lib.mma_m16n8k32_f8f6f4_f16_e4m3_e4m3_f16.argtypes = [ctypes.c_void_p] * 4
# sm_120a mxf8f6f4
lib.mma_m16n8k32_mxf8f6f4_block32_f32_e5m2_e5m2_f32_ue8m0.argtypes = [
    ctypes.c_void_p
] * 6
lib.mma_m16n8k32_mxf8f6f4_block32_f32_e4m3_e4m3_f32_ue8m0.argtypes = [
    ctypes.c_void_p
] * 6
# sm_120a mxf4 and nvf4
lib.mma_m16n8k64_mxf4_block32_f32_e2m1_e2m1_f32_ue8m0.argtypes = [ctypes.c_void_p] * 6
lib.mma_m16n8k64_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue8m0.argtypes = [
    ctypes.c_void_p
] * 6
lib.mma_m16n8k64_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue4m3.argtypes = [
    ctypes.c_void_p
] * 6
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

mma_intrinsic_impls = {
    # sm_120a f8f6f4
    "m16n8k32.f8f6f4.f32.e5m2.e5m2.f32": lib.mma_m16n8k32_f8f6f4_f32_e5m2_e5m2_f32,
    "m16n8k32.f8f6f4.f32.e4m3.e4m3.f32": lib.mma_m16n8k32_f8f6f4_f32_e4m3_e4m3_f32,
    "m16n8k32.f8f6f4.f16.e5m2.e5m2.f16": lib.mma_m16n8k32_f8f6f4_f16_e5m2_e5m2_f16,
    "m16n8k32.f8f6f4.f16.e4m3.e4m3.f16": lib.mma_m16n8k32_f8f6f4_f16_e4m3_e4m3_f16,
    # sm_120a mxf8f6f4
    "m16n8k32.mxf8f6f4.block32.f32.e5m2.e5m2.f32.ue8m0": lib.mma_m16n8k32_mxf8f6f4_block32_f32_e5m2_e5m2_f32_ue8m0,
    "m16n8k32.mxf8f6f4.block32.f32.e4m3.e4m3.f32.ue8m0": lib.mma_m16n8k32_mxf8f6f4_block32_f32_e4m3_e4m3_f32_ue8m0,
    # sm_120a mxf4 and nvf4
    "m16n8k64.mxf4.block32.f32.e2m1.e2m1.f32.ue8m0": lib.mma_m16n8k64_mxf4_block32_f32_e2m1_e2m1_f32_ue8m0,
    "m16n8k64.mxf4nvf4.block32.f32.e2m1.e2m1.f32.ue8m0": lib.mma_m16n8k64_mxf4nvf4_block32_f32_e2m1_e2m1_f32_ue8m0,
    "m16n8k64.mxf4nvf4.block16.f32.e2m1.e2m1.f32.ue4m3": lib.mma_m16n8k64_mxf4nvf4_block16_f32_e2m1_e2m1_f32_ue4m3,
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
mma_intrinsics = {
    qualifier: MMAIntrinsic("RTX Blackwell", qualifier, mma_intrinsic_impls[qualifier])
    for qualifier in mma_intrinsic_impls
}
