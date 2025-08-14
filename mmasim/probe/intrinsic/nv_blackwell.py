import ctypes
import pathlib

from . import NV_MMA, NV_TCGen05MMA


path = pathlib.Path(__file__).parent / "impl/nv_blackwell.so"
lib = ctypes.CDLL(str(path))

# sm_100a tf32
lib.tcgen05mma_tf32_m64n8k8_f32_tf32_tf32.argtypes = [ctypes.c_void_p] * 3
# sm_100a f16 and bf16
lib.tcgen05mma_f16_m64n8k16_f32_f16_f16.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_f16_m64n8k16_f32_bf16_bf16.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_f16_m64n8k16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 3
# sm_100a fp8 f32_output
lib.tcgen05mma_f8f6f4_m64n8k32_f32_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_f8f6f4_m64n8k32_f32_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3
# sm_100a fp8 f16_output
lib.tcgen05mma_f8f6f4_m64n8k32_f16_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_f8f6f4_m64n8k32_f16_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3
# sm_100a mxfp8
lib.tcgen05mma_mxf8f6f4_m128n8k32_f32_e5m2_e5m2_block32.argtypes = [ctypes.c_void_p] * 5
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


tcgen05mma_intrinsic_impls = {
    # sm_100a tf32
    "tf32.m64n8k8.f32.tf32.tf32": lib.tcgen05mma_tf32_m64n8k8_f32_tf32_tf32,
    # sm_100a f16 and bf16
    "f16.m64n8k16.f32.f16.f16": lib.tcgen05mma_f16_m64n8k16_f32_f16_f16,
    "f16.m64n8k16.f32.bf16.bf16": lib.tcgen05mma_f16_m64n8k16_f32_bf16_bf16,
    "f16.m64n8k16.f16.f16.f16": lib.tcgen05mma_f16_m64n8k16_f16_f16_f16,
    # sm_100a fp8 f32_output
    "f8f6f4.m64n8k32.f32.e5m2.e5m2": lib.tcgen05mma_f8f6f4_m64n8k32_f32_e5m2_e5m2,
    "f8f6f4.m64n8k32.f32.e4m3.e4m3": lib.tcgen05mma_f8f6f4_m64n8k32_f32_e4m3_e4m3,
    # sm_100a fp8 f16_output
    "f8f6f4.m64n8k32.f16.e5m2.e5m2": lib.tcgen05mma_f8f6f4_m64n8k32_f16_e5m2_e5m2,
    "f8f6f4.m64n8k32.f16.e4m3.e4m3": lib.tcgen05mma_f8f6f4_m64n8k32_f16_e4m3_e4m3,
    # sm_100a mxfp8
    "mxf8f6f4.m128n8k32.f32.e5m2.e5m2.block32": lib.tcgen05mma_mxf8f6f4_m128n8k32_f32_e5m2_e5m2_block32,
}
mma_intrinsic_impls = {
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
tcgen05mma_intrinsics = {
    qualifier: NV_TCGen05MMA(qualifier, tcgen05mma_intrinsic_impls[qualifier])
    for qualifier in tcgen05mma_intrinsic_impls
}
mma_intrinsics = {
    qualifier: NV_MMA(qualifier, mma_intrinsic_impls[qualifier])
    for qualifier in mma_intrinsic_impls
}
