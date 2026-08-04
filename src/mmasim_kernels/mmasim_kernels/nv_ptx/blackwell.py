import ctypes
from pathlib import Path

from torch.utils.cpp_extension import load
from . import MMA, TCGen05MMA, TCGen05MMABlockScale

dir = Path(__file__).with_name("kernels")
files = [dir / f for f in ["tcgen05mma_sm100a.cu", "mma_sm80.cu", "mma_sm75.cu"]]
kernel_path = load(
    "mmasim_kernels_nv_blackwell",
    [str(f) for f in files],
    extra_cuda_cflags=["-arch=compute_100a", "-code=sm_100a"],
    is_python_module=False,
)
lib = ctypes.CDLL(kernel_path)

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

# sm_100a tf32
lib.tcgen05mma_m64n8k8_f32_tf32_tf32.argtypes = [ctypes.c_void_p] * 3
# sm_100a f16 and bf16
lib.tcgen05mma_m64n8k16_f32_f16_f16.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_m64n8k16_f32_bf16_bf16.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_m64n8k16_f16_f16_f16.argtypes = [ctypes.c_void_p] * 3
# sm_100a fp8 f32_output
lib.tcgen05mma_m64n8k32_f32_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_m64n8k32_f32_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3
# sm_100a fp8 f16_output
lib.tcgen05mma_m64n8k32_f16_e5m2_e5m2.argtypes = [ctypes.c_void_p] * 3
lib.tcgen05mma_m64n8k32_f16_e4m3_e4m3.argtypes = [ctypes.c_void_p] * 3

# sm_100a mxf8f6f4
lib.tcgen05mma_m128n8k32_block32_f32_e5m2_e5m2_ue8m0.argtypes = [ctypes.c_void_p] * 5
lib.tcgen05mma_m128n8k32_block32_f32_e4m3_e4m3_ue8m0.argtypes = [ctypes.c_void_p] * 5
# sm_100a mxf4 and nvf4
lib.tcgen05mma_m128n8k64_block32_f32_e2m1_e2m1_ue8m0.argtypes = [ctypes.c_void_p] * 5
lib.tcgen05mma_m128n8k64_block16_f32_e2m1_e2m1_ue8m0.argtypes = [ctypes.c_void_p] * 5
lib.tcgen05mma_m128n8k64_block16_f32_e2m1_e2m1_ue4m3.argtypes = [ctypes.c_void_p] * 5

mma_cuda_kernels = {
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
tcgen05mma_cuda_kernels = {
    # sm_100a tf32
    "m64n8k8.f32.tf32.tf32": lib.tcgen05mma_m64n8k8_f32_tf32_tf32,
    # sm_100a f16 and bf16
    "m64n8k16.f32.f16.f16": lib.tcgen05mma_m64n8k16_f32_f16_f16,
    "m64n8k16.f32.bf16.bf16": lib.tcgen05mma_m64n8k16_f32_bf16_bf16,
    "m64n8k16.f16.f16.f16": lib.tcgen05mma_m64n8k16_f16_f16_f16,
    # sm_100a fp8 f32_output
    "m64n8k32.f32.e5m2.e5m2": lib.tcgen05mma_m64n8k32_f32_e5m2_e5m2,
    "m64n8k32.f32.e4m3.e4m3": lib.tcgen05mma_m64n8k32_f32_e4m3_e4m3,
    # sm_100a fp8 f16_output
    "m64n8k32.f16.e5m2.e5m2": lib.tcgen05mma_m64n8k32_f16_e5m2_e5m2,
    "m64n8k32.f16.e4m3.e4m3": lib.tcgen05mma_m64n8k32_f16_e4m3_e4m3,
}
tcgen05mma_block_scale_cuda_kernels = {
    # sm_100a mxf8f6f4
    "m128n8k32.block32.f32.e5m2.e5m2.ue8m0": lib.tcgen05mma_m128n8k32_block32_f32_e5m2_e5m2_ue8m0,
    "m128n8k32.block32.f32.e4m3.e4m3.ue8m0": lib.tcgen05mma_m128n8k32_block32_f32_e4m3_e4m3_ue8m0,
    # sm_100a mxf4 and nvf4
    "m128n8k64.block32.f32.e2m1.e2m1.ue8m0": lib.tcgen05mma_m128n8k64_block32_f32_e2m1_e2m1_ue8m0,
    "m128n8k64.block16.f32.e2m1.e2m1.ue8m0": lib.tcgen05mma_m128n8k64_block16_f32_e2m1_e2m1_ue8m0,
    "m128n8k64.block16.f32.e2m1.e2m1.ue4m3": lib.tcgen05mma_m128n8k64_block16_f32_e2m1_e2m1_ue4m3,
}

mma_kernels = {
    shape_and_type: MMA("Blackwell", shape_and_type, mma_cuda_kernels[shape_and_type])
    for shape_and_type in mma_cuda_kernels
}
tcgen05mma_kernels = {
    shape_and_type: TCGen05MMA(
        "Blackwell", shape_and_type, tcgen05mma_cuda_kernels[shape_and_type]
    )
    for shape_and_type in tcgen05mma_cuda_kernels
}
tcgen05mma_block_scale_kernels = {
    shape_and_type: TCGen05MMABlockScale(
        "Blackwell", shape_and_type, tcgen05mma_block_scale_cuda_kernels[shape_and_type]
    )
    for shape_and_type in tcgen05mma_block_scale_cuda_kernels
}
