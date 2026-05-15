import torch

nv_torch_dtype = {
    "f64": torch.float64,
    "f32": torch.float32,
    "tf32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    "ue8m0": torch.float8_e8m0fnu,
    "ue4m3": torch.float8_e4m3fn,
    "e2m1": torch.uint8,  # torch.float4_e2m1fn_x2 is not well-implemented
}


def nv_shape_to_mnk(shape: str) -> tuple[int, int, int]:
    mnk = shape.split("m")[1]
    m, nk = mnk.split("n")
    n, k = nk.split("k")
    return int(m), int(n), int(k)
