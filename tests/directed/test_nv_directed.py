import torch

from mmasim.nv_ptx import MMA, WGMMA, TCGen05MMA

# Expected D bits are measured on H200 (Hopper) and B300 (Blackwell)


def vector(dtype: torch.dtype, k: int, entries: dict[int, float]) -> torch.Tensor:
    x = torch.zeros(k, dtype=torch.float32)
    for i, v in entries.items():
        x[i] = v
    return x.to(dtype)


def raw(x: torch.Tensor) -> int:
    if x.dtype == torch.float16:
        return int(x.view(torch.int16).item()) & 0xFFFF
    return int(x.view(torch.int32).item()) & 0xFFFFFFFF


def directed_test(sim, a: dict, b: dict, c: float, expected: int):
    a_vec = vector(sim.a_type, sim.k, a)
    b_vec = vector(sim.b_type, sim.k, b)
    d = sim.dpa(a_vec, b_vec, torch.tensor(c, dtype=sim.c_type))
    assert raw(d) == expected, f"D_sim = {raw(d):#x}, D_gpu = {expected:#x}"


def test_zero_result_is_positive():
    # a nonzero negative sum that rounds to zero gives +0, not -0
    tiny = {0: -(2.0**-75)}, {0: 2.0**-75}, 0.0
    f16_tiny = {0: (1 + 2.0**-10) * 2.0**-5}, {0: (1 - 2.0**-10) * 2.0**-5}, -(2.0**-10)
    bf16_tiny = (
        {0: (1 + 2.0**-7) * 2.0**-75},
        {0: (1 + 2.0**-7) * 2.0**-75},
        -(2.0**-149),
    )
    for sim, (a, b, c) in [
        (WGMMA("Hopper", "m64n8k16.f32.bf16.bf16"), tiny),
        (MMA("Hopper", "m16n8k16.f32.bf16.bf16.f32"), tiny),
        (WGMMA("Hopper", "m64n8k16.f16.f16.f16"), f16_tiny),
        (MMA("Hopper", "m16n8k16.f16.f16.f16.f16"), f16_tiny),
        (TCGen05MMA("Blackwell", "m64n8k16.f16.f16.f16"), f16_tiny),
        (MMA("Blackwell", "m16n8k16.f16.f16.f16.f16"), f16_tiny),
        (TCGen05MMA("Blackwell", "m64n8k16.f32.bf16.bf16"), bf16_tiny),
        (MMA("Blackwell", "m16n8k16.f32.bf16.bf16.f32"), bf16_tiny),
    ]:
        print(f"Testing {sim.arch} {sim.shape_and_type}")
        directed_test(sim, a, b, c, 0)
