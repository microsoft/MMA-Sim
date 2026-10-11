import torch

from mmasim.nv_ptx import MMA, WGMMA, TCGen05MMA

# Expected D bits are measured on H200 (Hopper), B300 and B200 (Blackwell)


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
    print("Tests passed!")


def test_e_zero_f16_accumulator():
    # a zero f16 C anchors the fused sum at 2^-21
    hopper = [
        WGMMA("Hopper", "m64n8k16.f16.f16.f16"),
    ]
    for sim in hopper:
        print(f"Testing {sim.arch} {sim.shape_and_type}")
        # 2^-25 is a tie at the f16 subnormal step, 2^-q breaks it iff q <= 46
        for q, expected in [(46, 0x0001), (47, 0x0000)]:
            a, b = {0: 2.0**-12, 1: 2.0**-23}, {0: 2.0**-13, 1: 2.0 ** (23 - q)}
            directed_test(sim, a, b, 0.0, expected)
    blackwell = [
        TCGen05MMA("Blackwell", "m64n8k16.f16.f16.f16"),
        MMA("Blackwell", "m16n8k16.f16.f16.f16.f16"),
    ]
    for sim in blackwell:
        print(f"Testing {sim.arch} {sim.shape_and_type}")
        # 1.5 * 2^-23 + 2^-25 is a tie, -2^-47 is below the 2^-46 grid
        a = {0: 1.5 * 2.0**-12, 1: 2.0**-13, 2: -(2.0**-24)}
        b = {0: 2.0**-11, 1: 2.0**-12, 2: 2.0**-23}
        directed_test(sim, a, b, 0.0, 0x0004)
    print("Tests passed!")


def test_e2m1_f8f6f4():
    # a * b = 0.25 is 0.01b * 2^0, so C = -(0.25 - 2^-26) loses its last bit
    sim = TCGen05MMA("Blackwell", "m128n8k32.f32.e2m1.e2m1")
    print(f"Testing {sim.arch} {sim.shape_and_type}")
    a = torch.zeros(sim.k, dtype=torch.uint8)
    a[0] = 0x1 << 2  # e2m1 sits in bits 5:2 of the 8-bit container
    d = sim.dpa(a, a, torch.tensor(-(0.25 - 2.0**-26), dtype=torch.float32))
    assert raw(d) == 0x33000000, f"D_sim = {raw(d):#x}, D_gpu = 0x33000000"
    print("Tests passed!")


if __name__ == "__main__":
    test_zero_result_is_positive()
    test_e_zero_f16_accumulator()
    test_e2m1_f8f6f4()
