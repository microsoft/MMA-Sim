import math

import torch


volta_mma_qualifiers = [
    # sm_70 instructions
    "m8n8k4.f16.f16.f16.f16",
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f32.f16.f16.f32",
]
turing_mma_qualifiers = [
    # sm_75 instructions
    "m16n8k8.f16.f16.f16.f16",
    "m16n8k8.f32.f16.f16.f32",
]
ampere_mma_qualifiers = [
    # sm_75 instructions
    "m16n8k8.f16.f16.f16.f16",
    "m16n8k8.f32.f16.f16.f32",
    # sm_80 instructions
    "m16n8k16.f16.f16.f16.f16",
    "m16n8k16.f32.f16.f16.f32",
    "m16n8k8.f32.bf16.bf16.f32",
    "m16n8k16.f32.bf16.bf16.f32",
    "m16n8k4.f32.tf32.tf32.f32",
    "m16n8k8.f32.tf32.tf32.f32",
]
adalovelace_mma_qualifiers = [
    # sm_75 instructions
    "m16n8k8.f16.f16.f16.f16",
    "m16n8k8.f32.f16.f16.f32",
    # sm_80 instructions
    "m16n8k16.f16.f16.f16.f16",
    "m16n8k16.f32.f16.f16.f32",
    "m16n8k8.f32.bf16.bf16.f32",
    "m16n8k16.f32.bf16.bf16.f32",
    "m16n8k4.f32.tf32.tf32.f32",
    "m16n8k8.f32.tf32.tf32.f32",
    # sm_89 instructions
    "m16n8k32.f32.e5m2.e5m2.f32",
    "m16n8k32.f32.e5m2.e4m3.f32",
    "m16n8k32.f32.e4m3.e5m2.f32",
    "m16n8k32.f32.e4m3.e4m3.f32",
    # sm_89 instructions since PTX 8.7
    "m16n8k32.f16.e5m2.e5m2.f16",
    "m16n8k32.f16.e5m2.e4m3.f16",
    "m16n8k32.f16.e4m3.e5m2.f16",
    "m16n8k32.f16.e4m3.e4m3.f16",
    "m16n8k16.f32.e5m2.e5m2.f32",
    "m16n8k16.f32.e5m2.e4m3.f32",
    "m16n8k16.f32.e4m3.e5m2.f32",
    "m16n8k16.f32.e4m3.e4m3.f32",
    "m16n8k16.f16.e5m2.e5m2.f16",
    "m16n8k16.f16.e5m2.e4m3.f16",
    "m16n8k16.f16.e4m3.e5m2.f16",
    "m16n8k16.f16.e4m3.e4m3.f16",
]
hopper_mma_qualifiers = [
    # sm_75 instructions
    "m16n8k8.f16.f16.f16.f16",
    "m16n8k8.f32.f16.f16.f32",
    # sm_80 instructions
    "m16n8k16.f16.f16.f16.f16",
    "m16n8k16.f32.f16.f16.f32",
    "m16n8k8.f32.bf16.bf16.f32",
    "m16n8k16.f32.bf16.bf16.f32",
    "m16n8k4.f32.tf32.tf32.f32",
    "m16n8k8.f32.tf32.tf32.f32",
]
hopper_wgmma_qualifiers = (
    # sm_90a instructions
    [f"m64n{N}k16.f16.f16.f16" for N in range(8, 256 + 1, 8)]
    + [f"m64n{N}k16.f32.f16.f16" for N in range(8, 256 + 1, 8)]
    + [f"m64n{N}k16.f32.bf16.bf16" for N in range(8, 256 + 1, 8)]
    + [f"m64n{N}k8.f32.tf32.tf32" for N in range(8, 256 + 1, 8)]
    + [
        f"m64n{N}k32.{dtype}.{atype}.{btype}"
        for N in range(8, 256 + 1, 8)
        for dtype in ["f32", "f16"]
        for atype in ["e5m2", "e4m3"]
        for btype in ["e5m2", "e4m3"]
    ]
)

arch_mma_qualifiers = {
    "Volta": volta_mma_qualifiers,
    "Turing": turing_mma_qualifiers,
    "Ampere": ampere_mma_qualifiers,
    "Ada Lovelace": adalovelace_mma_qualifiers,
    "Hopper": hopper_mma_qualifiers,
}

arch_wgmma_qualifiers = {
    "Hopper": hopper_wgmma_qualifiers,
}


def shape_to_mnk(shape: str) -> tuple[int, int, int]:
    mnk = shape.split("m")[1]
    m, nk = mnk.split("n")
    n, k = nk.split("k")
    return int(m), int(n), int(k)


class TensorCoreInstruction:
    arch: str
    qualifier: str
    shape: str
    a_type: str
    b_type: str
    c_type: str
    d_type: str


torch_dtype = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "tf32": torch.float32,
    "e5m2": torch.float8_e5m2,
    "e4m3": torch.float8_e4m3fn,
}

min_exponent = {
    "f16": -14,
    "f32": -126,
    "bf16": -126,
    "tf32": -126,
    "e5m2": -14,
    "e4m3": -6,
}

arch_accum_fraction_bits = {
    "Volta": 23,
    "Turing": 24,
    "Ampere": 24,
    "Ada Lovelace": 24,
    "Hopper": 25,
}


def extract_significand_exponent(x: float, min_exponent=-126) -> tuple[float, int]:
    significand, exponent = math.frexp(x)
    significand *= 2  # 1 <= |significand| < 2
    exponent -= 1
    # subnormal
    if exponent < min_exponent:
        significand *= 2.0 ** (exponent - min_exponent)
        exponent = min_exponent
    if significand == 0.0:
        exponent = -999
    return significand, exponent


@torch.inference_mode()
def truncate_to_tf32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    x = x.view(torch.int32)  # uint32 operations are not supported by pytorch
    x = x >> 13 << 13  # truncate to tf32
    return x.view(torch.float32)


@torch.inference_mode()
def fused_dot_add(
    inst: TensorCoreInstruction, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    # handle nan and inf
    products = a.double() * b.double()
    special_sum = c + products.sum()
    if torch.isnan(special_sum).any():
        # NVIDIA encodes nan as 0x7fff ffff
        x = torch.tensor(0x7FFF_FFFF, dtype=torch.uint32)
        return x.view(torch.float32)
    elif torch.isinf(special_sum).any():
        return special_sum

    # handle finite numbers
    sc, ec = extract_significand_exponent(c.item(), min_exponent[inst.c_type])
    significands = [sc]
    exponents = [ec]
    for i in range(products.numel()):
        if products[i] == 0:
            continue
        sa, ea = extract_significand_exponent(a[i].item(), min_exponent[inst.a_type])
        sb, eb = extract_significand_exponent(b[i].item(), min_exponent[inst.b_type])
        significands.append(sa * sb)
        exponents.append(ea + eb)
    maxe = max(exponents)
    if inst.a_type in ["e4m3", "e5m2"]:
        n_accum_fraction_bits = 13
    else:
        n_accum_fraction_bits = arch_accum_fraction_bits[inst.arch]
    sum = 0
    for i in range(len(significands)):
        as_integer = math.trunc(
            significands[i] * 2 ** (n_accum_fraction_bits + exponents[i] - maxe)
        )
        sum += as_integer
    sum *= 2 ** (maxe - n_accum_fraction_bits)
    if inst.d_type == "f16":
        # RNE
        # note that direcctly converting f64 to f16 may be incorrect
        # as PyTorch-CPU computes f64 -> f32 -> f16 internally
        s, e = extract_significand_exponent(sum, min_exponent=-14)
        s = round(s * 2.0**10) * 2.0**-10
        return torch.tensor(s * 2.0**e, dtype=torch.float16)
    elif inst.d_type == "f32":
        # RZ
        s, e = extract_significand_exponent(sum, min_exponent=-126)
        n_fraction_bits = 13 if inst.a_type in ["e5m2", "e4m3"] else 23
        s = math.trunc(s * 2.0**n_fraction_bits) * 2.0**-n_fraction_bits
        return torch.tensor(s * 2.0**e, dtype=torch.float32)


class MMA(TensorCoreInstruction):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for MMA.\n"
            f"Supported architectures: {list(arch_mma_qualifiers.keys())}"
        )
        self.arch = arch
        supported_qualifiers = arch_mma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported MMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.qualifier = qualifier
        self.shape, self.d_type, self.a_type, self.b_type, self.c_type = (
            qualifier.split(".")
        )

    def is_split_k(self) -> bool:
        if self.arch in ["Ampere", "Ada Lovelace"] and self.qualifier in [
            "m16n8k16.f16.f16.f16.f16",
            "m16n8k16.f32.f16.f16.f32",
            "m16n8k16.f32.bf16.bf16.f32",
            "m16n8k8.f32.tf32.tf32.f32",
            "m16n8k32.f32.e5m2.e5m2.f32",
            "m16n8k32.f32.e5m2.e4m3.f32",
            "m16n8k32.f32.e4m3.e5m2.f32",
            "m16n8k32.f32.e4m3.e4m3.f32",
            "m16n8k32.f16.e5m2.e5m2.f16",
            "m16n8k32.f16.e5m2.e4m3.f16",
            "m16n8k32.f16.e4m3.e5m2.f16",
            "m16n8k32.f16.e4m3.e4m3.f16",
        ]:
            return True
        return False

    @torch.inference_mode()
    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = shape_to_mnk(self.shape)
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert c.shape == (m, n)
        assert a.dtype == torch_dtype[self.a_type]
        assert b.dtype == torch_dtype[self.b_type]
        assert c.dtype == torch_dtype[self.c_type]
        assert a.is_cpu and b.is_cpu and c.is_cpu
        if self.a_type == "tf32":
            a = truncate_to_tf32(a)
            b = truncate_to_tf32(b)
        d = torch.empty((m, n), dtype=torch_dtype[self.d_type])
        for i in range(m):
            for j in range(n):
                sum = c[i][j]
                if self.is_split_k():
                    sum = fused_dot_add(self, a[i, : k // 2], b[: k // 2, j], sum)
                    sum = fused_dot_add(self, a[i, k // 2 : k], b[k // 2 : k, j], sum)
                else:
                    sum = fused_dot_add(self, a[i, :], b[:, j], sum)
                d[i][j] = sum
        return d


class WGMMA(MMA):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_wgmma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for WGMMA.\n"
            f"Supported architectures: {list(arch_mma_qualifiers.keys())}"
        )
        self.arch = arch
        supported_qualifiers = arch_wgmma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported WGMMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.qualifier = qualifier
        self.shape, self.d_type, self.a_type, self.b_type = qualifier.split(".")
        self.c_type = self.d_type
