import torch

from .common import MatrixMultiplyAdd, nv_shape_to_mnk, nv_torch_dtype

volta_mma_qualifiers = [
    # sm_70 f16
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f32.f16.f16.f32",
    "m8n8k4.f16.f16.f16.f16",
]
turing_mma_qualifiers = [
    # sm_75 f16
    "m16n8k8.f32.f16.f16.f32",
    "m16n8k8.f16.f16.f16.f16",
]
ampere_mma_qualifiers = [
    # sm_80 tf32
    "m16n8k4.f32.tf32.tf32.f32",
    "m16n8k8.f32.tf32.tf32.f32",
    # sm_80 f16
    "m16n8k16.f16.f16.f16.f16",
    "m16n8k16.f32.f16.f16.f32",
    # sm_80 bf16
    "m16n8k8.f32.bf16.bf16.f32",
    "m16n8k16.f32.bf16.bf16.f32",
]
adalovelace_mma_qualifiers = [
    # sm_89 fp8 m16n8k32 f32_output
    "m16n8k32.f32.e5m2.e5m2.f32",
    "m16n8k32.f32.e5m2.e4m3.f32",
    "m16n8k32.f32.e4m3.e5m2.f32",
    "m16n8k32.f32.e4m3.e4m3.f32",
    # sm_89 fp8 m16n8k16 f32_output
    "m16n8k16.f32.e5m2.e5m2.f32",
    "m16n8k16.f32.e5m2.e4m3.f32",
    "m16n8k16.f32.e4m3.e5m2.f32",
    "m16n8k16.f32.e4m3.e4m3.f32",
    # sm_89 fp8 m16n8k32 f16_output
    "m16n8k32.f16.e5m2.e5m2.f16",
    "m16n8k32.f16.e5m2.e4m3.f16",
    "m16n8k32.f16.e4m3.e5m2.f16",
    "m16n8k32.f16.e4m3.e4m3.f16",
    # sm_89 fp8 m16n8k16 f16_output
    "m16n8k16.f16.e5m2.e5m2.f16",
    "m16n8k16.f16.e5m2.e4m3.f16",
    "m16n8k16.f16.e4m3.e5m2.f16",
    "m16n8k16.f16.e4m3.e4m3.f16",
]

arch_mma_qualifiers = {
    "Volta": volta_mma_qualifiers,
    "Turing": turing_mma_qualifiers,
    "Ampere": ampere_mma_qualifiers + turing_mma_qualifiers,
    "Ada Lovelace": adalovelace_mma_qualifiers
    + ampere_mma_qualifiers
    + turing_mma_qualifiers,
    "Hopper": ampere_mma_qualifiers + turing_mma_qualifiers,
    "Blackwell": ampere_mma_qualifiers + turing_mma_qualifiers,
    "RTX Blackwell": adalovelace_mma_qualifiers
    + ampere_mma_qualifiers
    + turing_mma_qualifiers,
}


class MMA(MatrixMultiplyAdd):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_mma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for MMA.\n"
            f"Supported architectures: {list(arch_mma_qualifiers.keys())}"
        )
        supported_qualifiers = arch_mma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported MMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.arch = arch
        self.qualifier = qualifier
        shape, d_type, a_type, b_type, c_type = qualifier.split(".")
        self.m, self.n, self.k = nv_shape_to_mnk(shape)
        self.a_type = nv_torch_dtype[a_type]
        self.b_type = nv_torch_dtype[b_type]
        self.c_type = nv_torch_dtype[c_type]
        self.d_type = nv_torch_dtype[d_type]

    def check_input(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        m, n, k = self.m, self.n, self.k
        assert A.shape == (m, k)
        assert B.shape == (k, n)
        assert C.shape == (m, n)
        assert A.dtype == self.a_type
        assert B.dtype == self.b_type
        assert C.dtype == self.c_type

    def dotadd(self, a: list[float], b: list[float], c: float) -> float:
        A = torch.zeros([self.m, self.k], dtype=self.a_type)
        B_T = torch.zeros([self.n, self.k], dtype=self.b_type)
        C = torch.zeros([self.m, self.n], dtype=self.c_type)
        for i in range(len(a)):
            A[0, i] = a[i]
            B_T[0, i] = b[i]
        C[0, 0] = c
        D = self(A, B_T.T, C)
        return D[0, 0].item()
    