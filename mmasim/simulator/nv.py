import torch

from .utils import truncate_to_tf32, nv_fused_dot_add
from ..isa import NV_MMABase, NV_WGMMABase, NV_TCGen05MMABase


volta_mma_qualifiers = [
    # sm_70 f16
    "m8n8k4.f16.f16.f16.f16",
    "m8n8k4.f32.f16.f16.f16",
    "m8n8k4.f32.f16.f16.f32",
]
turing_mma_qualifiers = [
    # sm_75 f16
    "m16n8k8.f16.f16.f16.f16",
    "m16n8k8.f32.f16.f16.f32",
]
ampere_mma_qualifiers = turing_mma_qualifiers + [
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
adalovelace_mma_qualifiers = ampere_mma_qualifiers + [
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
hopper_mma_qualifiers = ampere_mma_qualifiers
blackwell_mma_qualifiers = ampere_mma_qualifiers
rtxblackwell_mma_qualifiers = adalovelace_mma_qualifiers

hopper_wgmma_qualifiers = (
    # sm_90a tf32
    [f"m64n{N}k8.f32.tf32.tf32" for N in range(8, 256 + 1, 8)]
    # sm_90a f16
    + [f"m64n{N}k16.f16.f16.f16" for N in range(8, 256 + 1, 8)]
    + [f"m64n{N}k16.f32.f16.f16" for N in range(8, 256 + 1, 8)]
    # sm_90a bf16
    + [f"m64n{N}k16.f32.bf16.bf16" for N in range(8, 256 + 1, 8)]
    # sm_90a fp8
    + [
        f"m64n{N}k32.{dtype}.{atype}.{btype}"
        for N in range(8, 256 + 1, 8)
        for dtype in ["f32", "f16"]
        for atype in ["e5m2", "e4m3"]
        for btype in ["e5m2", "e4m3"]
    ]
)

blackwell_tcgen05mma_qualifiers = (
    # sm_100a tf32
    [f"m{M}n{N}k8.f32.tf32.tf32" for M in {64, 128} for N in range(8, 256 + 1, 8)]
    # sm_100a f16 or bf16
    + [f"m{M}n{N}k16.f16.f16.f16" for M in {64, 128} for N in range(8, 256 + 1, 8)]
    + [
        f"m{M}n{N}k16.f32.{atype}.{btype}"
        for M in {64, 128}
        for N in range(8, 256 + 1, 8)
        for atype in ["f16", "bf16"]
        for btype in ["f16", "bf16"]
    ]
    # sm_100a fp8
    + [
        f"m{M}n{N}k32.{dtype}.{atype}.{btype}"
        for M in {64, 128}
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
    "Blackwell": blackwell_mma_qualifiers,
    "RTX Blackwell": rtxblackwell_mma_qualifiers,
}

arch_wgmma_qualifiers = {
    "Hopper": hopper_wgmma_qualifiers,
}


arch_accum_fraction_bits = {
    "Volta": 23,
    "Turing": 24,
    "Ampere": 24,
    "Ada Lovelace": 24,
    "Hopper": 25,
    "Blackwell": 25,
    "RTX Blackwell": 25,
}


class MMA(NV_MMABase):
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
        NV_MMABase.__init__(self, qualifier)

        self.n_accum_fraction_bits = arch_accum_fraction_bits[arch]
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"
        if self.arch == "Ada Lovelace" and self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            self.n_accum_fraction_bits = 13
            if self.output_type == "f32":
                self.output_type = "f32_e8m13"
        self.is_split_k = self.arch in [
            "Ampere",
            "Ada Lovelace",
        ] and self.qualifier in [
            # tf32 k8
            "m16n8k8.f32.tf32.tf32.f32",
            # f16 k16
            "m16n8k16.f16.f16.f16.f16",
            "m16n8k16.f32.f16.f16.f32",
            # bf16 k16
            "m16n8k16.f32.bf16.bf16.f32",
            # fp8 k32
            "m16n8k32.f32.e5m2.e5m2.f32",
            "m16n8k32.f32.e5m2.e4m3.f32",
            "m16n8k32.f32.e4m3.e5m2.f32",
            "m16n8k32.f32.e4m3.e4m3.f32",
            "m16n8k32.f16.e5m2.e5m2.f16",
            "m16n8k32.f16.e5m2.e4m3.f16",
            "m16n8k32.f16.e4m3.e5m2.f16",
            "m16n8k32.f16.e4m3.e4m3.f16",
        ]

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert c.shape == (m, n)
        assert a.dtype == self.a_type
        assert b.dtype == self.b_type
        assert c.dtype == self.c_type
        a = a.cpu()
        b = b.cpu()
        c = c.cpu()
        d = torch.zeros((m, n), dtype=self.d_type)
        if self.a_type == torch.float32:  # tf32
            a = truncate_to_tf32(a)
            b = truncate_to_tf32(b)
        for i in range(m):
            for j in range(n):
                sum = c[i, j]
                if self.is_split_k:
                    sum = nv_fused_dot_add(
                        a[i, : k // 2],
                        b[: k // 2, j],
                        sum,
                        n_fraction_bits=self.n_accum_fraction_bits,
                        output_type=self.output_type,
                    )
                    sum = nv_fused_dot_add(
                        a[i, k // 2 : k],
                        b[k // 2 : k, j],
                        sum,
                        n_fraction_bits=self.n_accum_fraction_bits,
                        output_type=self.output_type,
                    )
                else:
                    sum = nv_fused_dot_add(
                        a[i, :],
                        b[:, j],
                        sum,
                        n_fraction_bits=self.n_accum_fraction_bits,
                        output_type=self.output_type,
                    )
                d[i][j] = sum
        return d


class WGMMA(NV_WGMMABase):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_wgmma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for WGMMA.\n"
            f"Supported architectures: {list(arch_wgmma_qualifiers.keys())}"
        )
        self.arch = arch
        supported_qualifiers = arch_wgmma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported WGMMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.qualifier = qualifier
        NV_WGMMABase.__init__(self, qualifier)

        self.n_accum_fraction_bits = arch_accum_fraction_bits[arch]
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"
        if self.a_type in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]:
            self.n_accum_fraction_bits = 13
            if self.output_type == "f32":
                self.output_type = "f32_e8m13"

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert c.shape == (m, n)
        assert a.dtype == self.a_type
        assert b.dtype == self.b_type
        assert c.dtype == self.c_type
        a = a.cpu()
        b = b.cpu()
        c = c.cpu()
        d = torch.zeros((m, n), dtype=self.d_type)
        if self.a_type == torch.float32:  # tf32
            a = truncate_to_tf32(a)
            b = truncate_to_tf32(b)
        for i in range(m):
            for j in range(n):
                sum = c[i, j]
                sum = nv_fused_dot_add(
                    a[i, :],
                    b[:, j],
                    sum,
                    n_fraction_bits=self.n_accum_fraction_bits,
                    output_type=self.output_type,
                )
                d[i][j] = sum
        return d


class TCGen05MMA(NV_TCGen05MMABase):
    def __init__(self, arch: str, qualifier: str):
        assert arch in arch_wgmma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for TCGen05MMA.\n"
            f"Supported architectures: {list(arch_wgmma_qualifiers.keys())}"
        )
        self.arch = arch
        supported_qualifiers = arch_wgmma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported TCGen05MMA qualifier {qualifier} for {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        self.qualifier = qualifier
        NV_TCGen05MMABase.__init__(self, qualifier)

        self.n_accum_fraction_bits = arch_accum_fraction_bits[arch]
        self.output_type = "f32" if self.d_type == torch.float32 else "f16"

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        m, n, k = self.m, self.n, self.k
        assert a.shape == (m, k)
        assert b.shape == (k, n)
        assert c.shape == (m, n)
        assert a.dtype == self.a_type
        assert b.dtype == self.b_type
        assert c.dtype == self.c_type
        a = a.cpu()
        b = b.cpu()
        c = c.cpu()
        d = torch.zeros((m, n), dtype=self.d_type)
        if self.a_type == torch.float32:  # tf32
            a = truncate_to_tf32(a)
            b = truncate_to_tf32(b)
        for i in range(m):
            for j in range(n):
                sum = c[i, j]
                sum = nv_fused_dot_add(
                    a[i, :],
                    b[:, j],
                    sum,
                    n_fraction_bits=self.n_accum_fraction_bits,
                    output_type=self.output_type,
                )
                d[i][j] = sum
        return d
