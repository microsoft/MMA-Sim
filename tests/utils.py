import torch

from mmasim.probe.intrinsic import Intrinsic
from mmasim.isa import MMAInstructionBase

dtype_n_bits = {
    torch.float32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float8_e5m2: 8,
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2fnuz: 8,
    torch.float8_e4m3fnuz: 8,
}

dtype_storage_type = {
    torch.float32: torch.uint32,
    torch.float16: torch.uint16,
    torch.bfloat16: torch.uint16,
    torch.float8_e5m2: torch.uint8,
    torch.float8_e4m3fn: torch.uint8,
    torch.float8_e5m2fnuz: torch.uint8,
    torch.float8_e4m3fnuz: torch.uint8,
}


def random_test(
    sim: MMAInstructionBase,
    real: Intrinsic,
    allow_different_nan: bool,
    trials: int,
):
    m, n, k = sim.m, sim.n, sim.k
    a_type = sim.a_type
    b_type = sim.b_type
    c_type = sim.c_type
    d_type = sim.d_type
    for _ in range(trials):
        A = torch.randint(
            0, 2 ** dtype_n_bits[a_type], (m, k), dtype=dtype_storage_type[a_type]
        ).view(a_type)
        B = torch.randint(
            0, 2 ** dtype_n_bits[b_type], (k, n), dtype=dtype_storage_type[b_type]
        ).view(b_type)
        C = torch.randint(
            0, 2 ** dtype_n_bits[c_type], (m, n), dtype=dtype_storage_type[c_type]
        ).view(c_type)
        D_sim = sim(A, B, C)
        D_sim_raw = D_sim.view(dtype_storage_type[d_type])
        D_real = real(A, B, C).cpu()
        D_real_raw = D_real.view(dtype_storage_type[d_type])

        is_different = D_sim_raw != D_real_raw
        if allow_different_nan:
            is_different &= ~D_real.isnan()
        if is_different.any():
            idx = is_different.nonzero()[0]
            i, j = idx
            a = A[i, :].tolist()
            b = B[:, j].tolist()
            c = float(C[i, j].item())
            d_sim = float(D_sim[i, j].item())
            d_real = float(D_real[i, j].item())
            print(f"Different results at ({i}, {j}):")
            print(f"    D_real[i, j] = {d_real.hex()}")
            print(f"    D_sim[i, j] = {d_sim.hex()}")
            print(f"    A[i, :] = {[x.hex() for x in a]}")
            print(f"    B[:, j] = {[x.hex() for x in b]}")
            print(f"    C[i, j] = {c.hex()}")
            print(
                f"    A[i, :] * B[:, j] = {[(a[i]*b[i]).hex() for i in range(len(a))]}"
            )
            raise Exception("Test failed")
