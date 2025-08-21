import torch

from mmasim.isa.common import MatrixMultiplyAdd, MatrixMultiplyAddWithBlockScale

storage_type = {4: torch.uint32, 2: torch.uint16, 1: torch.uint8}


def random_test(
    sim: MatrixMultiplyAdd | MatrixMultiplyAddWithBlockScale,
    real: MatrixMultiplyAdd | MatrixMultiplyAddWithBlockScale,
    allow_different_nan: bool,
    trials: int,
):
    m, n, k = sim.m, sim.n, sim.k
    a_type = sim.a_type
    b_type = sim.b_type
    c_type = sim.c_type
    d_type = sim.d_type
    has_block_scale = False
    packing = 1
    if isinstance(sim, MatrixMultiplyAddWithBlockScale):
        has_block_scale = sim.block_size > 0
        packing = sim.packing
    for _ in range(trials):
        A = torch.randint(
            0,
            256**a_type.itemsize,
            (m, k // packing),
            dtype=storage_type[a_type.itemsize],
        ).view(a_type)
        B_T = torch.randint(
            0,
            256**b_type.itemsize,
            (n, k // packing),
            dtype=storage_type[b_type.itemsize],
        ).view(b_type)
        B = B_T.T
        C = torch.randint(
            0, 256**c_type.itemsize, (m, n), dtype=storage_type[c_type.itemsize]
        ).view(c_type)
        if has_block_scale:
            assert isinstance(sim, MatrixMultiplyAddWithBlockScale)
            assert isinstance(real, MatrixMultiplyAddWithBlockScale)
            s_type, block_size = sim.s_type, sim.block_size
            scale_A = torch.randint(
                0,
                256**s_type.itemsize,
                (m, k // block_size),
                dtype=storage_type[s_type.itemsize],
            ).view(s_type)
            scale_B_T = torch.randint(
                0,
                256**s_type.itemsize,
                (n, k // block_size),
                dtype=storage_type[s_type.itemsize],
            ).view(s_type)
            scale_B = scale_B_T.T
            D_sim = sim(A, B, C, scale_A, scale_B)
            D_real = real(A, B, C, scale_A, scale_B).cpu()
        else:
            assert isinstance(sim, MatrixMultiplyAdd)
            assert isinstance(real, MatrixMultiplyAdd)
            D_sim = sim(A, B, C)
            D_real = real(A, B, C).cpu()
        D_sim_raw = D_sim.view(storage_type[d_type.itemsize])
        D_real_raw = D_real.view(storage_type[d_type.itemsize])
        is_different = D_sim_raw != D_real_raw
        if allow_different_nan:
            is_different &= ~D_real.isnan()
        if is_different.any():
            idx = is_different.nonzero()[0]
            i, j = idx
            print(f"Different results at ({i}, {j}):")
            print(f"    D_real[i, j] = {float(D_real[i,j].item()).hex()}")
            print(f"    D_sim[i, j] = {float(D_sim[i,j].item()).hex()}")
            print(f"    A_raw[i, :] = {A[i, :].view(storage_type[a_type.itemsize])}")
            print(f"    B_raw[:, j] = {B[:, j].view(storage_type[b_type.itemsize])}")
            print(f"    C_raw[i, j] = {C[i, j].view(storage_type[c_type.itemsize])}")
            if has_block_scale:
                print(
                    f"    scale_A[i, :] = {scale_A[i, :].view(storage_type[s_type.itemsize])}"  # type: ignore
                )
                print(
                    f"    scale_B[:, j] = {scale_B[:, j].view(storage_type[s_type.itemsize])}"  # type: ignore
                )
            raise Exception("Test failed")
