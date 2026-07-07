import torch

from mmasim import MMAOperation, MMABlockScaleOperation
from mmasim_kernels import MMAKernel, MMABlockScaleKernel

storage_type = {8: torch.uint64, 4: torch.uint32, 2: torch.uint16, 1: torch.uint8}


def random_test(
    sim: MMAOperation | MMABlockScaleOperation,
    kernel: MMAKernel | MMABlockScaleKernel,
    allow_different_nan: bool,
    trials: int,
):
    m, n, k = kernel.m, kernel.n, kernel.k
    a_type = kernel.a_type
    b_type = kernel.b_type
    c_type = kernel.c_type
    d_type = kernel.d_type
    has_block_scale = False
    packing = 1
    if isinstance(kernel, MMABlockScaleKernel):
        has_block_scale = kernel.block_size > 0
        packing = kernel.packing
    for _ in range(trials):
        A = (
            torch.randint(
                -(2**31),
                2**31,
                [m * k // packing * a_type.itemsize // 4],
                dtype=torch.int32,
                device="cuda",
            )
            .view(a_type)
            .view(m, k // packing)
        )
        B_T = (
            torch.randint(
                -(2**31),
                2**31,
                [n * k // packing * b_type.itemsize // 4],
                dtype=torch.int32,
                device="cuda",
            )
            .view(b_type)
            .view(n, k // packing)
        )
        B = B_T.T
        C = (
            torch.randint(
                -(2**31),
                2**31,
                [m * n * c_type.itemsize // 4],
                dtype=torch.int32,
                device="cuda",
            )
            .view(c_type)
            .view(m, n)
        )
        if not has_block_scale:
            assert isinstance(sim, MMAOperation)
            assert isinstance(kernel, MMAOperation)
            D_gpu = kernel(A, B, C).cpu()
            D_sim = sim(A, B, C).cpu()
        else:
            assert isinstance(sim, MMABlockScaleOperation)
            assert isinstance(kernel, MMABlockScaleKernel)
            s_type, block_size = kernel.s_type, kernel.block_size
            scale_A = (
                torch.randint(
                    -(2**31),
                    2**31,
                    [m * k // block_size * s_type.itemsize // 4],
                    dtype=torch.int32,
                    device="cuda",
                )
                .view(s_type)
                .view(m, k // block_size)
            )
            scale_B_T = (
                torch.randint(
                    -(2**31),
                    2**31,
                    [n * k // block_size * s_type.itemsize // 4],
                    dtype=torch.int32,
                    device="cuda",
                )
                .view(s_type)
                .view(n, k // block_size)
            )
            scale_B = scale_B_T.T
            D_gpu = kernel(A, B, C, scale_A, scale_B).cpu()
            D_sim = sim(A, B, C, scale_A, scale_B).cpu()
        D_sim_raw = D_sim.view(storage_type[d_type.itemsize])
        D_gpu_raw = D_gpu.view(storage_type[d_type.itemsize])
        is_different = D_sim_raw != D_gpu_raw
        if allow_different_nan:
            is_different &= ~D_gpu.isnan()
        if is_different.any():
            idx = is_different.nonzero()[0]
            i, j = idx
            print(f"Different results at ({i}, {j}):")
            print(
                f"    D_gpu[i, j] = {float(D_gpu[i,j].item()).hex()}, {D_gpu[i, j].view(storage_type[d_type.itemsize])}"
            )
            print(
                f"    D_sim[i, j] = {float(D_sim[i,j].item()).hex()}, {D_sim[i, j].view(storage_type[d_type.itemsize])}"
            )
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
