import torch

import mmasim


n_bits = {
    "f16": 16,
    "f32": 32,
    "bf16": 16,
    "tf32": 32,
    "e5m2": 8,
    "e4m3": 8,
}

storage_type = {
    "f16": torch.uint16,
    "f32": torch.uint32,
    "bf16": torch.uint16,
    "tf32": torch.uint32,
    "e5m2": torch.uint8,
    "e4m3": torch.uint8,
}


@torch.inference_mode()
def test_mfma(mfma_sim: mmasim.amd.MatrixCoreInstruction, mfma, trials: int):
    m, n, k = mmasim.amd.shape_to_mnk(mfma_sim.shape)
    a_type = mfma_sim.a_type
    b_type = mfma_sim.b_type
    c_type = mfma_sim.c_type
    d_type = mfma_sim.d_type
    for _ in range(trials):
        a = (
            torch.randint(0, 2 ** n_bits[a_type], (m, k), dtype=storage_type[a_type])
            .view(mmasim.amd.torch_dtype[a_type])
            .cuda()
        )
        b = (
            torch.randint(0, 2 ** n_bits[b_type], (k, n), dtype=storage_type[b_type])
            .view(mmasim.amd.torch_dtype[b_type])
            .cuda()
        )
        c = (
            torch.randint(0, 2 ** n_bits[c_type], (m, n), dtype=storage_type[c_type])
            .view(mmasim.amd.torch_dtype[c_type])
            .cuda()
        )
        d = torch.empty([m, n], dtype=mmasim.amd.torch_dtype[d_type]).cuda()
        mfma(d.data_ptr(), a.data_ptr(), b.data_ptr(), c.data_ptr())
        d_raw = d.view(storage_type[d_type])

        d_sim = mfma_sim(a.cpu(), b.cpu(), c.cpu())
        d_sim_raw = d_sim.view(storage_type[d_type])

        is_different = (d_raw.cpu() != d_sim_raw) & ~d_sim.isnan()
        if is_different.any():
            idx = is_different.nonzero()[0]
            i, j = idx
            A = a[i, :].tolist()
            B = b[:, j].tolist()
            C = c[i, j].item()
            print("a[i,:] =", [x.hex() for x in A])
            print("b[:,j] =", [x.hex() for x in B])
            print("c[i,j] =", C.hex())
            print("a[i,:]*b[:,j] =", [(A[i] * B[i]).hex() for i in range(len(A))])
            print("d[i,j] =", d[i, j].item().hex())
            print("d_sim[i,j] =", d_sim[i, j].item().hex())
            raise Exception(
                f"Different results at ({i}, {j}): "
                f"Matrix Core: {hex(d_raw[i, j].item())}, "
                f"Simulation: {hex(d_sim_raw[i, j].item())}"
            )
