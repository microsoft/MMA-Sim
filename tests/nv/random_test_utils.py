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
def test_mma(
    mma_sim: mmasim.nv.TensorCoreInstruction, mma, trials: int, is_wgmma: bool = False
):
    m, n, k = mmasim.nv.shape_to_mnk(mma_sim.shape)
    a_type = mma_sim.a_type
    b_type = mma_sim.b_type
    c_type = mma_sim.c_type
    d_type = mma_sim.d_type
    for _ in range(trials):
        a = (
            torch.randint(0, 2 ** n_bits[a_type], (m, k), dtype=storage_type[a_type])
            .view(mmasim.nv.torch_dtype[a_type])
            .cuda()
        )
        b_t = (
            torch.randint(0, 2 ** n_bits[b_type], (n, k), dtype=storage_type[b_type])
            .view(mmasim.nv.torch_dtype[b_type])
            .cuda()
        )
        b = b_t.T
        c = (
            torch.randint(0, 2 ** n_bits[c_type], (m, n), dtype=storage_type[c_type])
            .view(mmasim.nv.torch_dtype[c_type])
            .cuda()
        )
        if not is_wgmma:
            d = torch.empty([m, n], dtype=mmasim.nv.torch_dtype[d_type]).cuda()
            mma(d.data_ptr(), a.data_ptr(), b.data_ptr(), c.data_ptr())
        else:
            d = c.clone()
            mma(d.data_ptr(), a.data_ptr(), b.data_ptr())
        d_raw = d.view(storage_type[d_type])

        d_sim = mma_sim(a.cpu(), b.cpu(), c.cpu())
        d_sim_raw = d_sim.view(storage_type[d_type])

        is_different = d_raw.cpu() != d_sim_raw
        if is_different.any():
            idx = is_different.nonzero()[0]
            i, j = idx
            A = a[i, :].tolist()
            B = b[:, j].tolist()
            C = c[i, j].item()
            print([x.hex() for x in A])
            print([x.hex() for x in B])
            print(C.hex())
            print([(A[i] * B[i]).hex() for i in range(len(A))])
            print(d[i, j].item().hex())
            print(d_sim[i, j].item().hex())
            raise Exception(
                f"Different results at ({i}, {j}): "
                f"Tensor Core: {hex(d_raw[i, j].item())}, "
                f"Simulation: {hex(d_sim_raw[i, j].item())}"
            )
