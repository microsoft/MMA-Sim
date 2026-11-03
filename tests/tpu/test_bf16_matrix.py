import unittest
import json
from pathlib import Path
import torch
from mmasim.tpu import MMA, add_fp32_bits as add32

BASE = Path(__file__).parent


def instruction(setup):
    return MMA("TPU-v6e", "vmatmul.mubr.bf16.gmra", weight_load=setup)


class MeasuredMatrixTests(unittest.TestCase):
    def test_captured_shapes_matrix_outputs(self):
        fixture = json.loads((BASE / "v6-instruction-measured.json").read_text())
        for row in fixture["cases"]:
            _, k, n = row["shape"]
            bank = fixture["banks"][str(k)]
            a = (
                torch.tensor(bank["a"], dtype=torch.int64)
                .to(torch.int32)
                .view(torch.float32)[row["row_index"]]
            )
            b = (
                torch.tensor(bank["b"], dtype=torch.int64)
                .to(torch.int32)
                .view(torch.float32)[row["col_index"]]
                .T
            )
            op = instruction(row["weight_load"])
            if op.groups == 2 and k == 128:
                result = op(a.reshape(2, 16, 128), torch.stack((b, b))).reshape(32, 128)
            elif op.groups == 2:
                partial = op(
                    torch.stack((a[:, :128], a[:, 128:])),
                    torch.stack((b[:128], b[128:])),
                )
                words = (
                    partial.view(torch.int32)
                    .to(torch.int64)
                    .bitwise_and(0xFFFFFFFF)
                    .reshape(2, -1)
                    .tolist()
                )
                result = (
                    torch.tensor(
                        [add32(x, y) for x, y in zip(*words)], dtype=torch.int64
                    )
                    .to(torch.int32)
                    .view(torch.float32)
                    .reshape(16, 128)
                )
            else:
                # Padding is a compiler action; the wide instruction always sees N256.
                padded_b = torch.zeros(256, 256)
                padded_b[:, :n] = b
                result = op(a, padded_b)[:, :n]
            expected = torch.tensor(row["expected"], dtype=torch.int64).to(torch.int32)
            self.assertTrue(
                torch.equal(result.contiguous().view(torch.int32), expected),
                row["source"],
            )
