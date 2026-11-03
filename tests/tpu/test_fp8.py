"""Native K256 FP8 observations and instruction API contracts."""

import unittest
from pathlib import Path
import numpy as np
import torch
from mmasim.tpu import MMA


def make(fmt):
    kind = "f8e4m3fn" if fmt == "e4m3" else "f8e5m2"
    return MMA(
        "TPU-v7",
        f"vmatmul.mubr.{kind}.vlgmr.msra.gmra.mrb[0].mxu0",
        weight_load="vmatpush1",
    )


def words(values):
    return torch.tensor(values, dtype=torch.int64).to(torch.int32).view(torch.float32)


class FP8Tests(unittest.TestCase):
    def test_all_native_observations(self):
        with np.load(
            (Path(__file__).parent / "trace_data" / "v7-fp8-measured.npz"),
            allow_pickle=False,
        ) as data:
            for fmt, count in [("e4m3", 7740), ("e5m2", 9664)]:
                op = make(fmt)
                self.assertEqual(len(data[f"{fmt}_expected"]), count)
                for a, b, y, label in zip(
                    data[f"{fmt}_a"],
                    data[f"{fmt}_b"],
                    data[f"{fmt}_expected"],
                    data[f"{fmt}_source"],
                ):
                    self.assertEqual(
                        op.dot_bits(a.tolist(), b.tolist()), int(y), (fmt, label)
                    )

    def test_matrix_transport_native_rows_and_padding(self):
        with np.load(
            (Path(__file__).parent / "trace_data" / "v7-fp8-measured.npz"),
            allow_pickle=False,
        ) as data:
            for fmt in ["e4m3", "e5m2"]:
                op = make(fmt)
                # One native observation has 16 repeated rows and columns.
                # Replaying those rows validates observed outputs; inactive rows are zero.
                index = next(
                    i for i, s in enumerate(data[f"{fmt}_source"]) if "dense" in s
                )
                a = torch.zeros(op.a_shape)
                a[:16] = words(data[f"{fmt}_a"][index].tolist()).repeat(16, 1)
                b = words(data[f"{fmt}_b"][index].tolist())[:, None].expand(256, 256)
                out = op(a, b)
                self.assertTrue(
                    torch.all(
                        out[:16]
                        .view(torch.int32)
                        .to(torch.int64)
                        .bitwise_and(0xFFFFFFFF)
                        == int(data[f"{fmt}_expected"][index])
                    )
                )
                self.assertTrue(torch.all(out[16:] == 0))
                self.assertEqual(
                    op.reduction_groups[0], ((0, 1, 2, 3, 128, 129, 130, 131),)
                )

    def test_independent_rows_and_fp8_storage(self):
        for fmt, dtype in [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]:
            op = make(fmt)
            a = (torch.arange(32) % 3 - 1).float()[:, None].expand(32, 256).to(dtype)
            b = (torch.arange(256) % 3 - 1).float()[None, :].expand(256, 256).to(dtype)
            self.assertTrue(torch.equal(op(a, b), (a.float() @ b.float())))

    def test_reject_unsupported(self):
        op = make("e4m3")
        for value in [0x7F800000, 0x7FC00000, 0x3F800001, 0x44000000]:
            with self.assertRaises(ValueError):
                op.dot_bits([value] * 256, [0] * 256)
        with self.assertRaises(ValueError):
            op.dot_bits([0] * 128, [0] * 128)
        with self.assertRaises(NotImplementedError):
            op.dot_bits([0] * 256, [0] * 256, 0x3F800000)
        with self.assertRaises(ValueError):
            MMA("TPU-v7", "vmatmul.mubr.f8e5m2.gmra", weight_load="vmatpush3")
        with self.assertRaises(ValueError):
            MMA(
                "TPU-v7",
                "vmatmul.mubr.f8e5m2.gmra",
                weight_load="vmatpush1",
                shape=(16, 512, 256),
            )
