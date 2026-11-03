"""Generation-specific prepared instruction tiles and saved v7 hardware witnesses."""

import json
from collections import Counter
from pathlib import Path
import unittest
import torch
from mmasim.tpu import MMA
from mmasim.tpu import add_fp32_bits as add32

BASE = Path(__file__).parent / "trace_data"
OP = "vmatmul.mubr.bf16.gmra"


def make(arch, k):
    load = (
        "vmatpush"
        if arch == "TPU-v5e"
        else (
            "vmatpush3"
            if k == 128
            else "vmatpush1" if arch == "TPU-v7" else "vmatpush1+vmatpush2"
        )
    )
    return MMA(
        arch,
        OP.replace(".mubr", "") if arch == "TPU-v5e" else OP,
        weight_load=load,
        shape=(16, k, k),
    )


class GenerationTests(unittest.TestCase):
    def test_v7_hardware_witnesses(self):
        witnesses = json.loads((BASE / "v7-instruction-measured.json").read_text())
        self.assertEqual(
            Counter(row["shape"][1] for row in witnesses), {128: 12, 256: 16}
        )
        for row in witnesses:
            k = row["shape"][1]
            op = make("TPU-v7", k)
            self.assertEqual(
                op.dot_bits(row["a"], row["b"]),
                row["expected"],
                (row["source"], row["column"]),
            )
        # Check public matrix transport too, using captured bits, not a simulator oracle.
        for k in (128, 256):
            rows = [
                r
                for r in json.loads((BASE / "v7-instruction-measured.json").read_text())
                if r["shape"][1] == k and r["source"].endswith("dense-710.npz")
            ]
            words = (
                lambda v: torch.tensor(v, dtype=torch.int64)
                .to(torch.int32)
                .view(torch.float32)
            )
            a = words(rows[0]["a"]).repeat(16, 1)
            b = words([rows[j % len(rows)]["b"] for j in range(k)]).T.contiguous()
            expected = (
                torch.tensor(
                    [rows[j % len(rows)]["expected"] for j in range(k)],
                    dtype=torch.int64,
                )
                .to(torch.int32)
                .repeat(16, 1)
            )
            self.assertTrue(
                torch.equal(make("TPU-v7", k)(a, b).view(torch.int32), expected)
            )

    def test_v5_saved_schedule(self):
        witnesses = json.loads((BASE / "v5-shape-measured.json").read_text())
        self.assertEqual(sorted(r["k"] for r in witnesses), [512, 1024, 2048])
        for r in witnesses:
            acc = 0
            op = make("TPU-v5e", 128)
            for start in range(0, r["k"], 128):
                acc = add32(
                    acc,
                    op.dot_bits(
                        r["a"][start : start + 128], r["b"][start : start + 128]
                    ),
                )
            expected = r["expected"]
            self.assertEqual(
                acc, int(expected, 16) if isinstance(expected, str) else expected
            )

    def test_shapes_and_exact_matrix_transport(self):
        for arch, k in [
            ("TPU-v5e", 128),
            ("TPU-v6e", 128),
            ("TPU-v6e", 256),
            ("TPU-v7", 128),
            ("TPU-v7", 256),
        ]:
            op = make(arch, k)
            self.assertEqual(op.a_shape, (16, k))
            self.assertEqual(op.groups, 1)
            a = (torch.arange(16) % 3 - 1).float()[:, None].expand(16, k)
            b = (torch.arange(k) % 5 - 2).float()[None, :].expand(k, k)
            self.assertTrue(torch.equal(op(a, b), a @ b))
        narrow = make("TPU-v6e", 128)
        dual = MMA("TPU-v6e", OP, weight_load="vmatpush3")
        r = json.loads((BASE / "v7-instruction-measured.json").read_text())[0]
        self.assertEqual(narrow.dot_bits(r["a"], r["b"]), dual.dot_bits(r["a"], r["b"]))

    def test_grouping_and_instruction_spelling(self):
        v7 = make("TPU-v7", 256)
        self.assertEqual(
            v7.reduction_groups[0],
            ((0, 1, 2, 3, 128, 129, 130, 131), (4, 5, 6, 7, 132, 133, 134, 135)),
        )
        self.assertEqual(len(make("TPU-v6e", 256).reduction_groups[0]), 4)
        op = MMA(
            "TPU-v7",
            "vmatmul.mubr.bf16.vlgmr.msra.gmra.mrb[0].mxu0",
            weight_load="vmatpush1",
            shape=(16, 256, 256),
        )
        self.assertEqual(op.mxu, "mxu0")
        with self.assertRaises(ValueError):
            MMA("TPU-v7", OP, weight_load="vmatpush1+vmatpush2")
        with self.assertRaises(ValueError):
            MMA("TPU-v6e", OP, weight_load="vmatpush3", shape=(16, 256, 256))
        with self.assertRaises(ValueError):
            MMA(
                "TPU-v5e",
                "vmatmul.bf16.gmra",
                weight_load="vmatpush",
                shape=(16, 256, 256),
            )
