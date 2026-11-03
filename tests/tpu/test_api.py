"""Public boundary checks and explicit FP32-add policy."""

import unittest
import torch
from mmasim import MMAOperation
from mmasim.tpu import MMA, add_fp32_bits


class APITests(unittest.TestCase):
    def setUp(self):
        self.op = MMA(
            "TPU-v6e",
            "vmatmul.mubr.bf16.gmra",
            weight_load="vmatpush3",
            shape=(16, 128, 128),
        )

    def test_upstream_protocol_and_dtypes(self):
        self.assertIsInstance(self.op, MMAOperation)
        self.assertEqual(self.op.a_type, torch.bfloat16)
        self.assertEqual(self.op.b_type, torch.bfloat16)
        self.assertEqual(self.op.c_type, torch.float32)
        self.assertEqual(self.op.d_type, torch.float32)
        self.assertEqual(
            self.op.dpa(torch.ones(128), torch.ones(128), torch.tensor(0.0)).item(),
            128.0,
        )

    def test_nonrepresentable_or_special_operands(self):
        for x in [0x3F800001, 0x7F800000, 0x7FC00000, -1, 2**32, True]:
            with self.assertRaises(ValueError):
                self.op.dot_bits([x] * 128, [0] * 128)
        with self.assertRaises(ValueError):
            self.op.dot_bits([0] * 127, [0] * 128)
        with self.assertRaises(NotImplementedError):
            self.op.dot_bits([0] * 128, [0] * 128, 0x3F800000)

    def test_tensor_guards_and_no_mutation(self):
        a = torch.ones(self.op.a_shape)
        b = torch.ones(self.op.b_shape)
        old_a, old_b = a.clone(), b.clone()
        self.op(a, b, torch.zeros(self.op.d_shape))
        self.assertTrue(torch.equal(a, old_a))
        self.assertTrue(torch.equal(b, old_b))
        with self.assertRaises(ValueError):
            self.op(a[:-1], b)
        with self.assertRaises(ValueError):
            self.op(a.to(torch.int32), b)
        with self.assertRaises(ValueError):
            self.op(a.to("meta"), b)
        with self.assertRaises(ValueError):
            self.op(a, b, torch.zeros(self.op.d_shape, dtype=torch.bfloat16))
        with self.assertRaises(NotImplementedError):
            self.op(a, b, torch.ones(self.op.d_shape))
        with self.assertRaises(NotImplementedError):
            self.op(a.requires_grad_(), b)
        a64 = old_a.double()
        a64[0, 0] += 2.0**-30
        with self.assertRaises(ValueError):
            self.op(a64, b)

    def test_instruction_guards(self):
        with self.assertRaises(TypeError):
            MMA("TPU-v6e", "vmatmul.mubr.bf16.gmra")
        with self.assertRaises(NotImplementedError):
            MMA("TPU-v8", "vmatmul.bf16.gmra", weight_load="vmatpush")
        with self.assertRaises(NotImplementedError):
            MMA("TPU-v5e", "vmatmul.bf16.gmra.mxu1", weight_load="vmatpush")
        with self.assertRaises(NotImplementedError):
            MMA("TPU-v6e", "vmatmul.mubr.bf16.unknown", weight_load="vmatpush3")

    def test_fp32_add_policy(self):
        # Mathematical RNE ties and the documented input/output FTZ policy.
        self.assertEqual(add_fp32_bits(0x3F800000, 0x33800000), 0x3F800000)
        self.assertEqual(add_fp32_bits(0x3F800001, 0x33800000), 0x3F800002)
        self.assertEqual(add_fp32_bits(0x00000001, 0x00800000), 0x00800000)
        self.assertEqual(add_fp32_bits(0x00800001, 0x80800000), 0)

    def test_matrix_transport_and_independent_streams(self):
        for setup in ("vmatpush3", "vmatpush1+vmatpush2"):
            op = MMA("TPU-v6e", "vmatmul.mubr.bf16.gmra", weight_load=setup)
            # Different row/column patterns and signs; small exact integer dot.
            a = ((torch.arange(op.m) % 3)[:, None] - 1).expand(op.m, op.k).float()
            b = ((torch.arange(op.n) % 4)[None, :] - 2).expand(op.k, op.n).float()
            if op.groups == 2:
                a = torch.stack((a, -2 * a))
                b = torch.stack((b, 3 * b))
            a = a.to(torch.bfloat16)
            b = b.to(torch.bfloat16)
            before_a, before_b = a.clone(), b.clone()
            result = op(a, b, torch.zeros(op.d_shape))
            self.assertEqual(tuple(result.shape), op.d_shape)
            self.assertTrue(
                torch.equal(
                    result.view(torch.int32), (a.float() @ b.float()).view(torch.int32)
                )
            )
            self.assertTrue(torch.equal(a, before_a))
            self.assertTrue(torch.equal(b, before_b))
