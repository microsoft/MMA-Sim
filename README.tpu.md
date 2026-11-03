# TPU instruction-group numerical models

The TPU backend provides BF16 models for TPU v5e, v6e and v7, and E4M3FN/E5M2
models for TPU v7. It follows the architecture/instruction constructor and
PyTorch callable interface used by the GPU backends.

A call evaluates a **prepared computation group**: weights are already loaded,
MUBR preparation is included where applicable, and the result accumulator is
initially zero. The instruction spelling is captured LLO text. The additional
weight-load mode identifies the arithmetic layout; it is not an invented
assembly qualifier.

```python
import torch
from mmasim.tpu import MMA

bf16 = MMA("TPU-v6e", "vmatmul.mubr.bf16.gmra",
           weight_load="vmatpush1+vmatpush2")
A = torch.ones(bf16.a_shape, dtype=torch.bfloat16)
B = torch.ones(bf16.b_shape, dtype=torch.bfloat16)
D = bf16(A, B)  # FP32 [16,256], each entry 256

fp8 = MMA("TPU-v7", "vmatmul.mubr.f8e4m3fn.vlgmr.msra.gmra.mrb[0].mxu0",
          weight_load="vmatpush1")
A8 = torch.ones(fp8.a_shape, dtype=torch.float8_e4m3fn)
B8 = torch.ones(fp8.b_shape, dtype=torch.float8_e4m3fn)
D8 = fp8(A8, B8)  # FP32 [32,256] carrying the p9 result
```

## Supported configurations

Shape tuples below use **(M,K,N)**; GPU mnemonic suffixes conventionally use
M,N,K. `a_shape`, `b_shape`, `d_shape`, `m`, `n`, `k`, and dtype attributes are
available on each object.

| Architecture | Input | Prepared weight load | Logical shape |
|---|---|---|---|
| TPU-v5e | BF16 | `vmatpush` | (16,128,128) |
| TPU-v6e | BF16 | `vmatpush3` | Two independent (16,128,128) streams |
| TPU-v6e | BF16 | `vmatpush1+vmatpush2` | (16,256,256) |
| TPU-v7 | BF16 | `vmatpush3` | Two independent (16,128,128) streams |
| TPU-v7 | BF16 | `vmatpush1` | (16,256,256) |
| TPU-v7 | E4M3FN / E5M2 | `vmatpush1` | (32,256,256) |

For `vmatpush3`, pass `shape=(16,128,128)` to select one logical stream;
otherwise A, B and D have a leading dimension of two. Callers decide whether
those streams represent different output tiles or K partitions requiring a
subsequent FP32 merge. Inactive rows/columns are supplied as zeros and sliced
by the caller. Supported spellings and shape checks are in `isa_vmatmul.py`.

## Arithmetic

`dot_bits(a,b,c=0)` accepts unsigned FP32 encodings of already-quantized input
values and returns one unsigned FP32 result encoding. `dpa(a,b,c)` is the
single-output, one-dimensional tensor version. These methods evaluate the
complete K128/K256 output dot, not a standalone physical PE instruction.

- **v5e BF16:** DP4 Emax−26 alignment with the measured tail rule; two rounded
  FP32 DP4 results are merged, then added to the running FP32 result.
- **v6e BF16:** K256 groups combine DP4(k0:3) with DP4(k128:131), and DP4(k4:7)
  with DP4(k132:135); merge the pair results, then update the FP32 accumulator.
  The independent K128 streams use the v5e-style DP4 pairing.
- **v7 BF16:** each cross-half DP8 uses Emax−27 alignment without the v5e tail
  rule and rounds to FP32. Two DP8 results merge before the running FP32 add.
  K128 with inactive cross-half terms reduces to the corresponding DP4 pairs.
- **v7 FP8:** each DP8 contains four terms from each K128 half. Products align
  with RTZ at Emax−8; each partial normalizes with RTZ to nine significant bits
  (p9). The 32 partials accumulate serially with p9 RNE. The result is stored
  in FP32. E7M8 physical encoding and shared BF16/FP8 circuitry are not required
  by this numerical model.

BF16 FP32 additions use RNE with input/output flush-to-zero. The explicit
`add_fp32_bits(x,y)` helper exposes that policy for caller-defined compositions.
It does not select a compiler reduction order or assert independent exhaustive
hardware validation of every MRB and vector-add exceptional-value behavior.
FP8 array-local p9 additions are distinct from these external FP32 additions.

## API scope

The API accepts finite, already-quantized **CPU** tensors, including
FP32 tensors whose values are exactly representable in the selected input
format. It rejects nonrepresentable inputs, NaN/Inf operands, nonzero C and
autograd tensors. Subnormal BF16 operands follow the stated FTZ policy.
C is optional; when supplied, it must be a zero FP32 tensor of output shape.

The simulator models arithmetic, not cycle timing or persistent load queues.
Outer K reductions, explicit vector additions, padding and MXU assignment are
caller operations. Instruction groups are evaluated directly, without HLO profile lookup or
LLO/binary execution.

## Validation

Run from the repository root with Python 3.10+ and upstream's dependencies:

```sh
pip install -e src/mmasim numpy
python -m unittest discover -s tests/tpu -v
```

Tests use saved real-hardware outputs, plus API rejection tests and small exact
integer controls. The FP8 fixture contains all **17,404** K256 observations
(7,740 E4M3FN and 9,664 E5M2). Its extractor verifies every spatial output and
repeat is identical before compacting a probe to one dot result. Captured FP8
probes have 16 active repeated rows; tests replay those rows through the M32
interface with inactive rows padded. Independent M32 rows are additionally
checked with exact integer controls, not presented as new hardware captures.

BF16 fixtures cover all three generations, v6e matrix transport and explicit
composition of saved v5e longer-K results. See [fixture provenance](tests/tpu/README.md).
