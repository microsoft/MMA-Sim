# TPU upstream review branch

Base: Microsoft MMA-Sim main, v2.1.1 (`44b3ac0`, 2026-10-22).

Proposed scope: TPU v5e/v6e/v7 BF16 and TPU v7 E4M3FN/E5M2 prepared instruction
models. Existing GPU source files and package dependencies are unchanged.

Suggested review order:

1. `README.tpu.md`: public API, supported configurations and numerical grouping.
2. `src/mmasim/mmasim/tpu/isa_vmatmul.py`: instruction/load/shape binding.
3. `src/mmasim/mmasim/tpu/arithmetic.py`: BF16 pair merging and FP8 p9 chain.
4. `src/mmasim/mmasim/tpu/_bits.py`: exact integer alignment and FP32 rounding.
5. `tests/tpu/README.md` and tests: native-output provenance and API contracts.

Review decisions before opening the upstream PR:

- Keep the prepared-group API explicit rather than treating an isolated mnemonic
  as sufficient to identify weight state and arithmetic layout.
- Keep nonzero C unsupported until instruction-level accumulator initialization
  is established. Caller-defined FP32 merges remain separate.
- Keep the first version CPU-only and finite-input; performance/vectorization
  and full exceptional-input coverage can follow separately.
- Keep FP8 exponent storage as FP32 carrying p9 arithmetic; E7M8 is a separate
  RTL implementation hypothesis.

No upstream pull request is opened by preparing this branch.

Validation on 2026-11-03: all 15 CPU test methods passed (PyTorch 2.9.0),
including all 17,404 native FP8 observations. Wheel construction and imports
from the built wheel passed; smoke calls exercised all three BF16 generations
and both FP8 formats. Existing GPU hardware equivalence tests were not rerun;
no GPU implementation files changed.
