# Hardware fixtures

Expected values in these fixtures are saved device outputs, not values generated
by the simulator under test. All operand words are FP32 encodings of exact BF16
or FP8 values; this avoids a dependency on host FP8 serialization conventions.

| Fixture | Origin and use |
|---|---|
| `v5-shape-measured.json` | Preserved v5e shape-boundary dot outputs; complete K128 multiples are replayed with explicit outer FP32 additions. |
| `v6-instruction-measured.json` | v6e schedule-boundary capture, M/K/N 128/128/128, 128/256/128, 128/256/256 and 257/256/128. Input banks and row/column indices reconstruct 12,288 saved output entries. |
| `v7-instruction-measured.json` | 28 v7 BF16 dot witnesses from native M16 K128/K256 captures: dense, cross-cancellation and tile-boundary inputs. |
| `v7-fp8-measured.npz` | 17,404 v7 native M16 K256 N256 observations from `A-{e4m3,e5m2}-jax-16-256-256-core/batch-*.npz`. |

The repeated diagnostic rows/columns are preserved or compacted losslessly;
they are not counted as independent random trials. FP8 source labels retain the
batch filename, original probe ID and name. Extraction checks all three repeats
and all spatial output values before retaining a scalar expected result.

`build_v6_instruction_fixture.py`, `build_v7_instruction_fixture.py` and
`build_fp8_fixture.py` regenerate their fixtures from the corresponding native
capture directories. The tests themselves need only the committed fixtures.
The v5 fixture is the frozen selection from the earlier shape corpus.

The full FP8 file exercises sparse rounding/cancellation/exponent probes and
dense inputs. Matrix tests additionally exercise public tensor transport;
small-integer tests isolate row/column layout from numerical rounding.
