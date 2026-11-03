"""Extract uniform native FP8 instruction observations; never generate expected values.

Usage: python tests/tpu/build_fp8_fixture.py /path/to/v7-fp8-final
"""

import argparse
from pathlib import Path
import numpy as np


def build(source, destination):
    arrays = {}
    for fmt, expected_count in [("e4m3", 7740), ("e5m2", 9664)]:
        inputs_a, inputs_b, expected, labels = [], [], [], []
        directory = source / f"A-{fmt}-jax-16-256-256-core"
        for path in sorted(directory.glob("batch-*.npz")):
            with np.load(path, allow_pickle=False) as z:
                a, b, observed, ids, names = (
                    z[k] for k in ["a", "b", "output_bits", "ids", "names"]
                )
                for i in range(len(a)):
                    outputs = observed[i]
                    value = outputs.flat[0]
                    if not np.all(outputs == value):
                        raise ValueError(
                            f"Nonuniform or nondeterministic capture: {path.name}:{i}"
                        )
                    inputs_a.append(a[i].view(np.uint32).copy())
                    inputs_b.append(b[i].view(np.uint32).copy())
                    expected.append(value)
                    labels.append(f"{path.name}:{int(ids[i])}:{names[i]}")
        if len(expected) != expected_count:
            raise ValueError(f"{fmt}: expected {expected_count}, got {len(expected)}")
        arrays.update(
            {
                f"{fmt}_a": np.array(inputs_a, dtype=np.uint32),
                f"{fmt}_b": np.array(inputs_b, dtype=np.uint32),
                f"{fmt}_expected": np.array(expected, dtype=np.uint32),
                f"{fmt}_source": np.array(labels),
            }
        )
    np.savez_compressed(destination, **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    build(args.source, (Path(__file__).parent / "trace_data" / "v7-fp8-measured.npz"))
