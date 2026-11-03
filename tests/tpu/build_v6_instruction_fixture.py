"""Extract instruction-group fixtures from the preserved TPU boundary corpus."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "corpus", type=Path, help="directory containing inputs-k*.npz and m*-k*-n*.npz"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(Path(__file__).parent / "trace_data" / "v6-instruction-measured.json"),
    )
    args = parser.parse_args()
    out = {
        "source": "v6-schedule-boundaries: saved real-TPU BF16 DEFAULT outputs; input-bank indices preserved",
        "banks": {},
        "cases": [],
    }
    for k in (128, 256):
        with np.load(args.corpus / f"inputs-k{k}.npz") as bank:
            out["banks"][str(k)] = {
                name: bank[name].view(np.uint32).tolist() for name in ("a", "b")
            }
    cases = [
        (128, 128, 128, "vmatpush3"),
        (128, 256, 128, "vmatpush3"),
        (128, 256, 256, "vmatpush1+vmatpush2"),
        (257, 256, 128, "vmatpush1+vmatpush2"),
    ]
    for m, k, n, setup in cases:
        tag = f"m{m}-k{k}-n{n}"
        with np.load(args.corpus / (tag + ".npz")) as record:
            rows = 32 if k == 128 else 16
            assert record["bits"].shape == (m, n)
            out["cases"].append(
                {
                    "source": tag + ".npz",
                    "shape": [m, k, n],
                    "weight_load": setup,
                    "row_index": record["row_index"][:rows].tolist(),
                    "col_index": record["col_index"].tolist(),
                    "expected": record["bits"][:rows].tolist(),
                }
            )
    args.output.write_text(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
