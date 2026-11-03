"""Extract finite BF16 instruction witnesses from saved hardware NPZ captures."""

import argparse
import json
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    args = p.parse_args()
    cases = []
    for shape in ((16, 128, 128), (16, 256, 256)):
        directory = args.results / ("native-" + "-".join(map(str, shape)))
        for name in ("dense-710", "dense-711", "cross-cancel", "tile-boundary"):
            path = directory / (name + ".npz")
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as z:
                a = z["input0_fp32"].view(np.uint32)
                b = z["input1_fp32"].view(np.uint32)
                out = z["output_bits"]
                assert np.all(out == out[0])
                assert np.all(a == a[0])
                assert np.all(out[0, 0] == out[0, 0, 0])
                # Four columns retain cancellation witnesses and dense controls.
                for col in (0, 1, 7, 19):
                    cases.append(
                        dict(
                            shape=shape,
                            source=str(path.relative_to(args.results)),
                            column=col,
                            a=a[0].tolist(),
                            b=b[:, col].tolist(),
                            expected=int(out[0, 0, 0, col]),
                        )
                    )
    target = Path(__file__).parent / "trace_data" / "v7-instruction-measured.json"
    target.write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    print(len(cases), "hardware witnesses ->", target)
