"""Build a fixed TPU executable and audit its final instruction sequence.

Run this file in a separate compile-only environment, before device tests:
  python export.py --arch TPU-v6e --k 256 --out /path/to/bundle
Use JAX/jaxlib 0.5.1 + libtpu 0.0.10 for v5e/v6e. v7 export/load requires
its own matching runtime and --on-device export. The v7 direct-executable path
has been hardware-validated for BF16 K128/K256 and FP8 E4M3FN/E5M2 K256.
Export and execution must use separate processes.
For equivalence tests, name each output bundle ARCH-FORMAT-kK and set
MMA_SIM_TPU_BUNDLES to their parent directory. Example: TPU-v6e-bf16-k256.
The output directory must be new. Keep the entire bundle for reproducibility.
"""

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arch: str, k: int, fmt: str, out: Path, on_device: bool = False) -> None:
    if arch not in ("TPU-v5e", "TPU-v6e", "TPU-v7"):
        raise ValueError("Unsupported architecture")
    if k not in (128, 256) or (arch == "TPU-v5e" and k != 128):
        raise ValueError("Use K128 for v5e or K128/K256 for v6e/v7")
    if fmt not in ("bf16", "e4m3fn", "e5m2"):
        raise ValueError("Unsupported input format")
    if fmt != "bf16" and (arch != "TPU-v7" or k != 256):
        raise ValueError("FP8 export requires v7 K256")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not on_device:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["TPU_SKIP_MDS_QUERY"] = "true"
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "")
        + f" --xla_jf_dump_to={out / 'llo'} --xla_jf_dump_llo_text"
        + " --xla_jf_dump_llo_pass_label_regex=final_bundles"
    )
    import jax
    import jax.numpy as jnp
    from jax.experimental import serialize_executable, topologies

    versions = {n: importlib.metadata.version(n) for n in ("jax", "jaxlib", "libtpu")}
    topology = arch.removeprefix("TPU-") + ":2x2"
    if on_device:
        device = jax.local_devices(backend="tpu")[0]
        tags = {
            "TPU-v5e": ("v5",),
            "TPU-v6e": ("v6",),
            "TPU-v7": ("v7", "7x", "ironwood"),
        }[arch]
        if jax.process_count() != 1 or not any(
            tag in device.device_kind.lower() for tag in tags
        ):
            raise ValueError(
                "Export requires one process on the selected TPU generation"
            )
        topology = device.device_kind
    else:
        if arch == "TPU-v7":
            raise ValueError(
                "Use --on-device for v7 export; its offline topology is not validated"
            )
        device = topologies.get_topology_desc(
            platform="tpu", topology_name=topology
        ).devices[0]
    shard = jax.sharding.SingleDeviceSharding(device)
    m, n = (16 if fmt == "bf16" else 32), k
    dtype = {
        "bf16": jnp.bfloat16,
        "e4m3fn": jnp.float8_e4m3fn,
        "e5m2": jnp.float8_e5m2,
    }[fmt]

    def product(a: jax.Array, b: jax.Array) -> jax.Array:
        return jax.lax.dot_general(
            a,
            b,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

    def merge(x: jax.Array, c: jax.Array) -> jax.Array:
        return x + c

    instructions: dict[str, dict[str, int]] = {}
    for name, fn, shapes in (
        ("product", product, ((m, k, dtype), (k, n, dtype))),
        ("merge", merge, ((m, n, jnp.float32), (m, n, jnp.float32))),
    ):
        before = set((out / "llo").glob("*final_bundles.txt"))
        args = [jax.ShapeDtypeStruct((a, b), t) for a, b, t in shapes]
        lowered = jax.jit(fn, in_shardings=(shard, shard), out_shardings=shard).lower(
            *args
        )
        (out / f"{name}.stablehlo").write_text(str(lowered.compiler_ir()))
        compiled = lowered.compile()
        blob, _, _ = serialize_executable.serialize(compiled)
        (out / f"{name}.pickle").write_bytes(blob)
        (out / f"{name}.hlo").write_text(compiled.as_text())
        dumps = sorted(set((out / "llo").glob("*final_bundles.txt")) - before)
        if not dumps:
            raise RuntimeError(
                "No final-bundle instruction dump; refusing to certify executable"
            )
        text = "\n".join(p.read_text() for p in dumps)
        (out / f"{name}.llo").write_text(text)
        instructions[name] = dict(
            collections.Counter(
                re.findall(
                    r"\b(?:vmatmul|vmatpush\d*|vmatprep|vpop|vadd)[\w.\[\]]*", text
                )
            )
        )
    prefix = "vmatmul." + ("" if arch == "TPU-v5e" else "mubr.")
    mnemonic = prefix + ("bf16" if fmt == "bf16" else "f8" + fmt) + ".vlgmr.msra.gmra"
    if arch == "TPU-v7":
        mnemonic += ".mrb[0]"
    mnemonic += ".mxu0"
    actual = {
        n: c for n, c in instructions["product"].items() if n.startswith("vmatmul.")
    }
    if actual != {mnemonic: 1}:
        raise RuntimeError(
            f"Expected one {mnemonic}; emitted {actual}. No bundle manifest written."
        )
    load = (
        "vmatpush"
        if arch == "TPU-v5e"
        else (
            "vmatpush3"
            if k == 128
            else "vmatpush1" if arch == "TPU-v7" else "vmatpush1+vmatpush2"
        )
    )
    expected_push = (
        {"vmatpush": 8}
        if arch == "TPU-v5e"
        else (
            {"vmatpush3": 8}
            if k == 128
            else (
                {"vmatpush1": 16 if fmt == "bf16" else 8}
                if arch == "TPU-v7"
                else {"vmatpush1": 8, "vmatpush2": 8}
            )
        )
    )
    pushes: dict[str, int] = {}
    for name, count in instructions["product"].items():
        if name.startswith("vmatpush"):
            op = name.split(".")[0]
            pushes[op] = pushes.get(op, 0) + count
    if pushes != expected_push or instructions["product"].get("vadd.f32", 0):
        raise RuntimeError(
            f"Unexpected product load/reduction sequence: {instructions['product']}"
        )
    if any(n.startswith("vmatmul") for n in instructions["merge"]) or not instructions[
        "merge"
    ].get("vadd.f32"):
        raise RuntimeError("External merge must use vector FP32 add, without MXU")
    files = ["product.pickle", "merge.pickle", "product.llo", "merge.llo"]
    manifest = dict(
        schema=1,
        arch=arch,
        format=fmt,
        shape=[m, k, n],
        instruction=mnemonic,
        weight_load=load,
        versions=versions,
        topology=topology,
        instructions=instructions,
        files={name: digest(out / name) for name in files},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--format", choices=("bf16", "e4m3fn", "e5m2"), default="bf16")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--on-device", action="store_true")
    args = parser.parse_args()
    build(args.arch, args.k, args.format, args.out, args.on_device)
