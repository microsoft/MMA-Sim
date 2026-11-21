"""Direct execution of audited, serialized TPU instruction groups.

Build bundles beforehand with tpu/export.py. Loading/execution never falls back
on torch.mm or JAX compilation. Bundles contain executable pickle data and must
come from a trusted local build. Exact JAX/jaxlib/libtpu versions must match.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import torch

from mmasim.tpu.isa_vmatmul import MMA as Instruction
from .. import MMAKernel


def reject_compile(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Compilation is forbidden in the fixed TPU executable kernel")


@contextmanager
def no_compile() -> Iterator[None]:
    """Fail on JAX graph compilation while loading or executing a fixed binary."""
    import jax
    from jax._src import compiler

    backend_compile = (
        "backend_compile"
        if hasattr(compiler, "backend_compile")
        else "backend_compile_and_load"
    )
    with ExitStack() as stack:
        for owner, name in (
            (jax, "jit"),
            (compiler, backend_compile),
            (compiler, "compile_or_get_cached"),
        ):
            if not hasattr(owner, name):
                raise RuntimeError(f"Unsupported JAX compilation guard: {name}")
            stack.enter_context(patch.object(owner, name, reject_compile))
        yield


class MMA(MMAKernel):
    """One audited MXU group, then a separately serialized FP32 vector add.

    The selected kernel is the exact instruction sequence in its bundle. This
    class has no arbitrary mnemonic, MXU-selector, or compiler fallback option.
    Use one kernel invocation at a time: compilation guards are process-global.
    """

    def __init__(
        self, arch: str, k: int = 128, *, bundle: str | Path, format: str = "bf16"
    ) -> None:
        import jax
        import ml_dtypes
        import numpy as np
        from jax.experimental import serialize_executable

        directory = Path(bundle)
        manifest = json.loads((directory / "manifest.json").read_text())
        versions = {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "libtpu")
        }
        if manifest["schema"] != 1 or manifest["versions"] != versions:
            raise RuntimeError(
                "Bundle/runtime version mismatch; rebuild in a matching environment"
            )
        if (
            manifest["arch"] != arch
            or manifest["format"] != format
            or manifest["shape"][1] != k
        ):
            raise ValueError(
                "Bundle does not match requested architecture, format and K"
            )
        for name in ("product.pickle", "merge.pickle", "product.llo", "merge.llo"):
            if (
                hashlib.sha256((directory / name).read_bytes()).hexdigest()
                != manifest["files"][name]
            ):
                raise ValueError(f"Modified executable/evidence file: {name}")
        self.arch: str = arch
        self.instruction: str = manifest["instruction"]
        self.weight_load: str = manifest["weight_load"]
        shape = tuple(manifest["shape"])
        descriptor = Instruction(
            arch, self.instruction, weight_load=self.weight_load, shape=shape
        )
        self.m, self.n, self.k = descriptor.m, descriptor.n, descriptor.k
        self.a_type, self.b_type = descriptor.a_type, descriptor.b_type
        self.c_type, self.d_type = descriptor.c_type, descriptor.d_type
        expected = "vmatmul." + ("" if arch == "TPU-v5e" else "mubr.")
        expected += ("bf16" if format == "bf16" else "f8" + format) + ".vlgmr.msra.gmra"
        expected += (".mrb[0]" if arch == "TPU-v7" else "") + ".mxu0"
        actual = {
            name: count
            for name, count in manifest["instructions"]["product"].items()
            if name.startswith("vmatmul.")
        }
        if self.instruction != expected or actual != {expected: 1}:
            raise ValueError(
                "Bundle must contain exactly the selected MXU0 matrix instruction"
            )
        self.device: jax.Device = jax.local_devices(backend="tpu")[0]
        generation = {
            "TPU-v5e": ("v5",),
            "TPU-v6e": ("v6",),
            "TPU-v7": ("v7", "7x", "ironwood"),
        }[arch]
        if not any(tag in self.device.device_kind.lower() for tag in generation):
            raise ValueError(f"Wrong TPU generation: {self.device.device_kind}")
        if jax.process_count() != 1:
            raise ValueError(
                "Direct executable kernels require a single-process runtime"
            )
        self.input_dtype: type[np.generic] = {
            "bf16": ml_dtypes.bfloat16,
            "e4m3fn": ml_dtypes.float8_e4m3fn,
            "e5m2": ml_dtypes.float8_e5m2,
        }[format]
        with no_compile():
            a = jax.device_put(
                np.zeros((self.m, self.k), self.input_dtype), self.device
            )
            b = jax.device_put(
                np.zeros((self.k, self.n), self.input_dtype), self.device
            )
            c = jax.device_put(np.zeros((self.m, self.n), np.float32), self.device)
            output_tree = jax.tree_util.tree_structure(c)
            self.product: jax.stages.Compiled = (
                serialize_executable.deserialize_and_load(
                    (directory / "product.pickle").read_bytes(),
                    jax.tree_util.tree_structure(((a, b), {})),
                    output_tree,
                    backend=self.device.client,
                    execution_devices=[self.device],
                )
            )
            self.merge: jax.stages.Compiled = serialize_executable.deserialize_and_load(
                (directory / "merge.pickle").read_bytes(),
                jax.tree_util.tree_structure(((c, c), {})),
                output_tree,
                backend=self.device.client,
                execution_devices=[self.device],
            )

    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        import jax
        import numpy as np

        self.check_input(A, B, C)
        if any(x.requires_grad for x in (A, B, C)):
            raise ValueError(
                "Pass detached tensors; the hardware kernel has no autograd"
            )
        word_type = torch.int16 if self.a_type == torch.bfloat16 else torch.int8
        a = A.cpu().contiguous().view(word_type).numpy().view(self.input_dtype)
        b = B.cpu().contiguous().view(word_type).numpy().view(self.input_dtype)
        with no_compile():
            aa, bb, cc = (
                jax.device_put(x, self.device)
                for x in (a, b, C.cpu().contiguous().numpy())
            )
            result = self.product(aa, bb)
            result.block_until_ready()
            merged = self.merge(result, cc)
            output = np.asarray(merged.block_until_ready()).copy()
        return torch.from_numpy(output)
