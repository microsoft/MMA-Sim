"""Validate already-quantized finite FP32 operand encodings."""

import struct
import torch
from ..arithmetic.helper import dtype_subnormal_exponent
from ._bits import unpack

_FORMATS = {
    "bf16": (torch.bfloat16, 8, 127, None),
    "fp32": (torch.float32, 24, 127, None),
    "e4m3fn": (torch.float8_e4m3fn, 4, 8, 448),
    "e5m2": (torch.float8_e5m2, 3, 15, None),
}


def validate_words(words, fmt):
    dtype, p, emax, maxvalue = _FORMATS[fmt]
    emin = dtype_subnormal_exponent[dtype]
    out = []
    for word in words:
        if isinstance(word, bool) or int(word) != word or not 0 <= word < 2**32:
            raise ValueError("Expected unsigned FP32 bit encodings")
        word = int(word)
        n, e = unpack(word, 23, 127, ftz=False)  # rejects nonfinite operands
        if n:
            top = abs(n).bit_length() - 1 + e
            quantum = max(emin, top) - (p - 1)
            shift = quantum - e
            if top > emax or (shift > 0 and abs(n) % (1 << shift)):
                raise ValueError(f"Input is not exactly representable as {fmt}")
            if maxvalue is not None and abs(_float(word)) > maxvalue:
                raise ValueError(f"Input exceeds the finite range of {fmt}")
        out.append(word)
    return out


def _float(word):
    return struct.unpack("!f", int(word).to_bytes(4, "big"))[0]
