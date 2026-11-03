"""Validate already-quantized finite FP32 operand encodings."""

import struct
from ._bits import unpack

_FORMATS = {
    "bf16": (8, -126, 127, None),
    "fp32": (24, -126, 127, None),
    "e4m3fn": (4, -6, 8, 448),
    "e5m2": (3, -14, 15, None),
}


def validate_words(words, fmt):
    p, emin, emax, maxvalue = _FORMATS[fmt]
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
