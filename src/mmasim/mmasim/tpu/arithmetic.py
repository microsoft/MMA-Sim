"""Instruction-local arithmetic; all inputs are exact FP32 encodings.

BF16: rounded DP4/DP8 outputs, pair merge, then FP32 carried-state add.
FP8: K256 cross-half DP8 partials use Emax-8 RTZ and p9 RTZ;
32 partials update a p9 RNE accumulator serially. Output is carried in FP32.
No physical E7M8 exponent encoding or compiler-selected outer schedule is assumed.
"""

from ._bits import add32, dot_bits, unpack, shift_round, pack32


def fold(values):
    acc = 0
    for v in values:
        acc = add32(acc, v)
    return acc


def block256(a, b, mode="cross128"):
    a = list(a) + [0] * (256 - len(a))
    b = list(b) + [0] * (256 - len(b))
    p = [dot_bits(a[i : i + 4], b[i : i + 4]) for i in range(0, 256, 4)]
    if mode == "independent128":
        lo = fold([add32(p[i], p[i + 1]) for i in range(0, 32, 2)])
        hi = fold([add32(p[i], p[i + 1]) for i in range(32, 64, 2)])
        return add32(lo, hi)
    if mode != "cross128":
        raise ValueError(mode)
    return fold(
        [
            add32(add32(p[i], p[i + 32]), add32(p[i + 1], p[i + 33]))
            for i in range(0, 32, 2)
        ]
    )


def cross_block256(a, b):
    if len(a) != 256 or len(b) != 256:
        raise ValueError("Complete K256 block required")
    acc = 0
    for i in range(0, 128, 8):
        pair = []
        for h in (0, 4):
            aa = a[i + h : i + h + 4] + a[i + h + 128 : i + h + 132]
            bb = b[i + h : i + h + 4] + b[i + h + 128 : i + h + 132]
            pair.append(dot_bits(aa, bb, group=8, fraction=27, sticky=False))
        acc = add32(acc, add32(pair[0], pair[1]))
    return acc


def bf16_dot(a, b, arch):
    a, b = [x >> 16 for x in a], [x >> 16 for x in b]
    if arch == "TPU-v5e":
        return dot_bits(a, b, reduction="pair_chain")
    if arch == "TPU-v6e":
        return block256(a, b, "independent128" if len(a) == 128 else "cross128")
    if len(a) == 256:
        return cross_block256(a, b)
    return dot_bits(a, b, fraction=27, sticky=False, reduction="pair_chain")


def quantize_p9(n, e, mode):
    shift = max(0, abs(n).bit_length() - 9)
    return pack32(shift_round(n, -shift, mode), e + shift)


def fp8_dot(a, b):
    if len(a) != 256 or len(b) != 256:
        raise ValueError("FP8 instruction group requires K256")
    acc = 0
    for j in range(0, 128, 4):
        terms = []
        for i in list(range(j, j + 4)) + list(range(j + 128, j + 132)):
            an, ae = unpack(a[i], 23, 127)
            bn, be = unpack(b[i], 23, 127)
            if an * bn:
                terms.append((an * bn, ae + be))
        if terms:
            step = max(abs(n).bit_length() - 1 + e for n, e in terms) - 8
            part = quantize_p9(
                sum(shift_round(n, e - step, "rtz") for n, e in terms), step, "rtz"
            )
        else:
            part = 0
        an, ae = unpack(acc, 23, 127)
        bn, be = unpack(part, 23, 127)
        e = min(ae, be)
        acc = quantize_p9((an << (ae - e)) + (bn << (be - e)), e, "rne")
    return acc
