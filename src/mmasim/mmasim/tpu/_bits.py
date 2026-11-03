"""Integer arithmetic for TPU finite-input instruction references.

FP32 packing uses RNE; add32 uses input/output flush-to-zero by default.
Internal widths and grouping are configured by the instruction arithmetic.
"""


def shift_round(n, shift, mode):
    if shift >= 0:
        return n << shift
    sign = -1 if n < 0 else 1
    q, r = divmod(abs(n), 1 << -shift)
    if mode == "rne":
        half = 1 << (-shift - 1)
        q += r > half or (r == half and q % 2 == 1)
    elif mode != "rtz":
        raise ValueError(mode)
    return sign * q


def pack32(n, exponent):
    if not n:
        return 0
    sign = 0x80000000 if n < 0 else 0
    n = abs(n)
    e = max(-126, n.bit_length() - 1 + exponent)
    q = shift_round(n, exponent - e + 23, "rne")
    if q >= 1 << 24:
        q >>= 1
        e += 1
    if e > 127:
        return sign | 0x7F800000
    if q < 1 << 23:
        return sign | q
    return sign | ((e + 127) << 23) | (q - (1 << 23))


def pack32_tails(n, exponent, tails):
    """Candidate sign-separated sticky rule; sticky is not left-normalized."""
    if not n:
        return 0
    shift = abs(n).bit_length() - 24
    if shift > 0:
        return pack32(2 * n + tails, exponent - 1)
    if tails and (tails > 0) != (n > 0):
        n -= 1 if n > 0 else -1
    return pack32(n, exponent)


def unpack(bits, fraction_bits, bias, ftz=True):
    sign = -1 if bits >> (fraction_bits + 8) else 1
    e = (bits >> fraction_bits) & 255
    m = bits & ((1 << fraction_bits) - 1)
    if e == 255:
        raise ValueError("Special input not supported in finite-input model")
    if e == 0:
        return (0 if ftz else sign * m), 1 - bias - fraction_bits
    return sign * ((1 << fraction_bits) + m), e - bias - fraction_bits


def add32(a, b, ftz=True):
    aa, bb = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if aa > 0x7F800000 or bb > 0x7F800000:
        return 0x7FC00000
    if aa == 0x7F800000 or bb == 0x7F800000:
        if aa == bb and (a >> 31) != (b >> 31):
            return 0x7FC00000
        return a if aa == 0x7F800000 else b
    an, ae = unpack(a, 23, 127, ftz=ftz)
    bn, be = unpack(b, 23, 127, ftz=ftz)
    e = min(ae, be)
    result = pack32((an << (ae - e)) + (bn << (be - e)), e)
    return 0 if ftz and (result & 0x7FFFFFFF) < 0x00800000 else result


def bf16_partial(a, b, *, fraction=26, sticky=True):
    """One independent DP4/DP8 result with FP32 RNE output.

    BF16 inputs and tiny products flush to zero. Product overflow is handled
    before alignment. v5e/v6e use sign-separated tails; v7 disables them.
    """
    terms = []
    infinities = set()
    for ai, bi in zip(a, b):
        an, ae = unpack(int(ai), 7, 127)
        bn, be = unpack(int(bi), 7, 127)
        n, e = an * bn, ae + be
        if not n:
            continue
        pe = abs(n).bit_length() - 1 + e
        if pe > 127:
            infinities.add(int(n < 0))
        elif pe >= -126:
            terms.append((n, e, pe))
    if len(infinities) == 2:
        return 0x7FC00000
    if infinities:
        return (next(iter(infinities)) << 31) | 0x7F800000
    if not terms:
        return 0
    step = max(pe for _, _, pe in terms) - fraction
    aligned = [(shift_round(n, e - step, "rtz"), n, e) for n, e, _ in terms]
    total = sum(q for q, _, _ in aligned)
    tails = sum(
        {
            1 if n > 0 else -1
            for q, n, e in aligned
            if q and e < step and abs(n) % (1 << (step - e))
        }
    )
    result = pack32_tails(total, step, tails) if sticky else pack32(total, step)
    return 0 if (result & 0x7FFFFFFF) < 0x00800000 else result
