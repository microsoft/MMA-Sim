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
    elif mode == "odd":
        q |= bool(r)
    elif mode == "rdn":
        q += bool(r) and sign < 0
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


def dot_bits(
    a,
    b,
    *,
    group=4,
    fraction=26,
    alignment="rtz",
    normalized=True,
    input_ftz=True,
    output_ftz=True,
    product_ftz=True,
    late_acc=True,
    reduction="forward",
    sticky=True
):
    if reduction != "forward":
        if not late_acc:
            raise ValueError("Alternate reductions require independent partials")
        partials = [
            dot_bits(
                a[i : i + group],
                b[i : i + group],
                group=group,
                fraction=fraction,
                alignment=alignment,
                normalized=normalized,
                input_ftz=input_ftz,
                output_ftz=output_ftz,
                product_ftz=product_ftz,
                sticky=sticky,
            )
            for i in range(0, len(a), group)
        ]
        if reduction == "reverse":
            acc = 0
            for p in reversed(partials):
                acc = add32(acc, p, output_ftz)
            return acc
        if reduction == "pair_chain":
            acc = 0
            for i in range(0, len(partials), 2):
                pair = (
                    add32(partials[i], partials[i + 1], output_ftz)
                    if i + 1 < len(partials)
                    else partials[i]
                )
                acc = add32(acc, pair, output_ftz)
            return acc
        if reduction == "tree":
            while len(partials) > 1:
                partials = [
                    (
                        add32(partials[i], partials[i + 1], output_ftz)
                        if i + 1 < len(partials)
                        else partials[i]
                    )
                    for i in range(0, len(partials), 2)
                ]
            return partials[0] if partials else 0
        raise ValueError(reduction)
    acc = 0
    for start in range(0, len(a), group):
        incoming = acc
        if late_acc:
            acc = 0
        terms = []
        infinities = set()
        if (acc & 0x7FFFFFFF) == 0x7F800000:
            infinities.add(acc >> 31)
        elif (acc & 0x7FFFFFFF) > 0x7F800000:
            return 0x7FC00000
        else:
            n, e = unpack(acc, 23, 127, ftz=output_ftz)
            if n:
                terms.append((n, e, abs(n).bit_length() - 1 + e))
        for ai, bi in zip(a[start : start + group], b[start : start + group]):
            an, ae = unpack(int(ai), 7, 127, ftz=input_ftz)
            bn, be = unpack(int(bi), 7, 127, ftz=input_ftz)
            n, e = an * bn, ae + be
            if not n:
                continue
            pe = abs(n).bit_length() - 1 + e
            if pe > 127:
                infinities.add(int(n < 0))
                continue
            if product_ftz and pe < -126:
                continue
            align_e = (
                pe
                if normalized
                else abs(an).bit_length() + abs(bn).bit_length() - 2 + e
            )
            terms.append((n, e, align_e))
        if infinities:
            if len(infinities) == 2:
                return 0x7FC00000
            acc = (next(iter(infinities)) << 31) | 0x7F800000
            if late_acc:
                acc = add32(incoming, acc, output_ftz)
            continue
        if not terms:
            acc = incoming if late_acc else 0
            continue
        step = max(t[2] for t in terms) - fraction
        aligned = [(shift_round(n, e - step, alignment), n, e) for n, e, _ in terms]
        total = sum(q for q, _, _ in aligned)
        tail_signs = {
            1 if n > 0 else -1
            for q, n, e in aligned
            if q and e < step and abs(n) % (1 << (step - e))
        }
        tails = sum(tail_signs)
        acc = pack32_tails(total, step, tails) if sticky else pack32(total, step)
        if output_ftz and (acc & 0x7FFFFFFF) < 0x00800000:
            acc = 0
        if late_acc:
            acc = add32(incoming, acc, output_ftz)
    return acc
