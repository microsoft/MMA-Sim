import math

import torch

from mmasim.amd import shape_to_mnk, torch_dtype, min_exponent


class DotAdd:
    def __init__(
        self,
        mfma,
        qualifier: str,
    ):
        self.mfma = mfma
        qualifier = qualifier.split("_")
        if len(qualifier) == 3:
            self.d_type, self.shape, self.a_type = qualifier
            self.b_type = self.a_type
        else:
            self.d_type, self.shape, self.a_type, self.b_type = qualifier
        self.c_type = self.d_type

    def __call__(self, a: list[float], b: list[float], c: float = 0.0) -> float:
        m, n, k = shape_to_mnk(self.shape)
        A = torch.zeros([m, k], dtype=torch_dtype[self.a_type])
        B = torch.zeros([k, n], dtype=torch_dtype[self.b_type])
        for i in range(len(a)):
            A[0, i] = a[i]
            B[i, 0] = b[i]
        A = A.cuda()
        B = B.cuda()
        C = torch.zeros([m, n], dtype=torch_dtype[self.c_type])
        D = torch.empty([m, n], dtype=torch_dtype[self.d_type])
        C[0, 0] = c
        C = C.cuda()
        D = D.cuda()
        self.mfma(D.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr())
        return D[0, 0].item()


@torch.inference_mode()
def test_tf32_input_rounding(dot: DotAdd):
    # f32 -> tf32: truncation

    # normal numbers: RZ
    X = 1.0
    ulpX = 2.0**-10
    # not RN, RU or RA, where X + 0.75 * ulpX should become X + ulpX
    assert dot([X + 0.75 * ulpX], [1.0]) == X
    # not RN, RD or RA, where -X - 0.75 * ulpX should become -X - ulpX
    assert dot([-X - 0.75 * ulpX], [1.0]) == -X

    # numbers > largest normal value: RZ, no overflow
    X = (2.0 - 2.0**-10) * 2.0**127
    ulpX = 2.0 ** (127 - 10)
    # not RU, RA or RN, where X + 0.75 * ulpX should overflow
    assert dot([X + 0.75 * ulpX], [1.0]) == X
    # not RD, RA or RN, where -X - 0.75 * ulpX should overflow
    assert dot([-X - 0.75 * ulpX], [1.0]) == -X

    # numbers < smallest normal value and > largest subnormal value: RZ
    X = (1.0 - 2.0**-10) * 2.0**-126
    ulpX = 2.0 ** (-126 - 10)
    # not RU, RA or RN, where X + 0.75 * ulpX should become X + ulpX
    assert dot([X + 0.75 * ulpX], [1.0]) == X
    # not RD, RA or RN, where -X - 0.75 * ulpX should become -X - ulpX
    assert dot([-X - 0.75 * ulpX], [1.0]) == -X

    # subnormal numbers: RZ
    X = 2.0**-130
    ulpX = 2.0 ** (-126 - 10)
    # not RU, RA or RN, where X + 0.75 * ulpX should become X + ulpX
    assert dot([X + 0.75 * ulpX], [1.0]) == X
    # not RD, RA or RN, where -X - 0.75 * ulpX should become -X - ulpX
    assert dot([-X - 0.75 * ulpX], [1.0]) == -X

    # numbers < smallest subnormal value: RZ
    X = 2.0 ** (-126 - 10)
    # not RU, RA or RN, where 0.75 * X should become X
    assert dot([0.75 * X], [1.0]) == 0.0
    # not RD, RA or RN, where -0.75 * X should become -X
    assert dot([-0.75 * X], [1.0]) == 0.0

    # special values: nan can be converted to inf because of truncation
    X = torch.tensor(0x7F80_0001, dtype=torch.uint32).view(torch.float32)
    assert X.isnan()
    assert dot([X], [1.0]) == math.inf


# applicable to f16, bf16, tf32, e5m2, and e4m3fn
@torch.inference_mode()
def test_fused_dot_add(
    dotadd: DotAdd,
    n_fraction_bits: int,
    split_k: bool = False,
):
    k = shape_to_mnk(dotadd.shape)[2]

    # sumfmation order
    Large = 2.0**7
    small = 2.0**-9
    if not split_k:
        # (k+1)-term fused sumfmation
        # c and other k terms are fused summed
        for i in range(k):
            Xs = [small] * k
            Xs[i] = Large
            assert dotadd(Xs, Xs, -Large * Large) == 0.0
        # aibi and other k terms are fused summed, i = 0, 1, ..., k-1
        for i in range(k):
            for j in range(i + 1, k):
                Xs = [small] * k
                Xs[i] = Large
                Xs[j] = -Large
                Ys = Xs.copy()
                Ys[j] = Large
                assert dotadd(Xs, Ys, small * small) == 0.0
    else:
        l = k // 2
        # (l+1)-term fused sumfmation
        # c and first l terms are fused summed
        for i in range(l):
            Xs = [small] * k
            Xs[i] = Large
            assert dotadd(Xs, Xs, -Large * Large) == l * small * small
        for i in range(l):
            for j in range(i + 1, l):
                Xs = [small] * k
                Xs[i] = Large
                Xs[j] = -Large
                Ys = Xs.copy()
                Ys[j] = Large
                assert dotadd(Xs, Ys, small * small) == l * small * small
        # and then the sum and other l terms are fused summed
        for i in range(l, k):
            Xs = [small] * k
            Xs[i] = Large
            assert dotadd(Xs, Xs, -Large * Large) == 0.0
        for i in range(l, k):
            for j in range(i + 1, k):
                Xs = [small] * k
                Xs[i] = Large
                Xs[j] = -Large
                Ys = Xs.copy()
                Ys[j] = Large
                assert dotadd(Xs, Ys, small * small) == 0.0

    # precision of accumulator
    X = (2.0 - 2.0**-10) * 2.0**-14
    ulpX = 2.0**-24
    p = -14
    Y, Z = 2.0**-7, 2.0**-7  # Y*Z == 2.0**p
    while dotadd([Y, -Y], [Z, Z], X) == X:
        p += 1
        if p % 2 == 1:
            Y *= 2.0
        else:
            Z *= 2.0
    # fraction bits: (p-1) - (-24)
    assert (p - 1) - (-24) == n_fraction_bits
    # p == -10 if n_fraction_bits == 13
    # p == 0 if n_fraction_bits == 23
    # p == 1 if n_fraction_bits == 24
    # p == 2 if n_fraction_bits == 25

    # rounding of terms: RZ (round to zero)
    # not RU or RA, where X + YZ - YZ should become X + ulpX
    assert dotadd([Y, -Y], [Z, Z], X) == X - ulpX
    # not RD or RA, where -X - YZ + YZ should become -X - ulpX
    assert dotadd([Y, -Y], [Z, Z], -X) == -X + ulpX
    # not RN, where X + 2YZ - 2YZ should become X + ulpX
    # and -X - 2YZ + 2YZ should become -X - ulpX
    assert dotadd([Y, -Y], [2.0 * Z, 2.0 * Z], X) == X - 3 * ulpX
    assert dotadd([Y, -Y], [2.0 * Z, 2.0 * Z], -X) == -X + 3 * ulpX

    # alignment of terms: max({exponentA+exponentB}, exponentC)
    # not max({exponentAB}, exponentC), where X + 1.5Y*1.5Z - 1.5Y*1.5Z should become X - 3 * ulpX
    assert dotadd([2.25 * Y, -2.25 * Y], [Z, Z], X) == X - 3 * ulpX
    assert dotadd([1.5 * Y, -1.5 * Y], [1.5 * Z, 1.5 * Z], X) == X - ulpX

    # exponents of subnormal numbers: always min_exponent
    X = 2.0 ** min_exponent[dotadd.a_type]
    Y = 2.0 ** min_exponent[dotadd.b_type]
    if dotadd.d_type == "f16":
        if X * Y < 2.0**-25:
            Y = 2.0**-25 / X
    else:  # dotadd.d_type == "f32"
        if X * Y < 2.0**-126:
            Y = 2.0**-126 / X
    Z = Y * 2.0 ** (n_fraction_bits + 1)
    assert dotadd([X, X, X, X], [Z, -Z, Y, Y]) == 0.0
    assert dotadd([X, X, X, X], [0.5 * Z, -0.5 * Z, Y, Y]) == 2 * X * Y
    assert (
        dotadd([0.5 * X, 0.5 * X, X, X], [Z, -Z, Y, Y]) == 0.0
    )  # exponent of 0.5*X is still min_exponent


# applicable to f16, bf16, tf32, e5m2, and e4m3fn
@torch.inference_mode()
def test_f32_output_rounding(dotadd: DotAdd):
    # f16, bf16, tf32:
    #   accumulator -> f32: RZ (round to zero)
    # fp8:
    #   accumulator -> f32_e8m13: RZ (round to zero)

    k = shape_to_mnk(dotadd.shape)[2]
    a_type = dotadd.a_type

    # normal numbers: RZ
    X = 1.0
    ulpX = 2.0**-23 if a_type in ["f16", "bf16", "tf32"] else 2.0**-13
    # not RU, RA or RN, where X + 0.75 * ulpX should become X + ulpX
    assert dotadd([0.25] * 4, [X] * 4, 0.75 * ulpX) == X
    # not RD, RA or RN, where -X - 0.75 * ulpX should become -X - ulpX
    assert dotadd([-0.25] * 4, [X] * 4, -0.75 * ulpX) == -X

    if a_type in ["bf16", "tf32"]:
        # numbers > largest normal value: RZ, do no overflow
        X = torch.finfo(torch.float32).max
        ufpX = 2.0**127
        # not RU, RA or RN, where 1.5ufpX + 0.25X should overflow
        assert dotadd([0.75, 0.75], [ufpX, ufpX], 0.25 * X) == X
        # not RD, RA or RN, where -1.5ufpX - 0.25X should overflow
        assert dotadd([-0.75, -0.75], [ufpX, ufpX], -0.25 * X) == -X
        # numbers >= 2^128: overflow
        assert dotadd([1.0, 1.0], [ufpX, ufpX]) == math.inf
        assert dotadd([-1.0, -1.0], [ufpX, ufpX]) == -math.inf

        # numbers < smallest normal value and > largest subnormal number: RZ
        X = (1.0 - 2.0**-23) * 2.0**-126
        ulpX = 2.0**-149
        # not RU, RA or RN, where X + 0.75 * ulpX should become X + ulpX
        assert dotadd([0.75 * 2.0**-23], [ulpX * 2.0**23], X) == X
        # not RD, RA or RN, where -X - 0.75 * ulpX should become -X - ulpX
        assert dotadd([-0.75 * 2.0**-23], [ulpX * 2.0**23], -X) == -X

        # subnormal numbers: RZ
        X = 2.0**-148
        ulpX = 2.0**-149
        # not RU, RA or RN, where X + 0.75 * ulpX should become X + ulpX
        assert dotadd([0.75 * 2.0**-23], [ulpX * 2.0**23], X) == X
        # not RD, RA or RN, where -X - 0.75 * ulpX should become -X - ulpX
        assert dotadd([-0.75 * 2.0**-23], [ulpX * 2.0**23], -X) == -X

        # numbers < smallest subnormal value: RZ
        # not RU, RA or RN, where 0.75 * ulpX should become ulpX
        assert dotadd([0.75 * 2.0**-23], [ulpX * 2.0**23]) == 0.0
        # not RD, RA or RN, where -0.75 * ulpX should become -ulpX
        assert dotadd([-0.75 * 2.0**-23], [ulpX * 2.0**23]) == 0.0
    else:
        pass  # cannot be obtained through fp16 or fp8

    # minus zero -> +0.0
    ret = dotadd([-0.0] * k, [1.0] * k, -0.0)
    assert ret == 0.0 and math.copysign(1.0, ret) == 1.0


# applicable to f16, e5m2, and e4m3fn
@torch.inference_mode()
def test_f16_output_rounding(dotadd: DotAdd):
    # accumulator -> f16: RNE (round to nearest, ties to even)

    k = shape_to_mnk(dotadd.shape)[2]
    a_type = dotadd.a_type
    b_type = dotadd.b_type

    # normal numbers: RNE
    X = 2.0**10
    ulpX = 1.0
    # not RU, RA, RNU, RNA or RNO, where X + 0.5 * ulpX should become X + ulpX
    assert dotadd([0.5], [ulpX], X) == X
    # not RD, RA, RND, RNA or RNO, where -X - 0.5 * ulpX should become -X - ulpX
    assert dotadd([-0.5], [ulpX], -X) == -X
    # not RD, RZ, RND, RNZ or RNO, where X + 1.5 * ulpX should become X + ulpX
    assert dotadd([1.5], [ulpX], X) == X + 2 * ulpX
    # not RU, RZ, RNU, RNZ or RNO, where -X - 1.5 * ulpX should become -X - ulpX
    assert dotadd([-1.5], [ulpX], -X) == -X - 2 * ulpX

    # no intermediate rounding
    if a_type == "f16":
        # not accumulator -> f32 -> f16
        # where X + 0.5 * ulpX + 0.5 * ulpX_f32 should become X
        X = 1.0
        ulpX = 2.0**-10
        ulpX_f32 = 2.0**-23
        assert dotadd([0.5] * 4, [X, X, ulpX, ulpX_f32]) == X + ulpX
    else:
        # not accumulator -> f32_e8m13 -> f16
        # where X + 0.5 * ulpX + 0.5 * ulpX_f32 should become X
        X = 2.0**5
        ulpX = 2.0**-5
        ulpX_f32 = 2.0**-8
        assert dotadd([0.125] * 8 + [0.5, 0.5], [X] * 8 + [ulpX, ulpX_f32]) == X + ulpX

    # numbers > largest normal value: comply with RNE
    X = torch.finfo(torch.float16).max
    ulpX = 2.0 ** (15 - 10)
    # numbers < X + 0.5 * ulpX -> X
    assert dotadd([0.25], [ulpX], X) == X
    assert dotadd([-0.25], [ulpX], -X) == -X
    # nubers >= X + 0.5 * ulpX -> inf
    assert dotadd([0.5], [ulpX], X) == math.inf
    assert dotadd([-0.5], [ulpX], -X) == -math.inf
    assert dotadd([0.75], [ulpX], X) == math.inf
    assert dotadd([-0.75], [ulpX], -X) == -math.inf

    # numbers < smallest normal value and > largest subnormal number: comply with RNE
    if a_type == "f16":
        X = (1.0 - 2.0**-10) * 2.0**-14
        ulpX = 2.0**-24
        # numbers >= X + 0.5 * ulpX -> X + ulpX
        assert dotadd([0.75], [ulpX], X) == X + ulpX
        assert dotadd([-0.75], [ulpX], -X) == -X - ulpX
        assert dotadd([0.5], [ulpX], X) == X + ulpX
        assert dotadd([-0.5], [ulpX], -X) == -X - ulpX
        # numbers < X + 0.5 * ulpX -> X
        assert dotadd([0.25], [ulpX], X) == X
        assert dotadd([-0.25], [ulpX], -X) == -X
    elif a_type == "e4m3" and b_type == "e4m3":
        pass  # cannot be obtained through e4m3fn
    else:
        X = (1.0 - 2.0**-10) * 2.0**-14
        Y, Z = (2.0**-8, 2.0**-16) if a_type == "e4m3" else (2.0**-15, 2.0**-9)
        ulpX = 2.0**-24  # YZ == ulpX
        assert dotadd([0.5 * Y], [Z], X) == X + ulpX
        assert dotadd([-0.5 * Y], [Z], -X) == -X - ulpX

    # subnormal numbers: RNE
    if a_type == "e4m3" and b_type == "e4m3":
        pass  # cannot be obtained through e4m3fn
    else:
        X = 2.0**-23
        Y, Z = (2.0**-8, 2.0**-16) if a_type == "e4m3" else (2.0**-15, 2.0**-9)
        ulpX = 2.0**-24  # YZ == ulpX
        # not RU, RA, RNU, RNA or RNO, where X + 0.5 * ulpX should become X + ulpX
        assert dotadd([0.5 * Y], [Z], X) == X
        # not RD, RA, RND, RNA or RNO, where -X - 0.5 * ulpX should become -X - ulpX
        assert dotadd([-0.5 * Y], [Z], -X) == -X
        # not RD, RZ, RND, RNZ or RNO, where X + 1.5 * ulpX should become X + ulpX
        assert dotadd([1.5 * Y], [Z], X) == X + 2 * ulpX
        # not RU, RZ, RNU, RNZ or RNO, where -X - 1.5 * ulpX should become -X - ulpX
        assert dotadd([-1.5 * Y], [Z], -X) == -X - 2 * ulpX

    # numbers < smallest subnormal value: comply with RNE
    if a_type == "f16":
        ulpX = 2.0**-24
        # numbers > 0.5 * ulpX -> ulpX
        assert dotadd([0.75], [ulpX]) == ulpX
        assert dotadd([-0.75], [ulpX]) == -ulpX
        # numbers <= 0.5 * ulpX -> 0
        assert dotadd([0.5], [ulpX]) == 0.0
        assert dotadd([-0.5], [ulpX]) == 0.0
        assert dotadd([0.25], [ulpX]) == 0.0
        assert dotadd([-0.25], [ulpX]) == 0.0
    elif a_type == "e4m3" and b_type == "e4m3":
        pass  # cannot be obtained through e4m3fn
    else:
        Y, Z = (2.0**-8, 2.0**-16) if a_type == "e4m3" else (2.0**-15, 2.0**-9)
        assert dotadd([0.5 * Y], [Z]) == 0.0
        assert dotadd([-0.5 * Y], [Z]) == 0.0

    # minus zero -> +0.0
    ret = dotadd([-0.0] * k, [1.0] * k, -0.0)
    assert ret == 0.0 and math.copysign(1.0, ret) == 1.0
