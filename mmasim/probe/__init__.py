from .intrinsic import Intrinsic


def is_fused_dot_add(intrinsic: Intrinsic, group_size: int) -> bool:
    # summation tree: (group_size+1)-way tree
    # c group_0
    # | ////
    # +
    # | group_1
    # | ////
    # +
    # ...
    # | group_(K/group_size-1)
    # | ////
    # +

    # verify via FPRev's method
    K = intrinsic.k
    dotadd = intrinsic.dotadd
    X = 2.0**7
    y = 2.0**-9
    i = 0
    while i < K:
        for j in range(group_size):
            a = [y] * K
            a[i + j] = X
            b = a.copy()
            c = -X * X
            #  c                      #(i+j)
            # -X*X + y*y + ... + y*y + X*X + y*y + ...
            if dotadd(a, b, c) != (K - i - group_size) * y * y:
                return False
            for k in range(j):
                a[i + k] = X
                b[i + k] = -X
                c = y * y
                # c                 #(i+k)                  #(i+j)
                # y*y + ... + y*y + -X*X + y*y + ... + y*y + X*X + y*y + ...
                if dotadd(a, b, c) != (K - i - group_size) * y * y:
                    return False
                a[i + k] = b[i + k] = y
        i += group_size
    return True


def is_pairwise_sum(intrinsic: Intrinsic, group_size: int) -> bool:
    # summation tree:
    # c group_0_sum
    # | /
    # + group_1_sum
    # | /
    # +
    # ...
    # | group_(K/group_size-1)_sum
    # | /
    # +

    # each group_sum is a pairwise sum of the group_size elements
    # like
    # #0 #1 #2 #3 #4 #5 #6 #7
    #  \ /   \ /   \ /   \ /
    #   +     +     +     +
    #    \   /       \   /
    #      +           +
    #         \     /
    #            +

    # verify via FPRev's method
    K = intrinsic.k
    dotadd = intrinsic.dotadd
    X = 2.0**7
    y = 2.0**-9
    i = 0
    while i < K:
        for j in range(group_size):
            a = [y] * K
            a[i + j] = X
            b = a.copy()
            c = -X * X
            #  c                      #(i+j)
            # -X*X + y*y + ... + y*y + X*X + y*y + ...
            if dotadd(a, b, c) != (K - i - group_size) * y * y:
                return False
            for k in range(j):
                a[i + k] = X
                b[i + k] = -X
                c = y * y
                # c                 #(i+k)                  #(i+j)
                # y*y + ... + y*y + -X*X + y*y + ... + y*y + X*X + y*y + ...
                l = j ^ k
                # calc next power of 2 of l
                l |= l >> 1
                l |= l >> 2
                l |= l >> 4
                l |= l >> 8
                l |= l >> 16
                l += 1
                if dotadd(a, b, c) != (K - l) * y * y:
                    return False
                a[i + k] = b[i + k] = y
        i += group_size
    return True


def probe_rounding_helper1(res1: bool, res2: bool, res3: bool) -> str:
    #             |RU |RD |RZ |RA |RN |
    # +0.75 * ulp | + | - | - | + | + |
    # -0.75 * ulp | + | - | + | - | - |
    # +0.25 * ulp | + | - | - | + | - |
    if res1 and res2:
        return "RU"
    elif not res1 and not res2:
        return "RD"
    elif not res1 and res2:
        return "RZ"
    elif res3:
        return "RA"
    else:
        return "RN"


def probe_rounding_helper2(res1: bool, res2: bool, res3: bool) -> str:
    #             |RNU|RND|RNZ|RNA|RNE|RNO|
    # +0.5 * ulp  | + | - | - | + | - | + |
    # -0.5 * ulp  | + | - | + | - | + | - |
    # +1.5 * ulp  | + | - | - | + | + | - |
    if res1 and res2:
        return "RNU"
    elif not res1 and not res2:
        return "RND"
    elif res2 and not res3:
        return "RNZ"
    elif not res2 and res3:
        return "RNA"
    elif res2 and res3:
        return "RNE"
    else:
        return "RNO"


class ProbeFusedDotAdd:
    def __init__(self, intrinsic: Intrinsic):
        self.intrinsic = intrinsic
        self.dotadd = intrinsic.dotadd
        self.precision = []
        self.rounding_mode = []
        self.M, self.u1, self.u2 = 0.0, 0.0, 0.0

    def probe_product_rounding(self, i: int) -> tuple[int, str]:
        # rounding precision
        M = 2.0**7
        e = 14
        if i == 0:
            A = [0.0, M, -M]
            B = [0.0, M, M]
        elif i == 1:
            A = [M, 0.0, 0.0, -M]  # consider CDNA3 fp8
            B = [M, 0.0, 0.0, M]
        else:
            A = [M, -M] + [0.0] * (i - 1)
            B = [M, M] + [0.0] * (i - 1)
        while True:
            A[i] = 2.0 ** (e // 2)
            B[i] = 2.0 ** (e - e // 2)
            if self.dotadd(A, B) != 2.0**e:
                break
            e -= 1
        e += 1
        n_fraction_bits = 14 - e
        u1, u2 = 2.0 ** (e // 2), 2.0 ** (e - e // 2)  # u1 * u2 == ulp(M*M)
        self.precision.append(n_fraction_bits)
        self.M, self.u1, self.u2 = M, u1, u2
        # rounding mode
        A[i], B[i] = 0.75 * u1, u2
        res1 = self.dotadd(A, B) == u1 * u2
        A[i], B[i] = -0.75 * u1, u2
        res2 = self.dotadd(A, B) == 0.0
        A[i], B[i] = 0.25 * u1, u2
        res3 = self.dotadd(A, B) == u1 * u2
        rounding = probe_rounding_helper1(res1, res2, res3)
        if rounding == "RN":
            A[i], B[i] = 0.5 * u1, u2
            res1 = self.dotadd(A, B) == u1 * u2
            A[i], B[i] = -0.5 * u1, u2
            res2 = self.dotadd(A, B) == 0.0
            A[i], B[i] = 1.5 * u1, u2
            res3 = self.dotadd(A, B) == 2.0 * u1 * u2
            rounding = probe_rounding_helper2(res1, res2, res3)
        self.rounding_mode.append(rounding)
        assert n_fraction_bits == self.precision[0]
        assert rounding == self.rounding_mode[0]
        return n_fraction_bits, rounding

    def probe_c_rounding(self) -> tuple[int, str]:
        # rounding precision
        M = 2.0**7
        e = 14
        while self.dotadd([M, -M], [M, M], 2.0**e) == 2.0**e:
            e -= 1
        e += 1
        n_fraction_bits = 14 - e
        ulp = 2.0**e  # ulp(M*M)
        self.c_precision = n_fraction_bits
        # rounding mode
        res1 = self.dotadd([M, -M], [M, M], 0.75 * ulp) == ulp
        res2 = self.dotadd([M, -M], [M, M], -0.75 * ulp) == 0.0
        res3 = self.dotadd([M, -M], [M, M], 0.25 * ulp) == ulp
        rounding = probe_rounding_helper1(res1, res2, res3)
        if rounding == "RN":
            res1 = self.dotadd([M, -M], [M, M], 0.5 * ulp) == ulp
            res2 = self.dotadd([M, -M], [M, M], -0.5 * ulp) == 0.0
            res3 = self.dotadd([M, -M], [M, M], 1.5 * ulp) == 2.0 * ulp
            rounding = probe_rounding_helper2(res1, res2, res3)
        self.c_rounding = rounding
        return n_fraction_bits, rounding

    def probe_output_rounding(self) -> tuple[int, str]:
        # rounding precision
        p = 0
        while self.dotadd([1.0], [1.0], 2.0**p) == 1.0 + 2.0**p:
            p -= 1
        p += 1
        n_fraction_bits = -p
        self.output_precision = n_fraction_bits
        M = 2.0**7
        A = [M / 2] * 4
        B = [-M / 2] * 4
        ulp = M * M * 2.0**p  # ulp(M*M)
        # rounding mode
        res1 = self.dotadd(A, A, 0.75 * ulp) == M * M + ulp
        res2 = self.dotadd(B, A, -0.75 * ulp) == -M * M
        res3 = self.dotadd(A, A, 0.25 * ulp) == M * M + ulp
        rounding = probe_rounding_helper1(res1, res2, res3)
        if rounding == "RN":
            res1 = self.dotadd(A, A, 0.5 * ulp) == M * M + ulp
            res2 = self.dotadd(B, A, -0.5 * ulp) == -M * M
            res3 = self.dotadd(A, A, 1.5 * ulp) == M * M + 2 * ulp
            rounding = probe_rounding_helper2(res1, res2, res3)
        self.output_rounding = rounding
        return n_fraction_bits, rounding

    def is_product_normalized(self) -> bool:
        # align according to exponent(a)+exponent(b) instead of exponent(a*b)
        M1, M2 = 2.0**6, 2.0**7
        ulp = M1 * M2 * 2.0**-self.c_precision
        assert self.dotadd([M1, -M1], [M2, M2], ulp) == ulp
        assert self.dotadd([2.25 * M1, -2.25 * M1], [M2, M2], ulp) == 0.0
        return self.dotadd([1.5 * M1, -1.5 * M1], [1.5 * M2, 1.5 * M2], ulp) == 0.0

    # TODO
    # def is_c_accumulated_at_last(self, group_size: int) -> bool:
    #     M, u1, u2 = self.M, self.u1, self.u2
    #     assert self.dotadd([u1], [u2/2], -M*M) + M*M == 0.0
    #     assert self.dotadd([u1], [u2], -M*M) + M*M == u1 * u2
    #     A = [u1]*group_size
    #     B = [u2/2]*group_size
    #     res = self.dotadd(A, B, -M*M)+M*M
    #     return res == group_size * u1 * u2 / 2

    # def is_all_products_fused_summed(self, group_size: int, include_c: bool) -> bool:
    #     M, u1, u2 = self.M, self.u1, self.u2
    #     assert self.dotadd([-M, u1], [M, u2/2]) + M*M == 0.0
    #     assert self.dotadd([-M, u1], [M, u2]) + M*M == u1 * u2

    # def test_internal_order(self, group_size: int, internal_order: str) -> bool:
    #     # is c accumulated last?
    #     M, u1, u2 = self.M, self.u1, self.u2
    #     half_ulp = u1*u2/2
    #     A = [u1]*group_size
    #     B = [u2/2]*group_size
    #     res_c = (self.dotadd(A, B, M*M)-M*M) / half_ulp
    #     res = []
    #     half_ulp = M * u1 * u2 * 2.0**-self.precision
    #     for i in range(group_size):
    #         A[i] = B[i] = M
    #         res.append((self.dotadd(A, B) - M*M) / half_ulp)
    #         A[i] = u1
    #         B[i] = u2/2
    #     if internal_order == "all fused":
    #         return res_c == 0.0 and all(r == 0.0 for r in res)
    #     elif internal_order == "products fused and c last":
