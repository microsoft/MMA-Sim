from mmasim.intrinsic.nv_rtxblackwell import mma_intrinsics
from mmasim.probe import ProbeFusedDotAdd, is_fused_dot_add


if __name__ == "__main__":
    for qualifier, intrinsic in mma_intrinsics.items():
        print(f"Testing RTX Blackwell instruction mma.{qualifier}")
        gsz = intrinsic.k
        assert is_fused_dot_add(intrinsic, gsz)
        print("    Type: fused dot-add operations")
        print(
            f"    Accumulation order: {intrinsic.k//gsz}-level ({gsz}+1)-term fused summation"
        )

        probe = ProbeFusedDotAdd(intrinsic)
        for i in range(gsz):
            probe.probe_product_rounding(i)
        probe.probe_c_rounding()
        print("    Summand", *range(gsz), "c", sep="\t")
        print("    Rounding", *probe.rounding_mode, probe.c_rounding, sep="\t")
        print("    Precision", *probe.precision, probe.c_precision, sep="\t")

        product_normalized = probe.is_product_normalized()
        print("    Product normalized for alignment:", product_normalized)

        output_rounding = probe.probe_output_rounding()
        print("    Output rounding:", output_rounding)
