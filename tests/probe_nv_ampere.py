from mmasim.kernels.nv_ampere import mma_kernels
from mmasim.probe import ProbeFusedDotAdd, is_fused_dot_add


if __name__ == "__main__":
    for qualifier, intrinsic in mma_kernels.items():
        print(f"Testing Ampere instruction mma.{qualifier}")
        gsz = intrinsic.k
        if "tf32" in qualifier:  # tf32
            if intrinsic.k == 8:
                gsz = 4
        else:  # f16 or bf16
            if intrinsic.k == 16:
                gsz = 8
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
