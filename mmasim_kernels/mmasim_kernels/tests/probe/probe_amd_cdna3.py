from mmasim.kernels.amd_cdna3 import mfma_kernels
from mmasim.probe import ProbeFusedDotAdd, is_fused_dot_add, is_pairwise_sum


if __name__ == "__main__":
    for qualifier, intrinsic in mfma_kernels.items():
        print(f"Testing CDNA3 instruction mfma_{qualifier}")
        if qualifier.endswith(("_xf32", "_f16", "_bf16", "_fp8", "_bf8")):
            if qualifier.endswith("_xf32"):
                gsz = 4
            elif qualifier.endswith(("_f16", "_bf16")):
                gsz = min(8, intrinsic.k)
            else:  # fp8 or bf8
                gsz = 16
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
        else:
            assert is_pairwise_sum(intrinsic, 1)
            print("    Type: standard floating-point operations")
            print(f"    Accumulation order: sequential")
