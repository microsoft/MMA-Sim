from mmasim.probe.intrinsic.amd_cdna2 import mfma_intrinsics
from mmasim.probe import is_pairwise_sum


if __name__ == "__main__":
    for qualifier, intrinsic in mfma_intrinsics.items():
        print(f"Testing CDNA2 instruction mfma_{qualifier}")
        if qualifier.endswith("f32"):
            gsz = 1
        elif qualifier.endswith("bf16"):  # CDNA1 bf16
            gsz = 2
        elif qualifier.endswith("_1k"):  # CDNA2 bf16
            gsz = 4
        else:  # f16
            gsz = 4
        assert is_pairwise_sum(intrinsic, gsz)
        print("    Type: standard floating-point operations")
        if gsz == 1:
            print("    Accumulation order: sequential")
        else:
            print(
                f"    Accumulation order: sequential sum of c and {intrinsic.k//gsz} partial sum(s), each computed as the pairwise sum of {gsz} terms"
            )
