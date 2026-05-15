from typing import Callable


from mmasim.amd import isa_mfma
from .. import MMAKernel


class MFMA(isa_mfma.MFMA, MMAKernel):
    def __init__(self, arch: str, suffix: str, kernel: Callable):
        super().__init__(arch, suffix)
        self.kernel = kernel
