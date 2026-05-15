from typing import Callable


from mmasim.nv_ptx import isa_mma, isa_wgmma, isa_tcgen05mma
from .. import MMAKernel, MMABlockScaleKernel


class MMA(isa_mma.MMA, MMAKernel):
    def __init__(self, arch: str, shape_and_type: str, kernel: Callable):
        super().__init__(arch, shape_and_type)
        self.kernel = kernel


class MMABlockScale(isa_mma.MMABlockScale, MMABlockScaleKernel):
    def __init__(self, arch: str, shape_and_type: str, kernel: Callable):
        super().__init__(arch, shape_and_type)
        self.kernel = kernel


class WGMMA(isa_wgmma.WGMMA, MMAKernel):
    def __init__(self, arch: str, shape_and_type: str, kernel: Callable):
        super().__init__(arch, shape_and_type)
        self.kernel = kernel


class TCGen05MMA(isa_tcgen05mma.TCGen05MMA, MMAKernel):
    def __init__(self, arch: str, shape_and_type: str, kernel: Callable):
        super().__init__(arch, shape_and_type)
        self.kernel = kernel


class TCGen05MMABlockScale(isa_tcgen05mma.TCGen05MMABlockScale, MMABlockScaleKernel):
    def __init__(self, arch: str, shape_and_type: str, kernel: Callable):
        super().__init__(arch, shape_and_type)
        self.kernel = kernel
