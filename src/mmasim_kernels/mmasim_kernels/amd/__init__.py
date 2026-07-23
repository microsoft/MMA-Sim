from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ctypes import _NamedFuncPointer
else:
    _NamedFuncPointer = object

from mmasim.amd import isa_mfma
from .. import MMAKernel


class MFMA(isa_mfma.MFMA, MMAKernel):
    def __init__(self, arch: str, suffix: str, kernel: _NamedFuncPointer):
        super().__init__(arch, suffix)
        self.kernel = kernel
