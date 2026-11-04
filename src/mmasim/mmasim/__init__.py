import torch


class MMAOperation:
    def __call__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor: ...

    def dpa(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor: ...


class MMABlockScaleOperation:
    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor: ...

    def dpa(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor: ...


class DPOperation:
    """Dot-product operation without an initial accumulator operand."""

    def dp(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor: ...
