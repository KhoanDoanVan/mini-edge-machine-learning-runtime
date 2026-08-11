"""Backend boundary used by the high-level session API"""

from __future__ import annotations

from typing import Protocol

from .tensor import Tensor


class Backend(Protocol):
    """Something that can execute a compiled model."""

    @property
    def name(self) -> str:
        ...


    def run(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        ...

    def close(self) -> None:
        ...