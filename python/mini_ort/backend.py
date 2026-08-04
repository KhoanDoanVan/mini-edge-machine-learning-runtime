"""Backend boundary used by the high-level session API"""

from __future__ import annotations

from typing import Mapping, Protocol

from .tensor import Tensor


class Backend(Protocol):
    """Something that can execute a compiled model."""

    @property
    def name(self) -> str:
        ...


    def run(
            self,
            feeds: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        ...