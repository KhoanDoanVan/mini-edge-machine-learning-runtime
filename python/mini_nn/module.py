"""Base abstraction shared by all model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from mini_ort import Tensor


class Module(ABC):

    @abstractmethod
    def forward(
        self,
        input_tensor: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def __call__(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        return self.forward(input_tensor)

    def named_children(self) -> Iterator[tuple[str, Module]]:
        return iter(())

    def named_parameters(
            self,
            prefix: str = ""
    ) -> Iterator[tuple[str, Tensor]]:
        for child_name, child in self.named_children():
            yield from child.named_parameters(f"{prefix}{child_name}.")


    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.named_parameters())