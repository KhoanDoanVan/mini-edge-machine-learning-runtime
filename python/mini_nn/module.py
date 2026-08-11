"""Base abstraction shared by all model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from mini_ort import Tensor
from mini_ort.model import LayerSpec


class Module(ABC):

    def forward(
            self,
            input_tensor: Tensor
    ) -> Tensor:

        from mini_ort import InferenceSession

        with InferenceSession(self) as session:
            return session.run(input_tensor)

    def __call__(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        return self.forward(input_tensor)


    @abstractmethod
    def layer_specs(self) -> tuple[LayerSpec, ...]:
        raise NotImplementedError

    def named_children(self) -> Iterator[
        tuple[
            str,
            Module
        ]
    ]:
        return iter(())

    def named_parameters(
            self,
            prefix: str = ""
    ) -> Iterator[
        tuple[
            str,
            Module
        ]
    ]:
        for child_name, child in self.named_children():
            yield from child.named_parameters(f"{prefix}{child_name}.")

    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.named_parameters())

    def save(
            self,
            path: str | Path
    ):
        
        from mini_ort.serialization import save_model

        return save_model(
            self,
            path
        )