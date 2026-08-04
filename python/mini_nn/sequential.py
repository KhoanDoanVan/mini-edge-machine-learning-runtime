"""A container that evaluates modules in declaration order."""

from __future__ import annotations

from collections.abc import Iterator

from mini_ort import Tensor
from .module import Module


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        if not modules:
            raise ValueError("Sequential requires at least one module")

        if any(not isinstance(module, Module) for module in modules):
            raise TypeError("Sequential accepts Module instances only")

        self._modules = tuple(modules)


    def forward(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        output = input_tensor
        for module in self._modules:
            output = module(output)

        return output


    def named_chilren(self) -> Iterator[tuple[str, Module]]:
        for index, module in enumerate(self._modules):
            yield str(index), module


    def __len__(self) -> int:
        return len(self._modules)


    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)