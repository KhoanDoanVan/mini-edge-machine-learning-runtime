"""A container that evaluates modules in declaration order."""

from __future__ import annotations

from collections.abc import Iterator

from mini_ort.model import LayerSpec
from .module import Module




class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        if not modules:
            raise ValueError("Sequential requires at least one module")

        if any(not isinstance(module, Module) for module in modules):
            raise TypeError("Sequential accepts Module instances only")

        self._modules = tuple(modules)


    def layer_specs(self) -> tuple[LayerSpec, ...]:
        return tuple(
            layer for module in self._modules for layer in module.layer_specs()
        )


    def named_chilren(self) -> Iterator[tuple[str, Module]]:
        for index, module in enumerate(self._modules):
            yield str(index), module


    def __len__(self) -> int:
        return len(self._modules)


    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)