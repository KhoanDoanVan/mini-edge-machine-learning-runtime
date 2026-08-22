"""A container that evaluates modules in declaration order."""

from __future__ import annotations

from collections.abc import Iterator

from mini_ort.model import LayerSpec, LinearSpec
from .module import Module




class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        if not modules:
            raise ValueError("Sequential requires at least one module")

        if any(not isinstance(module, Module) for module in modules):
            raise TypeError("Sequential accepts Module instanc/es only")

        self._modules = tuple(modules)

        current_features: int | None = None

        for layer in self.layer_specs():
            if isinstance(layer, LinearSpec):
                if (current_features is not None and current_features != layer.in_features):
                    raise ValueError("adjacent Linear layers have incompatible features counts")
            current_features = layer.out_features


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

    def __getitem__(self, index: int) -> Module:
        return self._modules[index]