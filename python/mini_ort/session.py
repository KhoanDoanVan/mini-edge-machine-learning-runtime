"""Thin public session API by the native C++ runtime."""


from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .backend import Backend
from .tensor import Tensor
from .model import LayerSpec
from .native import NativeBackend


class LayerProvider(Protocol):
    def layer_specs(self) -> tuple[LayerSpec, ...]:
        ...


class InferenceSession:

    def __init__(
            self,
            model: LayerProvider | str | Path,
            backend: Backend | None = None
    ) -> None:
        if backend is not None:
            self._backend = backend
        elif isinstance(model, (str, Path)):
            self._backend = NativeBackend.from_file(model)
        else:
            self._backend = NativeBackend(model.layer_specs())

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def run(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        return self._backend.run(input_tensor)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> InferenceSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()