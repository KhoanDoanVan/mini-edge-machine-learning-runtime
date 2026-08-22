"""Thin public session API by the native C++ runtime."""


from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .backend import Backend
from .tensor import Tensor
from .model import LayerSpec
from .native import NativeBackend

DynamicShape = tuple[
    int | None,
    ...
]


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

    @property
    def backend(self) -> str:
        return self.backend_name

    @property
    def input_shape(self) -> DynamicShape | None:
        features = self._backend.input_features
        return None if features is None else (None, features)

    @property
    def output_shape(self) -> DynamicShape | None:
        features = self._backend.output_features
        return None if features is None else (None, features)


    def output_shape_for(
            self,
            input_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        input_features = self._backend.input_features
        output_features = self._backend.output_features

        if input_features is None or output_features is None:
            raise RuntimeError("native model shape metadata is unavailable")

        if len(input_shape) != 2 or input_shape[1] != input_features:
            raise ValueError(
                f"expected input shape (batch, {input_features}), got {input_shape}"
            )

        return (
            input_shape[0],
            output_features
        )
    

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