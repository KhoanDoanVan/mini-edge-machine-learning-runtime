"""Backend adapter that executes complete models in the C++ runtime."""

from __future__ import annotations

from pathlib import Path

from ..model import LayerSpec
from ..tensor import Tensor
from .ffi import NativeSessionHandle


class NativeBackend:

    def __init__(
            self,
            layers: tuple[LayerSpec, ...]
    ) -> None:
        self._session = NativeSessionHandle(layers)


    @classmethod
    def from_file(
        cls,
        model_path: str | Path
    ) -> NativeBackend:
        backend = cls.__new__(cls)
        backend._session = NativeSessionHandle.from_file(model_path)
        return backend

    @property
    def name(self) -> str:
        return "cpp-native"

    @property
    def input_features(self) -> int | None:
        return self._session.input_features

    @property
    def output_features(self) -> int | None:
        return self._session.output_features

    def run(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        return self._session.run(input_tensor)

    def close(self) -> None:
        self._session.close()