"""Thin public session API; execution belongs to a selected backend."""


from __future__ import annotations

from pathlib import Path
from typing import Mapping

from mini_onnx import Model, load_model

from .backend import Backend
from .reference.backend import ReferenceBackend
from .tensor import Tensor


class InferenceSession:
    def __init__(
            self,
            model: Model | str | Path,
            backend: Backend | None = None
    ) -> None:
        loaded_model = load_model(model) if isinstance(model, (str, Path)) else model
        # NativeBackend will replace this default after the C ABI milestone.
        self._backend = backend or ReferenceBackend(loaded_model)


    @property
    def backend_name(self) -> str:
        return self._backend.name

    def run(
            self,
            feeds: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        return self._backend.run(feeds)

    