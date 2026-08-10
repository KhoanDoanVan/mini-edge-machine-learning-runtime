"""Language-neural layer descriptions passed to the native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .tensor import Tensor


@dataclass(frozen=True, slots=True)
class LinearSpec:
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor | None


@dataclass(frozen=True, slots=True)
class ReluSpec:
    pass


LayerSpec: TypeAlias = LinearSpec | ReluSpec