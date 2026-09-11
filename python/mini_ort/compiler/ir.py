"""Small, target-independent model representation used by compiler passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class LinearLayer:
    index: int
    in_features: int
    out_features: int
    weights: tuple[float, ...]
    bias: tuple[float, ...]


@dataclass(frozen=True)
class ReluLayer:
    index: int


Layer: TypeAlias = LinearLayer | ReluLayer


@dataclass(frozen=True)
class Model:
    layers: tuple[Layer, ...]
    input_features: int
    output_features: int
