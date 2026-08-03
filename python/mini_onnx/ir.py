"""In-memory representation of a small static inference graph"""


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Mapping


class DType(
    str,
    Enum
):
    FLOAT32 = "float32"


Shape = tuple[
    int, 
    ...
]


def element_count(
        shape: Shape
) -> int:
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"shape dimensions must be non-negative, got {shape}")

    return reduce(mul, shape, 1)


@dataclass(frozen=True, slots=True)
class ValueInfo:
    name: str
    dtype: DType
    shape: Shape


@dataclass(frozen=True, slots=True)
class TensorProto:
    name: str
    dtype: DType
    shape: Shape
    data: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Graph:
    name: str
    inputs: tuple[ValueInfo, ...]
    outputs: tuple[ValueInfo, ...]
    nodes: tuple[Node, ...]
    initializers: tuple[TensorProto, ...] = ()


@dataclass(frozen=True, slots=True)
class Model:
    graph: Graph
    ir_version: int = 1
    opset_version: int = 1
    producer_name: str = "mini-edge-runtime"