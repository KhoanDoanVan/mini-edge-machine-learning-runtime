"""Portable high-level tensor value used at the backend boundary."""


from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from struct import pack, unpack
from typing import Iterable
from functools import reduce
from enum import Enum
from operator import mul



class DType(
    str,
    Enum
):
    FLOAT32 = "float32"



Shape = tuple[int, ...]


def _element_count(shape: Shape) -> int:
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in shape):
        raise TypeError("tensor dimensions must be integers")

    if any(dimension < 0 for dimension in shape):
        raise ValueError("tensor dimensions must be non-negative")

    return reduce(
        mul,
        shape,
        1
    )



@dataclass(frozen=True, slots=True)
class Tensor:
    shape: Shape
    data: tuple[float, ...]
    dtype: DType = DType.FLOAT32


    def __post_init__(self) -> None:

        if self.dtype is not DType.FLOAT32:
            raise TypeError("the Python runtime currently supports float32 only")

        expected_size = _element_count(self.shape)

        if len(self.data) != expected_size:
            raise ValueError(f"shape {self.shape} needs {expected_size} values, got {len(self.data)}")

        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.data):
            raise ValueError("tensor data must contain real numbers")

        try:
            normalized = tuple(unpack("!f", pack("!f", float(value)))[0] for value in self.data)
        except OverflowError as error:
            raise ValueError("tensor value is outside the float32 range") from error

        object.__setattr__(self, "data", normalized)


    @classmethod
    def from_list(
        cls, 
        shape: Shape,
        values: Iterable[float] 
    ) -> Tensor:
        return cls(shape, tuple(float(value) for value in values))

    @property
    def size(self) -> int:
        return len(self.data)

    def to_list(self) -> list[float]:
        return list(self.data)
    