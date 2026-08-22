"""Dependency-free parameter initialization for small dense models."""

from __future__ import annotations

from math import sqrt
from random import Random
from typing import Literal, TypeAlias

from mini_ort import Tensor


Initializer: TypeAlias = Literal[
    "xavier_uniform",
    "zeros",
    "ones"
]


def initializer_weight(
        in_features: int,
        out_features: int,
        method: Initializer,
        seed: int | None
) -> Tensor:

    element_count = in_features * out_features

    if method == "xavier_uniform":
        limit = sqrt(6.0 / (in_features + out_features))
        generator = Random(seed)
        values = tuple(
            generator.uniform(-limit, limit) for _ in range(element_count)
        )

    elif method == "zeros":
        values = (0.0,) * element_count

    elif method == "ones":
        values = (1.0,) * element_count

    else:
        raise ValueError(f"unsupported Linear initialization: {method}")

    return Tensor(
        (in_features, out_features),
        values
    )