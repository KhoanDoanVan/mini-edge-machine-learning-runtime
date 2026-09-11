"""Semantic validation for the compiler IR."""

from __future__ import annotations

from collections.abc import Sequence

from .errors import ModelFormatError
from .ir import Layer, LinearLayer, Model, ReluLayer


def validate_model(
        layers: Sequence[Layer]
) -> Model:
    if not layers:
        raise ModelFormatError("model has no layers")

    current_features: int | None = None
    input_features: int | None = None

    for layer in layers:
        if isinstance(layer, LinearLayer):
            if current_features is not None and layer.in_features != current_features:
                raise ModelFormatError(
                    f"Linear layer {layer.index} input shape does not match"
                )

            if input_features is None:
                input_features = layer.in_features

            current_features = layer.out_features
        elif isinstance(layer, ReluLayer) and current_features is None:
            raise ModelFormatError(
                "Micro/AOT v1 does not support ReLU before Linear"
            )

    if input_features is None or current_features is None:
        raise ModelFormatError("Micro/AOT v1 requires at least one Linear layer")

    return Model(tuple(layers), input_features, current_features)