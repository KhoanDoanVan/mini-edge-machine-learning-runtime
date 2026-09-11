"""Strict reader for the versioned little-endian .mer model format."""

from __future__ import annotations

import math
import struct
from pathlib import Path

from .errors import ModelFormatError
from .ir import Layer, LinearLayer, Model, ReluLayer
from .validation import validate_model


MAGIC = b"MERMDL1\0"
VERSION = 1
LINEAR = 1
RELU = 2
FILE_HEADER = struct.Struct("<8sII")
LAYER_HEADER = struct.Struct("<IIQQQQ")


def _read_exact(
        data: bytes,
        offset: int,
        byte_count: int
) -> tuple[bytes, int]:
    end = offset + byte_count
    if end > len(data):
        raise ModelFormatError("model is truncated")
    return data[offset:end], end


def _read_floats(
        data: bytes,
        offset: int,
        count: int
) -> tuple[
    tuple[float, ...],
    int
]:
    payload, offset = _read_exact(
        data,
        offset,
        count * 4
    )
    values = tuple(
        value[0] for value in struct.iter_unpack("<f", payload)
    )

    if any(not math.isfinite(value) for value in values):
        raise ModelFormatError("model contains a non-finite float")

    return values, offset


def load_model(path: Path) -> Model:
    data = path.read_bytes()

    if len(data) < FILE_HEADER.size:
        raise ModelFormatError("model is smaller than the file header")

    magic, version, layer_count = FILE_HEADER.unpack_from(data)

    if magic != MAGIC:
        raise ModelFormatError("invalid .mer magic")

    if version != VERSION:
        raise ModelFormatError(f"unsupported .mer version: {version}")

    if layer_count == 0:
        raise ModelFormatError("model has no layers")

    layers: list[Layer] = []
    offset = FILE_HEADER.size

    for index in range(layer_count):
        header, offset = _read_exact(
            data,
            offset,
            LAYER_HEADER.size
        )

        (
            layer_type,
            reserved,
            in_features,
            out_features,
            weight_count,
            bias_count
        ) = LAYER_HEADER.unpack(header)

        if reserved != 0:
            raise ModelFormatError(f"layer {index} has a nonzero reserved field")

        if layer_type == LINEAR:
            if in_features == 0 or out_features == 0:
                raise ModelFormatError(f"Linear layer {index} has an empty dimension")

            if in_features > ((1 << 64) - 1) // out_features:
                raise ModelFormatError(f"Linear layer {index} dimensions overflow")

            if weight_count != in_features * out_features:
                raise ModelFormatError(f"Linear layer {index} has an invalid bias count")

            weights, offset = _read_floats(data, offset, weight_count)
            bias, offset = _read_floats(data, offset, bias_count)
            layers.append(
                LinearLayer(
                    index=index,
                    in_features=in_features,
                    out_features=out_features,
                    weights=weights,
                    bias=bias
                )
            )
        elif layer_type == RELU:
            if any((in_features, out_features, weight_count, bias_count)):
                raise ModelFormatError(f"ReLU layer {index} contains unexpected payload metadata")
            layers.append(
                ReluLayer(index=index)
            )
        else:
            raise ModelFormatError(f"unsupported layer type {layer_type} at layer {index}")

    if offset != len(data):
        raise ModelFormatError("model contains trailing bytes")

    return validate_model(layers)