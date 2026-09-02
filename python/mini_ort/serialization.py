"""Writer for the compact, versioned MER model format."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Protocol

from .model import LayerSpec, LinearSpec, ReluSpec


_MAGIC = b"MERMDL1\0"
_VERSION = 1
_LINEAR = 1
_RELU = 2
_HEADER = struct.Struct("<8sII")
_LAYER_HEADER = struct.Struct("<IIQQQQ")

class LayerProvider(Protocol):

    def layer_specs(self) -> tuple[LayerSpec, ...]:
        ...



def _write_float32(
        file: BinaryIO,
        values: tuple[float, ...]
) -> None:
    if values:
        file.write(
            struct.pack(f"<{len(values)}f", *values)
        )


def save_model(
        model: LayerProvider,
        path: str | Path
) -> Path:
    layers = model.layer_specs()

    if not layers:
        raise ValueError("a MER model requires at least one layer")

    if len(layers) > 0xFFFFFFFF:
        raise ValueError("the model has too many layers")


    destination = Path(path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with destination.open("wb") as file:

        file.write(
            _HEADER.pack(
                _MAGIC,
                _VERSION,
                len(layers)
            )
        )

        for layer in layers:
            if isinstance(layer, LinearSpec):
                bias_count = 0 if layer.bias is None else layer.bias.size

                file.write(
                    _LAYER_HEADER.pack(
                        _LINEAR,
                        0,
                        layer.in_features,
                        layer.out_features,
                        layer.weight.size,
                        bias_count
                    )
                )

                _write_float32(
                    file,
                    layer.weight.data
                )

                if layer.bias is not None:
                    _write_float32(file, layer.bias.data)
            elif isinstance(
                layer, 
                ReluSpec
            ):
                file.write(
                    _LAYER_HEADER.pack(
                        _RELU,
                        0,
                        0,
                        0,
                        0,
                        0
                    )
                )
            else:
                raise TypeError(f"unsupported MER layer: {type(layer).__name__}")
    return destination