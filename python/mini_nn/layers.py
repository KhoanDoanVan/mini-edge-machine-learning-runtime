"""Small set of inference-only neural-network layers."""

from __future__ import annotations

from collections.abc import Iterator

from mini_ort import Tensor
from mini_ort.model import LinearSpec, ReluSpec

from .module import Module


class Linear(Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            weight: Tensor,
            bias: Tensor | None = None
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Lienar feature counts must be positive")

        if weight.shape != (in_features, out_features):
            raise ValueError(
                f"Linear weight must have shape {(in_features, out_features)}, got {weight.shape}."
            )

        if bias is not None and bias.shape != (out_features,):
            raise ValueError(f"Linear bias must have shape {(out_features,)} got {bias.shape}.")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias


    def layer_specs(self) -> tuple[LinearSpec, ...]:
        return (
            LinearSpec(
                self.in_features,
                self.out_features,
                self.weight,
                self.bias
            )
        )

    def named_parameters(
            self, 
            prefix = ""
    ) -> Iterator[
        tuple[
            str,
            Tensor
        ]
    ]:
        yield f"{prefix}weight", self.weight
        if self.bias is not None:
            yield f"{prefix}bias", self.bias
    

class ReLU(Module):
    def layer_specs(self) -> tuple[ReluSpec, ...]:
        return (
            ReluSpec()
        )