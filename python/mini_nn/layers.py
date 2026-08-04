"""Small set of inference-only neural-network layers."""

from __future__ import annotations

from collections.abc import Iterator

from mini_ort import Tensor
from mini_ort.reference import add, matmul, relu

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


    def forward(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        if len(input_tensor.shape) != 2 or input_tensor.shape[1] != self.in_features:
            raise ValueError(
                f"Linear input must have shape (batch, {self.in_features}), got {input_tensor.shape}"
            )

        output = matmul(input_tensor, self.weight)
        return add(output, self.bias) if self.bias is not None else output


    def named_parameters(self, prefix = "") -> Iterator[tuple[str, Tensor]]:
        yield f"{prefix}weight", self.weight
        if self.bias is not None:
            yield f"{prefix}bias", self.bias


class ReLU(Module):
    def forward(
            self, 
            input_tensor: Tensor
    ) -> Tensor:
        return relu(input_tensor)