"""Readable golden operators; these are not production CPU Kernels"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from ..tensor import Tensor


def matmul(
        lhs: Tensor,
        rhs: Tensor
) -> Tensor:
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise ValueError("Matmul expects rank-2 tensors")

    rows, inner = lhs.shape

    rhs_inner, columns = rhs.shape

    if inner != rhs_inner:
        raise ValueError(f"MatMul inner dimensions differ: {lhs.shape} and {rhs.shape}")

    output = [0.0] * (rows * columns)

    # i-k-j ordering reuses lhs values and walks rhs/output contigously.
    for row in range(rows):
        lhs_row = row * inner
        output_row = row * columns
        for reduction in range(inner):
            lhs_value = lhs.data[lhs_row + reduction]
            rhs_row = reduction * columns
            for column in range(columns):
                output[output_row + column] += lhs_value * rhs.data[rhs_row + column]

    return Tensor.from_list(
        (rows, columns),
        output
    )


def add(
        lhs: Tensor,
        rhs: Tensor
) -> Tensor:
    if lhs.shape == rhs.shape:
        output = [
            left + right for left, right in zip(lhs.data, rhs.data, strict=True)
        ]
    elif rhs.shape == ():
        output = [
            value + rhs.data[0] for value in lhs.data
        ]
    elif len(rhs.shape) == 1 and lhs.shape and rhs.shape[0] == lhs.shape[-1]:
        width = rhs.shape[0]
        output = [
            value + rhs.data[index % width] for index, value in enumerate(lhs.data)
        ]
    else:
        raise ValueError(f"Add supports equal shapes, scaler, or last-axis bias; got {lhs.shape} and {rhs.shape}")

    return Tensor.from_list(lhs.shape, output)


def relu(
        source: Tensor
) -> Tensor:
    return Tensor.from_list(
        source.shape,
        (max(0.0, value) for value in source.data)
    )



class Kernel(ABC):
    @abstractmethod
    def compute(
        self,
        inputs: Sequence[Tensor],
        attributes: Mapping[str, Any]
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError


class KernelRegistry:
    def __init__(self) -> None:
        self._kernels: dict[str, Kernel] = {}

    def register(
            self,
            op_type: str,
            kernel: Kernel
    ) -> None:
        if op_type in self._kernels:
            raise ValueError(f"kernel already registered: {op_type}")
        self._kernels[op_type] = kernel

    def resolve(
            self,
            op_type: str
    ) -> Kernel:
        try:
            return self._kernels[op_type]
        except KeyError as error:
            raise LookupError(f"no reference kernel registered for {op_type}") from error



class _MatMulKernel(Kernel):
    def compute(
            self,
            inputs: Sequence[Tensor],
            attributes: Mapping[str, Any]
    ) -> tuple[Tensor, ...]:
        return (
            matmul(
                inputs[0],
                inputs[1]
            )
        )


class _AddKernel(Kernel):
    def compute(
            self,
            inputs: Sequence[Tensor],
            attributes: Mapping[str, Any]
    ) -> tuple[Tensor, ...]:
        return (
            add(
                inputs[0],
                inputs[1]
            )
        )


class _ReluKernel(Kernel):
    def compute(
            self,
            inputs: Sequence[Tensor],
            attributes: Mapping[str, Any]
    ) -> tuple[Tensor, ...]:
        return (
            relu(
                inputs[0]
            )
        )


def create_reference_registry() -> KernelRegistery:
    registry = KernelRegistery()
    registry.register("MatMul", _MatMulKernel())
    registry.register("Add", _AddKernel())
    registry.register("Relu", _ReluKernel())
    return registry