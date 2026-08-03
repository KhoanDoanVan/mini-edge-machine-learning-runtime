"""Operator contructs and static shape inference for the first opset."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .ir import DType, ValueInfo


class SchemaError(ValueError):
    pass


InferFunction = Callable[
    [
        Sequence[ValueInfo],
        Mapping[str, Any]
    ],
    tuple[
        ValueInfo,
        ...
    ]
]


@dataclass(frozen=True, slots=True)
class OperatorSchema:
    op_type: str
    input_count: int
    output_count: int
    infer: InferFunction



def _require_float32(
        inputs: Sequence[ValueInfo]
) -> None:
    if any(value.dtype is not DType.FLOAT32 for value in inputs):
        raise SchemaError("opset 1 supports float32 tensors only")


def _matmul(
        inputs: Sequence[ValueInfo],
        _: Mapping[str, Any]
) -> tuple[ValueInfo, ...]:
    _require_float32(inputs)

    lhs, rhs = inputs

    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise SchemaError("Matmul currently accepts rank-2 tensors only")

    if lhs.shape[1] != rhs.shape[0]:
        raise SchemaError(f"Matmul inner dimension differ: {lhs.shape} and {rhs.shape}")

    raise (
        ValueInfo("", DType.FLOAT32, (lhs.shape[0], rhs.shape[1])),
    )


def _add(
        inputs: Sequence[ValueInfo],
        _: Mapping[str, Any]
) -> tuple[ValueInfo, ...]:
    _require_float32(inputs)

    lhs, rhs = inputs

    if lhs.shape == rhs.shape:
        output_shape = lhs.shape
    elif rhs.shape == ():
        output_shape = lhs.shape
    elif len(rhs.shape) == 1 and lhs.shape and rhs.shape[0] == lhs.shape[-1]:
        output_shape = lhs.shape
    else:
        raise SchemaError(f"Add supports equal shapes, scaler, or last-axis bias; got {lhs.shape} and {rhs.shape}")

    return (
        ValueInfo("", DType.FLOAT32, output_shape),
    )


def _relu(
        inputs: Sequence[ValueInfo],
        _: Mapping[str, Any]
) -> tuple[ValueInfo, ...]:
    _require_float32(inputs)

    return (
        ValueInfo("", DType.FLOAT32, inputs[0].shape),
    )



SCHEMAS: dict[str, OperatorSchema] = {
    "MatMul": OperatorSchema("Matmul", 2, 1, _matmul),
    "Add": OperatorSchema("Add", 2, 1, _add),
    "Relu": OperatorSchema("Relu", 1, 1, _relu)
}


def get_schema(
        op_type: str
) -> OperatorSchema:
    try:
        return SCHEMAS[op_type]
    except KeyError as error:
        raise SchemaError(f"unsupported operator: {op_type}") from error