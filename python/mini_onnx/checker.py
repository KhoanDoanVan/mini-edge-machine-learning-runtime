"""Structural and type validation for model graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Model, ValueInfo, element_count
from .schemas import SchemaError, get_schema


class ModelValidationError(ValueError):
    pass



@dataclass(frozen=True, slots=True)
class CheckedModel:
    model: Model
    values: dict[str, ValueInfo]


def _unique_names(
        items: object,
        kind: str
) -> None:
    names = [
        item.name for item in items
    ]

    if len(names) != len(set(names)):
        raise ModelValidationError(f"{kind} names must be unique")

    if any(not name for name in names):
        raise ModelValidationError(f"{kind} names must not be empty")


def check_model(
        model: Model
) -> CheckedModel:
    if model.ir_version != 1 or model.opset_version != 1:
        raise ModelValidationError("only IR version 1 and opset version 1 are supported")

    graph = model.graph

    _unique_names(graph.inputs, "graph input")
    _unique_names(graph.outputs, "graph output")
    _unique_names(graph.initializers, "initializer")
    _unique_names(graph.nodes, "node")

    values: dict[str, ValueInfo] = {}

    # INPUTS
    for value in graph.inputs:
        if any(dimension < 0 for dimension in value.shape):
            raise ModelValidationError(f"{value.name}: dynamic or negative dimensions are not supported")

        values[value.name] = value

    # INITIALIZERS
    for initializer in graph.initializers:
        if initializer.name in values:
            raise ModelValidationError(f"duplicate graph value: {initializer.name}")

        expected_size = element_count(initializer.shape)

        if len(initializer.data) != expected_size:
            raise ModelValidationError(f"{initializer.name}: expected {expected_size} values, got {len(initializer.data)}")

        values[initializer.name] = ValueInfo(
            initializer.name,
            initializer.dtype,
            initializer.shape
        )

    # NODES
    for node in graph.nodes:
        if not node.outputs or any(not name for name in node.outputs):
            raise ModelValidationError(f"{node.name}: outputs must be named")

        missing = [
            name for name in node.inputs if name in node.outputs
        ]

        if missing:
            raise ModelValidationError(f"{node.name}: inputs are unavailable or graph is not topological: {missing}")

        if any(name in values for name in node.outputs):
            raise ModelValidationError(f"{node.name}: output names must be globally unique")

        try:
            schema = get_schema(node.op_type)

            if len(node.inputs) != schema.input_count or len(node.outputs) != schema.output_count:
                raise SchemaError(
                    f"expected {schema.input_count} inputs and {schema.output_count} outputs, "
                    f"got {len(node.inputs)} and {len(node.outputs)}."
                )

            inferred = schema.infer(
                [
                    values[name] for name in node.inputs
                ],
                node.attributes
            )

        except SchemaError as error:
            raise ModelValidationError(f"{node.name} ({node.op_type}): {error}") from error

        for output_name, output_info in zip(node.outputs, inferred, strict=True):
            values[output_name] = ValueInfo(
                output_name,
                output_info.dtype,
                output_info.shape
            )

    # OUTPUTS
    for declared_output in graph.outputs:
        inferred_output = values.get(declared_output.name)
        if inferred_output is None:
            raise ModelValidationError(f"graph output is not produced: {declared_output.name}")

        if inferred_output.dtype != declared_output.dtype or inferred_output.shape != declared_output.shape:
            raise ModelValidationError(
                f"graph output {declared_output.name} declares {declared_output.dtype.value}{declared_output.shape}, "
                f"inferred {inferred_output.dtype.name}{inferred_output.shape}"
            )

    return CheckedModel(model, values)