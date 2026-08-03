"""Executable pure-Python backend that acts as the correctness oracle."""

from __future__ import annotations

from typing import Mapping

from mini_onnx import Model, check_model
from mini_onnx.ir import ValueInfo

from ..tensor import Tensor
from .kernels import create_reference_registry
from .planner import build_execution_plan


class ReferenceBackend:
    def __init__(
            self,
            model: Model
    ) -> None:
        self._model = model
        self._checked = check_model(model)
        self._plan = build_execution_plan(
            self._checked,
            create_reference_registry()
        )
        self._initializers = {
            initializer.name: Tensor.from_proto(initializer) for initializer in model.graph.initializers
        }


    @property
    def name(self) -> str:
        return "python-reference"

    @staticmethod
    def _validate_tensor(
        name: str,
        tensor: Tensor,
        expected: ValueInfo
    ) -> None:
        if tensor.dtype != expected.dtype or tensor.shape != expected.shape:
            raise ValueError(
                f"{name}: expected {expected.dtype.value}{expected.shape}"
                f"got {tensor.dtype.value}{tensor.shape}"
            )


    def run(
            self,
            feeds: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        expected_input = set(self._plan.input_names)

        if set(feeds) != expected_input:
            missing = sorted(expected_input - set(feeds))
            extra = sorted(set(feeds) - expected_input)
            raise ValueError(
                f"feed names do not match graph input; missing={missing}, extra={extra}"
            )

        values = dict(self._initializers)

        for name, tensor in feeds.items():
            self._validate_tensor(
                name,
                tensor,
                self._checked.values[name]
            )

        for step in self._plan.steps:
            inputs = [
                values[name] for name in step.node.inputs
            ]
            outputs = step.kernel.compute(
                inputs,
                step.node.attributes
            )

            if len(outputs) != len(step.node.outputs):
                raise RuntimeError(f"{step.node.name}: kernel returned an invalid output count")

            for output_name, tensor in zip(step.node.outputs, outputs, strict=True):
                self._validate_tensor(
                    output_name,
                    tensor,
                    self._checked.values[output_name]
                )
                values[output_name] = tensor

            for value_name in step.release_after:
                del values[value_name]

        return {
            name: values[name] for name in self._plan.output_names
        }