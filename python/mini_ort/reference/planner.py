"""Golden sequential planner with simple intermediate-value liveness"""

from __future__ import annotations

from dataclasses import dataclass

from mini_onnx.checker import CheckedModel
from mini_onnx.ir import Node


from .kernels import Kernel, KernelRegistry


@dataclass(frozen=True, slots=True)
class PlanStep:
    node: Node
    kernel: Kernel
    release_after: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    steps: tuple[PlanStep, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


def build_execution_plan(
        checked: CheckedModel,
        registry: KernelRegistry
) -> ExecutionPlan:
    graph = checked.model.graph
    output_names = tuple(value.name for value in graph.outputs)
    retained = set(output_names)
    last_use: dict[str, int] = {}

    for index, node in enumerate(graph.nodes):
        for input_name in node.inputs:
            last_use[input_name] = index

    steps: list[PlanStep] = []

    for index, node in enumerate(graph.nodes):
        releasable = tuple(
            dict.fromkeys(
                input_name for input_name in node.inputs if last_use[input_name] == index and input_name not in retained
            )
        )
        steps.append(
            PlanStep(
                node,
                registry.resolve(node.op_type),
                releasable
            )
        )

    return ExecutionPlan(
        tuple(steps),
        tuple(value.name for value in graph.inputs),
        output_names
    )