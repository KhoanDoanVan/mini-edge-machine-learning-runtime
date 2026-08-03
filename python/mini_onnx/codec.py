"""Deterministic JSON Serilization for the learning IR(not the ONNX format)"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checker import check_model
from .ir import DType, Graph, Model, Node, TensorProto, ValueInfo


def _value_to_dict(
        value: ValueInfo
) -> dict[str, Any]:
    return {
        "name": value.name,
        "dtype": value.dtype.value,
        "shape": list(value.shape)
    }



def model_to_dict(
        model: Model
) -> dict[str, Any]:

    graph = model.graph

    return {
        "ir_version": model.ir_version,
        "opset_version": model.opset_version,
        "producer_name": model.producer_name,
        "graph": {
            "name": graph.name,
            "inputs": [_value_to_dict(value) for value in graph.inputs],
            "outputs": [_value_to_dict(value) for value in graph.outputs],
            "initializers": [
                {
                    "name": tensor.name,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "data": list(tensor.data)
                } for tensor in graph.initializers
            ],
            "nodes": [
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.inputs),
                    "outputs": list(node.outputs),
                    "attributes": dict(node.attributes),
                } for node in graph.nodes
            ]
        }
    }


def _value_from_dict(
        payload: dict[str, Any]
) -> ValueInfo:
    return ValueInfo(
        payload["name"],
        DType(
            payload["dtype"]
        ),
        tuple(
            payload["shape"]
        )
    )



def model_from_dict(
        payload: dict[str, Any]
) -> Model:
    graph_payload = payload["graph"]

    model = Model(
        graph=Graph(
            name=graph_payload["name"],
            inputs=tuple(
                _value_from_dict(value) for value in graph_payload["inputs"]
            ),
            outputs=tuple(
                _value_from_dict(value) for value in graph_payload["outputs"]
            ),
            initializers=tuple(
                TensorProto(
                    tensor["name"],
                    DType(tensor["dtype"]),
                    tuple(tensor["shape"]),
                    tuple(
                        float(value) for value in tensor["data"]
                    )
                ) for tensor in graph_payload["initializers"]
            ),
            nodes=tuple(
                Node(
                    node["name"],
                    node["op_type"],
                    tuple(node["inputs"]),
                    tuple(node["outputs"]),
                    node.get(
                        "attributes", {}
                    )
                ) for node in graph_payload["nodes"]
            )
        ),
        ir_version=payload["ir_version"],
        opset_version=payload["opset_version"],
        producer_name=payload["producer_name"]
    )



def save_model(
        model: Model,
        path: str | Path
) -> None:
    check_model(model)
    Path(path).write_text(
        json.dumps(
            model_to_dict(model),
            indent=2
        ) + "\n",
        encoding="utf-8"
    )


def load_model(
        path: str | Path
) -> Model:
    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, dict):
        raise ValueError("model root must be an object")

    return model_from_dict(payload)