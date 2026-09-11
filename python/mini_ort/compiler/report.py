"""Machine-readable compiler report generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .ir import LinearLayer, Model
from .memory_planner import MemoryPlan
from .mer_reader import VERSION


GENERATOR_VERSION = 1


def create_report(
    model: Model, 
    memory_plan: MemoryPlan, 
    source_model: Path
) -> dict[str, Any]:
    destinations = memory_plan.destinations

    return {
        "format": "mer",
        "format_version": VERSION,
        "generator_version": GENERATOR_VERSION,
        "source_model": source_model.name,
        "source_model_sha256": hashlib.sha256(
            source_model.read_bytes()
        ).hexdigest(),
        "input_elements": model.input_features,
        "output_elements": model.output_features,
        "layer_count": len(model.layers),
        "alignment": memory_plan.alignment,
        "arena_bytes": memory_plan.arena_bytes,
        "scratch_bytes": 0,
        "weight_bytes": sum(
            (len(layer.weights) + len(layer.bias)) * 4
            for layer in model.layers
            if isinstance(layer, LinearLayer)
        ),
        "layers": [
            {
                "index": layer.index,
                "type": "Linear" if isinstance(layer, LinearLayer) else "ReLU",
                **(
                    {
                        "in_features": layer.in_features,
                        "out_features": layer.out_features,
                        "has_bias": bool(layer.bias),
                        "destination": destinations[layer.index].kind,
                        "arena_offset": destinations[layer.index].offset
                        if destinations[layer.index].kind == "arena"
                        else None,
                    }
                    if isinstance(layer, LinearLayer)
                    else {}
                ),
            }
            for layer in model.layers
        ],
    }


def write_report(
    report: dict[str, Any], 
    path: Path
) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
