"""Static activation placement for the sequential Micro/AOT graph."""

from __future__ import annotations

from dataclasses import dataclass

from .ir import LinearLayer, Model


DEFAULT_ALIGNMENT = 16
FLOAT32_BYTES = 4


@dataclass(frozen=True)
class Destination:
    kind: str
    offset: int
    byte_count: int


@dataclass(frozen=True)
class MemoryPlan:
    destinations: dict[int, Destination]
    arena_bytes: int
    alignment: int


def _align_up(
    value: int, 
    alignment: int
) -> int:
    return (value + alignment - 1) // alignment * alignment


def plan_memory(
    model: Model, 
    alignment: int = DEFAULT_ALIGNMENT
) -> MemoryPlan:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("memory alignment must be a positive power of two")


    linear_layers = [
        layer for layer in model.layers if isinstance(layer, LinearLayer)
    ]

    last_linear_index = linear_layers[-1].index
    slot_sizes = [0, 0]
    slot_for_layer: dict[int, int] = {}
    next_slot = 0

    for layer in linear_layers:
        if layer.index == last_linear_index:
            continue

        byte_count = layer.out_features * FLOAT32_BYTES
        slot_for_layer[layer.index] = next_slot
        slot_sizes[next_slot] = max(
            slot_sizes[next_slot], 
            byte_count
        )
        next_slot = 1 - next_slot

    slot_offsets = [0, _align_up(slot_sizes[0], alignment)]
    arena_bytes = _align_up(slot_offsets[1] + slot_sizes[1], alignment)
    destinations: dict[int, Destination] = {}

    for layer in linear_layers:
        byte_count = layer.out_features * FLOAT32_BYTES

        if layer.index == last_linear_index:
            destinations[layer.index] = Destination(
                "output", 
                0, 
                byte_count
            )

        else:
            slot = slot_for_layer[layer.index]
            destinations[layer.index] = Destination(
                "arena", 
                slot_offsets[slot], 
                byte_count
            )

    return MemoryPlan(
        destinations,
        arena_bytes,
        alignment
    )