#!/usr/bin/env python3
"""Export a local .mer v1 fixture without external dependencies."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


MAGIC = b"MERMDL1\0"
FILE_HEADER = struct.Struct("<8sII")
LAYER_HEADER = struct.Struct("<IIQQQQ")


def linear(
    in_features: int,
    out_features: int,
    weights: list[float],
    bias: list[float],
) -> bytes:
    return b"".join(
        (
            LAYER_HEADER.pack(
                1,
                0,
                in_features,
                out_features,
                len(weights),
                len(bias),
            ),
            struct.pack(f"<{len(weights)}f", *weights),
            struct.pack(f"<{len(bias)}f", *bias),
        )
    )


def relu() -> bytes:
    return LAYER_HEADER.pack(2, 0, 0, 0, 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    first_weights = [
        1.0, 0.0, -1.0, 0.5,
        0.0, 1.0, 1.0, -1.0,
        1.0, 1.0, 0.0, 0.5,
    ]
    second_weights = [
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        -1.0, 0.5,
    ]
    artifact = b"".join(
        (
            FILE_HEADER.pack(MAGIC, 1, 3),
            linear(3, 4, first_weights, [0.0, 0.0, 0.0, 0.0]),
            relu(),
            linear(4, 2, second_weights, [0.5, -0.5]),
        )
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(artifact)
    print(f"wrote {arguments.output} ({len(artifact)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
