"""Command-line interface for Micro/AOT compilation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .errors import ModelFormatError
from .pipeline import compile_model


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> int:
    parser = create_parser()
    arguments = parser.parse_args()
    try:
        compile_model(arguments.model, arguments.output_dir)
    except (OSError, ModelFormatError) as error:
        parser.error(str(error))
    return 0
