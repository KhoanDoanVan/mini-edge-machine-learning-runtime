#!/usr/bin/env python3
"""Command-line entrypoint for the Mini Edge Runtime AOT compiler."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from mini_ort.compiler.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())