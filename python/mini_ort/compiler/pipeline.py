"""Compiler orchestration independent from command-line concerns."""

from __future__ import annotations

from pathlib import Path

from .codegen import generate_header, generate_source
from .memory_planner import plan_memory
from .mer_reader import load_model
from .report import create_report, write_report


def compile_model(
        source_model: Path,
        output_dir: Path
) -> None:
    model = load_model(source_model)
    memory_plan = plan_memory(model)

    include_dir = output_dir / "include" / "mini_ort_generated_model"
    include_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (include_dir / "model.h").write_text(
        generate_header(
            model,
            memory_plan
        ),
        encoding="utf-8"
    )

    (output_dir / "model.c").write_text(
        generate_source(
            model,
            memory_plan
        ),
        encoding="utf-8"
    )

    write_report(
        create_report(
            model,
            memory_plan,
            source_model
        ),
        output_dir / "model_report.json"
    )