"""Deterministic C literal formatting."""


def float32_literal(value: float) -> str:
    return f"{value.hex()}F"


def format_float32_array(
    values: tuple[float, ...], 
    indent: str = "    "
) -> str:
    rows: list[str] = []
    row: list[str] = []

    for value in values:
        row.append(
            float32_literal(value)
        )
        row = []

    if row:
        rows.append(indent + ", ".join(row) + ",")

    return "\n".join(rows)


def aligned_float32_array(
    name: str,
    values: tuple[float, ...]
) -> str:
    return (
        f"_Alignas(MER_MODEL_ALIGNMENT) static const float "
        f"{name}[{len(values)}] = {{\n{format_float32_array(values)}\n}};"
    )