"""Pie chart parser — standard categorical_x format (label=slice, value=percentage)."""
from __future__ import annotations
from pathlib import Path
from ._helpers import parse_standard, show_standard_table
from ..rms.types import AxisRanges
from ..row_types import ChartRow, StandardRow
from ..rms.types import AxisRanges

_T = "pie"


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    return parse_standard(data, _T, filepath)


def get_ranges(data: dict) -> AxisRanges:
    return AxisRanges(val_relative=True)


def show_table(data: dict) -> str:
    rows = [r for r in parse(data) if isinstance(r, StandardRow)]
    return show_standard_table(rows)
