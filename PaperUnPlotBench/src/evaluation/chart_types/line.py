"""
Line chart parser.

Categorical x-axis (x_axis has null min/max): StandardRow(series, label, value)
  label = x_value string, value = y_value float  (standard bar-like format)

Numeric x-axis (x_axis has min/max): ScatterRow(series, x, y)
  Both axes are numeric — matched via value-proximity (scatter_degenerate in core.py)
"""
from __future__ import annotations
from pathlib import Path
from ._helpers import (
    to_float, warn, parse_standard, ranges_auto, ranges_scatter,
    show_standard_table, _normalize_label,
)
from ..row_types import ChartRow, MetaRow, ScatterRow, StandardRow
from ..rms.types import AxisRanges

_T = "line"


def _x_is_numeric(data: dict) -> bool:
    """True if x_axis has explicit min/max → x is a numeric axis."""
    x = data.get("x_axis") or {}
    return x.get("min") is not None and x.get("max") is not None


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    if not _x_is_numeric(data):
        return parse_standard(data, _T, filepath)

    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if data.get("x_axis") is None or data.get("y_axis") is None:
        warn(_T, "x_axis o y_axis mancante", filepath)

    for dp in data.get("data_points", []):
        series = _normalize_label(dp.get("series_name", ""))
        x_raw = dp.get("x_value")
        y_raw = dp.get("y_value")

        xf = to_float(x_raw)
        yf = to_float(y_raw)

        if xf is None or yf is None:
            warn(_T, f"x_value o y_value non numerico: x={x_raw!r} y={y_raw!r}", filepath)
            continue

        rows.append(ScatterRow(series=series, x=xf, y=yf))

    return rows


def get_ranges(data: dict) -> AxisRanges:
    if _x_is_numeric(data):
        return ranges_scatter(data)
    return ranges_auto(data)


def show_table(data: dict) -> str:
    rows_parsed = parse(data)
    scatter_rows = [r for r in rows_parsed if isinstance(r, ScatterRow)]
    std_rows     = [r for r in rows_parsed if isinstance(r, StandardRow)]

    if scatter_rows:
        w_s = max((len(r.series) for r in scatter_rows), default=6)
        w_s = max(w_s, 6)
        header = f"{'Series':<{w_s}}  {'X':>14}  {'Y':>14}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in scatter_rows:
            lines.append(f"{r.series:<{w_s}}  {r.x:>14.6f}  {r.y:>14.6f}")
        return "\n".join(lines)

    if std_rows:
        return show_standard_table(std_rows)

    return "(no data)"
