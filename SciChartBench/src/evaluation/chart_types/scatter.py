"""
Scatter chart parser.

Normal scatter  (both axes numeric): produces ScatterRow(series, x, y).
Categorical scatter (x_axis has null min/max): produces StandardRow(series, label, value)
  treating x_value as the categorical label and y_value as the numeric value.
"""
from __future__ import annotations
from pathlib import Path
from ._helpers import to_float, warn, ranges_scatter, ranges_from_y, ranges_from_x, show_standard_table
from ..row_types import ChartRow, MetaRow, ScatterRow, StandardRow
from ..rms.types import AxisRanges

_T = "scatter"


def _x_is_categorical(data: dict) -> bool:
    """True if x_axis has null min/max → x is categorical."""
    x = data.get("x_axis") or {}
    return x.get("min") is None and x.get("max") is None


def _y_is_categorical(data: dict) -> bool:
    """True if y_axis has null min/max → y is categorical."""
    y = data.get("y_axis") or {}
    return y.get("min") is None and y.get("max") is None


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if data.get("x_axis") is None or data.get("y_axis") is None:
        warn(_T, "x_axis o y_axis mancante", filepath)

    if _x_is_categorical(data):
        # Categorical x: label=x_value, value=y_value (numeric)
        for dp in data.get("data_points", []):
            series = str(dp.get("series_name", ""))
            x_raw  = dp.get("x_value", "")
            y_raw  = dp.get("y_value")
            if y_raw is None:
                warn(_T, f"y_value mancante in data_point: {dp}", filepath)
                continue
            yf = to_float(y_raw)
            if yf is None:
                warn(_T, f"y_value non numerico: {y_raw!r}", filepath)
                continue
            rows.append(StandardRow(series=series, label=str(x_raw), value=yf))
        return rows

    if _y_is_categorical(data):
        # Categorical y: label=y_value, value=x_value (numeric)
        for dp in data.get("data_points", []):
            series = str(dp.get("series_name", ""))
            x_raw  = dp.get("x_value")
            y_raw  = dp.get("y_value", "")
            if x_raw is None:
                warn(_T, f"x_value mancante in data_point: {dp}", filepath)
                continue
            xf = to_float(x_raw)
            if xf is None:
                warn(_T, f"x_value non numerico: {x_raw!r}", filepath)
                continue
            rows.append(StandardRow(series=series, label=str(y_raw), value=xf))
        return rows

    for dp in data.get("data_points", []):
        series = str(dp.get("series_name", ""))
        x_raw  = dp.get("x_value")
        y_raw  = dp.get("y_value")

        xf = to_float(x_raw)
        yf = to_float(y_raw)

        if xf is None or yf is None:
            warn(_T, f"x_value o y_value non numerico: x={x_raw!r} y={y_raw!r}", filepath)
            continue

        rows.append(ScatterRow(series=series, x=xf, y=yf))

    return rows


def get_ranges(data: dict) -> AxisRanges:
    if _x_is_categorical(data):
        return ranges_from_y(data)
    if _y_is_categorical(data):
        return ranges_from_x(data)
    return ranges_scatter(data)


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
