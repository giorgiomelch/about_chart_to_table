"""
Histogram parser.

Orientation and bin format auto-detected from axis metadata:

  Horizontal (y_axis has null min/max):
    label = y_value (bin label string), value = x_value (float count)
    → StandardRow — same as bar/line categorical

  Vertical, numeric bins (x_axis has min/max, x_value convertible to float):
    → ScatterRow(series, x_bin, y_count)
    Avoids "5.0" vs "5" key mismatches; uses value-proximity matching.

  Vertical, string bins (x_axis has min/max, x_value NOT a float, e.g. "0-200"):
    label = x_value (range string), value = y_value (float count)
    → StandardRow — key matching on string label
"""
from __future__ import annotations
from pathlib import Path
from ._helpers import (
    to_float, warn, ranges_auto, ranges_scatter,
    show_standard_table, _normalize_label, _is_horizontal,
)
from ..row_types import ChartRow, MetaRow, ScatterRow, StandardRow
from ..rms.types import AxisRanges

_T = "histogram"


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if data.get("x_axis") is None or data.get("y_axis") is None:
        warn(_T, "x_axis o y_axis mancante", filepath)

    horizontal = _is_horizontal(data)

    for dp in data.get("data_points", []):
        series = _normalize_label(dp.get("series_name", ""))
        x_raw = dp.get("x_value")
        y_raw = dp.get("y_value")

        if horizontal:
            # y is the categorical bin label, x is the numeric count
            label_raw = y_raw if y_raw is not None else ""
            val_raw = x_raw
            vf = to_float(val_raw)
            if vf is None:
                warn(_T, f"x_value non numerico (orizzontale): {val_raw!r}", filepath)
                continue
            rows.append(StandardRow(series=series,
                                    label=_normalize_label(label_raw), value=vf))
        else:
            # Vertical: x is bin, y is count
            yf = to_float(y_raw)
            if yf is None:
                warn(_T, f"y_value non numerico: {y_raw!r}", filepath)
                continue
            x_axis = data.get("x_axis") or {}
            x_is_numeric = x_axis.get("min") is not None
            xf = to_float(x_raw)
            if x_is_numeric and xf is not None:
                # x_axis has explicit numeric range → ScatterRow for proximity matching
                rows.append(ScatterRow(series=series, x=xf, y=yf))
            else:
                # Categorical or string bin (e.g. "0-200") → StandardRow
                rows.append(StandardRow(series=series,
                                        label=_normalize_label(x_raw) if x_raw is not None else "",
                                        value=yf))

    return rows


def get_ranges(data: dict) -> AxisRanges:
    # For ScatterRow bins use x+y ranges; for StandardRow use auto (y range)
    x_axis = data.get("x_axis") or {}
    if not _is_horizontal(data) and x_axis.get("min") is not None:
        return ranges_scatter(data)
    return ranges_auto(data)


def show_table(data: dict) -> str:
    rows_parsed = parse(data)
    scatter_rows = [r for r in rows_parsed if isinstance(r, ScatterRow)]
    std_rows     = [r for r in rows_parsed if isinstance(r, StandardRow)]

    if scatter_rows:
        w_s = max((len(r.series) for r in scatter_rows), default=6)
        w_s = max(w_s, 6)
        header = f"{'Series':<{w_s}}  {'Bin (X)':>14}  {'Count (Y)':>14}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in scatter_rows:
            lines.append(f"{r.series:<{w_s}}  {r.x:>14.4f}  {r.y:>14.4f}")
        return "\n".join(lines)

    if std_rows:
        return show_standard_table(std_rows)

    return "(no data)"
