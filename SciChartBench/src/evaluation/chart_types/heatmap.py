"""
Heatmap parser — StandardRow as special case: series=row_label, label=col_label.

Handles two input formats:
  GT format   : data_points with x_value, y_value, cell_value
  Matrix fmt  : x_categories, y_categories, matrix  (LLM output)
"""
from __future__ import annotations
import math
from pathlib import Path
from ._helpers import to_float, warn, _normalize_label
from ..row_types import ChartRow, MetaRow, StandardRow
from ..rms.types import AxisRanges

_T = "heatmap"


def _from_matrix(data: dict, filepath) -> list[ChartRow]:
    rows: list[ChartRow] = []
    x_cats = data.get("x_categories", [])
    y_cats = data.get("y_categories", [])
    matrix = data.get("matrix", [])

    if not x_cats or not y_cats or not matrix:
        warn(_T, "x_categories, y_categories o matrix mancante/vuoto", filepath)
        return rows

    for ri, row_lbl in enumerate(y_cats):
        if ri >= len(matrix):
            break
        for ci, col_lbl in enumerate(x_cats):
            if ci >= len(matrix[ri]):
                break
            val = matrix[ri][ci]
            if val is None:
                continue
            vf = to_float(val)
            if vf is None or math.isnan(vf):
                continue
            rows.append(StandardRow(series=_normalize_label(row_lbl),
                                    label=_normalize_label(col_lbl), value=round(vf, 4)))

    return rows


def _from_datapoints(data: dict, filepath) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for dp in data.get("data_points", []):
        x_raw = dp.get("x_value")
        y_raw = dp.get("y_value")
        cell_raw = dp.get("cell_value")

        if cell_raw is None:
            warn(_T, f"cell_value mancante in data_point: {dp}", filepath)
            continue

        vf = to_float(cell_raw)
        if vf is None:
            warn(_T, f"cell_value non numerico: {cell_raw!r}", filepath)
            continue

        rows.append(StandardRow(series=_normalize_label(y_raw),
                                label=_normalize_label(x_raw), value=vf))

    return rows


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if "matrix" in data and "x_categories" in data and "y_categories" in data:
        rows.extend(_from_matrix(data, filepath))
    else:
        if not data.get("data_points"):
            warn(_T, "data_points mancante o vuoto (e non è formato matrix)", filepath)
        rows.extend(_from_datapoints(data, filepath))

    return rows


def get_ranges(data: dict) -> AxisRanges:
    """Range extracted from cell_axis (GT only; predictions may not have it)."""
    cell = data.get("cell_axis") or {}
    mn, mx = cell.get("min"), cell.get("max")
    is_log = bool(cell.get("is_log", False))
    if mn is not None and mx is not None:
        mn, mx = float(mn), float(mx)
        if is_log:
            r = abs(math.log10(mx) - math.log10(mn)) if mn > 0 and mx > 0 else None
        else:
            r = abs(mx - mn) or None
    else:
        r = None
    return AxisRanges(val=r, val_log=is_log)


def show_table(data: dict) -> str:
    rows = [r for r in parse(data) if isinstance(r, StandardRow)]
    if not rows:
        return "(no data)"
    w_s = max((len(r.series) for r in rows), default=6)
    w_l = max((len(r.label) for r in rows), default=5)
    w_s, w_l = max(w_s, 9), max(w_l, 8)
    header = f"{'Row label':<{w_s}}  {'Col label':<{w_l}}  {'Cell value':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(f"{r.series:<{w_s}}  {r.label:<{w_l}}  {r.value:>12.4f}")
    return "\n".join(lines)
