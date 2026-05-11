"""
Bubble chart parser — BubbleRow format.

Orientation is auto-detected from axis metadata and normalised:
  label  = value on the categorical axis (x_value or y_value)
  value  = numeric position on the non-categorical axis
  z, w   = optional size / colour dimensions
"""
from __future__ import annotations
from pathlib import Path
from ._helpers import to_float, warn, ranges_bubble, _normalize_label
from ..row_types import BubbleRow, ChartRow, MetaRow
from ..rms.types import AxisRanges

_T = "bubble"


def _categorical_axis(data: dict) -> str:
    """Return 'y' if y_axis is categorical (null min/max), 'x' if x_axis is categorical,
    or 'none' when both axes are numeric."""
    y = data.get("y_axis") or {}
    if y.get("min") is None and y.get("max") is None:
        return "y"
    x = data.get("x_axis") or {}
    if x.get("min") is None and x.get("max") is None:
        return "x"
    return "none"  # both axes numeric


def _float_label(v) -> str:
    """Convert a numeric value to a canonical float string (avoids '2.0' vs '2' mismatches)."""
    f = to_float(v)
    if f is None:
        return str(v)
    return f"{f:g}"


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if data.get("x_axis") is None or data.get("y_axis") is None:
        warn(_T, "x_axis o y_axis mancante", filepath)

    cat_axis = _categorical_axis(data)

    for dp in data.get("data_points", []):
        series = _normalize_label(str(dp.get("series_name", "")))
        x_raw = dp.get("x_value")
        y_raw = dp.get("y_value")
        z_raw = dp.get("z_value")
        w_raw = dp.get("w_value")

        if cat_axis == "y":
            label = _normalize_label(y_raw) if y_raw is not None else ""
            num_val = to_float(x_raw)
        elif cat_axis == "none":
            # Both axes numeric: normalize x as canonical float string to avoid
            # "2.0" vs "2" key mismatches; y is the matched numeric value.
            label = _float_label(x_raw) if x_raw is not None else ""
            num_val = to_float(y_raw)
        else:  # cat_axis == "x"
            label = _normalize_label(x_raw) if x_raw is not None else ""
            num_val = to_float(y_raw)

        if num_val is None:
            warn(_T, f"valore numerico mancante/non valido: {dp}", filepath)
            continue

        zf = to_float(z_raw) if z_raw is not None else None
        wf = to_float(w_raw) if w_raw is not None else None

        rows.append(BubbleRow(series=series, label=label, value=num_val, z=zf, w=wf))

    return rows


def get_ranges(data: dict) -> AxisRanges:
    cat_axis = _categorical_axis(data)
    if cat_axis == "y":
        numeric_axis = "x"
    else:
        numeric_axis = "y"  # covers cat_axis == "x" and cat_axis == "none"
    return ranges_bubble(data, numeric_axis)


def show_table(data: dict) -> str:
    rows = [r for r in parse(data) if isinstance(r, BubbleRow)]
    if not rows:
        return "(no data)"
    w_s = max((len(r.series) for r in rows), default=6)
    w_l = max((len(r.label) for r in rows), default=5)
    w_s, w_l = max(w_s, 6), max(w_l, 5)
    header = f"{'Series':<{w_s}}  {'Label':<{w_l}}  {'Value':>10}  {'Z':>8}  {'W':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        z_s = f"{r.z:>8.4f}" if r.z is not None else "    None"
        w_s2 = f"{r.w:>8.4f}" if r.w is not None else "    None"
        lines.append(f"{r.series:<{w_s}}  {r.label:<{w_l}}  {r.value:>10.4f}  {z_s}  {w_s2}")
    return "\n".join(lines)
