"""
Shared utilities for chart_types parsers.
Not part of the public registry — imported by individual chart type modules.
"""
from __future__ import annotations

import math
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

from ..row_types import ChartRow, MetaRow, StandardRow
from ..rms.types import AxisRanges


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SI_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9, "t": 1e12}

# Matches "digit(s) whitespace letter" — e.g. "1 µM", "500 nM" → "1µM", "500nM"
_DIGIT_SPACE_UNIT = re.compile(r"(\d)\s+([A-Za-zµμnNpPkKmM])")


def _normalize_label(s) -> str:
    """Normalize a text label stored in a row.

    - NFKC: collapses visually identical characters (µ U+00B5 → μ U+03BC, etc.)
    - Digit-space-unit: '1 µM' → '1µM', '500 nM' → '500nM'
    """
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = _DIGIT_SPACE_UNIT.sub(r"\1\2", s)
    return s


def to_float(v: Any) -> float | None:
    """Convert v to float, handling Unicode scientific notation, % suffix, and SI suffixes (K/M/B/G/T)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    if not isinstance(v, str):
        return None
    s = v.strip().translate(_SUPERSCRIPT_MAP)
    s = s.replace("×10", "e").replace("x10", "e")
    if s.endswith("%"):
        s = s[:-1]
    elif s and s[-1].lower() in _SI_SUFFIX:
        mult = _SI_SUFFIX[s[-1].lower()]
        try:
            return float(s[:-1]) * mult
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return None


def warn(chart_type: str, msg: str, filepath, stacklevel: int = 3) -> None:
    loc = f"\n  File: {filepath}" if filepath else ""
    warnings.warn(f"[{chart_type}] {msg}{loc}", UserWarning, stacklevel=stacklevel)


# ---------------------------------------------------------------------------
# Range extraction helpers
# ---------------------------------------------------------------------------

def _axis_range(axis_dict: dict | None) -> tuple[float | None, bool]:
    """Return (range_value, is_log) from an axis metadata dict."""
    if not axis_dict:
        return None, False
    mn, mx = axis_dict.get("min"), axis_dict.get("max")
    is_log = bool(axis_dict.get("is_log", False))
    if mn is None or mx is None:
        return None, is_log
    mn, mx = float(mn), float(mx)
    if is_log:
        if mn <= 0 or mx <= 0:
            return None, is_log
        r = abs(math.log10(mx) - math.log10(mn))
    else:
        r = abs(mx - mn)
    return (r if r > 0 else None), is_log


def _is_horizontal(data: dict) -> bool:
    """True if y_axis is explicitly categorical (present but null min/max) → label on y, value on x.
    Returns False when y_axis is absent (predictions often omit axis metadata — default to vertical)."""
    y_axis = data.get("y_axis")
    if y_axis is None:
        return False
    return y_axis.get("min") is None and y_axis.get("max") is None


def ranges_from_y(data: dict) -> AxisRanges:
    """AxisRanges for categorical_x charts: value range = y_axis."""
    r, log = _axis_range(data.get("y_axis"))
    return AxisRanges(val=r, val_log=log)


def ranges_from_x(data: dict) -> AxisRanges:
    """AxisRanges for categorical_y charts: value range = x_axis."""
    r, log = _axis_range(data.get("x_axis"))
    return AxisRanges(val=r, val_log=log)


def ranges_auto(data: dict) -> AxisRanges:
    """AxisRanges that auto-detects the value axis from metadata."""
    if _is_horizontal(data):
        return ranges_from_x(data)
    return ranges_from_y(data)


def ranges_scatter(data: dict) -> AxisRanges:
    x_r, x_log = _axis_range(data.get("x_axis"))
    y_r, y_log = _axis_range(data.get("y_axis"))
    return AxisRanges(x=x_r, x_log=x_log, y=y_r, y_log=y_log)


def ranges_bubble(data: dict, numeric_axis: str) -> AxisRanges:
    """numeric_axis: 'x' or 'y' — which axis holds the numeric position."""
    num_r, num_log = _axis_range(data.get(f"{numeric_axis}_axis"))
    z_r, z_log = _axis_range(data.get("z_axis"))
    w_r, w_log = _axis_range(data.get("w_axis"))
    return AxisRanges(x=num_r, x_log=num_log, z=z_r, z_log=z_log, w=w_r, w_log=w_log)


# ---------------------------------------------------------------------------
# Standard categorical_x parser (shared by bar, line, pie, radar, histogram)
# ---------------------------------------------------------------------------

def parse_standard(
    data: dict,
    chart_type: str,
    filepath=None,
) -> list[ChartRow]:
    """
    Parse a categorical JSON to [MetaRow?] + [StandardRow ...].

    Orientation is auto-detected from axis metadata:
      vertical  (default): label = x_value, value = y_value
      horizontal (y_axis has null min/max): label = y_value, value = x_value
    """
    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if data.get("x_axis") is None or data.get("y_axis") is None:
        warn(chart_type, "x_axis o y_axis mancante", filepath)

    horizontal = _is_horizontal(data)

    for dp in data.get("data_points", []):
        series = str(dp.get("series_name", ""))

        if horizontal:
            label_raw = dp.get("y_value", "")
            val_raw   = dp.get("x_value")
        else:
            label_raw = dp.get("x_value", "")
            val_raw   = dp.get("y_value")

        if val_raw is None:
            axis = "x" if horizontal else "y"
            warn(chart_type, f"{axis}_value mancante in data_point: {dp}", filepath)
            continue

        vf = to_float(val_raw)
        if vf is None:
            if horizontal:
                # VLMs (e.g. DePlot) output labels in x_value and counts in y_value
                # regardless of orientation → swap: label←x_value, value←y_value.
                alt = to_float(label_raw)
                if alt is not None:
                    label_raw, vf = val_raw, alt
            if vf is None:
                axis = "x" if horizontal else "y"
                warn(chart_type, f"{axis}_value non numerico: {val_raw!r}", filepath)
                continue

        rows.append(StandardRow(series=_normalize_label(series),
                                label=_normalize_label(label_raw), value=vf))

    return rows


# ---------------------------------------------------------------------------
# Table display helper
# ---------------------------------------------------------------------------

def show_standard_table(rows: list[StandardRow]) -> str:
    if not rows:
        return "(no data)"
    w_s = max(len(r.series) for r in rows)
    w_l = max(len(r.label) for r in rows)
    w_s, w_l = max(w_s, 6), max(w_l, 5)
    header = f"{'Series':<{w_s}}  {'Label':<{w_l}}  {'Value':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(f"{r.series:<{w_s}}  {r.label:<{w_l}}  {r.value:>12.4f}")
    return "\n".join(lines)
