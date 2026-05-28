"""
Box plot parser — BoxRow format.

y_value must be a dict with keys: min, q1, median, q3, max (any subset accepted).
"""
from __future__ import annotations
from pathlib import Path
from ._helpers import to_float, warn, ranges_auto, _is_horizontal, _normalize_label
from ..row_types import BoxRow, ChartRow, MetaRow
from ..rms.types import AxisRanges

_T = "box"


def parse(data: dict, filepath: str | Path | None = None) -> list[ChartRow]:
    rows: list[ChartRow] = []

    title = data.get("chart_title")
    if title is not None:
        rows.append(MetaRow(field="chart_title", value=str(title)))

    if data.get("x_axis") is None or data.get("y_axis") is None:
        warn(_T, "x_axis o y_axis mancante", filepath)

    horizontal = _is_horizontal(data)

    for dp in data.get("data_points", []):
        series = str(dp.get("series_name", ""))

        if horizontal:
            label_raw = dp.get("y_value", "")
            val_raw   = dp.get("x_value")
        else:
            label_raw = dp.get("x_value", "")
            val_raw   = dp.get("y_value")

        if not isinstance(val_raw, dict):
            axis = "x" if horizontal else "y"
            warn(_T, f"{axis}_value non è un dict strutturato: {val_raw!r}", filepath)
            continue

        def _f(k):
            v = val_raw.get(k)
            return to_float(v) if v is not None else None

        rows.append(BoxRow(
            series=_normalize_label(series),
            label=_normalize_label(label_raw),
            min=_f("min"), q1=_f("q1"), median=_f("median"),
            q3=_f("q3"), max=_f("max"),
        ))

    return rows


def get_ranges(data: dict) -> AxisRanges:
    return ranges_auto(data)


def show_table(data: dict) -> str:
    rows = [r for r in parse(data) if isinstance(r, BoxRow)]
    if not rows:
        return "(no data)"
    w_s = max((len(r.series) for r in rows), default=6)
    w_l = max((len(r.label) for r in rows), default=5)
    w_s, w_l = max(w_s, 6), max(w_l, 5)
    header = (f"{'Series':<{w_s}}  {'Label':<{w_l}}"
              f"  {'min':>8}  {'q1':>8}  {'median':>8}  {'q3':>8}  {'max':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        def _fmt(v): return f"{v:>8.4f}" if v is not None else "    None"
        lines.append(
            f"{r.series:<{w_s}}  {r.label:<{w_l}}"
            f"  {_fmt(r.min)}  {_fmt(r.q1)}  {_fmt(r.median)}  {_fmt(r.q3)}  {_fmt(r.max)}"
        )
    return "\n".join(lines)
