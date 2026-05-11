"""
Core RMS computation: Hungarian matching, precision/recall/F1, public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .distance import _vd_scatter, d_theta, nl_tau, val_distance
from .types import AxisRanges
from ..row_types import (
    BoxRow, BubbleRow, ChartRow, ErrorRow,
    MetaRow, ScatterRow, StandardRow,
)


# ---------------------------------------------------------------------------
# Axis metadata injection
# ---------------------------------------------------------------------------

def _inject_axis_meta(pred: dict, gt: dict) -> dict:
    """Copy axis metadata from GT into pred (predictions never include axis ranges).

    Orientation detection (_is_horizontal, _x_is_categorical, etc.) reads from
    axis metadata.  Without this injection, predictions parsed in isolation would
    always appear to have all-null axes and pick the wrong orientation.
    """
    if not isinstance(pred, dict) or not isinstance(gt, dict):
        return pred
    result = dict(pred)
    for key in ("x_axis", "y_axis", "z_axis", "w_axis", "cell_axis"):
        if result.get(key) is None and gt.get(key) is not None:
            result[key] = gt[key]
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_series(rows: list[ChartRow]) -> list[ChartRow]:
    """Replace series with '' for rows that carry a label (ignores series dimension)."""
    result = []
    for row in rows:
        if isinstance(row, StandardRow):
            result.append(StandardRow(series="", label=row.label, value=row.value))
        elif isinstance(row, ErrorRow):
            result.append(ErrorRow(series="", label=row.label,
                                   min=row.min, median=row.median, max=row.max))
        elif isinstance(row, BoxRow):
            result.append(BoxRow(series="", label=row.label,
                                 min=row.min, q1=row.q1, median=row.median,
                                 q3=row.q3, max=row.max))
        elif isinstance(row, BubbleRow):
            result.append(BubbleRow(series="", label=row.label,
                                    value=row.value, z=row.z, w=row.w))
        else:
            result.append(row)  # ScatterRow, MetaRow — unchanged
    return result


def _transpose_rows(rows: list[ChartRow]) -> list[ChartRow]:
    """Swap series ↔ label (equivalent to transposing row/col in old Mapping)."""
    result = []
    for row in rows:
        if isinstance(row, StandardRow):
            result.append(StandardRow(series=row.label, label=row.series, value=row.value))
        elif isinstance(row, ErrorRow):
            result.append(ErrorRow(series=row.label, label=row.series,
                                   min=row.min, median=row.median, max=row.max))
        elif isinstance(row, BoxRow):
            result.append(BoxRow(series=row.label, label=row.series,
                                 min=row.min, q1=row.q1, median=row.median,
                                 q3=row.q3, max=row.max))
        elif isinstance(row, BubbleRow):
            result.append(BubbleRow(series=row.label, label=row.series,
                                    value=row.value, z=row.z, w=row.w))
        else:
            result.append(row)  # ScatterRow, MetaRow — unchanged
    return result


def _distinct_series(rows: list[ChartRow]) -> set[str]:
    return {r.series for r in rows if not isinstance(r, (MetaRow, ScatterRow))}


def _all_main(rows: list[ChartRow]) -> bool:
    """True when every labeled row has series_name == 'Main' (single-series sentinel)."""
    labeled = [r for r in rows if not isinstance(r, (MetaRow, ScatterRow))]
    return bool(labeled) and all(r.series == "Main" for r in labeled)


def _scatter_entry_sim(p: ScatterRow, t: ScatterRow,
                       theta: float, ranges: AxisRanges | None) -> float:
    """
    Entry similarity for scatter-degenerate rows: (1-dx) * (1-dy).

    Theta is checked independently per axis: a point beyond theta in x contributes
    zero x-credit regardless of how well y matches, and vice versa.
    """
    if ranges is None or ranges.x is None or ranges.y is None:
        return 1.0 if (p.x == t.x and p.y == t.y) else 0.0
    dx = d_theta(p.x, t.x, theta, ranges.x, ranges.x_log)
    dy = d_theta(p.y, t.y, theta, ranges.y, ranges.y_log)
    return (1.0 - dx) * (1.0 - dy)


def _rms_single(P: list[ChartRow], T: list[ChartRow],
                tau: float, theta: float, ranges: AxisRanges | None,
                scatter_type: bool) -> dict:
    """
    Compute RMS scores for one orientation of predicted vs ground-truth rows.
    Returns dict: precision, recall, f1, matched_sim, pairs.
    """
    N, M = len(P), len(T)

    if N == 0 and M == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched_sim": 0.0, "pairs": []}
    if N == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "matched_sim": 0.0, "pairs": []}
    if M == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "matched_sim": 0.0, "pairs": []}

    # Single-series normalization: drop series from key when either side has <=1 distinct
    # series, or when the GT uses the 'Main' sentinel (meaning no series dimension).
    if len(_distinct_series(P)) <= 1 or len(_distinct_series(T)) <= 1 or _all_main(T):
        P = _strip_series(P)
        T = _strip_series(T)

    # Degenerate scatter: all data rows are ScatterRow with ≤1 series.
    # MetaRows (chart_title) are excluded from this check — they are always
    # matched via nl_tau regardless of the scatter_degenerate flag.
    _p_data = [r for r in P if not isinstance(r, MetaRow)]
    _t_data = [r for r in T if not isinstance(r, MetaRow)]
    scatter_degenerate = (
        bool(_p_data) and bool(_t_data)
        and all(isinstance(r, ScatterRow) for r in _p_data + _t_data)
        and len({r.series for r in _p_data}) <= 1
        and len({r.series for r in _t_data}) <= 1
    )

    # Key-similarity matrix (N x M)
    key_sim = np.zeros((N, M))
    for i, p in enumerate(P):
        for j, t in enumerate(T):
            if scatter_degenerate and isinstance(p, ScatterRow) and isinstance(t, ScatterRow):
                # entry_sim = (1-dx)*(1-dy): theta checked independently per axis.
                # This is the same value used as pair_sim, so key_sim doubles as the
                # assignment cost and the final similarity (no val_distance step).
                key_sim[i, j] = _scatter_entry_sim(p, t, theta, ranges)
            else:
                key_sim[i, j] = 1.0 - nl_tau(p.key(), t.key(), tau)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(1.0 - key_sim)

    if scatter_degenerate:
        # ScatterRow pairs: pair_sim encoded in key_sim, no second val_distance pass.
        # MetaRow pairs: key_sim is nl_tau-based (key = field name), val_distance
        # carries the actual title-value comparison — preserve it.
        pair_sims = [
            (key_sim[i, j] * (1.0 - val_distance(P[i], T[j], theta, ranges))
             if isinstance(P[i], MetaRow)
             else key_sim[i, j])
            for i, j in zip(row_ind, col_ind)
        ]
    else:
        pair_sims = [
            key_sim[i, j] * (1.0 - val_distance(P[i], T[j], theta, ranges))
            for i, j in zip(row_ind, col_ind)
        ]
    total_sim = sum(pair_sims)

    precision = total_sim / N
    recall    = total_sim / M
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "matched_sim": total_sim,
        "pairs": list(zip(row_ind.tolist(), col_ind.tolist(), pair_sims)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rms(
    predicted: dict,
    target: dict,
    chart_type: str,
    tau: float = 0.5,
    theta: float = 0.1,
    try_transpose: bool = False,
    debug: bool = False,
) -> dict:
    """
    Compute RMS between two chart JSON dicts.

    Parameters
    ----------
    predicted  : dict   Model-predicted chart JSON.
    target     : dict   Ground-truth chart JSON.
    chart_type : str    Chart type key (e.g. 'bar', 'scatter', 'box').
    tau        : float  NL distance threshold (default 0.5).
    theta      : float  Numeric distance threshold as fraction of axis range
                        (default 0.1 = 10%).
    try_transpose : bool  Try both series<->label orientations and return best.
                          Enable only for DePlot. Default False.
    debug      : bool   If True, add pred_rows, gt_rows, pred_table, gt_table.

    Returns
    -------
    dict: precision, recall, f1, orientation, chart_type, normal, transposed,
          (optional debug keys)
    """
    from ..chart_types import get_parser  # deferred to avoid circular import
    parser = get_parser(chart_type)
    predicted_enriched = _inject_axis_meta(predicted, target)
    T = parser.parse(target)
    P = parser.parse(predicted_enriched)
    ranges = parser.get_ranges(target)

    scatter_type = (chart_type == "scatter")

    if scatter_type or not try_transpose:
        orientations = [False]
    else:
        orientations = [False, True]

    results: dict = {}
    for transpose in orientations:
        rows = _transpose_rows(P) if transpose else P
        key  = "transposed" if transpose else "normal"
        results[key] = _rms_single(rows, T, tau, theta, ranges, scatter_type)

    if "transposed" not in results:
        results["transposed"] = results["normal"]

    best = max(results, key=lambda k: results[k]["f1"])
    out = {
        "precision":   results[best]["precision"],
        "recall":      results[best]["recall"],
        "f1":          results[best]["f1"],
        "orientation": best,
        "chart_type":  chart_type,
        "normal":      results["normal"],
        "transposed":  results["transposed"],
    }

    if debug:
        out["pred_rows"]  = P
        out["gt_rows"]    = T
        out["pred_table"] = parser.show_table(predicted_enriched)
        out["gt_table"]   = parser.show_table(target)

    return out
