"""
Distance and similarity functions for the RMS metric.

All numeric distances are range-normalised using axis metadata from the
ground-truth JSON.  Axis ranges must always be provided; this module raises
ValueError if a numeric comparison is attempted without a valid range.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from .types import AxisRanges

if TYPE_CHECKING:
    from ..row_types import (
        BoxRow, BubbleRow, ChartRow, ErrorRow,
        MetaRow, ScatterRow, StandardRow,
    )


# ---------------------------------------------------------------------------
# String distance
# ---------------------------------------------------------------------------

def normalized_levenshtein(s1: str, s2: str) -> float:
    """Normalized Damerau-Levenshtein (OSA) distance in [0, 1].

    Counts adjacent transpositions as 1 edit (not 2), which tolerates common
    typos such as swapped characters (e.g. PPAGRC1A1 vs PPARGC1A1).
    Also applies NFKC normalization for Unicode equivalence (e.g. µ vs μ).
    """
    import unicodedata
    s1 = unicodedata.normalize("NFKC", str(s1).lower().strip())
    s2 = unicodedata.normalize("NFKC", str(s2).lower().strip())
    if s1 == s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)
    return dp[n][m] / max(n, m)


def nl_tau(s1: str, s2: str, tau: float) -> float:
    """Normalized Levenshtein clipped at tau."""
    if tau <= 0:
        return float(normalized_levenshtein(s1, s2) > 0)
    return min(1.0, normalized_levenshtein(s1, s2) / tau)


# ---------------------------------------------------------------------------
# Relative numeric distance (no axis-window normalisation)
# ---------------------------------------------------------------------------

def d_relative(p: Any, t: Any) -> float:
    """Relative distance: |p - t| / |t|, clipped to [0, 1].

    Uses the GT value as reference.  More appropriate than a range-window when
    values span orders of magnitude (e.g. pie slice percentages where a 1 %
    error on a 2 % slice is just as significant as a 5 % error on a 10 % slice).

    Edge cases:
      t == 0, p == 0  →  0.0  (perfect match)
      t == 0, p != 0  →  1.0  (undefined, worst distance)
    """
    if not (_is_numeric(p) and _is_numeric(t)):
        return normalized_levenshtein(str(p), str(t))
    pf, tf = _to_float(p), _to_float(t)
    if pf is None or tf is None or math.isnan(pf) or math.isnan(tf):
        return 1.0
    denom = abs(tf)
    if denom == 0:
        return 0.0 if abs(pf) == 0 else 1.0
    return min(1.0, abs(pf - tf) / denom)


# ---------------------------------------------------------------------------
# Scalar numeric distance
# ---------------------------------------------------------------------------

def d_theta(p: Any, t: Any, theta: float,
            val_range: float | None = None, is_log: bool = False) -> float:
    """
    Range-normalised numeric distance clipped at theta; NL distance for strings.

    raw = |p - t| / val_range

    Within theta:  D = raw          →  in [0, theta]
    Beyond theta:  D = 1.0

    D is 0 for a perfect match, rises linearly up to theta, then jumps to 1.0.

    Raises ValueError if val_range is None or ≤ 0 for a numeric comparison.
    Ensure axis metadata (x_axis / y_axis with non-null min and max) is
    present in the ground-truth JSON so _extract_ranges can compute a range.
    """
    if not (_is_numeric(p) and _is_numeric(t)):
        return normalized_levenshtein(str(p), str(t))

    pf, tf = _to_float(p), _to_float(t)
    if pf is None or tf is None or math.isnan(pf) or math.isnan(tf):
        return 1.0

    if val_range is None or val_range <= 0:
        raise ValueError(
            f"val_range must be a positive number, got {val_range!r}. "
            "Ensure axis metadata (x_axis / y_axis with non-null min and max) "
            "is present in the ground-truth JSON."
        )

    if is_log:
        if pf <= 0 or tf <= 0:
            return 1.0
        raw = abs(math.log10(pf) - math.log10(tf)) / val_range
    else:
        raw = abs(pf - tf) / val_range

    return 1.0 if raw > theta else raw


# ---------------------------------------------------------------------------
# Per-row-type value distance (new API)
# ---------------------------------------------------------------------------

def val_distance(p: "ChartRow", t: "ChartRow",
                 theta: float, ranges: AxisRanges | None) -> float:
    """
    Dispatch value distance based on the concrete row type pair.

    Returns a distance in [0, 1]:
      0.0 → identical values
      1.0 → maximally different (or type mismatch)
    """
    from ..row_types import (
        BoxRow, BubbleRow, ErrorRow, MetaRow, ScatterRow, StandardRow,
    )

    if isinstance(p, StandardRow) and isinstance(t, StandardRow):
        return _vd_standard(p, t, theta, ranges)
    if isinstance(p, ErrorRow) and isinstance(t, ErrorRow):
        return _vd_error(p, t, theta, ranges)
    if isinstance(p, BoxRow) and isinstance(t, BoxRow):
        return _vd_box(p, t, theta, ranges)
    if isinstance(p, BubbleRow) and isinstance(t, BubbleRow):
        return _vd_bubble(p, t, theta, ranges)
    if isinstance(p, ScatterRow) and isinstance(t, ScatterRow):
        return _vd_scatter(p, t, theta, ranges)
    if isinstance(p, MetaRow) and isinstance(t, MetaRow):
        return normalized_levenshtein(p.value, t.value)
    return 1.0  # type mismatch


def _vd_standard(p: "StandardRow", t: "StandardRow",
                  theta: float, ranges: AxisRanges | None) -> float:
    if ranges and ranges.val_relative:
        return d_relative(p.value, t.value)
    vr = ranges.val if ranges else None
    vl = ranges.val_log if ranges else False
    return d_theta(p.value, t.value, theta, vr, vl)


def _vd_error(p: "ErrorRow", t: "ErrorRow",
               theta: float, ranges: AxisRanges | None) -> float:
    vr = ranges.val if ranges else None
    vl = ranges.val_log if ranges else False
    pairs = [(p.min, t.min), (p.median, t.median), (p.max, t.max)]
    present = [(pv, tv) for pv, tv in pairs if pv is not None or tv is not None]
    if not present:
        return 0.0
    total = sum(
        d_theta(pv, tv, theta, vr, vl) if pv is not None and tv is not None else 1.0
        for pv, tv in present
    )
    return total / len(present)


def _vd_box(p: "BoxRow", t: "BoxRow",
             theta: float, ranges: AxisRanges | None) -> float:
    vr = ranges.val if ranges else None
    vl = ranges.val_log if ranges else False
    pairs = [
        (p.min, t.min), (p.q1, t.q1), (p.median, t.median),
        (p.q3, t.q3), (p.max, t.max),
    ]
    present = [(pv, tv) for pv, tv in pairs if pv is not None or tv is not None]
    if not present:
        return 0.0
    total = sum(
        d_theta(pv, tv, theta, vr, vl) if pv is not None and tv is not None else 1.0
        for pv, tv in present
    )
    return total / len(present)


def _vd_bubble(p: "BubbleRow", t: "BubbleRow",
                theta: float, ranges: AxisRanges | None) -> float:
    x_r  = ranges.x     if ranges else None
    x_l  = ranges.x_log if ranges else False
    z_r  = ranges.z     if ranges else None
    z_l  = ranges.z_log if ranges else False
    w_r  = ranges.w     if ranges else None
    w_l  = ranges.w_log if ranges else False

    dims: list[float] = [d_theta(p.value, t.value, theta, x_r, x_l)]

    for pv, tv, vr, vl in ((p.z, t.z, z_r, z_l), (p.w, t.w, w_r, w_l)):
        if pv is None and tv is None:
            continue
        elif pv is None or tv is None:
            dims.append(1.0)
        else:
            dims.append(d_theta(pv, tv, theta, vr, vl))

    return sum(dims) / len(dims)


def _vd_scatter(p: "ScatterRow", t: "ScatterRow",
                 theta: float, ranges: AxisRanges | None) -> float:
    x_r = ranges.x     if ranges else None
    x_l = ranges.x_log if ranges else False
    y_r = ranges.y     if ranges else None
    y_l = ranges.y_log if ranges else False
    return 0.5 * d_theta(p.x, t.x, theta, x_r, x_l) + \
           0.5 * d_theta(p.y, t.y, theta, y_r, y_l)


# ---------------------------------------------------------------------------
# Private helpers (also used by d_theta)
# ---------------------------------------------------------------------------

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SI_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9, "t": 1e12}


def _to_float(v: Any) -> float | None:
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


def _is_numeric(v: Any) -> bool:
    return _to_float(v) is not None
