"""
Data structures shared across all RMS modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AxisRanges:
    """
    Per-axis effective numeric ranges extracted from ground-truth axis metadata.

    Linear:      raw = |p - t| / range;  D = raw if raw ≤ theta else 1.0
    Logarithmic: raw = |log10(p) - log10(t)| / range;  D = raw if raw ≤ theta else 1.0

    Fields:
        val / val_log  — value axis (standard, errorpoint, box, heatmap)
        val_relative   — if True, use relative distance (|p-t|/|t|) instead of range-window
        x   / x_log   — x axis (scatter x, bubble numeric position)
        y   / y_log   — y axis (scatter y)
        z   / z_log   — bubble size axis
        w   / w_log   — bubble colour/weight axis
    """
    val:          float | None = None
    val_log:      bool         = False
    val_relative: bool         = False
    x:       float | None = None
    x_log:   bool         = False
    y:       float | None = None
    y_log:   bool         = False
    z:       float | None = None
    z_log:   bool         = False
    w:       float | None = None
    w_log:   bool         = False
