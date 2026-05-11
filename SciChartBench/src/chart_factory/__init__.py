"""
chart_factory — Synthetic chart generator package.

Modules:
  common            – Shared themes, helpers, JSON builders
  bar_chart         – Bar chart (6 subtypes)
  bubble_chart      – Bubble chart (3 subtypes)
  box_chart         – Box plot (4 subtypes)
  errorpoint_chart  – Error point chart (3 subtypes)
  pie_chart         – Pie chart (4 subtypes)
  scatter_chart     – Scatter plot (4 subtypes)
  radar_chart       – Radar chart (3 subtypes)
  line_chart        – Line chart (5 subtypes)
  histogram_chart   – Histogram (3 subtypes)
  heatmap_chart     – Heatmap (4 subtypes)

Entry point:
  generate_all.py   – CLI: python src/chart_factory/generate_all.py --n 50 --types all
"""

from . import (
    bar_chart,
    bubble_chart,
    box_chart,
    errorpoint_chart,
    pie_chart,
    scatter_chart,
    radar_chart,
    line_chart,
    histogram_chart,
    heatmap_chart,
)

__all__ = [
    "bar_chart",
    "bubble_chart",
    "box_chart",
    "errorpoint_chart",
    "pie_chart",
    "scatter_chart",
    "radar_chart",
    "line_chart",
    "histogram_chart",
    "heatmap_chart",
]
