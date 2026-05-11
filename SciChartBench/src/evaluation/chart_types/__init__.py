"""
Chart type parser registry.

Usage
-----
    from src.evaluation.chart_types import get_parser

    parser = get_parser("box")
    rows   = parser.parse(gt_json, filepath=path)
    ranges = parser.get_ranges(gt_json)
    print(parser.show_table(gt_json))
"""

from . import (
    bar, bubble, box, errorpoint, heatmap,
    histogram, line, pie, radar, scatter,
)

REGISTRY: dict = {
    "bar":        bar,
    "line":       line,
    "scatter":    scatter,
    "bubble":     bubble,
    "box":        box,
    "errorpoint": errorpoint,
    "histogram":  histogram,
    "pie":        pie,
    "radar":      radar,
    "heatmap":    heatmap,
}


def get_parser(chart_type: str):
    """Return the parser module for *chart_type*. Raises ValueError if unknown."""
    if chart_type not in REGISTRY:
        raise ValueError(
            f"Tipo chart sconosciuto: {chart_type!r}. "
            f"Disponibili: {sorted(REGISTRY)}"
        )
    return REGISTRY[chart_type]
