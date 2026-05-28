"""Shared utilities for GT-based chart renderers."""

import random
import matplotlib.pyplot as plt

# 12 perceptually distinct colors (tab10 + 2 extras)
PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#bcbd22",  # yellow-green
    "#393b79",  # dark blue
    "#843c39",  # dark red
    "#7f7f7f",  # gray
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def get_colors(n: int, rng: random.Random) -> list:
    """Return n colors from the palette, offset by rng for variety."""
    offset = rng.randrange(len(PALETTE))
    return [PALETTE[(offset + i) % len(PALETTE)] for i in range(n)]


def show_legend(series_names: list) -> bool:
    """Return False when the only series label is 'Main' (legend adds no information)."""
    unique = list(dict.fromkeys(series_names))
    return unique != ["Main"]


def xtick_rotation(n_cats: int, max_label_len: int = 0) -> tuple:
    """Return (rotation_degrees, ha) for categorical x-tick labels."""
    if n_cats > 50:
        return 90, "center"
    if max_label_len > 6:
        return 35, "right"
    return 0, "center"


def figsize_for(n_cats: int, n_series: int, cat_labels=None) -> tuple:
    w, h = 9.0, 5.5
    if n_series > 5:
        w, h = 11.0, 6.0
    if cat_labels:
        avg_len = sum(len(str(c)) for c in cat_labels) / max(len(cat_labels), 1)
        if avg_len > 8 and n_cats > 4:
            w = max(w, 11.0)
    return w, h


def setup_style(fig, ax, data: dict, rng: random.Random, *, polar: bool = False) -> None:
    """Apply common visual style: title, labels, spines, optional grid."""
    title = data.get("chart_title")
    if title:
        ax.set_title(str(title), fontsize=13, pad=10)

    if not polar:
        x_label = data.get("x_axis_label")
        y_label = data.get("y_axis_label")
        if x_label:
            ax.set_xlabel(str(x_label), fontsize=11)
        if y_label:
            ax.set_ylabel(str(y_label), fontsize=11)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if rng.random() < 0.65:
        ax.grid(True, color="#cccccc", alpha=0.45, linewidth=0.7, zorder=0)


# Import renderers last to avoid circular imports
from . import bar, line, scatter, pie, box, errorpoint, histogram, heatmap, bubble, radar  # noqa: E402

RENDERERS: dict = {
    "bar":        bar.render,
    "line":       line.render,
    "scatter":    scatter.render,
    "pie":        pie.render,
    "box":        box.render,
    "errorpoint": errorpoint.render,
    "histogram":  histogram.render,
    "heatmap":    heatmap.render,
    "bubble":     bubble.render,
    "radar":      radar.render,
}
