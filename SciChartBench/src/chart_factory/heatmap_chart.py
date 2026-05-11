"""
heatmap_chart.py — Heatmap Generator

Subtypes:
  standard     – Sequential colormap heatmap
  correlation  – Correlation matrix (values in [-1, 1])
  diverging    – Diverging colormap (pos/neg values)
  triangle     – Lower-triangular heatmap
"""

import random
import string
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, apply_theme, save_outputs,
)

CHART_TYPE       = "heatmap"
SHOW_VALUES_PROB = 0.6   # 60 % of charts annotate cell values

SUBTYPES = ["standard", "correlation", "diverging", "triangle"]

DATASET_TEMPLATES = [
    {
        "theme": "Temperature Anomaly",
        "x_label": "Station", "y_label": "Month",
        "n_rows": (4, 8), "n_cols": (5, 10),
        "value_range": (-4.5, 4.5), "fmt": ".1f",
    },
    {
        "theme": "Sales Performance",
        "x_label": "Product", "y_label": "Region",
        "n_rows": (3, 7), "n_cols": (4, 9),
        "value_range": (5.0, 120.0), "fmt": ".0f",
    },
    {
        "theme": "Risk Assessment",
        "x_label": "Factor", "y_label": "Category",
        "n_rows": (3, 7), "n_cols": (4, 8),
        "value_range": (0.0, 10.0), "fmt": ".1f",
    },
    {
        "theme": "Gene Expression",
        "x_label": "Sample", "y_label": "Gene",
        "n_rows": (6, 14), "n_cols": (5, 11),
        "value_range": (-3.0, 3.0), "fmt": ".2f",
    },
    {
        "theme": "Survey Results",
        "x_label": "Topic", "y_label": "Group",
        "n_rows": (3, 6), "n_cols": (4, 8),
        "value_range": (1.0, 5.0), "fmt": ".1f",
    },
    {
        "theme": "Network Traffic",
        "x_label": "Source", "y_label": "Dest",
        "n_rows": (4, 8), "n_cols": (4, 8),
        "value_range": (0.0, 1000.0), "fmt": ".0f",
    },
    {
        "theme": "Model Accuracy",
        "x_label": "Model", "y_label": "Dataset",
        "n_rows": (3, 6), "n_cols": (4, 8),
        "value_range": (0.0, 1.0), "fmt": ".2f",
    },
    {
        "theme": "Confusion Matrix",
        "x_label": "Predicted", "y_label": "Actual",
        "n_rows": (3, 6), "n_cols": (3, 6),
        "value_range": (0.0, 250.0), "fmt": ".0f",
    },
    {
        "theme": "Pollution Index",
        "x_label": "Site", "y_label": "Pollutant",
        "n_rows": (4, 8), "n_cols": (5, 9),
        "value_range": (0.0, 200.0), "fmt": ".1f",
    },
    {
        "theme": "Energy Consumption",
        "x_label": "Hour", "y_label": "Day",
        "n_rows": (5, 8), "n_cols": (6, 12),
        "value_range": (50.0, 800.0), "fmt": ".0f",
    },
    {
        "theme": "Budget Variance",
        "x_label": "Dept", "y_label": "Quarter",
        "n_rows": (3, 5), "n_cols": (4, 7),
        "value_range": (-40.0, 40.0), "fmt": ".1f",
    },
    {
        "theme": "User Engagement",
        "x_label": "Feature", "y_label": "Cohort",
        "n_rows": (4, 7), "n_cols": (5, 9),
        "value_range": (0.0, 100.0), "fmt": ".1f",
    },
]

MIN_DIM = 3
MAX_DIM = 7

SEQUENTIAL_CMAPS = [
    "YlOrRd", "Blues", "Greens", "Purples", "YlGnBu",
    "OrRd", "PuBu", "BuGn", "Oranges", "Reds", "GnBu",
]
DIVERGING_CMAPS = [
    "RdBu_r", "coolwarm", "PiYG", "PRGn", "BrBG", "seismic", "RdYlGn",
]
CORRELATION_CMAPS = ["RdBu_r", "coolwarm", "PiYG", "RdYlGn"]


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def _rand_labels(n: int, max_len: int = 5) -> list:
    """Generate n unique random uppercase strings of length 1..max_len."""
    seen, labels = set(), []
    while len(labels) < n:
        lbl = "".join(random.choices(string.ascii_uppercase,
                                     k=random.randint(1, max_len)))
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)
    return labels


def _text_color(cmap_name: str, val: float, vmin: float, vmax: float) -> str:
    """Return black or white based on the luminance of the colormap color at val."""
    norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    norm = float(np.clip(norm, 0.0, 1.0))
    r, g, b, _ = plt.get_cmap(cmap_name)(norm)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#000000" if luminance > 0.45 else "#FFFFFF"


def _annotate(ax, matrix: np.ndarray, fmt: str, cmap_name: str,
              vmin: float, vmax: float) -> None:
    rows, cols = matrix.shape
    fsize = max(5, min(8, int(80 / max(rows, cols))))
    for i in range(rows):
        for j in range(cols):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = _text_color(cmap_name, val, vmin, vmax)
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=fsize, color=color, fontweight="normal")


def _style_colorbar(cbar, theme: dict) -> None:
    cbar.ax.tick_params(colors=theme["tick"], labelsize=8)
    cbar.ax.yaxis.label.set_color(theme["label"])
    cbar.outline.set_edgecolor(theme["grid"])


_CATEGORICAL_AXIS = {"min": None, "max": None, "is_log": False}


def _build_json(title, x_label, y_label, col_labels, row_labels,
                matrix: np.ndarray, vmin: float, vmax: float) -> dict:
    dp = []
    for ri, row_lbl in enumerate(row_labels):
        for ci, col_lbl in enumerate(col_labels):
            val = matrix[ri, ci]
            if not np.isnan(val):
                dp.append({"x_value": col_lbl, "y_value": row_lbl,
                           "cell_value": round(float(val), 4)})
    return {
        "chart_title":  title,
        "x_axis_label": x_label,
        "y_axis_label": y_label,
        "x_axis":       dict(_CATEGORICAL_AXIS),
        "y_axis":       dict(_CATEGORICAL_AXIS),
        "cell_axis":    {"min": round(float(vmin), 4),
                         "max": round(float(vmax), 4),
                         "is_log": False},
        "data_points":  dp,
    }


# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_standard(tmpl, theme, idx, show_values):
    n_rows = random.randint(MIN_DIM, MAX_DIM)
    n_cols = random.randint(MIN_DIM, MAX_DIM)
    row_labels = _rand_labels(n_rows)
    col_labels = _rand_labels(n_cols)

    lo, hi = tmpl["value_range"]
    # Add structured variation: some rows have a trend
    matrix = np.random.uniform(lo, hi, (n_rows, n_cols))
    if random.random() > 0.5:
        bias = np.linspace(0, (hi - lo) * 0.3, n_rows)[:, None]
        matrix = np.clip(matrix + bias * random.choice([-1, 1]), lo, hi)
    matrix = matrix.round(4)

    cmap = random.choice(SEQUENTIAL_CMAPS)
    fig_w = max(7, n_cols * 1.0 + 2.5)
    fig_h = max(4, n_rows * 0.85 + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    apply_theme(fig, ax, theme)
    ax.grid(False)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    _style_colorbar(cbar, theme)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       fontsize=9, color=theme["tick"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9, color=theme["tick"])
    ax.set_title(tmpl["theme"], fontsize=13, fontweight="bold", pad=12,
                 color=theme["title"])
    ax.set_xlabel(tmpl["x_label"], fontsize=10, color=theme["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=10, color=theme["label"])
    ax.tick_params(length=0)

    if show_values:
        _annotate(ax, matrix, tmpl["fmt"], cmap, lo, hi)

    plt.tight_layout()
    js = _build_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"],
                     col_labels, row_labels, matrix, lo, hi)
    return fig, js


def _render_correlation(tmpl, theme, idx, show_values):
    n = random.randint(MIN_DIM, MAX_DIM)
    labels = _rand_labels(n)

    # Random positive semi-definite correlation matrix
    A = np.random.randn(n, n)
    cov = A @ A.T
    D = np.sqrt(np.diag(cov))
    matrix = (cov / D[:, None] / D[None, :]).round(4)
    np.fill_diagonal(matrix, 1.0)

    cmap = random.choice(CORRELATION_CMAPS)
    fig_sz = max(5.5, n * 1.0 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
    apply_theme(fig, ax, theme)
    ax.grid(False)

    im = ax.imshow(matrix, cmap=cmap, aspect="equal", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    _style_colorbar(cbar, theme)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=9, color=theme["tick"])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9, color=theme["tick"])
    title = "Correlation — " + tmpl["theme"]
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12,
                 color=theme["title"])
    ax.tick_params(length=0)

    if show_values:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = _text_color(cmap, val, -1, 1)
                fsize = max(5, min(8, int(80 / n)))
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=fsize, color=color)

    plt.tight_layout()
    js = _build_json(title, tmpl["x_label"], tmpl["y_label"],
                     labels, labels, matrix, -1.0, 1.0)
    return fig, js


def _render_diverging(tmpl, theme, idx, show_values):
    n_rows = random.randint(MIN_DIM, MAX_DIM)
    n_cols = random.randint(MIN_DIM, MAX_DIM)
    row_labels = _rand_labels(n_rows)
    col_labels = _rand_labels(n_cols)

    lo, hi = tmpl["value_range"]
    absmax = max(abs(lo), abs(hi))
    matrix = np.random.uniform(-absmax, absmax, (n_rows, n_cols))
    # Add some spatial structure
    if random.random() > 0.4:
        col_bias = np.linspace(-absmax * 0.4, absmax * 0.4, n_cols)
        matrix = matrix + col_bias[None, :]
    matrix = np.clip(matrix, -absmax, absmax).round(4)

    cmap = random.choice(DIVERGING_CMAPS)
    fig_w = max(7, n_cols * 1.0 + 2.5)
    fig_h = max(4, n_rows * 0.85 + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    apply_theme(fig, ax, theme)
    ax.grid(False)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-absmax, vmax=absmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    _style_colorbar(cbar, theme)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       fontsize=9, color=theme["tick"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9, color=theme["tick"])
    title = tmpl["theme"] + " (Δ)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12,
                 color=theme["title"])
    ax.set_xlabel(tmpl["x_label"], fontsize=10, color=theme["label"])
    ax.set_ylabel(tmpl["y_label"], fontsize=10, color=theme["label"])
    ax.tick_params(length=0)

    if show_values:
        _annotate(ax, matrix, tmpl["fmt"], cmap, -absmax, absmax)

    plt.tight_layout()
    js = _build_json(title, tmpl["x_label"], tmpl["y_label"],
                     col_labels, row_labels, matrix, -absmax, absmax)
    return fig, js


def _render_triangle(tmpl, theme, idx, show_values):
    n = random.randint(MIN_DIM, MAX_DIM)
    labels = _rand_labels(n)

    lo, hi = tmpl["value_range"]
    full = np.random.uniform(lo, hi, (n, n))
    matrix = full.astype(float)
    # Mask upper triangle (excluding diagonal or not — randomize)
    k = random.choice([0, 1])   # 0 → include diagonal, 1 → exclude it
    mask = np.triu(np.ones((n, n), dtype=bool), k=k)
    matrix[mask] = np.nan
    matrix = np.where(np.isnan(matrix), np.nan, matrix.round(4))

    cmap = random.choice(SEQUENTIAL_CMAPS + DIVERGING_CMAPS)
    fig_sz = max(5.5, n * 1.0 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
    apply_theme(fig, ax, theme)
    ax.grid(False)

    matrix_masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(matrix_masked, cmap=cmap, aspect="equal", vmin=lo, vmax=hi)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    _style_colorbar(cbar, theme)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=9, color=theme["tick"])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9, color=theme["tick"])
    title = tmpl["theme"] + " (▽)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12,
                 color=theme["title"])
    ax.tick_params(length=0)

    if show_values:
        _annotate(ax, matrix, tmpl["fmt"], cmap, lo, hi)

    plt.tight_layout()
    js = _build_json(title, tmpl["x_label"], tmpl["y_label"],
                     labels, labels, matrix, lo, hi)
    return fig, js


RENDERERS = {
    "standard":    _render_standard,
    "correlation": _render_correlation,
    "diverging":   _render_diverging,
    "triangle":    _render_triangle,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    """Generate n heatmap charts (random subtype, template, and theme each time)."""
    for i in range(1, n + 1):
        random.seed(i * 13 + 5)
        np.random.seed(i * 13 + 5)
        subtype    = random.choice(SUBTYPES)
        tmpl       = random.choice(DATASET_TEMPLATES)
        theme      = random.choice(CHART_THEMES)
        show_vals  = random.random() < SHOW_VALUES_PROB
        fig, js    = RENDERERS[subtype](tmpl, theme, i, show_vals)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
