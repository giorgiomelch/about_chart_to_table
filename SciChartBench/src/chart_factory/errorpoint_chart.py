"""
errorpoint_chart.py — Error Point Chart Generator

Subtypes:
  vertical      – Vertical error bars, categorical X axis
  horizontal    – Horizontal error bars, categorical Y axis
  multi_series  – Multiple series with vertical error bars
"""

import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, rv, apply_theme,
    get_axis_limits, build_standard_json, save_outputs,
)

CHART_TYPE = "errorpoint"
MAX_PTS    = 12

SUBTYPES = ["vertical", "horizontal", "multi_series"]

DATASET_TEMPLATES = [
    {"theme": "Experimental Measurements", "x_label": "Experiment", "y_label": "Value (μm)",
     "categories": ["Exp 1", "Exp 2", "Exp 3", "Exp 4", "Exp 5", "Exp 6", "Exp 7", "Exp 8"],
     "series": ["Main"], "value_range": (10, 90), "error_range": (2, 12)},
    {"theme": "Clinical Trial Results", "x_label": "Treatment Group", "y_label": "Response Rate (%)",
     "categories": ["Placebo", "Dose 1mg", "Dose 5mg", "Dose 10mg", "Dose 20mg"],
     "series": ["Week 4", "Week 8", "Week 12"], "value_range": (20, 95), "error_range": (3, 15)},
    {"theme": "Sensor Readings Across Locations", "x_label": "Location", "y_label": "Temperature (°C)",
     "categories": ["Station A", "Station B", "Station C", "Station D", "Station E", "Station F"],
     "series": ["Morning", "Afternoon", "Evening"], "value_range": (5, 40), "error_range": (1, 5)},
    {"theme": "Survey Confidence Intervals", "x_label": "Category", "y_label": "Agreement Score",
     "categories": ["Policy A", "Policy B", "Policy C", "Policy D", "Policy E"],
     "series": ["Age 18-30", "Age 31-50", "Age 51+"], "value_range": (30, 90), "error_range": (5, 20)},
    {"theme": "Material Tensile Strength", "x_label": "Material", "y_label": "Tensile Strength (MPa)",
     "categories": ["Steel", "Aluminum", "Carbon Fiber", "Titanium", "Copper", "Brass"],
     "series": ["Main"], "value_range": (100, 1500), "error_range": (20, 100)},
    {"theme": "Crop Yield by Irrigation Method", "x_label": "Method", "y_label": "Yield (tons/ha)",
     "categories": ["Drip", "Sprinkler", "Flood", "Subsurface", "Rainfed"],
     "series": ["Wheat", "Corn", "Soybean"], "value_range": (2, 12), "error_range": (0.3, 1.5)},
    {"theme": "Execution Time by Algorithm", "x_label": "Algorithm", "y_label": "Time (ms)",
     "categories": ["BFS", "DFS", "Dijkstra", "A*", "Bellman-Ford"],
     "series": ["Small Graph", "Medium Graph", "Large Graph"], "value_range": (1, 500), "error_range": (5, 50)},
    {"theme": "Protein Expression Levels", "x_label": "Protein", "y_label": "Expression (AU)",
     "categories": ["Prot-A", "Prot-B", "Prot-C", "Prot-D", "Prot-E", "Prot-F"],
     "series": ["Control", "Treatment"], "value_range": (0.5, 5.0), "error_range": (0.1, 0.6)},
]

# ─────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────

def _gen_errorpoint(center: float, err_lo: float, err_hi: float) -> dict:
    lo = round(max(0.0, center - random.uniform(err_lo, err_hi)), 4)
    hi = round(center + random.uniform(err_lo, err_hi), 4)
    return {"min": lo, "median": round(center, 4), "max": hi}

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_vertical(tmpl, theme, idx):
    cats = tmpl["categories"][:MAX_PTS]
    lo, hi   = tmpl["value_range"]
    elo, ehi = tmpl["error_range"]
    centers  = rv(len(cats), lo, hi)
    eps      = [_gen_errorpoint(c, elo, ehi) for c in centers]
    yerr_lo  = [c - ep["min"]  for c, ep in zip(centers, eps)]
    yerr_hi  = [ep["max"] - c  for c, ep in zip(centers, eps)]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    ax.errorbar(cats, centers, yerr=[yerr_lo, yerr_hi],
                fmt="o", color=theme["palette"][0],
                ecolor=theme["palette"][2 % len(theme["palette"])],
                elinewidth=2, capsize=5, capthick=2, markersize=7, zorder=3)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": c, "y_value": ep}
          for c, ep in zip(cats, eps)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_horizontal(tmpl, theme, idx):
    cats = tmpl["categories"][:MAX_PTS]
    lo, hi   = tmpl["value_range"]
    elo, ehi = tmpl["error_range"]
    centers  = rv(len(cats), lo, hi)
    eps      = [_gen_errorpoint(c, elo, ehi) for c in centers]
    xerr_lo  = [c - ep["min"]  for c, ep in zip(centers, eps)]
    xerr_hi  = [ep["max"] - c  for c, ep in zip(centers, eps)]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    ax.errorbar(centers, cats, xerr=[xerr_lo, xerr_hi],
                fmt="o", color=theme["palette"][1 % len(theme["palette"])],
                ecolor=theme["palette"][3 % len(theme["palette"])],
                elinewidth=2, capsize=5, capthick=2, markersize=7, zorder=3,
                linestyle="none")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["y_label"], fontsize=11)
    ax.set_ylabel(tmpl["x_label"], fontsize=11)
    plt.tight_layout()
    # x is value axis, y is categorical
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y", is_categorical=True)
    # For horizontal: x_value holds the errorpoint object, y_value holds the category
    dp = [{"series_name": "Main", "x_value": ep, "y_value": c}
          for c, ep in zip(cats, eps)]
    return fig, build_standard_json(tmpl["theme"], tmpl["y_label"], tmpl["x_label"], x_lim, y_lim, dp)


def _render_multi_series(tmpl, theme, idx):
    cats    = tmpl["categories"][:8]
    series  = tmpl["series"][:3]
    lo, hi  = tmpl["value_range"]
    elo, ehi = tmpl["error_range"]
    offsets = np.linspace(-0.15, 0.15, len(series))
    x_pos   = np.arange(len(cats))

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)

    series_eps = {}
    for si, ser in enumerate(series):
        centers = rv(len(cats), lo, hi)
        eps     = [_gen_errorpoint(c, elo, ehi) for c in centers]
        series_eps[ser] = list(zip(cats, eps, centers))
        yerr_lo = [c - ep["min"]  for c, ep in zip(centers, eps)]
        yerr_hi = [ep["max"] - c  for c, ep in zip(centers, eps)]
        ax.errorbar(x_pos + offsets[si], centers, yerr=[yerr_lo, yerr_hi],
                    fmt="o", color=theme["palette"][si % len(theme["palette"])],
                    ecolor=theme["palette"][si % len(theme["palette"])],
                    elinewidth=1.8, capsize=4, capthick=1.8, markersize=6,
                    label=ser, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": ser, "x_value": cat, "y_value": ep}
          for ser, triples in series_eps.items()
          for cat, ep, _ in triples]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


RENDERERS = {
    "vertical":     _render_vertical,
    "horizontal":   _render_horizontal,
    "multi_series": _render_multi_series,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 17 + 4)
        np.random.seed(i * 17 + 4)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
