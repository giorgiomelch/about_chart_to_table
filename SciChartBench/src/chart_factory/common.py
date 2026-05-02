"""
common.py — Shared utilities for the chart factory.

Provides:
- CHART_THEMES: 8 visual themes
- IMAGE_BASE / JSON_BASE: output directory roots
- rv(n, lo, hi): random float list
- apply_theme(fig, ax, theme): applies visual styling to a matplotlib figure
- get_axis_limits(ax, axis, is_categorical): extracts axis limits from matplotlib
- build_standard_json(...): constructs the standard JSON structure
- build_bubble_json(...): constructs the bubble-chart JSON structure
- save_outputs(fig, json_data, chart_type, idx, subtype): saves PNG + JSON
"""

import os
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  OUTPUT PATHS
# ─────────────────────────────────────────

IMAGE_BASE = "data/data/synthetic"
JSON_BASE  = "data/groundtruth/synthetic"

# ─────────────────────────────────────────
#  VISUAL THEMES
# ─────────────────────────────────────────

CHART_THEMES = [
    {
        "name": "corporate_blue",
        "bg": "#FFFFFF", "grid": "#E5E5E5", "title": "#1A237E",
        "label": "#37474F", "tick": "#546E7A", "spine": True,
        "palette": ["#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#90CAF9", "#64B5F6", "#0D47A1", "#82B1FF"],
    },
    {
        "name": "dark_pro",
        "bg": "#1E1E2E", "grid": "#313244", "title": "#CDD6F4",
        "label": "#BAC2DE", "tick": "#A6ADC8", "spine": False,
        "palette": ["#89B4FA", "#A6E3A1", "#FAB387", "#F38BA8", "#CBA6F7", "#94E2D5", "#F9E2AF", "#74C7EC"],
    },
    {
        "name": "pastel_soft",
        "bg": "#FAFAFA", "grid": "#F0F0F0", "title": "#4A4A4A",
        "label": "#6A6A6A", "tick": "#8A8A8A", "spine": False,
        "palette": ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E8BAFF", "#FFC8DD", "#BDE0FE"],
    },
    {
        "name": "vibrant_modern",
        "bg": "#F8F9FA", "grid": "#DEE2E6", "title": "#212529",
        "label": "#495057", "tick": "#6C757D", "spine": True,
        "palette": ["#E63946", "#F4A261", "#2A9D8F", "#264653", "#E9C46A", "#8338EC", "#FB5607", "#3A86FF"],
    },
    {
        "name": "minimal_mono",
        "bg": "#FFFFFF", "grid": "#F0F0F0", "title": "#000000",
        "label": "#333333", "tick": "#666666", "spine": False,
        "palette": ["#000000", "#333333", "#555555", "#777777", "#999999", "#BBBBBB", "#222222", "#444444"],
    },
    {
        "name": "sunset_gradient",
        "bg": "#FFF8F0", "grid": "#FFE0CC", "title": "#7B2D00",
        "label": "#8B4513", "tick": "#A0522D", "spine": True,
        "palette": ["#FF6B35", "#F7C59F", "#004E89", "#1A936F", "#88D498", "#C84B31", "#F3722C", "#577590"],
    },
    {
        "name": "neon_dark",
        "bg": "#0D0D0D", "grid": "#1A1A1A", "title": "#00FF88",
        "label": "#CCCCCC", "tick": "#888888", "spine": False,
        "palette": ["#00FF88", "#FF006E", "#3A86FF", "#FFBE0B", "#FB5607", "#8338EC", "#00B4D8", "#FF9F1C"],
    },
    {
        "name": "nature_green",
        "bg": "#F1F8E9", "grid": "#DCEDC8", "title": "#1B5E20",
        "label": "#2E7D32", "tick": "#388E3C", "spine": True,
        "palette": ["#2E7D32", "#388E3C", "#43A047", "#66BB6A", "#A5D6A7", "#81C784", "#1B5E20", "#558B2F"],
    },
]

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def rv(n: int, lo: float, hi: float) -> list:
    """Return n random floats in [lo, hi], rounded to 4 decimal places."""
    return [round(random.uniform(lo, hi), 4) for _ in range(n)]


def apply_theme(fig, ax, theme: dict) -> None:
    """Apply a visual theme to a matplotlib figure and axes."""
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])
    ax.title.set_color(theme["title"])
    ax.xaxis.label.set_color(theme["label"])
    ax.yaxis.label.set_color(theme["label"])
    ax.tick_params(colors=theme["tick"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.7, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(theme["spine"])
        if theme["spine"]:
            sp.set_edgecolor(theme["grid"])


def get_axis_limits(ax, axis: str, is_categorical: bool = False) -> dict:
    """
    Extract axis limits from a matplotlib Axes after layout is applied.

    Parameters
    ----------
    ax            : matplotlib Axes
    axis          : "x" or "y"
    is_categorical: True when the axis carries string tick labels
                    (min/max will be null in the JSON)

    Returns
    -------
    {"min": float|None, "max": float|None, "is_log": bool}
    """
    if is_categorical:
        return {"min": None, "max": None, "is_log": False}

    if axis == "x":
        lo, hi  = ax.get_xlim()
        is_log  = ax.get_xscale() == "log"
    else:
        lo, hi  = ax.get_ylim()
        is_log  = ax.get_yscale() == "log"

    return {
        "min":    round(lo, 6),
        "max":    round(hi, 6),
        "is_log": is_log,
    }

# ─────────────────────────────────────────
#  JSON BUILDERS
# ─────────────────────────────────────────

def build_standard_json(
    title,
    x_label,
    y_label,
    x_limits: dict,
    y_limits: dict,
    data_points: list,
) -> dict:
    """Build the standard JSON structure (bar/line/scatter/histogram/etc.)."""
    return {
        "chart_title":    title,
        "x_axis_label":   x_label,
        "y_axis_label":   y_label,
        "x_axis_limits":  x_limits,
        "y_axis_limits":  y_limits,
        "data_points":    data_points,
    }


def build_bubble_json(
    title,
    x_label,
    y_label,
    x_limits: dict,
    y_limits: dict,
    z_limits: dict,
    w_limits: dict,
    data_points: list,
) -> dict:
    """Build the bubble-chart JSON structure."""
    return {
        "chart_title":    title,
        "x_axis_label":   x_label,
        "y_axis_label":   y_label,
        "x_axis_limits":  x_limits,
        "y_axis_limits":  y_limits,
        "z_axis_limits":  z_limits,
        "w_axis_limits":  w_limits,
        "data_points":    data_points,
    }

# ─────────────────────────────────────────
#  SAVE UTILITIES
# ─────────────────────────────────────────

def save_outputs(
    fig,
    json_data: dict,
    chart_type: str,
    idx: int,
    subtype: str,
) -> None:
    """
    Save the matplotlib figure as PNG and the JSON data.

    PNG  → data/synthetic/{chart_type}/chart_{idx:03d}_{subtype}.png
    JSON → data/groundtruth/synthetic/{chart_type}/chart_{idx:03d}_{subtype}.json
    """
    img_dir  = os.path.join(IMAGE_BASE, chart_type)
    json_dir = os.path.join(JSON_BASE,  chart_type)
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    stem     = f"chart_{idx:03d}_{subtype}"
    img_path = os.path.join(img_dir,  f"{stem}.png")
    js_path  = os.path.join(json_dir, f"{stem}.json")

    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
