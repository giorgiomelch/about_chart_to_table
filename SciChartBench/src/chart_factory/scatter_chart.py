"""
scatter_chart.py — Scatter Plot Generator

Subtypes:
  simple       – Single-series scatter
  multi_class  – Multiple colour-coded series
  regression   – Scatter with regression line
  quadrant     – Quadrant plot with reference lines
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

CHART_TYPE = "scatter"
MAX_PTS    = 15

SUBTYPES = ["simple", "multi_class", "regression", "quadrant"]

DATASET_TEMPLATES = [
    {"theme": "Height vs Weight", "x_label": "Height (cm)", "y_label": "Weight (kg)",
     "series": [{"name": "Adults", "x": (150, 200), "y": (45, 120), "n": 12}]},
    {"theme": "Study Hours vs Exam Score", "x_label": "Study Hours/Week", "y_label": "Exam Score (%)",
     "series": [{"name": "Students", "x": (0, 40), "y": (30, 100), "n": 12}]},
    {"theme": "Income vs Life Satisfaction", "x_label": "Annual Income (K$)", "y_label": "Life Satisfaction (1-10)",
     "series": [{"name": "Survey", "x": (20, 200), "y": (3, 9.5), "n": 14}]},
    {"theme": "Advertising Spend vs Revenue", "x_label": "Ad Spend (K$)", "y_label": "Revenue (K$)",
     "series": [{"name": "Campaigns", "x": (5, 100), "y": (50, 2000), "n": 12}]},
    {"theme": "Temperature vs Ice Cream Sales", "x_label": "Temperature (°C)", "y_label": "Sales (units)",
     "series": [{"name": "Daily", "x": (10, 40), "y": (50, 500), "n": 14}]},
    {"theme": "Sleep Hours vs Productivity", "x_label": "Sleep (hours)", "y_label": "Productivity Score",
     "series": [{"name": "Employees", "x": (4, 10), "y": (20, 100), "n": 12}]},
    {"theme": "CO₂ vs GDP", "x_label": "GDP per Capita (K$)", "y_label": "CO₂ per Capita (t)",
     "series": [{"name": "Countries", "x": (1, 80), "y": (0.2, 20), "n": 14}]},
    {"theme": "Age vs Marathon Time", "x_label": "Age (years)", "y_label": "Marathon Time (min)",
     "series": [{"name": "Runners", "x": (18, 65), "y": (150, 350), "n": 12}]},
]

MULTI_TEMPLATES = [
    {"theme": "Species Classification", "x_label": "Sepal Length (cm)", "y_label": "Petal Length (cm)",
     "series": [
         {"name": "Species A", "x": (4.5, 5.5), "y": (1.0, 2.0), "n": 10},
         {"name": "Species B", "x": (5.5, 7.0), "y": (3.0, 5.0), "n": 10},
         {"name": "Species C", "x": (6.0, 8.0), "y": (4.5, 7.0), "n": 10},
     ]},
    {"theme": "Customer Segments", "x_label": "Annual Spend ($)", "y_label": "Visit Frequency",
     "series": [
         {"name": "Budget",    "x": (100, 500),   "y": (1, 5),   "n": 10},
         {"name": "Regular",   "x": (500, 2000),  "y": (5, 20),  "n": 10},
         {"name": "Premium",   "x": (2000, 8000), "y": (15, 50), "n": 8},
     ]},
    {"theme": "Software Bug Distribution", "x_label": "Lines of Code (K)", "y_label": "Bug Count",
     "series": [
         {"name": "Frontend", "x": (5, 30),  "y": (2, 40),  "n": 8},
         {"name": "Backend",  "x": (10, 80), "y": (5, 80),  "n": 10},
         {"name": "Mobile",   "x": (3, 20),  "y": (1, 25),  "n": 8},
     ]},
    {"theme": "Player Stats by Position", "x_label": "Speed Score", "y_label": "Strength Score",
     "series": [
         {"name": "Forwards",   "x": (60, 90), "y": (50, 80), "n": 8},
         {"name": "Midfielders","x": (65, 85), "y": (45, 70), "n": 8},
         {"name": "Defenders",  "x": (55, 80), "y": (60, 90), "n": 8},
     ]},
]

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_simple(tmpl, theme, idx):
    ser = tmpl["series"][0]
    xs  = rv(min(ser["n"], MAX_PTS), *ser["x"])
    ys  = rv(min(ser["n"], MAX_PTS), *ser["y"])
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_theme(fig, ax, theme)
    ax.scatter(xs, ys, color=theme["palette"][0], s=70, alpha=0.8, edgecolors=theme["bg"],
               linewidths=0.5, zorder=3)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": ser["name"], "x_value": x, "y_value": y} for x, y in zip(xs, ys)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_multi_class(tmpl, theme, idx):
    fig, ax = plt.subplots(figsize=(10, 7))
    apply_theme(fig, ax, theme)
    dp = []
    for i, ser in enumerate(tmpl["series"]):
        n  = min(ser["n"], MAX_PTS)
        xs = rv(n, *ser["x"])
        ys = rv(n, *ser["y"])
        ax.scatter(xs, ys, color=theme["palette"][i % len(theme["palette"])],
                   s=70, alpha=0.8, edgecolors=theme["bg"], linewidths=0.5,
                   label=ser["name"], zorder=3)
        dp.extend([{"series_name": ser["name"], "x_value": x, "y_value": y}
                   for x, y in zip(xs, ys)])
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_regression(tmpl, theme, idx):
    ser = tmpl["series"][0]
    n   = min(ser["n"], MAX_PTS)
    xs  = sorted(rv(n, *ser["x"]))
    # Add slight linear correlation
    slope  = (ser["y"][1] - ser["y"][0]) / (ser["x"][1] - ser["x"][0])
    ys     = [round(ser["y"][0] + slope * (x - ser["x"][0]) + random.gauss(0, (ser["y"][1]-ser["y"][0])*0.15), 4)
              for x in xs]

    fig, ax = plt.subplots(figsize=(9, 6))
    apply_theme(fig, ax, theme)
    ax.scatter(xs, ys, color=theme["palette"][0], s=70, alpha=0.8, edgecolors=theme["bg"],
               linewidths=0.5, zorder=3, label="Data")
    # Regression line
    m, b = np.polyfit(xs, ys, 1)
    x_line = np.linspace(min(xs), max(xs), 100)
    ax.plot(x_line, m * x_line + b, color=theme["palette"][2 % len(theme["palette"])],
            linewidth=2, linestyle="--", label="Regression", zorder=4)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": ser["name"], "x_value": x, "y_value": y} for x, y in zip(xs, ys)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_quadrant(tmpl, theme, idx):
    ser  = tmpl["series"][0]
    n    = min(ser["n"], MAX_PTS)
    xs   = rv(n, *ser["x"])
    ys   = rv(n, *ser["y"])
    x_mid = (ser["x"][0] + ser["x"][1]) / 2
    y_mid = (ser["y"][0] + ser["y"][1]) / 2

    fig, ax = plt.subplots(figsize=(9, 7))
    apply_theme(fig, ax, theme)
    ax.scatter(xs, ys, color=theme["palette"][0], s=70, alpha=0.8, edgecolors=theme["bg"],
               linewidths=0.5, zorder=3)
    ax.axvline(x_mid, color=theme["tick"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(y_mid, color=theme["tick"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": ser["name"], "x_value": x, "y_value": y} for x, y in zip(xs, ys)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


RENDERERS = {
    "simple":      _render_simple,
    "multi_class": _render_multi_class,
    "regression":  _render_regression,
    "quadrant":    _render_quadrant,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 23 + 6)
        subtype = random.choice(SUBTYPES)
        if subtype == "multi_class":
            tmpl = random.choice(MULTI_TEMPLATES)
        else:
            tmpl = random.choice(DATASET_TEMPLATES)
        theme = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
