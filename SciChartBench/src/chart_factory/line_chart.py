"""
line_chart.py — Line Chart Generator

Subtypes:
  simple    – Single-series line chart
  multi     – Multiple line series
  area      – Area (filled) line chart
  step      – Step line chart
  dual_y    – Dual Y-axis line chart
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

CHART_TYPE = "line"
MAX_PTS    = 15

SUBTYPES = ["simple", "multi", "area", "step", "dual_y"]

DATASET_TEMPLATES = [
    {"theme": "Monthly Revenue Trend", "x_label": "Month", "y_label": "Revenue (K€)",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
     "series": ["2022", "2023", "2024"], "value_range": (50, 300)},
    {"theme": "Website Traffic Over Time", "x_label": "Week", "y_label": "Unique Visitors",
     "categories": [f"W{i}" for i in range(1, 13)],
     "series": ["Organic", "Paid", "Social"], "value_range": (1000, 25000)},
    {"theme": "Average Daily Temperature", "x_label": "Month", "y_label": "Temperature (°C)",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
     "series": ["London", "Rome", "Helsinki"], "value_range": (-5, 35)},
    {"theme": "CPU Usage by Service", "x_label": "Hour", "y_label": "CPU Usage (%)",
     "categories": [f"{h:02d}:00" for h in range(0, 24, 2)],
     "series": ["Service A", "Service B", "Service C"], "value_range": (5, 95)},
    {"theme": "Stock Price History", "x_label": "Quarter", "y_label": "Price ($)",
     "categories": ["Q1 21", "Q2 21", "Q3 21", "Q4 21", "Q1 22", "Q2 22", "Q3 22", "Q4 22",
                    "Q1 23", "Q2 23", "Q3 23", "Q4 23"],
     "series": ["Stock A", "Stock B"], "value_range": (50, 500)},
    {"theme": "Renewable Energy Generation", "x_label": "Month", "y_label": "GWh",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
     "series": ["Wind", "Solar", "Hydro"], "value_range": (100, 900)},
    {"theme": "User Signups Growth", "x_label": "Month", "y_label": "New Users",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
     "series": ["Mobile App", "Web"], "value_range": (200, 5000)},
    {"theme": "Pollution Index by Season", "x_label": "Month", "y_label": "PM2.5 (μg/m³)",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
     "series": ["City Center", "Suburbs"], "value_range": (10, 120)},
    {"theme": "Product Sales by Quarter", "x_label": "Quarter", "y_label": "Units Sold (K)",
     "categories": ["Q1", "Q2", "Q3", "Q4"],
     "series": ["Product A", "Product B", "Product C", "Product D"], "value_range": (5, 80)},
    {"theme": "Lab Measurement Over Runs", "x_label": "Run", "y_label": "Measured Value",
     "categories": [f"Run {i}" for i in range(1, 11)],
     "series": ["Sample 1", "Sample 2"], "value_range": (0.5, 10.0)},
]

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def _smooth(vals, alpha=0.4):
    """Exponential smoothing for nicer-looking lines."""
    smoothed = [vals[0]]
    for v in vals[1:]:
        smoothed.append(round(alpha * v + (1 - alpha) * smoothed[-1], 4))
    return smoothed

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_simple(tmpl, theme, idx):
    cats = tmpl["categories"][:MAX_PTS]
    vals = _smooth(rv(len(cats), *tmpl["value_range"]))
    x    = range(len(cats))
    fig, ax = plt.subplots(figsize=(11, 6))
    apply_theme(fig, ax, theme)
    ax.plot(list(x), vals, color=theme["palette"][0], linewidth=2.5, marker="o",
            markersize=5, markerfacecolor=theme["palette"][0], zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_multi(tmpl, theme, idx):
    cats   = tmpl["categories"][:MAX_PTS]
    series = tmpl["series"][:5]
    x      = range(len(cats))
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    dp = []
    for si, ser in enumerate(series):
        vals = _smooth(rv(len(cats), *tmpl["value_range"]))
        ax.plot(list(x), vals, color=theme["palette"][si % len(theme["palette"])],
                linewidth=2, marker="o", markersize=4, label=ser, zorder=3)
        dp.extend([{"series_name": ser, "x_value": c, "y_value": v}
                   for c, v in zip(cats, vals)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_area(tmpl, theme, idx):
    cats   = tmpl["categories"][:MAX_PTS]
    series = tmpl["series"][:3]
    x      = list(range(len(cats)))
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    dp = []
    for si, ser in enumerate(series):
        vals = _smooth(rv(len(cats), *tmpl["value_range"]))
        color = theme["palette"][si % len(theme["palette"])]
        ax.plot(x, vals, color=color, linewidth=1.5, zorder=3)
        ax.fill_between(x, vals, alpha=0.3, color=color, label=ser, zorder=2)
        dp.extend([{"series_name": ser, "x_value": c, "y_value": v}
                   for c, v in zip(cats, vals)])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_step(tmpl, theme, idx):
    cats = tmpl["categories"][:MAX_PTS]
    vals = rv(len(cats), *tmpl["value_range"])
    x    = list(range(len(cats)))
    fig, ax = plt.subplots(figsize=(11, 6))
    apply_theme(fig, ax, theme)
    ax.step(x, vals, where="mid", color=theme["palette"][0], linewidth=2.5, zorder=3)
    ax.scatter(x, vals, color=theme["palette"][0], s=50, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_dual_y(tmpl, theme, idx):
    cats   = tmpl["categories"][:MAX_PTS]
    x      = list(range(len(cats)))
    series = tmpl["series"][:2]
    lo, hi = tmpl["value_range"]
    vals1  = _smooth(rv(len(cats), lo, hi))
    vals2  = _smooth(rv(len(cats), lo * 0.1, hi * 0.1))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax1, theme)
    ax2 = ax1.twinx()
    ax2.set_facecolor(theme["bg"])

    c1 = theme["palette"][0]
    c2 = theme["palette"][1 % len(theme["palette"])]
    ax1.plot(x, vals1, color=c1, linewidth=2.5, marker="o", markersize=5,
             label=series[0], zorder=3)
    ax2.plot(x, vals2, color=c2, linewidth=2.5, marker="s", markersize=5,
             label=series[1], linestyle="--", zorder=3)
    ax2.tick_params(colors=theme["tick"])
    ax2.yaxis.label.set_color(theme["label"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=30, ha="right")
    ax1.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel(tmpl["x_label"], fontsize=11)
    ax1.set_ylabel(tmpl["y_label"] + f" ({series[0]})", fontsize=11, color=c1)
    ax2.set_ylabel(tmpl["y_label"] + f" ({series[1]})", fontsize=11, color=c2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()

    # For JSON: use ax1's y-axis limits (primary)
    x_lim = get_axis_limits(ax1, "x", is_categorical=True)
    y_lim = get_axis_limits(ax1, "y")
    dp = [{"series_name": series[0], "x_value": c, "y_value": v}
          for c, v in zip(cats, vals1)]
    dp += [{"series_name": series[1], "x_value": c, "y_value": v}
           for c, v in zip(cats, vals2)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


RENDERERS = {
    "simple": _render_simple,
    "multi":  _render_multi,
    "area":   _render_area,
    "step":   _render_step,
    "dual_y": _render_dual_y,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 31 + 8)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
