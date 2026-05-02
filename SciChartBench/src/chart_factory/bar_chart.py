"""
bar_chart.py — Bar Chart Generator

Subtypes:
  simple       – Vertical bars, single series
  horizontal   – Horizontal bars, single series
  grouped      – Grouped multi-series vertical bars
  stacked      – Stacked multi-series vertical bars
  diverging    – Diverging bars (positive/negative values)
  lollipop     – Lollipop chart (single series)
"""

import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, rv, apply_theme,
    get_axis_limits, build_standard_json, save_outputs,
)

CHART_TYPE = "bar"
MAX_PTS    = 15

SUBTYPES = ["simple", "horizontal", "grouped", "stacked", "diverging", "lollipop"]

DATASET_TEMPLATES = [
    {"theme": "Quarterly Sales by Region", "x_label": "Quarter", "y_label": "Sales (€)",
     "categories": ["Q1", "Q2", "Q3", "Q4"],
     "series": ["North", "Center", "South"], "value_range": (50_000, 300_000)},
    {"theme": "Monthly Energy Consumption", "x_label": "Month", "y_label": "kWh",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
     "series": ["2022", "2023", "2024"], "value_range": (100, 900)},
    {"theme": "Programming Language Popularity", "x_label": "Language", "y_label": "Index (%)",
     "categories": ["Python", "JavaScript", "Java", "C++", "Rust", "Go", "TypeScript"],
     "series": ["Main"], "value_range": (2, 35)},
    {"theme": "Average Score by Subject", "x_label": "Subject", "y_label": "Average Score",
     "categories": ["Math", "Physics", "Chemistry", "Biology", "History", "English"],
     "series": ["Class A", "Class B"], "value_range": (55, 98)},
    {"theme": "Regional Agricultural Production", "x_label": "Region", "y_label": "Tonnes (k)",
     "categories": ["Sicily", "Puglia", "Campania", "Lombardy", "Veneto", "Emilia-Romagna"],
     "series": ["Wheat", "Corn", "Tomatoes"], "value_range": (10, 500)},
    {"theme": "Active Users by Platform", "x_label": "Platform", "y_label": "Active Users (M)",
     "categories": ["Mobile", "Desktop", "Tablet", "Smart TV", "Console"],
     "series": ["2023", "2024"], "value_range": (5, 200)},
    {"theme": "CO₂ Emissions by Sector", "x_label": "Sector", "y_label": "MtCO₂e",
     "categories": ["Transport", "Industry", "Agriculture", "Energy", "Buildings", "Waste"],
     "series": ["Main"], "value_range": (20, 450)},
    {"theme": "Departmental Budget", "x_label": "Department", "y_label": "Budget (M€)",
     "categories": ["R&D", "Marketing", "HR", "IT", "Operations", "Legal", "Finance"],
     "series": ["Planned", "Actual"], "value_range": (1, 80)},
    {"theme": "Monthly Average Temperature", "x_label": "Month", "y_label": "Temperature (°C)",
     "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
     "series": ["Rome", "Milan", "Palermo"], "value_range": (0, 38)},
    {"theme": "Employment Rate by Age Group", "x_label": "Age Group", "y_label": "Rate (%)",
     "categories": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
     "series": ["Men", "Women"], "value_range": (20, 85)},
]

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_simple(tmpl, theme, idx):
    cats  = tmpl["categories"][:MAX_PTS]
    vals  = rv(len(cats), *tmpl["value_range"])
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    ax.bar(cats, vals, color=theme["palette"][0], edgecolor=theme["bg"], width=0.6, zorder=3)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_horizontal(tmpl, theme, idx):
    cats  = tmpl["categories"][:MAX_PTS]
    vals  = rv(len(cats), *tmpl["value_range"])
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    ax.barh(cats, vals, color=theme["palette"][1 % len(theme["palette"])],
            edgecolor=theme["bg"], height=0.6, zorder=3)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["y_label"], fontsize=11)
    ax.set_ylabel(tmpl["x_label"], fontsize=11)
    plt.tight_layout()
    # x is now the value axis, y is categorical
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y", is_categorical=True)
    dp = [{"series_name": "Main", "x_value": v, "y_value": c} for c, v in zip(cats, vals)]
    return fig, build_standard_json(tmpl["theme"], tmpl["y_label"], tmpl["x_label"], x_lim, y_lim, dp)


def _render_grouped(tmpl, theme, idx):
    cats    = tmpl["categories"][:8]
    series  = tmpl["series"][:4]
    n_cats  = len(cats)
    n_ser   = len(series)
    x       = np.arange(n_cats)
    width   = 0.8 / n_ser
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    series_vals = {}
    for i, ser in enumerate(series):
        vals = rv(n_cats, *tmpl["value_range"])
        series_vals[ser] = vals
        offset = (i - n_ser / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=theme["palette"][i % len(theme["palette"])],
               label=ser, edgecolor=theme["bg"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": ser, "x_value": c, "y_value": v}
          for ser in series for c, v in zip(cats, series_vals[ser])]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_stacked(tmpl, theme, idx):
    cats   = tmpl["categories"][:8]
    series = tmpl["series"][:4]
    x      = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_theme(fig, ax, theme)
    bottoms = np.zeros(len(cats))
    series_vals = {}
    for i, ser in enumerate(series):
        vals = rv(len(cats), *tmpl["value_range"])
        series_vals[ser] = vals
        ax.bar(x, vals, bottom=bottoms,
               color=theme["palette"][i % len(theme["palette"])],
               label=ser, edgecolor=theme["bg"], width=0.6, zorder=3)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": ser, "x_value": c, "y_value": v}
          for ser in series for c, v in zip(cats, series_vals[ser])]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_diverging(tmpl, theme, idx):
    cats = tmpl["categories"][:MAX_PTS]
    lo, hi = tmpl["value_range"]
    mid  = (lo + hi) / 2
    vals = [round(random.uniform(lo - mid, hi - mid), 4) for _ in cats]
    colors = [theme["palette"][0] if v >= 0 else theme["palette"][3 % len(theme["palette"])]
              for v in vals]
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    ax.bar(cats, vals, color=colors, edgecolor=theme["bg"], width=0.6, zorder=3)
    ax.axhline(0, color=theme["tick"], linewidth=1.2)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x", is_categorical=True)
    y_lim = get_axis_limits(ax, "y")
    dp = [{"series_name": "Main", "x_value": c, "y_value": v} for c, v in zip(cats, vals)]
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_lollipop(tmpl, theme, idx):
    cats = tmpl["categories"][:MAX_PTS]
    vals = rv(len(cats), *tmpl["value_range"])
    x    = range(len(cats))
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    ax.vlines(x, 0, vals, color=theme["palette"][0], linewidth=2, zorder=3)
    ax.scatter(x, vals, color=theme["palette"][0], s=80, zorder=4)
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


RENDERERS = {
    "simple":     _render_simple,
    "horizontal": _render_horizontal,
    "grouped":    _render_grouped,
    "stacked":    _render_stacked,
    "diverging":  _render_diverging,
    "lollipop":   _render_lollipop,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    """Generate n bar charts (random subtype, template, and theme each time)."""
    for i in range(1, n + 1):
        random.seed(i * 7 + 1)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
