"""
radar_chart.py — Radar / Spider Chart Generator

Subtypes:
  simple       – Single-series radar
  multi_series – Multiple series on the same radar
  filled       – Filled area radar
"""

import random
import warnings
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, rv, apply_theme, save_outputs, build_standard_json,
)

CHART_TYPE = "radar"
MAX_DIMS   = 8

SUBTYPES = ["simple", "multi_series", "filled"]

DATASET_TEMPLATES = [
    {"theme": "Athlete Skills Assessment", "x_label": None, "y_label": None,
     "dimensions": ["Speed", "Strength", "Endurance", "Agility", "Technique", "Teamwork"],
     "series": ["Athlete A", "Athlete B", "Athlete C"], "value_range": (30, 100)},
    {"theme": "Product Feature Comparison", "x_label": None, "y_label": None,
     "dimensions": ["Performance", "Design", "Durability", "Price", "Support", "Usability"],
     "series": ["Product X", "Product Y", "Product Z"], "value_range": (20, 100)},
    {"theme": "Department KPI Radar", "x_label": None, "y_label": None,
     "dimensions": ["Efficiency", "Quality", "Cost Control", "Innovation", "Compliance", "Customer Sat"],
     "series": ["Q1", "Q2", "Q3", "Q4"], "value_range": (40, 95)},
    {"theme": "Country Development Indicators", "x_label": None, "y_label": None,
     "dimensions": ["Education", "Healthcare", "Infrastructure", "Economy", "Safety", "Environment"],
     "series": ["Country A", "Country B", "Country C"], "value_range": (25, 95)},
    {"theme": "Player Ability Radar", "x_label": None, "y_label": None,
     "dimensions": ["Shooting", "Passing", "Dribbling", "Defense", "Physical", "Pace"],
     "series": ["Player 1", "Player 2"], "value_range": (40, 99)},
    {"theme": "Software Architecture Quality", "x_label": None, "y_label": None,
     "dimensions": ["Scalability", "Security", "Maintainability", "Performance", "Reliability", "Testability"],
     "series": ["System A", "System B"], "value_range": (30, 90)},
    {"theme": "Marketing Channel Effectiveness", "x_label": None, "y_label": None,
     "dimensions": ["Reach", "Engagement", "Conversion", "Retention", "Cost Efficiency", "Brand Impact"],
     "series": ["Social", "Email", "SEO", "Paid"], "value_range": (20, 90)},
    {"theme": "Candidate Competency Profile", "x_label": None, "y_label": None,
     "dimensions": ["Communication", "Leadership", "Technical", "Problem Solving", "Adaptability", "Teamwork"],
     "series": ["Candidate A", "Candidate B"], "value_range": (40, 100)},
]

# ─────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────

def _make_radar_axes(fig, n_dims):
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close polygon
    ax = fig.add_subplot(111, polar=True)
    return ax, angles


def _plot_radar(ax, angles, values, color, alpha=1.0, linewidth=2, fill=False, label=None):
    values_closed = values + values[:1]
    ax.plot(angles, values_closed, color=color, linewidth=linewidth, label=label)
    if fill:
        ax.fill(angles, values_closed, color=color, alpha=alpha)

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_simple(tmpl, theme, idx):
    dims   = tmpl["dimensions"][:MAX_DIMS]
    n      = len(dims)
    vals   = rv(n, *tmpl["value_range"])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor(theme["bg"])
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(theme["bg"])
    _plot_radar(ax, angles, vals, theme["palette"][0], linewidth=2.5)
    ax.fill(angles, vals + vals[:1], color=theme["palette"][0], alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, color=theme["label"], fontsize=10)
    ax.yaxis.set_tick_params(labelcolor=theme["tick"])
    ax.grid(True, color=theme["grid"], linewidth=0.7)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=20,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    dp = [{"series_name": "Main", "x_value": d, "y_value": v} for d, v in zip(dims, vals)]
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


def _render_multi_series(tmpl, theme, idx):
    dims    = tmpl["dimensions"][:MAX_DIMS]
    series  = tmpl["series"][:4]
    n       = len(dims)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor(theme["bg"])
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(theme["bg"])

    dp = []
    for si, ser in enumerate(series):
        vals = rv(n, *tmpl["value_range"])
        color = theme["palette"][si % len(theme["palette"])]
        _plot_radar(ax, angles, vals, color, linewidth=2, label=ser)
        dp.extend([{"series_name": ser, "x_value": d, "y_value": v} for d, v in zip(dims, vals)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, color=theme["label"], fontsize=10)
    ax.yaxis.set_tick_params(labelcolor=theme["tick"])
    ax.grid(True, color=theme["grid"], linewidth=0.7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=20,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


def _render_filled(tmpl, theme, idx):
    dims    = tmpl["dimensions"][:MAX_DIMS]
    series  = tmpl["series"][:3]
    n       = len(dims)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor(theme["bg"])
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(theme["bg"])

    dp = []
    for si, ser in enumerate(series):
        vals  = rv(n, *tmpl["value_range"])
        color = theme["palette"][si % len(theme["palette"])]
        _plot_radar(ax, angles, vals, color, alpha=0.25, linewidth=2, fill=True, label=ser)
        dp.extend([{"series_name": ser, "x_value": d, "y_value": v} for d, v in zip(dims, vals)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, color=theme["label"], fontsize=10)
    ax.yaxis.set_tick_params(labelcolor=theme["tick"])
    ax.grid(True, color=theme["grid"], linewidth=0.7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=20,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


RENDERERS = {
    "simple":       _render_simple,
    "multi_series": _render_multi_series,
    "filled":       _render_filled,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 29 + 7)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
