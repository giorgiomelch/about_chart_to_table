"""
pie_chart.py — Pie Chart Generator

Subtypes:
  classic          – Standard pie chart
  donut            – Donut chart (pie with hole)
  exploded         – Exploded pie (one slice pulled out)
  rose_nightingale – Rose / Nightingale chart (polar bar)
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
    CHART_THEMES, rv, apply_theme, save_outputs,
)

CHART_TYPE = "pie"
MAX_SLICES = 10

SUBTYPES = ["classic", "donut", "exploded", "rose_nightingale"]

DATASET_TEMPLATES = [
    {"theme": "Market Share by Company", "slices": ["Apple", "Samsung", "Xiaomi", "Huawei", "OPPO", "Others"],
     "weights": [28, 22, 13, 10, 9, 18]},
    {"theme": "Budget Allocation by Department", "slices": ["R&D", "Marketing", "HR", "IT", "Operations", "Legal"],
     "weights": [30, 20, 10, 15, 18, 7]},
    {"theme": "Global Energy Mix 2024", "slices": ["Coal", "Natural Gas", "Oil", "Nuclear", "Hydro", "Wind", "Solar"],
     "weights": [27, 23, 31, 5, 7, 4, 3]},
    {"theme": "Traffic Sources", "slices": ["Organic Search", "Direct", "Social", "Referral", "Email", "Paid"],
     "weights": [40, 25, 15, 10, 6, 4]},
    {"theme": "Population by Continent", "slices": ["Asia", "Africa", "Europe", "Americas", "Oceania"],
     "weights": [59, 18, 10, 13, 1]},
    {"theme": "Programming Language Usage", "slices": ["Python", "JavaScript", "Java", "C/C++", "C#", "PHP", "Others"],
     "weights": [30, 24, 17, 10, 7, 5, 7]},
    {"theme": "Causes of Website Downtime", "slices": ["Hardware", "Software", "Human Error", "Network", "Security", "Other"],
     "weights": [22, 28, 18, 15, 10, 7]},
    {"theme": "Investment Portfolio Allocation", "slices": ["Stocks", "Bonds", "Real Estate", "Commodities", "Cash", "Crypto"],
     "weights": [40, 25, 15, 8, 7, 5]},
]

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_classic(tmpl, theme, idx):
    slices  = tmpl["slices"][:MAX_SLICES]
    weights = tmpl["weights"][:MAX_SLICES]
    total   = sum(weights)
    vals    = [round(w / total * 100, 4) for w in weights]
    colors  = (theme["palette"] * 3)[:len(slices)]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])
    wedges, texts, autotexts = ax.pie(
        vals, labels=slices, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": theme["bg"], "linewidth": 1.5},
        textprops={"color": theme["label"]},
    )
    for at in autotexts:
        at.set_color(theme["bg"])
        at.set_fontsize(9)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=14,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    dp = [{"series_name": "Main", "x_value": s, "y_value": v}
          for s, v in zip(slices, vals)]
    from .common import build_standard_json
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


def _render_donut(tmpl, theme, idx):
    slices  = tmpl["slices"][:MAX_SLICES]
    weights = tmpl["weights"][:MAX_SLICES]
    total   = sum(weights)
    vals    = [round(w / total * 100, 4) for w in weights]
    colors  = (theme["palette"] * 3)[:len(slices)]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])
    wedges, texts, autotexts = ax.pie(
        vals, labels=slices, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": theme["bg"], "linewidth": 1.5, "width": 0.6},
        textprops={"color": theme["label"]},
    )
    for at in autotexts:
        at.set_color(theme["bg"])
        at.set_fontsize(9)
    ax.text(0, 0, f"{sum(weights)}", ha="center", va="center",
            fontsize=16, fontweight="bold", color=theme["title"])
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=14,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    dp = [{"series_name": "Main", "x_value": s, "y_value": v}
          for s, v in zip(slices, vals)]
    from .common import build_standard_json
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


def _render_exploded(tmpl, theme, idx):
    slices  = tmpl["slices"][:MAX_SLICES]
    weights = tmpl["weights"][:MAX_SLICES]
    total   = sum(weights)
    vals    = [round(w / total * 100, 4) for w in weights]
    colors  = (theme["palette"] * 3)[:len(slices)]

    # Explode the largest slice
    max_idx = vals.index(max(vals))
    explode = [0.0] * len(slices)
    explode[max_idx] = 0.12

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])
    wedges, texts, autotexts = ax.pie(
        vals, labels=slices, colors=colors,
        explode=explode, autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": theme["bg"], "linewidth": 1.5},
        textprops={"color": theme["label"]},
    )
    for at in autotexts:
        at.set_color(theme["bg"])
        at.set_fontsize(9)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=14,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    dp = [{"series_name": "Main", "x_value": s, "y_value": v}
          for s, v in zip(slices, vals)]
    from .common import build_standard_json
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


def _render_rose_nightingale(tmpl, theme, idx):
    slices  = tmpl["slices"][:MAX_SLICES]
    weights = tmpl["weights"][:MAX_SLICES]
    total   = sum(weights)
    vals    = [round(w / total * 100, 4) for w in weights]
    n       = len(slices)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)
    colors  = (theme["palette"] * 3)[:n]

    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor(theme["bg"])
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor(theme["bg"])
    width = 2 * np.pi / n

    bars = ax.bar(angles, vals, width=width * 0.85, bottom=0.0,
                  color=colors, edgecolor=theme["bg"], linewidth=1.2, alpha=0.85)
    ax.set_xticks(angles)
    ax.set_xticklabels(slices, color=theme["label"], fontsize=9)
    ax.yaxis.set_tick_params(labelcolor=theme["tick"])
    ax.grid(True, color=theme["grid"], linewidth=0.6)
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=20,
                 color=theme["title"])
    plt.tight_layout()

    null_lim = {"min": None, "max": None, "is_log": False}
    dp = [{"series_name": "Main", "x_value": s, "y_value": v}
          for s, v in zip(slices, vals)]
    from .common import build_standard_json
    return fig, build_standard_json(tmpl["theme"], None, None, null_lim, null_lim, dp)


RENDERERS = {
    "classic":          _render_classic,
    "donut":            _render_donut,
    "exploded":         _render_exploded,
    "rose_nightingale": _render_rose_nightingale,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 19 + 5)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
