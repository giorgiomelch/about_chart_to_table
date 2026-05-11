"""
histogram_chart.py — Histogram Generator

Subtypes:
  simple       – Single-series histogram (frequency count)
  multi        – Multiple overlapping histograms
  cumulative   – Cumulative distribution histogram
"""

import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

from .common import (
    CHART_THEMES, apply_theme,
    get_axis_limits, build_standard_json, save_outputs,
)

CHART_TYPE = "histogram"
N_BINS     = 12   # default bins

SUBTYPES = ["simple", "multi", "cumulative"]

DATASET_TEMPLATES = [
    {"theme": "Age Distribution", "x_label": "Age (years)", "y_label": "Frequency",
     "series": [{"name": "Population", "dist": "normal", "mu": 38, "sigma": 14, "n": 200}]},
    {"theme": "Annual Salary Distribution", "x_label": "Salary (K€)", "y_label": "Count",
     "series": [
         {"name": "Engineers", "dist": "normal", "mu": 65,  "sigma": 15, "n": 150},
         {"name": "Managers",  "dist": "normal", "mu": 90,  "sigma": 20, "n": 100},
     ]},
    {"theme": "Exam Score Distribution", "x_label": "Score (%)", "y_label": "Students",
     "series": [
         {"name": "Class A", "dist": "normal", "mu": 72, "sigma": 12, "n": 100},
         {"name": "Class B", "dist": "normal", "mu": 65, "sigma": 15, "n": 100},
     ]},
    {"theme": "Server Response Time", "x_label": "Response Time (ms)", "y_label": "Requests",
     "series": [{"name": "API Gateway", "dist": "lognormal", "mu": 4.5, "sigma": 0.6, "n": 300}]},
    {"theme": "Product Review Ratings", "x_label": "Rating (1-5)", "y_label": "Count",
     "series": [{"name": "Reviews", "dist": "skewed", "mu": 4.1, "sigma": 0.8, "n": 200}]},
    {"theme": "Body Mass Index Distribution", "x_label": "BMI", "y_label": "Frequency",
     "series": [
         {"name": "Male",   "dist": "normal", "mu": 25.5, "sigma": 4.0, "n": 200},
         {"name": "Female", "dist": "normal", "mu": 24.0, "sigma": 3.8, "n": 200},
     ]},
    {"theme": "Transaction Amount Distribution", "x_label": "Amount ($)", "y_label": "Transactions",
     "series": [{"name": "Payments", "dist": "lognormal", "mu": 4.0, "sigma": 1.2, "n": 400}]},
    {"theme": "Commute Time Distribution", "x_label": "Time (min)", "y_label": "Employees",
     "series": [
         {"name": "City A", "dist": "normal", "mu": 35, "sigma": 12, "n": 200},
         {"name": "City B", "dist": "normal", "mu": 55, "sigma": 18, "n": 200},
     ]},
]

# ─────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────

def _sample(ser: dict) -> np.ndarray:
    """Generate synthetic samples from a distribution spec."""
    dist = ser["dist"]
    n    = ser["n"]
    mu, sigma = ser["mu"], ser["sigma"]
    if dist == "normal":
        return np.random.normal(mu, sigma, n)
    elif dist == "lognormal":
        return np.random.lognormal(mu, sigma, n)
    else:  # skewed
        return np.clip(np.random.normal(mu, sigma, n) - 0.3 * abs(np.random.normal(0, sigma, n)), 1, 5)


def _hist_data_points(counts, bin_edges, series_name, round_to_int=True) -> list:
    """Convert histogram (counts, bin_edges) to data_points list."""
    dp = []
    for cnt, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
        mid = round((left + right) / 2, 4)
        y = int(cnt) if round_to_int else round(float(cnt), 4)
        dp.append({"series_name": series_name, "x_value": mid, "y_value": y})
    return dp

# ─────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────

def _render_simple(tmpl, theme, idx):
    ser  = tmpl["series"][0]
    data = _sample(ser)
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    counts, bin_edges, patches = ax.hist(
        data, bins=N_BINS,
        color=theme["palette"][0], edgecolor=theme["bg"],
        alpha=0.85, zorder=3
    )
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    dp    = _hist_data_points(counts, bin_edges, ser["name"])
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_multi(tmpl, theme, idx):
    series = tmpl["series"][:3]
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    dp = []
    for si, ser in enumerate(series):
        data = _sample(ser)
        counts, bin_edges, _ = ax.hist(
            data, bins=N_BINS,
            color=theme["palette"][si % len(theme["palette"])],
            edgecolor=theme["bg"], alpha=0.6, label=ser["name"], zorder=3
        )
        dp.extend(_hist_data_points(counts, bin_edges, ser["name"]))
    ax.set_title(tmpl["theme"], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel(tmpl["y_label"], fontsize=11)
    ax.legend(facecolor=theme["bg"], labelcolor=theme["label"], fontsize=9)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    return fig, build_standard_json(tmpl["theme"], tmpl["x_label"], tmpl["y_label"], x_lim, y_lim, dp)


def _render_cumulative(tmpl, theme, idx):
    ser  = tmpl["series"][0]
    data = _sample(ser)
    fig, ax = plt.subplots(figsize=(10, 6))
    apply_theme(fig, ax, theme)
    counts, bin_edges, _ = ax.hist(
        data, bins=N_BINS, cumulative=True, density=True,
        color=theme["palette"][0], edgecolor=theme["bg"],
        alpha=0.85, zorder=3
    )
    ax.set_title(tmpl["theme"] + " (Cumulative)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(tmpl["x_label"], fontsize=11)
    ax.set_ylabel("Cumulative Frequency", fontsize=11)
    plt.tight_layout()
    x_lim = get_axis_limits(ax, "x")
    y_lim = get_axis_limits(ax, "y")
    dp    = _hist_data_points(counts, bin_edges, ser["name"], round_to_int=False)
    title = tmpl["theme"] + " (Cumulative)"
    return fig, build_standard_json(title, tmpl["x_label"], "Cumulative Frequency", x_lim, y_lim, dp)


RENDERERS = {
    "simple":     _render_simple,
    "multi":      _render_multi,
    "cumulative": _render_cumulative,
}

# ─────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────

def generate_charts(n: int) -> None:
    for i in range(1, n + 1):
        random.seed(i * 37 + 9)
        np.random.seed(i * 37 + 9)
        subtype = random.choice(SUBTYPES)
        tmpl    = random.choice(DATASET_TEMPLATES)
        theme   = random.choice(CHART_THEMES)
        fig, js = RENDERERS[subtype](tmpl, theme, i)
        save_outputs(fig, js, CHART_TYPE, i, subtype)
