import math
import random
import numpy as np
import matplotlib.pyplot as plt
from . import setup_style, get_colors, show_legend, xtick_rotation


def _bar_params_linear(xs: list):
    """Left edges and widths so bars fill their bins edge-to-edge (no gaps, no overlap)."""
    n = len(xs)
    if n == 1:
        return [xs[0] - 0.5], [1.0]
    lefts = []
    for i in range(n):
        if i == 0:
            lefts.append(xs[0] - (xs[1] - xs[0]) / 2)
        else:
            lefts.append((xs[i - 1] + xs[i]) / 2)
    last_right = xs[-1] + (xs[-1] - xs[-2]) / 2
    rights = lefts[1:] + [last_right]
    widths = [r - l for l, r in zip(lefts, rights)]
    return lefts, widths


def _bar_params_log(xs: list):
    """Left edges and widths computed in log-space so bars look visually equal-width."""
    n = len(xs)
    if n == 1:
        return [xs[0] / 1.5], [xs[0]]
    lxs = [math.log10(x) for x in xs]
    ledges = []
    for i in range(n):
        if i == 0:
            ledges.append(lxs[0] - (lxs[1] - lxs[0]) / 2)
        else:
            ledges.append((lxs[i - 1] + lxs[i]) / 2)
    last_lright = lxs[-1] + (lxs[-1] - lxs[-2]) / 2
    lefts = [10 ** l for l in ledges]
    rights = [10 ** l for l in ledges[1:] + [last_lright]]
    widths = [r - l for l, r in zip(lefts, rights)]
    return lefts, widths


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    x_axis_meta = data.get("x_axis", {})
    y_axis_meta = data.get("y_axis", {})

    # Axis type from metadata: null min AND max → categorical axis.
    # Also treat as categorical if any actual values are strings (e.g. bin ranges "0-200").
    x_is_cat = (
        (x_axis_meta.get("min") is None and x_axis_meta.get("max") is None)
        or any(isinstance(p["x_value"], str) for p in dps)
    )
    y_is_cat = (
        (y_axis_meta.get("min") is None and y_axis_meta.get("max") is None)
        or any(isinstance(p["y_value"], str) for p in dps)
    )

    # Transposed: y labels categorical, x numeric → horizontal bars
    transposed = y_is_cat and not x_is_cat

    x_is_log = x_axis_meta.get("is_log", False) and not x_is_cat and not transposed
    y_is_log = y_axis_meta.get("is_log", False)

    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    n_series = len(series_names)
    colors = get_colors(n_series, rng)
    multi = n_series > 1

    # Draw largest-total series first so smaller ones stay visible on top when overlapping
    def _series_total(sname):
        key = "y_value" if not transposed else "x_value"
        return sum(float(p[key]) for p in dps if p["series_name"] == sname)

    draw_order = sorted(range(n_series),
                        key=lambda i: _series_total(series_names[i]),
                        reverse=True)

    if transposed:
        categories = list(dict.fromkeys(str(p["y_value"]) for p in dps))
        n_cats = len(categories)
        vals = {}
        for p in dps:
            vals.setdefault(p["series_name"], {})[str(p["y_value"])] = float(p["x_value"])

        h = max(5.5, 0.4 * n_cats + 1.5)
        fig, ax = plt.subplots(figsize=(9.0, h))
        y = np.arange(n_cats)

        for i in draw_order:
            sname = series_names[i]
            x_vals = [vals.get(sname, {}).get(c, 0.0) for c in categories]
            ax.barh(y, x_vals, height=0.85, color=colors[i],
                    label=sname if multi else None,
                    alpha=0.65 if multi else 0.92,
                    edgecolor="white", linewidth=0.7)

        ax.set_yticks(y)
        ax.set_yticklabels(categories, fontsize=9)

    elif x_is_cat:
        categories = list(dict.fromkeys(str(p["x_value"]) for p in dps))
        n_cats = len(categories)
        vals = {}
        for p in dps:
            vals.setdefault(p["series_name"], {})[str(p["x_value"])] = float(p["y_value"])

        w = max(9.0, n_cats * 0.45)
        fig, ax = plt.subplots(figsize=(w, 5.5))
        x = np.arange(n_cats)

        for i in draw_order:
            sname = series_names[i]
            y_vals = [vals.get(sname, {}).get(c, 0.0) for c in categories]
            ax.bar(x, y_vals, width=0.85, color=colors[i],
                   label=sname if multi else None,
                   alpha=0.65 if multi else 0.92,
                   edgecolor="white", linewidth=0.7, align="center")

        max_len = max(len(str(c)) for c in categories)
        rot, ha = xtick_rotation(n_cats, max_len)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9, rotation=rot, ha=ha)

    else:
        # Numeric x axis: compute per-bar widths from bin midpoints
        all_xs = sorted(set(float(p["x_value"]) for p in dps))
        vals = {}
        for p in dps:
            vals.setdefault(p["series_name"], {})[float(p["x_value"])] = float(p["y_value"])

        fig, ax = plt.subplots(figsize=(9.0, 5.5))

        for i in draw_order:
            sname = series_names[i]
            xs = sorted(vals.get(sname, {}).keys())
            ys = [vals[sname][x] for x in xs]
            lefts, widths = (_bar_params_log(xs) if x_is_log
                             else _bar_params_linear(xs))
            ax.bar(lefts, ys, width=widths, color=colors[i],
                   label=sname if multi else None,
                   alpha=0.65 if multi else 0.92,
                   edgecolor="white", linewidth=0.7, align="edge")

        # Numeric axis: let matplotlib auto-pick ticks (no forced tick per bin)

    if multi and show_legend(series_names):
        ax.legend(fontsize=9, framealpha=0.7)

    setup_style(fig, ax, data, rng)

    if y_is_log:
        (ax.set_xscale if transposed else ax.set_yscale)("log")
    if x_is_log:
        ax.set_xscale("log")

    plt.tight_layout()
    return fig
