import random
import math
import numpy as np
import matplotlib.pyplot as plt
from . import setup_style, get_colors, show_legend


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    spokes = list(dict.fromkeys(str(p["x_value"]) for p in dps))
    n_spokes = len(spokes)
    n_series = len(series_names)

    if n_spokes < 3:
        # Fall back to bar chart for degenerate radar
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = get_colors(n_series, rng)
        for i, sname in enumerate(series_names):
            vals = [float(p["y_value"]) for p in dps if p["series_name"] == sname]
            ax.bar(range(len(vals)), vals, color=colors[i], label=sname)
        if show_legend(series_names):
            ax.legend(fontsize=9)
        setup_style(fig, ax, data, rng)
        plt.tight_layout()
        return fig

    colors = get_colors(n_series, rng)
    angles = [2 * math.pi * i / n_spokes for i in range(n_spokes)]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    y_axis = data.get("y_axis", {})
    y_min = y_axis.get("min")
    y_max = y_axis.get("max")

    for i, sname in enumerate(series_names):
        vals_map = {str(p["x_value"]): float(p["y_value"])
                    for p in dps if p["series_name"] == sname}
        vals = [vals_map.get(s, 0.0) for s in spokes]
        vals_closed = vals + [vals[0]]

        ax.plot(angles_closed, vals_closed, color=colors[i], linewidth=2.0, label=sname)
        ax.fill(angles_closed, vals_closed, color=colors[i], alpha=0.12)

    ax.set_xticks(angles)
    ax.set_xticklabels(spokes, fontsize=9)

    if y_min is not None and y_max is not None:
        ax.set_ylim(float(y_min), float(y_max))

    setup_style(fig, ax, data, rng, polar=True)

    title = data.get("chart_title")
    if title:
        ax.set_title(str(title), fontsize=13, pad=18)

    if show_legend(series_names):
        ax.legend(fontsize=9, framealpha=0.7, loc="upper right",
                  bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()
    return fig
