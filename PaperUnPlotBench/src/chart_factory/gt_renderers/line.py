import random
import matplotlib.pyplot as plt
from . import setup_style, get_colors, show_legend, xtick_rotation, MARKERS


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    n_series = len(series_names)
    colors = get_colors(n_series, rng)

    # Determine if x-axis is categorical
    all_x = [p["x_value"] for p in dps]
    x_is_cat = any(isinstance(x, str) for x in all_x)
    if x_is_cat:
        x_order = list(dict.fromkeys(str(p["x_value"]) for p in dps))
        x_map = {c: i for i, c in enumerate(x_order)}

    w = 10.0 if n_series > 4 else 9.0
    fig, ax = plt.subplots(figsize=(w, 5.5))

    for i, sname in enumerate(series_names):
        pts = [(p["x_value"], float(p["y_value"])) for p in dps if p["series_name"] == sname]
        if x_is_cat:
            pts.sort(key=lambda t: x_map.get(str(t[0]), 0))
            xs = [x_map[str(t[0])] for t in pts]
        else:
            pts.sort(key=lambda t: float(t[0]))
            xs = [float(t[0]) for t in pts]
        ys = [t[1] for t in pts]

        ax.plot(xs, ys, color=colors[i], marker=MARKERS[i % len(MARKERS)],
                markersize=5, linewidth=1.8, label=sname)

    if x_is_cat:
        max_len = max((len(c) for c in x_order), default=0)
        rot, ha = xtick_rotation(len(x_order), max_len)
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order, rotation=rot, ha=ha, fontsize=9)
    else:
        # Force ticks only at the x-values actually present in the data,
        # preventing matplotlib from inserting intermediate ticks.
        x_vals_unique = sorted(set(float(p["x_value"]) for p in dps))
        tick_labels = [str(int(v)) if v == int(v) else f"{v:g}" for v in x_vals_unique]
        rot, ha = xtick_rotation(len(x_vals_unique),
                                 max((len(l) for l in tick_labels), default=0))
        ax.set_xticks(x_vals_unique)
        ax.set_xticklabels(tick_labels, rotation=rot, ha=ha, fontsize=9)

    if show_legend(series_names):
        legend_outside = n_series > 4
        if legend_outside:
            ax.legend(fontsize=9, framealpha=0.7, bbox_to_anchor=(1.01, 1), loc="upper left")
        else:
            ax.legend(fontsize=9, framealpha=0.7)

    setup_style(fig, ax, data, rng)

    x_axis = data.get("x_axis", {})
    y_axis = data.get("y_axis", {})
    if x_axis.get("is_log") and not x_is_cat:
        ax.set_xscale("log")
    if y_axis.get("is_log"):
        ax.set_yscale("log")

    plt.tight_layout()
    return fig
