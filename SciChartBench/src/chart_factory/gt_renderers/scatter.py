import random
import matplotlib.pyplot as plt
from . import setup_style, get_colors, show_legend, xtick_rotation, MARKERS


def _annotate_point(ax, x, y, label: str, color: str):
    """Place series name next to a single isolated point."""
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=9,
        color=color,
    )


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    n_series = len(series_names)
    colors = get_colors(n_series, rng)

    all_x = [p["x_value"] for p in dps]
    all_y = [p["y_value"] for p in dps]
    x_is_cat = any(isinstance(x, str) for x in all_x)
    y_is_cat = any(isinstance(y, str) for y in all_y)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Double marker size when the whole chart has fewer than 10 points
    size_mult = 2 if len(dps) < 10 else 1
    s_std   = 50 * size_mult   # standard numeric scatter
    s_strip = 40 * size_mult   # strip / dot plots

    has_legend_entry = False   # becomes True when any series gets a legend label

    if not x_is_cat and not y_is_cat:
        # Standard numeric scatter
        for i, sname in enumerate(series_names):
            pts = [(float(p["x_value"]), float(p["y_value"]))
                   for p in dps if p["series_name"] == sname]
            if not pts:
                continue
            xs, ys = zip(*pts)
            single = len(pts) == 1
            lbl = None if single else sname
            if lbl:
                has_legend_entry = True
            ax.scatter(xs, ys, color=colors[i], marker=MARKERS[i % len(MARKERS)],
                       s=s_std, label=lbl, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
            if single:
                _annotate_point(ax, xs[0], ys[0], sname, colors[i])

        x_axis = data.get("x_axis", {})
        y_axis = data.get("y_axis", {})
        if x_axis.get("is_log"):
            ax.set_xscale("log")
        if y_axis.get("is_log"):
            ax.set_yscale("log")

    elif x_is_cat and not y_is_cat:
        # Strip/dot plot: x categorical, y numeric
        x_order = list(dict.fromkeys(str(p["x_value"]) for p in dps))
        x_map = {c: i for i, c in enumerate(x_order)}
        series_gap = 0.15

        for i, sname in enumerate(series_names):
            pts = [(str(p["x_value"]), float(p["y_value"]))
                   for p in dps if p["series_name"] == sname]
            if not pts:
                continue
            offset = (i - (n_series - 1) / 2) * series_gap
            jitter = [rng.gauss(0, 0.04) for _ in pts]
            xs = [x_map[t[0]] + offset + j for t, j in zip(pts, jitter)]
            ys = [t[1] for t in pts]
            single = len(pts) == 1
            lbl = None if single else sname
            if lbl:
                has_legend_entry = True
            ax.scatter(xs, ys, color=colors[i], marker=MARKERS[i % len(MARKERS)],
                       s=s_strip, label=lbl, alpha=0.8,
                       edgecolors="white", linewidths=0.4)
            if single:
                _annotate_point(ax, xs[0], ys[0], sname, colors[i])

        max_len = max((len(c) for c in x_order), default=0)
        rot, ha = xtick_rotation(len(x_order), max_len)
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels(x_order, rotation=rot, ha=ha, fontsize=9)
        y_axis = data.get("y_axis", {})
        if y_axis.get("is_log"):
            ax.set_yscale("log")

    else:
        # y categorical (or both): horizontal strip plot
        y_order = list(dict.fromkeys(str(p["y_value"]) for p in dps))
        y_map = {c: i for i, c in enumerate(y_order)}
        series_gap = 0.15

        for i, sname in enumerate(series_names):
            pts = [(float(p["x_value"]) if not isinstance(p["x_value"], str) else 0,
                    str(p["y_value"]))
                   for p in dps if p["series_name"] == sname]
            if not pts:
                continue
            offset = (i - (n_series - 1) / 2) * series_gap
            jitter = [rng.gauss(0, 0.04) for _ in pts]
            xs = [t[0] for t in pts]
            ys = [y_map[t[1]] + offset + j for t, j in zip(pts, jitter)]
            single = len(pts) == 1
            lbl = None if single else sname
            if lbl:
                has_legend_entry = True
            ax.scatter(xs, ys, color=colors[i], marker=MARKERS[i % len(MARKERS)],
                       s=s_strip, label=lbl, alpha=0.8,
                       edgecolors="white", linewidths=0.4)
            if single:
                _annotate_point(ax, xs[0], ys[0], sname, colors[i])

        ax.set_yticks(range(len(y_order)))
        ax.set_yticklabels(y_order, fontsize=9)
        x_axis = data.get("x_axis", {})
        if x_axis.get("is_log"):
            ax.set_xscale("log")

    if has_legend_entry and show_legend(series_names):
        ax.legend(fontsize=9, framealpha=0.7)
    setup_style(fig, ax, data, rng)
    plt.tight_layout()
    return fig
