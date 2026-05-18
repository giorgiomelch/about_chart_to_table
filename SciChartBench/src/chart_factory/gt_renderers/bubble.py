import random
import math
import matplotlib.pyplot as plt
from . import setup_style


def render(data: dict, rng: random.Random):
    dps = data["data_points"]

    raw_x = [p["x_value"] for p in dps]
    raw_y = [p["y_value"] for p in dps]
    x_is_cat = any(isinstance(x, str) for x in raw_x)
    y_is_cat = any(isinstance(y, str) for y in raw_y)

    if x_is_cat and not y_is_cat:
        # Axes swapped: x_value=category, y_value=numeric → rotate
        y_cats = list(dict.fromkeys(str(p["x_value"]) for p in dps))
        y_map = {c: i for i, c in enumerate(y_cats)}
        xs = [float(p["y_value"]) for p in dps]
        ys = [y_map[str(p["x_value"])] for p in dps]
    elif y_is_cat:
        y_cats = list(dict.fromkeys(str(p["y_value"]) for p in dps))
        y_map = {c: i for i, c in enumerate(y_cats)}
        xs = [float(p["x_value"]) for p in dps]
        ys = [y_map[str(p["y_value"])] for p in dps]
    else:
        xs = [float(p["x_value"]) for p in dps]
        ys = [float(p["y_value"]) for p in dps]
        y_cats = None

    # Bubble size from z_value (area in pt²: 30–730)
    z_vals = [float(p.get("z_value") or 1) for p in dps]
    z_min, z_max = min(z_vals), max(z_vals)
    z_range = z_max - z_min if z_max > z_min else 1.0
    sizes = [30 + 700 * (z - z_min) / z_range for z in z_vals]

    # Bubble color from w_value
    w_vals = [float(p.get("w_value") or 0) for p in dps]
    w_axis = data.get("w_axis", {})
    vmin = w_axis.get("min", min(w_vals))
    vmax = w_axis.get("max", max(w_vals))

    n_cats = len(y_cats) if y_cats is not None else 0
    h = max(5.5, 0.45 * n_cats + 1.5) if y_cats is not None else 5.5
    fig, ax = plt.subplots(figsize=(9.0, h))

    sc = ax.scatter(xs, ys, s=sizes, c=w_vals, cmap="plasma",
                    vmin=vmin, vmax=vmax, alpha=0.85,
                    edgecolors="white", linewidths=0.5, zorder=3)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, shrink=0.7)
    cbar.ax.tick_params(labelsize=8)

    if y_cats is not None:
        ax.set_yticks(range(len(y_cats)))
        ax.set_yticklabels(y_cats, fontsize=9)

    # --- Size legend ---
    # Cap display area to 150 pt² so circles in legend don't overlap.
    # Compute each entry's display size proportionally, then scale so max → 150 pt².
    MAX_LEGEND_S = 150.0
    legend_z_vals = sorted(set(z_vals))
    raw_legend_s = [30 + 700 * (z - z_min) / z_range for z in legend_z_vals]
    scale = MAX_LEGEND_S / max(raw_legend_s)
    legend_s = [max(12.0, s * scale) for s in raw_legend_s]

    # labelspacing: center-to-center distance = fontsize*(1+labelspacing)
    # must be ≥ diameter of largest circle = sqrt(MAX_LEGEND_S)
    fontsize_leg = 8
    min_center_gap = math.sqrt(MAX_LEGEND_S)  # ≈ 12.2 pt
    label_spacing = max(1.2, (min_center_gap / fontsize_leg) - 1 + 0.5)

    legend_handles = [
        plt.scatter([], [], s=s, color="gray", alpha=0.75,
                    edgecolors="white", linewidths=0.5)
        for s in legend_s
    ]
    z_title = str(data.get("z_axis_label") or "z")
    size_leg = ax.legend(
        legend_handles,
        [f"{z:.3g}" for z in legend_z_vals],
        title=z_title,
        fontsize=fontsize_leg,
        title_fontsize=fontsize_leg,
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),   # right of colorbar
        framealpha=0.85,
        labelspacing=label_spacing,
        borderpad=0.9,
        handletextpad=1.2,
    )
    setup_style(fig, ax, data, rng)
    # Always show grid for bubble charts, regardless of rng
    ax.grid(True, color="#cccccc", alpha=0.45, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig
