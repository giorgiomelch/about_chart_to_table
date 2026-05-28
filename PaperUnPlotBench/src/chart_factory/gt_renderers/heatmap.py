import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from . import setup_style


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    x_cats = list(dict.fromkeys(str(p["x_value"]) for p in dps))
    y_cats = list(dict.fromkeys(str(p["y_value"]) for p in dps))
    n_x, n_y = len(x_cats), len(y_cats)

    x_idx = {c: i for i, c in enumerate(x_cats)}
    y_idx = {c: i for i, c in enumerate(y_cats)}

    matrix = np.full((n_y, n_x), np.nan)
    for p in dps:
        try:
            val = float(p["cell_value"])
        except (ValueError, TypeError):
            continue
        xi = x_idx[str(p["x_value"])]
        yi = y_idx[str(p["y_value"])]
        matrix[yi, xi] = val

    # Choose colormap
    valid = matrix[~np.isnan(matrix)]
    has_neg = len(valid) > 0 and valid.min() < 0
    cmap_name = "coolwarm" if has_neg else "viridis"

    cell_axis = data.get("cell_axis", {})
    vmin = cell_axis.get("min")
    vmax = cell_axis.get("max")
    if vmin is None and len(valid) > 0:
        vmin = float(valid.min())
    if vmax is None and len(valid) > 0:
        vmax = float(valid.max())
    vmin = vmin if vmin is not None else 0.0
    vmax = vmax if vmax is not None else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Adaptive figure size based on matrix dimensions
    cell_px = 0.55
    w = max(6.0, n_x * cell_px + 2.5)
    h = max(4.5, n_y * cell_px + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(range(n_x))
    ax.set_xticklabels(x_cats, fontsize=8, rotation=40, ha="right")
    ax.set_yticks(range(n_y))
    ax.set_yticklabels(y_cats, fontsize=8)

    # White cell boundaries via minor-tick gridlines
    ax.set_xticks(np.arange(-0.5, n_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_y, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cells with luminance-aware text color
    if n_x * n_y <= 200:
        cmap_obj = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for yi in range(n_y):
            for xi in range(n_x):
                val = matrix[yi, xi]
                if not np.isnan(val):
                    rgba = cmap_obj(norm(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "black" if lum > 0.45 else "white"
                    ax.text(xi, yi, f"{val:.2g}", ha="center", va="center",
                            fontsize=7, color=text_color)

    setup_style(fig, ax, data, rng)
    # Heatmaps never show background grid
    ax.grid(False)
    # Restore cell boundary gridlines suppressed by setup_style
    ax.grid(which="minor", color="white", linewidth=0.8)
    plt.tight_layout()
    return fig
