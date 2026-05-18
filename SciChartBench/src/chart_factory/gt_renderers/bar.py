import random
import numpy as np
import matplotlib.pyplot as plt
from . import setup_style, get_colors, figsize_for, show_legend, MARKERS

# Numeric sentinel strings that must NOT be treated as category labels
_NUMERIC_SENTINELS = {"INF", "+INF", "-INF", "INFINITY", "+INFINITY", "-INFINITY", "NAN"}


def _parse_val(v) -> float:
    """Convert a bar value to float, mapping 'INF' variants to ±10^9."""
    s = str(v).strip().upper()
    if s in ("INF", "+INF", "INFINITY", "+INFINITY"):
        return 1e9
    if s in ("-INF", "-INFINITY"):
        return -1e9
    return float(v)


def _is_cat(v) -> bool:
    """True only for genuine category strings (not numeric sentinel codes)."""
    if not isinstance(v, str):
        return False
    return str(v).strip().upper() not in _NUMERIC_SENTINELS


def render(data: dict, rng: random.Random):
    dps = data["data_points"]

    y_vals_raw = [p["y_value"] for p in dps]
    x_vals_raw = [p["x_value"] for p in dps]
    # Exclude sentinel strings like "INF" from categorical detection
    y_is_cat = all(_is_cat(y) for y in y_vals_raw)
    x_is_cat = all(_is_cat(x) for x in x_vals_raw)
    transposed = y_is_cat and not x_is_cat

    if transposed:
        # x_value = bar length (numeric), y_value = category label → forced horizontal
        categories = list(dict.fromkeys(str(p["y_value"]) for p in dps))
        series_names = list(dict.fromkeys(p["series_name"] for p in dps))
        vals = {}
        for p in dps:
            vals.setdefault(p["series_name"], {})[str(p["y_value"])] = _parse_val(p["x_value"])
        forced_horizontal = True
    else:
        categories = list(dict.fromkeys(str(p["x_value"]) for p in dps))
        series_names = list(dict.fromkeys(p["series_name"] for p in dps))
        vals = {}
        for p in dps:
            vals.setdefault(p["series_name"], {})[str(p["x_value"])] = _parse_val(p["y_value"])
        forced_horizontal = False

    n_series = len(series_names)
    n_cats = len(categories)
    colors = get_colors(n_series, rng)

    avg_len = sum(len(c) for c in categories) / max(n_cats, 1)
    horizontal = forced_horizontal or (avg_len > 10 and n_cats >= 4)

    w, h = figsize_for(n_cats, n_series, categories)
    if horizontal:
        h = max(h, 0.45 * n_cats + 1.5)
    fig, ax = plt.subplots(figsize=(w, h))

    x = np.arange(n_cats)
    bar_w = min(0.8 / n_series, 0.8) if n_series > 1 else 0.6

    for i, sname in enumerate(series_names):
        offset = (i - (n_series - 1) / 2) * bar_w
        y_vals = [vals.get(sname, {}).get(c, float("nan")) for c in categories]
        pos = x + offset
        label = sname if n_series > 1 else None

        if horizontal:
            ax.barh(pos, y_vals, height=bar_w, color=colors[i], label=label,
                    edgecolor="white", linewidth=0.4)
        else:
            ax.bar(pos, y_vals, width=bar_w, color=colors[i], label=label,
                   edgecolor="white", linewidth=0.4)

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(categories, fontsize=9)
    else:
        if n_cats > 50:
            rot, ha = 90, "center"
        elif n_cats > 10:
            rot, ha = 30, "right"
        elif avg_len > 7 and n_cats > 5:
            rot, ha = 40, "right"
        else:
            rot, ha = 0, "center"
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=rot, ha=ha, fontsize=9)

    if n_series > 1 and show_legend(series_names):
        ax.legend(fontsize=9, framealpha=0.7)

    setup_style(fig, ax, data, rng)

    y_axis = data.get("y_axis", {})
    if y_axis.get("is_log"):
        (ax.set_xscale if horizontal else ax.set_yscale)("log")

    plt.tight_layout()
    return fig
