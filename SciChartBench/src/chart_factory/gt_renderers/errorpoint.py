import random
import matplotlib.pyplot as plt
from . import setup_style, get_colors, show_legend, xtick_rotation, MARKERS


def _resolve(d: dict) -> tuple:
    """Return (center, err_lo, err_hi, has_median) from a {min, median, max} dict.

    has_median=False means the caller should draw only the range line (no marker).
    Returns center=None when no numeric value at all exists.
    """
    med   = d.get("median")
    min_v = d.get("min")
    max_v = d.get("max")

    med   = float(med)   if med   is not None else None
    min_v = float(min_v) if min_v is not None else None
    max_v = float(max_v) if max_v is not None else None

    has_median = med is not None

    if has_median:
        center = med
    elif min_v is not None and max_v is not None:
        center = (min_v + max_v) / 2
    elif min_v is not None:
        center = min_v
    elif max_v is not None:
        center = max_v
    else:
        return None, 0, 0, False

    err_lo = max(center - min_v, 0) if min_v is not None else 0
    err_hi = max(max_v - center, 0) if max_v is not None else 0
    return center, err_lo, err_hi, has_median


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    if not dps:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        setup_style(fig, ax, data, rng)
        plt.tight_layout()
        return fig

    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    n_series = len(series_names)
    colors = get_colors(n_series, rng)

    # Detect orientation: horizontal when x_value is a range dict
    horizontal = isinstance(dps[0]["x_value"], dict)

    if horizontal:
        # Forest-plot style: y categorical, x has error range
        y_cats = list(dict.fromkeys(p["y_value"] for p in dps))
        n_cats = len(y_cats)
        y_map = {c: i for i, c in enumerate(y_cats)}

        w = 9.0
        h = max(5.5, 0.45 * n_cats + 1.5)
        fig, ax = plt.subplots(figsize=(w, h))

        series_offset = 0.18

        for j, sname in enumerate(series_names):
            pts = [p for p in dps if p["series_name"] == sname]
            y_offset = (j - (n_series - 1) / 2) * series_offset

            for p in pts:
                center, err_lo, err_hi, has_median = _resolve(p["x_value"])
                if center is None:
                    continue
                yi = y_map[p["y_value"]] + y_offset
                ax.errorbar(
                    center, yi,
                    xerr=[[err_lo], [err_hi]],
                    fmt=MARKERS[j % len(MARKERS)] if has_median else "none",
                    color=colors[j],
                    capsize=4, capthick=1.2, elinewidth=1.2, markersize=6,
                    label=sname if p is pts[0] else None,
                )

        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(y_cats, fontsize=9)
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    else:
        # Vertical error bars: x is scalar, y_value is {min, median, max} or scalar
        all_x = [p["x_value"] for p in dps]
        x_is_cat = any(isinstance(x, str) for x in all_x)
        if x_is_cat:
            x_order = list(dict.fromkeys(str(p["x_value"]) for p in dps))
            x_map = {c: i for i, c in enumerate(x_order)}

        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        series_offset = 0.15

        for j, sname in enumerate(series_names):
            pts = [p for p in dps if p["series_name"] == sname]
            x_offset = (j - (n_series - 1) / 2) * series_offset

            for p in pts:
                yv = p["y_value"]
                if isinstance(yv, dict):
                    center, err_lo, err_hi, has_median = _resolve(yv)
                    if center is None:
                        continue
                else:
                    center, err_lo, err_hi, has_median = float(yv), 0, 0, True

                xi = (x_map[str(p["x_value"])] if x_is_cat else float(p["x_value"])) + x_offset
                ax.errorbar(
                    xi, center,
                    yerr=[[err_lo], [err_hi]],
                    fmt=MARKERS[j % len(MARKERS)] if has_median else "none",
                    color=colors[j],
                    capsize=4, capthick=1.2, elinewidth=1.2, markersize=6,
                    label=sname if p is pts[0] else None,
                )

        if x_is_cat:
            max_len = max(len(c) for c in x_order)
            rot, ha = xtick_rotation(len(x_order), max_len)
            ax.set_xticks(range(len(x_order)))
            ax.set_xticklabels(x_order, rotation=rot, ha=ha, fontsize=9)

        y_axis = data.get("y_axis", {})
        if y_axis.get("is_log"):
            ax.set_yscale("log")

    if n_series > 1 and show_legend(series_names):
        ax.legend(fontsize=9, framealpha=0.7)

    setup_style(fig, ax, data, rng)
    plt.tight_layout()
    return fig
