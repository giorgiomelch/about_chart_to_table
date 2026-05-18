import random
import numpy as np
import matplotlib.pyplot as plt
from . import setup_style, get_colors, show_legend, xtick_rotation


def _make_bxp_stat(yv: dict) -> dict:
    """Convert a GT y_value dict to an ax.bxp() stats dict."""
    def _f(v, fallback):
        # yv.get returns None both when key is absent AND when key is null
        return float(v) if v is not None else fallback

    med = _f(yv.get("median"), 0.0)
    q1 = _f(yv.get("q1"), med)
    q3 = _f(yv.get("q3"), med)
    whislo = _f(yv.get("min"), q1)
    whishi = _f(yv.get("max"), q3)
    return {"med": med, "q1": q1, "q3": q3, "whislo": whislo, "whishi": whishi}


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    categories = list(dict.fromkeys(str(p["x_value"]) for p in dps))
    n_series = len(series_names)
    n_cats = len(categories)
    colors = get_colors(n_series, rng)

    h = max(5.5, 0.4 * n_cats + 1.5)
    fig, ax = plt.subplots(figsize=(max(9.0, n_cats * 0.9), h))

    group_w = 1.0
    box_w = group_w / (n_series + 1)

    for j, sname in enumerate(series_names):
        positions = []
        stats = []
        for i, cat in enumerate(categories):
            pts = [p for p in dps if p["series_name"] == sname and str(p["x_value"]) == cat]
            if not pts:
                continue
            yv = pts[0]["y_value"]
            if not isinstance(yv, dict):
                continue
            stat = _make_bxp_stat(yv)
            if "med" not in stat:
                continue
            offset = (j - (n_series - 1) / 2) * box_w
            positions.append(i + offset)
            stats.append(stat)

        if not stats:
            continue

        bxp = ax.bxp(
            stats,
            positions=positions,
            widths=box_w * 0.85,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in bxp["boxes"]:
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.75)
        for part in ("medians", "whiskers", "caps"):
            for line in bxp[part]:
                line.set_color("black")
                line.set_linewidth(1.2)

        # Invisible scatter for legend entry
        ax.scatter([], [], color=colors[j], label=sname, s=60, alpha=0.75)

    max_len = max((len(c) for c in categories), default=0)
    rot, ha = xtick_rotation(n_cats, max_len)
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=rot, ha=ha, fontsize=9)

    if n_series > 1 and show_legend(series_names):
        ax.legend(fontsize=9, framealpha=0.7)

    setup_style(fig, ax, data, rng)

    y_axis = data.get("y_axis", {})
    if y_axis.get("is_log"):
        ax.set_yscale("log")

    plt.tight_layout()
    return fig
