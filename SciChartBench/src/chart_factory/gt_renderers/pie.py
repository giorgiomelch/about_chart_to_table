import math
import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from . import setup_style, get_colors


def _parse(v):
    s = str(v).strip().rstrip("%")
    return abs(float(s))


def _fmt_val(v: float) -> str:
    """Format a value as-is, with M suffix for millions and K for thousands."""
    if abs(v) >= 1e6:
        return f"{v / 1e6:.3g}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.3g}K"
    return f"{v:g}"


def _is_pct_data(values: list, tol: float = 0.5) -> bool:
    """True when values already represent percentages (sum within tol of 100)."""
    return abs(sum(values) - 100.0) <= tol


def _reorder_slices(labels: list, values: list) -> tuple:
    """Interleave small slices (< 2 %) between larger ones so their labels
    never cluster together in the circular pie layout."""
    total = sum(values) or 1.0
    SMALL = 0.02
    large = sorted(
        [(l, v) for l, v in zip(labels, values) if v / total >= SMALL],
        key=lambda x: x[1], reverse=True,
    )
    small = sorted(
        [(l, v) for l, v in zip(labels, values) if v / total < SMALL],
        key=lambda x: x[1],
    )
    if not large:
        return labels, values
    if not small:
        lbl, val = zip(*large)
        return list(lbl), list(val)

    result: list = []
    i = j = 0
    while i < len(large) or j < len(small):
        if i < len(large):
            result.append(large[i]); i += 1
        if j < len(small):
            result.append(small[j]); j += 1

    lbl, val = zip(*result)
    return list(lbl), list(val)


def _annotate_inside(ax, theta: float, r: float, text: str, color: str, fs: int):
    """Draw text inside a wedge with luminance-aware ink color."""
    rgba = mcolors.to_rgba(color)
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    ink = "white" if lum < 0.5 else "#222222"
    ax.text(
        r * math.cos(theta), r * math.sin(theta),
        text, ha="center", va="center",
        fontsize=fs, color=ink, fontweight="bold",
    )


def _annotate_edge(ax, theta: float, r_conn: float, r_text: float,
                   text: str, fs: int):
    """Draw text outside a wedge with a leader line."""
    xc = r_conn * math.cos(theta)
    yc = r_conn * math.sin(theta)
    xl = r_text * math.cos(theta)
    yl = r_text * math.sin(theta)
    ax.annotate(
        text,
        xy=(xc, yc), xytext=(xl, yl),
        ha="left" if xl >= 0 else "right", va="center",
        fontsize=fs,
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.75),
    )


def _pie_on_ax(ax, labels: list, values: list, colors: list,
               use_labels: bool, as_pct: bool = False):
    """Draw one pie on ax.

    use_labels=True  → category labels + values outside with leader lines.
    use_labels=False → values only (inside or on edge); returns wedges for legend.
    """
    total = sum(values) or 1.0
    n = len(labels)
    fracs = [v / total for v in values]
    explode = [0.05 if f < 0.05 else 0.0 for f in fracs]
    THIN = 0.04   # fraction below which a slice is too thin for inside text

    if use_labels:
        wedges, _ = ax.pie(
            values, labels=None, explode=explode, colors=colors,
            autopct=None, startangle=90,
        )

        r_label = 1.35 if n <= 6 else 1.52
        lbl_fs = 10 if n <= 8 else 9
        val_fs = 9 if n <= 8 else 8

        for i, (wedge, label, frac, color, val) in enumerate(
                zip(wedges, labels, fracs, colors, values)):
            theta = math.radians((wedge.theta1 + wedge.theta2) / 2)
            val_str = f"{val:g}%" if as_pct else _fmt_val(val)
            r_conn = 1.05 + explode[i]

            if frac >= THIN:
                # Value inside + label outside via leader line
                r_in = 0.65 if frac > 0.15 else 0.52
                _annotate_inside(ax, theta, r_in, val_str, color, val_fs)
                annot_text = label
            else:
                # Slice too thin → combine label + value in the outside annotation
                annot_text = f"{label}\n{val_str}"

            # Outside label with leader line
            xl = r_label * math.cos(theta)
            yl = r_label * math.sin(theta)
            ax.annotate(
                annot_text,
                xy=(r_conn * math.cos(theta), r_conn * math.sin(theta)),
                xytext=(xl, yl),
                ha="left" if xl >= 0 else "right", va="center",
                fontsize=lbl_fs,
                arrowprops=dict(
                    arrowstyle="-", color="#888888",
                    lw=0.85, connectionstyle="arc3,rad=0.0",
                ),
            )

        return None  # no legend needed

    else:
        # Legend mode: values inside large slices, edge annotation for thin ones
        wedges, _ = ax.pie(
            values, labels=None, explode=explode, colors=colors,
            autopct=None, startangle=90,
        )

        for i, (wedge, frac, color, val) in enumerate(
                zip(wedges, fracs, colors, values)):
            if val == 0:
                continue
            theta = math.radians((wedge.theta1 + wedge.theta2) / 2)
            val_str = f"{val:g}%" if as_pct else _fmt_val(val)

            if frac >= THIN:
                r_in = 0.72
                _annotate_inside(ax, theta, r_in, val_str, color, 9)
            else:
                _annotate_edge(ax, theta,
                               r_conn=1.05 + explode[i],
                               r_text=1.22,
                               text=val_str, fs=8)

        return wedges


def _apply_title(fig, data: dict):
    title = data.get("chart_title")
    if title:
        fig.suptitle(str(title), fontsize=20, fontweight="bold")
        return True
    return False


def render(data: dict, rng: random.Random):
    dps = data["data_points"]
    series_names = list(dict.fromkeys(p["series_name"] for p in dps))
    n_series = len(series_names)

    use_labels_base = rng.random() < 0.8

    if n_series == 1:
        labels_raw = [str(p["x_value"]) for p in dps]
        values_raw = [_parse(p["y_value"]) for p in dps]

        has_zero = any(v == 0.0 for v in values_raw)
        use_labels = use_labels_base and not has_zero
        as_pct = _is_pct_data(values_raw)

        if use_labels:
            labels, values = _reorder_slices(labels_raw, values_raw)
        else:
            labels, values = labels_raw, values_raw

        n = len(labels)
        colors = get_colors(n, rng)
        max_lbl = max((len(l) for l in labels), default=0)
        w = max(8.5, 7.0 + 0.3 * max_lbl) if use_labels else 8.0
        fig, ax = plt.subplots(figsize=(w, 6.5))

        wedges = _pie_on_ax(ax, labels, values, colors, use_labels, as_pct)
        if wedges is not None:
            ax.legend(wedges, labels, loc="lower center",
                      bbox_to_anchor=(0.5, -0.08),
                      ncol=min(n, 4), fontsize=10, framealpha=0.7)

        setup_style(fig, ax, {**data, "chart_title": None}, rng)
        has_title = _apply_title(fig, data)
        rect = [0, 0, 1, 0.87] if has_title else [0, 0, 1, 1]
        plt.tight_layout(rect=rect)

    else:
        ncols = min(n_series, 3)
        nrows = math.ceil(n_series / ncols)
        fig, axes_arr = plt.subplots(
            nrows, ncols,
            figsize=(8.0 * ncols, 6.5 * nrows),
            squeeze=False,
        )
        axes_flat = [axes_arr[r][c] for r in range(nrows) for c in range(ncols)]

        all_labels = list(dict.fromkeys(str(p["x_value"]) for p in dps))
        base_colors = get_colors(len(all_labels), rng)
        label_color = dict(zip(all_labels, base_colors))

        for idx, sname in enumerate(series_names):
            ax = axes_flat[idx]
            pts = [p for p in dps if p["series_name"] == sname]
            lbls_raw = [str(p["x_value"]) for p in pts]
            vals_raw = [_parse(p["y_value"]) for p in pts]

            has_zero = any(v == 0.0 for v in vals_raw)
            use_labels = use_labels_base and not has_zero
            as_pct = _is_pct_data(vals_raw)

            if use_labels:
                lbls, vals = _reorder_slices(lbls_raw, vals_raw)
            else:
                lbls, vals = lbls_raw, vals_raw

            clrs = [label_color[l] for l in lbls]
            wedges = _pie_on_ax(ax, lbls, vals, clrs, use_labels, as_pct)
            if wedges is not None:
                ax.legend(wedges, lbls, loc="lower center",
                          bbox_to_anchor=(0.5, -0.08),
                          ncol=min(len(lbls), 4), fontsize=9, framealpha=0.7)
            ax.set_title(sname, fontsize=15, fontweight="bold")

        for idx in range(n_series, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        setup_style(fig, axes_flat[0], {**data, "chart_title": None}, rng)
        has_title = _apply_title(fig, data)
        rect = [0, 0, 1, 0.90] if has_title else [0, 0, 1, 1]
        plt.tight_layout(rect=rect)

    return fig
