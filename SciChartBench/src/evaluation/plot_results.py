import statistics
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

from src.config import METRICS_DIR as METRICS_OUTPUT


# Commercial models get hatch patterns to distinguish them visually.
# Matching is case-insensitive and substring-based; more specific names first.
_COMMERCIAL_HATCHES = [
    ('gpt4o mini', 'xx'),
    ('gpt-4o-mini', 'xx'),
    ('gpt4o', '\\\\'),
    ('gpt-4o', '\\\\'),
    ('openai', '\\\\'),
    ('gemini', '///'),
    ('claude', '***'),
]

# Models in the same family share a hue; shades are assigned in sorted order.
# Each entry: (keyword, family_id).  More specific keywords come first.
_FAMILY_KEYWORDS = [
    ('gpt4o mini', 'gpt'),
    ('gpt-4o-mini', 'gpt'),
    ('gpt4o',       'gpt'),
    ('gpt-4o',      'gpt'),
    ('openai',      'gpt'),
]

# Color shades per family (dark → light, for sorted model order within family).
_FAMILY_PALETTES: dict[str, list[str]] = {
    'gpt': ['#1565C0', '#64B5F6'],   # deep blue → sky blue
}

# tab10 indices to use for models outside any defined family.
# Skip index 0 (#1f77b4 blue) and 9 (#17becf cyan) to avoid clashing with GPT blues.
_TAB10_FREE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]


_DEPLOT_EDGE = ('#222222', 2.0)   # (edgecolor, linewidth) for DePlot bars

CANONICAL_CHART_CLASSES = [
    'bar', 'box', 'bubble', 'errorpoint', 'heatmap',
    'histogram', 'line', 'pie', 'radar', 'scatter',
]


def _edge_for_model(model_name: str) -> tuple[str, float] | tuple[None, float]:
    if 'deplot' in model_name.lower():
        return _DEPLOT_EDGE
    return (None, 0.0)


def _hatch_for_model(model_name: str) -> str | None:
    lower = model_name.lower()
    for keyword, hatch in _COMMERCIAL_HATCHES:
        if keyword in lower:
            return hatch
    return None


def _build_color_map(model_names: list[str]) -> dict[str, str]:
    """Assign a color to every model. Models in the same family share a hue."""
    tab10 = plt.cm.tab10.colors

    # Group family members (sorted for deterministic shade assignment).
    family_members: dict[str, list[str]] = defaultdict(list)
    for m in sorted(model_names):
        lower = m.lower()
        for keyword, fam in _FAMILY_KEYWORDS:
            if keyword in lower:
                family_members[fam].append(m)
                break

    # Assign family shades.
    color_map: dict[str, str] = {}
    for fam, members in family_members.items():
        palette = _FAMILY_PALETTES.get(fam, [])
        for k, m in enumerate(members):
            color_map[m] = palette[k % len(palette)] if palette else tab10[k % len(tab10)]

    # Assign tab10 to the rest, cycling through the free indices.
    free = [tab10[i] for i in _TAB10_FREE_INDICES]
    free_idx = 0
    for m in sorted(model_names):
        if m not in color_map:
            color_map[m] = free[free_idx % len(free)]
            free_idx += 1

    return color_map


def salva_grafico_comparativo(dati_f1, dataset_label):
    """Vertical grouped bar chart comparing model F1 scores across chart types."""
    if not dati_f1:
        return

    METRICS_OUTPUT.mkdir(parents=True, exist_ok=True)

    chart_classes = CANONICAL_CHART_CLASSES
    model_names = sorted(next(iter(dati_f1.values())).keys())

    num_classes = len(chart_classes)
    num_models = len(model_names)

    bar_width = 0.95 / num_models
    fig_width = max(16, num_classes * num_models * 0.22 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 5), layout='constrained')

    x_positions = np.arange(num_classes)
    global_avg_x = num_classes + 1
    color_map = _build_color_map(model_names)

    for i, model in enumerate(model_names):
        means = []
        stdevs = []
        all_scores = []

        for cc in chart_classes:
            scores_tuples = dati_f1.get(cc, {}).get(model, [])
            scores = [t[1] for t in scores_tuples]
            all_scores.extend(scores)

            if len(scores) > 1:
                means.append(statistics.mean(scores))
                stdevs.append(statistics.stdev(scores))
            elif len(scores) == 1:
                means.append(scores[0])
                stdevs.append(0.0)
            else:
                means.append(0.0)
                stdevs.append(0.0)

        offset = (i - num_models / 2 + 0.5) * bar_width
        hatch = _hatch_for_model(model)
        edgecolor, linewidth = _edge_for_model(model)

        lower_errors = [min(m, s) for m, s in zip(means, stdevs)]
        upper_errors = [min(100.0 - m, s) for m, s in zip(means, stdevs)]

        rects = ax.bar(x_positions + offset, means, bar_width, label=model,
                       color=color_map[model], alpha=0.85,
                       hatch=hatch, edgecolor=edgecolor, linewidth=linewidth)

        ax.errorbar(x_positions + offset, means,
                    yerr=[lower_errors, upper_errors],
                    fmt='none', capsize=3, ecolor='black', elinewidth=0.9)

        for rect in rects:
            height = rect.get_height()
            if height > 3:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height + 1.0,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    rotation=90,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.75),
                )

        global_mean = statistics.mean(all_scores) if all_scores else 0.0
        ax.bar(global_avg_x + offset, global_mean, bar_width,
               color=color_map[model], alpha=0.85,
               hatch=hatch, edgecolor=edgecolor, linewidth=linewidth)
        if global_mean > 3:
            ax.text(
                global_avg_x + offset,
                global_mean + 1.5,
                f'{global_mean:.1f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold',
                rotation=90,
            )

    ax.axvline(x=num_classes + 0.5, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_ylabel('Mean F1 Score (%)', fontsize=15)
    ax.set_title(
        f'F1 Score by Chart Type (Mean ± Std. Dev.) — {dataset_label.upper()} Dataset',
        fontsize=20,
    )

    all_x_positions = list(x_positions) + [global_avg_x]
    all_x_labels = [cc.replace('_', ' ') for cc in chart_classes] + ['Global\nAverage']
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_x_labels, rotation=40, ha='right', fontsize=15)
    ax.tick_params(axis='y', labelsize=15)

    ax.legend(
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=10, title='Models', title_fontsize=11,
        frameon=True,
    )
    ax.set_ylim(bottom=0, top=105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    output_path = METRICS_OUTPUT / f"f1_barplot_stdev_{dataset_label.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Bar chart saved: {output_path}")