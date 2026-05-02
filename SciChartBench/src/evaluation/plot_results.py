import statistics
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict

from src.config import METRICS_DIR as METRICS_OUTPUT


def salva_grafico_comparativo(dati_f1, dataset_label):
    """Vertical grouped bar chart comparing model F1 scores across chart types."""
    if not dati_f1:
        return

    METRICS_OUTPUT.mkdir(parents=True, exist_ok=True)

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    num_classes = len(chart_classes)
    num_models = len(model_names)

    bar_width = 0.9 / num_models
    fig_width = max(16, num_classes * num_models * 0.22 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 8), layout='constrained')

    x_positions = np.arange(num_classes)
    global_avg_x = num_classes + 1
    colors = plt.cm.tab10.colors

    for i, model in enumerate(model_names):
        means = []
        stdevs = []
        all_scores = []

        for cc in chart_classes:
            scores_tuples = dati_f1[cc].get(model, [])
            scores = [f1 for _, f1 in scores_tuples]
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

        lower_errors = [min(m, s) for m, s in zip(means, stdevs)]
        upper_errors = [min(100.0 - m, s) for m, s in zip(means, stdevs)]

        rects = ax.bar(x_positions + offset, means, bar_width, label=model,
                       color=colors[i % len(colors)], alpha=0.85)

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
                )

        global_mean = statistics.mean(all_scores) if all_scores else 0.0
        ax.bar(global_avg_x + offset, global_mean, bar_width,
               color=colors[i % len(colors)], alpha=0.95,
               edgecolor='white', linewidth=0.8)
        if global_mean > 3:
            ax.text(
                global_avg_x + offset,
                global_mean + 1.0,
                f'{global_mean:.1f}',
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold',
                rotation=90,
            )

    ax.axvline(x=num_classes + 0.5, color='#888888', linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_ylabel('Mean F1 Score (%)', fontsize=13)
    ax.set_title(
        f'F1 Score by Chart Type (Mean ± Std. Dev.) — {dataset_label.upper()} Dataset',
        fontsize=14,
    )

    all_x_positions = list(x_positions) + [global_avg_x]
    all_x_labels = [cc.replace('_', ' ') for cc in chart_classes] + ['Global\nAverage']
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_x_labels, rotation=40, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.legend(
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=10, title='Models', title_fontsize=11,
        frameon=True,
    )
    ax.set_ylim(bottom=0, top=120)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    output_path = METRICS_OUTPUT / f"f1_barplot_stdev_{dataset_label.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Bar chart saved: {output_path}")


def salva_grafico_facet_elementi(dati_f1, dataset_label):
    if not dati_f1:
        return

    METRICS_OUTPUT.mkdir(parents=True, exist_ok=True)

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    num_classes = len(chart_classes)
    cols = min(3, num_classes)
    rows = math.ceil(num_classes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), layout=None)

    if num_classes == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.tab10.colors

    for idx, chart_class in enumerate(chart_classes):
        ax = axes[idx]

        for i, model in enumerate(model_names):
            scores_data = dati_f1[chart_class].get(model, [])

            raggruppamento = defaultdict(list)
            for num_elem, f1 in scores_data:
                raggruppamento[num_elem].append(f1)

            if not raggruppamento:
                continue

            x_vals = sorted(raggruppamento.keys())
            y_means = [statistics.mean(raggruppamento[x]) for x in x_vals]

            ax.plot(x_vals, y_means, marker='o', linestyle='-', linewidth=2,
                    color=colors[i % len(colors)], label=model)

        ax.set_title(chart_class.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Number of data points', fontsize=11)
        ax.set_ylabel('Mean F1 Score (%)', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_ylim(bottom=0, top=105)
        ax.grid(True, linestyle='--', alpha=0.6)

    for idx in range(num_classes, len(axes)):
        fig.delaxes(axes[idx])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    title_str = f'F1 Score vs. Number of Data Points — {dataset_label.upper()} Dataset'
    fig.suptitle(title_str, fontsize=14, y=0.975)

    handles, labels = axes[num_classes - 1].get_legend_handles_labels()

    fig.legend(handles, labels,
               loc='upper left',
               bbox_to_anchor=(1.02, 1.0),
               ncol=1,
               fontsize=11,
               title='Models',
               title_fontsize=12,
               frameon=True,
               shadow=False)

    output_path = METRICS_OUTPUT / f"f1_facet_elements_{dataset_label.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Facet chart saved: {output_path}")
