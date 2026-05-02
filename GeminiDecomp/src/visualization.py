"""
Funzioni di visualizzazione per la valutazione delle performance di Gemini.
Tutte le funzioni restituiscono un oggetto Figure e salvano su disco se save_path è fornito.
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

_CLASS_COLORS = {
    "chart":     "#FF6B35",
    "non_chart": "#4ECDC4",
    "all":       "#6C757D",
}
_METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1-Score"}


def _save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_metrics_vs_iou(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Tre subplot (Precision / Recall / F1) con una linea per ogni classe + "all",
    al variare della soglia IoU.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    classes = df["class"].unique()
    metrics = ["precision", "recall", "f1"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Metriche al variare della soglia IoU", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        for cls in classes:
            sub = df[df["class"] == cls].sort_values("iou_threshold")
            ls  = "--" if cls == "all" else "-"
            lw  = 1.5 if cls == "all" else 2.2
            ax.plot(
                sub["iou_threshold"], sub[metric],
                color=_CLASS_COLORS.get(cls, "gray"),
                linestyle=ls, linewidth=lw,
                marker="o", markersize=4,
                label=cls,
            )
        ax.set_title(_METRIC_LABELS[metric])
        ax.set_xlabel("Soglia IoU")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.legend(title="Classe", fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_summary_heatmap(
    df: pd.DataFrame,
    metric: str = "f1",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Heatmap [classe × soglia IoU] con valori del metric scelto (f1/precision/recall).
    """
    sns.set_theme(style="white", font_scale=1.1)
    pivot = df.pivot(index="class", columns="iou_threshold", values=metric)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".3f",
        cmap="YlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": _METRIC_LABELS.get(metric, metric)},
    )
    ax.set_title(
        f"{_METRIC_LABELS.get(metric, metric)} per classe e soglia IoU",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Soglia IoU")
    ax.set_ylabel("Classe")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_iou_histogram(
    iou_scores: Dict[str, List[float]],
    iou_threshold: float = 0.5,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Distribuzione degli IoU score sui match TP, una barra per bin IoU.

    Args:
        iou_scores: output di metrics.collect_iou_scores()
        iou_threshold: soglia usata per raccogliere i match (mostrata come linea verticale)
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    bins = np.linspace(iou_threshold, 1.0, 21)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    fig.suptitle(
        f"Distribuzione IoU score dei match TP (soglia={iou_threshold})",
        fontsize=13, fontweight="bold",
    )

    for ax, (cls, scores) in zip(axes, iou_scores.items()):
        color = _CLASS_COLORS.get(cls, "steelblue")
        if scores:
            ax.hist(scores, bins=bins, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(f"Classe: {cls}  (n={len(scores)})")
        ax.set_xlabel("IoU score")
        ax.set_ylabel("Conteggio match")
        ax.axvline(iou_threshold, color="black", linewidth=1.2, linestyle="--",
                   label=f"soglia={iou_threshold}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pr_curve(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Curva Precision-Recall parametrizzata per soglia IoU: ogni punto è una
    soglia IoU, le frecce indicano la direzione di IoU crescente.
    Una curva per ogni classe.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    classes = [c for c in df["class"].unique() if c != "all"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        "Precision-Recall al variare della soglia IoU",
        fontsize=13, fontweight="bold",
    )

    for cls in classes:
        sub = df[df["class"] == cls].sort_values("iou_threshold")
        r = sub["recall"].values
        p = sub["precision"].values
        thrs = sub["iou_threshold"].values
        color = _CLASS_COLORS.get(cls, "gray")

        ax.plot(r, p, color=color, linewidth=2, marker="o", markersize=5, label=cls)

        # Annota alcuni punti con il valore IoU
        for i, thr in enumerate(thrs):
            if thr in (0.1, 0.5, 0.9):
                ax.annotate(
                    f"IoU={thr}",
                    xy=(r[i], p[i]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=color,
                )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(title="Classe", fontsize=10)
    ax.plot([0, 1], [1, 0], color="lightgray", linewidth=0.8, linestyle=":")  # guida visiva

    fig.tight_layout()
    _save(fig, save_path)
    return fig
