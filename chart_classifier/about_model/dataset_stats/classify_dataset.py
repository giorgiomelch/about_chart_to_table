import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import argparse
import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from torch.utils.data import DataLoader

import config
from model import ChartClassifier
from dataset import get_val_transforms
from inference import InferenceDataset, run_batch_inference

EXCLUDED_CLASSES = {'image', 'map', 'molecule', 'diagram', 'table'}
IMAGE_EXTS = ('**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.bmp')


def collect_images(path: str) -> list[str]:
    file_list = []
    for ext in IMAGE_EXTS:
        file_list.extend(glob.glob(os.path.join(path, ext), recursive=True))
        file_list.extend(glob.glob(os.path.join(path, ext.upper()), recursive=True))
    return sorted(set(file_list))


def run_inference(path: str, batch_size: int = 16, model_path: str | None = None, num_workers: int = 4):
    device = config.DEVICE
    file_list = collect_images(path)
    if not file_list:
        print(f"[ERR] Nessuna immagine trovata in: {path}")
        return []

    checkpoint = model_path or os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME)
    print(f"[INFO] Caricamento modello: {checkpoint}")
    model = ChartClassifier(checkpoint_path=checkpoint).to(device)

    dataset = InferenceDataset(file_list, transform=get_val_transforms())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return run_batch_inference(model, dataloader, device, config.CLASS_NAMES)


def compute_stats(results: list, exclude: set | None = None) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for _, label, _ in results:
        if exclude and label in exclude:
            continue
        counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    rows = [
        {"class": cls, "count": cnt, "percentage": round(cnt / total * 100, 2) if total else 0.0}
        for cls, cnt in counts.items()
    ]
    df = pd.DataFrame(rows, columns=["class", "count", "percentage"])
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="simple", showindex=False,
                       floatfmt=(".0f", ".0f", ".2f")))
    except ImportError:
        print(df.to_string(index=False))
    print(f"  Totale immagini: {df['count'].sum()}")


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"[INFO] CSV salvato: {output_path}")


def _bar_panel(ax, df: pd.DataFrame, title: str) -> None:
    classes = df["class"].tolist()
    counts = df["count"].tolist()
    pcts = df["percentage"].tolist()

    cmap = cm.get_cmap("tab20", len(classes))
    colors = [cmap(i) for i in range(len(classes))]

    y = np.arange(len(classes))
    bars = ax.barh(y, counts, color=colors, edgecolor="white", linewidth=0.5)

    for bar, cnt, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt} ({pct:.1f}%)",
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.25)


def plot_stats(df_all: pd.DataFrame, df_filtered: pd.DataFrame, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    _bar_panel(ax1, df_all, "All classes")
    _bar_panel(
        ax2, df_filtered,
        "Filtered classes\n(excl. image, map, molecule, diagram, table)"
    )

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Grafico salvato: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferenza su cartella di immagini con statistiche per classe"
    )
    parser.add_argument("--path", type=str, required=True,
                        help="Cartella radice contenente le immagini")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Cartella di output per CSV e PNG (default: results_stats/output)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Percorso ai pesi .pth (opzionale)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    results = run_inference(
        args.path,
        batch_size=args.batch_size,
        model_path=args.model_path,
        num_workers=args.workers,
    )
    if not results:
        return

    df_all = compute_stats(results)
    df_filtered = compute_stats(results, exclude=EXCLUDED_CLASSES)

    print_table(df_all, "Tutte le classi")
    print_table(df_filtered, "Classi filtrate (escluse: image, map, molecule, diagram, table)")

    save_csv(df_all, os.path.join(args.output_dir, "stats_all.csv"))
    save_csv(df_filtered, os.path.join(args.output_dir, "stats_filtered.csv"))

    plot_stats(df_all, df_filtered, os.path.join(args.output_dir, "classification_stats.png"))


if __name__ == "__main__":
    main()
