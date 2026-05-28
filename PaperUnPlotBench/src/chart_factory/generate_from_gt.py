#!/usr/bin/env python3
"""Generate chart images from ground-truth JSON files.

Usage:
    python src/chart_factory/generate_from_gt.py
    python src/chart_factory/generate_from_gt.py --datasets PMCharts --types bar,line
    python src/chart_factory/generate_from_gt.py --datasets arXiv --types all
"""

import argparse
import json
import pathlib
import random
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.legend as mlegend  # noqa: E402

from src.chart_factory.gt_renderers import RENDERERS  # noqa: E402

GT_BASE = pathlib.Path("data/groundtruth")
IMG_BASE = pathlib.Path("data/images_")

ALL_DATASETS = ["PMCharts", "arXiv"]
ALL_TYPES = list(RENDERERS.keys())


def generate(datasets: list, chart_types: list) -> None:
    total = 0
    errors = 0

    for dataset in datasets:
        for chart_type in chart_types:
            class_dir = GT_BASE / dataset / chart_type
            if not class_dir.exists():
                continue

            out_dir = IMG_BASE / f"{dataset}_synthetic" / chart_type
            out_dir.mkdir(parents=True, exist_ok=True)

            json_files = sorted(class_dir.glob("*.json"))
            print(f"  {dataset}/{chart_type}: {len(json_files)} charts ...", flush=True)

            for json_file in json_files:
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)

                    rng = random.Random(hash(json_file.name))
                    fig = RENDERERS[chart_type](data, rng)

                    out_path = out_dir / json_file.with_suffix(".png").name
                    # Include any legends placed outside the axes in the tight bbox
                    extra = fig.findobj(mlegend.Legend)
                    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                                bbox_extra_artists=extra if extra else None)
                    plt.close(fig)
                    total += 1

                except Exception as e:
                    errors += 1
                    print(f"    ERROR {json_file.name}: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        plt.close("all")
                    except Exception:
                        pass

    print(f"\nDone: {total} images generated, {errors} errors.")
    if errors:
        print(f"  Output directories: {IMG_BASE}/{{dataset}}_synthetic/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render GT JSON annotations as chart images."
    )
    parser.add_argument(
        "--datasets", default="PMCharts,arXiv",
        help="Comma-separated dataset names or 'all' (default: PMCharts,arXiv)",
    )
    parser.add_argument(
        "--types", default="all",
        help=f"Comma-separated chart types or 'all'. Available: {','.join(ALL_TYPES)}",
    )
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == "all" else args.datasets.split(",")
    chart_types = ALL_TYPES if args.types == "all" else args.types.split(",")

    for d in datasets:
        if d not in ALL_DATASETS:
            print(f"Unknown dataset '{d}'. Available: {ALL_DATASETS}", file=sys.stderr)
            sys.exit(1)
    for t in chart_types:
        if t not in ALL_TYPES:
            print(f"Unknown chart type '{t}'. Available: {ALL_TYPES}", file=sys.stderr)
            sys.exit(1)

    print(f"Datasets : {datasets}")
    print(f"Types    : {chart_types}")
    print(f"Output   : {IMG_BASE}/{{dataset}}_synthetic/\n")

    generate(datasets, chart_types)


if __name__ == "__main__":
    main()
