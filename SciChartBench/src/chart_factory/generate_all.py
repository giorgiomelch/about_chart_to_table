"""
generate_all.py — Chart Factory CLI entry point

Usage:
    python src/chart_factory/generate_all.py --n 50 --types all
    python src/chart_factory/generate_all.py --n 20 --types bar,line,scatter
    python src/chart_factory/generate_all.py --n 10 --types bubble,box,errorpoint,pie,radar,histogram
"""

import argparse
import sys
import time

# Ensure package imports work when run as a script from the project root.
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.chart_factory import (
    bar_chart,
    bubble_chart,
    box_chart,
    errorpoint_chart,
    pie_chart,
    scatter_chart,
    radar_chart,
    line_chart,
    histogram_chart,
    heatmap_chart,
)

GENERATORS = {
    "bar":        bar_chart.generate_charts,
    "bubble":     bubble_chart.generate_charts,
    "box":        box_chart.generate_charts,
    "errorpoint": errorpoint_chart.generate_charts,
    "pie":        pie_chart.generate_charts,
    "scatter":    scatter_chart.generate_charts,
    "radar":      radar_chart.generate_charts,
    "line":       line_chart.generate_charts,
    "histogram":  histogram_chart.generate_charts,
    "heatmap":    heatmap_chart.generate_charts,
}

ALL_TYPES = list(GENERATORS.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chart images (PNG) with ground-truth JSON files."
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of charts to generate per type (default: 50)"
    )
    parser.add_argument(
        "--types", type=str, default="all",
        help=(
            "Comma-separated list of chart types to generate, or 'all'. "
            f"Available: {', '.join(ALL_TYPES)}"
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.types.strip().lower() == "all":
        selected = ALL_TYPES
    else:
        selected = [t.strip().lower() for t in args.types.split(",")]
        invalid  = [t for t in selected if t not in GENERATORS]
        if invalid:
            print(f"[ERROR] Unknown chart type(s): {', '.join(invalid)}")
            print(f"        Valid types: {', '.join(ALL_TYPES)}")
            sys.exit(1)

    print(f"Chart Factory — generating {args.n} chart(s) × {len(selected)} type(s) "
          f"= {args.n * len(selected)} total")
    print()

    total_start = time.time()
    for chart_type in selected:
        t0 = time.time()
        print(f"  [{chart_type:>12s}]  generating {args.n} chart(s)...", end=" ", flush=True)
        GENERATORS[chart_type](args.n)
        elapsed = time.time() - t0
        print(f"done  ({elapsed:.1f}s)")

    total = time.time() - total_start
    print()
    print(f"All done in {total:.1f}s.")
    print(f"  Images  → data/images/synthetic2/{{type}}/")
    print(f"  JSONs   → data/groundtruth/synthetic2/{{type}}/")


if __name__ == "__main__":
    main()
