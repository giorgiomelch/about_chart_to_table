"""
Dump ground-truth JSON files as parsed row tables for visual inspection.

For every GT JSON file in GROUNDTRUTH_DIR/<dataset>/<chart_class>/ the script
calls get_parser(chart_class).show_table() and writes the result to:

    outputs/parsed_gt/<dataset>/<chart_class>/<filename>.txt

Warnings emitted during parsing are captured and appended at the bottom of
each file under a "WARNINGS" section.

Usage (from project root):
    python src/evaluation/dump_gt_rows.py
    python src/evaluation/dump_gt_rows.py --out /custom/output/path
"""

from __future__ import annotations

import argparse
import json
import warnings
import sys
from pathlib import Path

# Ensure project root is in sys.path so package imports work.
_proj_root = Path(__file__).resolve().parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from src.config import GROUNDTRUTH_DIR
from src.evaluation.chart_types import get_parser


def _dump_file(gt_file: Path, chart_class: str, out_file: Path) -> int:
    """
    Parse one GT file, write the table to out_file.
    Returns the number of warnings captured.
    """
    with open(gt_file, encoding="utf-8") as f:
        data = json.load(f)

    captured: list[str] = []

    def _warn_handler(message, category, filename, lineno, file=None, line=None):
        captured.append(str(message))

    orig = warnings.showwarning
    warnings.showwarning = _warn_handler
    try:
        parser = get_parser(chart_class)
        table = parser.show_table(data)
    finally:
        warnings.showwarning = orig

    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"Source : {gt_file}",
        f"Class  : {chart_class}",
        "",
        table,
    ]

    if captured:
        lines += ["", "─" * 60, "WARNINGS", "─" * 60]
        lines += captured

    out_file.write_text("\n".join(lines), encoding="utf-8")
    return len(captured)


def main():
    parser = argparse.ArgumentParser(description="Dump GT rows to .txt files.")
    parser.add_argument(
        "--out",
        type=Path,
        default=_proj_root / "outputs" / "parsed_gt",
        help="Root output directory (default: outputs/parsed_gt/)",
    )
    args = parser.parse_args()

    out_root: Path = args.out

    total_files = 0
    total_warnings = 0
    skipped = 0

    for dataset_dir in sorted(GROUNDTRUTH_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for class_dir in sorted(dataset_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            chart_class = class_dir.name

            try:
                get_parser(chart_class)
            except KeyError:
                print(f"  [SKIP] unknown chart type: {chart_class}")
                skipped += 1
                continue

            json_files = sorted(class_dir.rglob("*.json"))
            n_class = 0
            n_warn = 0

            for gt_file in json_files:
                rel = gt_file.relative_to(class_dir)
                out_file = out_root / dataset / chart_class / rel.with_suffix(".txt")
                n_warn += _dump_file(gt_file, chart_class, out_file)
                n_class += 1

            total_files += n_class
            total_warnings += n_warn
            status = f"  {n_warn} warnings" if n_warn else "  ok"
            print(f"{dataset}/{chart_class:12s}  {n_class:4d} files{status}")

    print()
    print(f"Written : {total_files} files  →  {out_root}")
    if total_warnings:
        print(f"Warnings: {total_warnings} (see WARNINGS section in affected files)")
    if skipped:
        print(f"Skipped : {skipped} unknown chart classes")


if __name__ == "__main__":
    main()
