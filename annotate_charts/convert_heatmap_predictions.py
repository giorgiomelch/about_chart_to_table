#!/usr/bin/env python3
"""
Convert heatmap predictions from matrix format (heatmap_raw/) to GT data_points format (heatmap/).

Input  (heatmap_raw/): {x_categories, y_categories, matrix, chart_title?, ...}
Output (heatmap/)    : {data_points, cell_axis, x_axis, y_axis, chart_title?, ...}

Skips DePlot (its predictions are already in text/table format).
Skips files that already exist in the output heatmap/ folder.
"""

import json
import math
import sys
from pathlib import Path

PREDICTIONS_DIR = Path("outputs/predictions")
SKIP_MODELS = {"DePlot"}


def _to_float(val):
    """Return float or None if val is missing, NaN, or non-numeric."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _label(val):
    """Stringify a category label; empty string for None."""
    return "" if val is None else str(val)


def convert(data: dict) -> tuple[dict | None, list[str]]:
    """
    Convert matrix-format dict to GT data_points format.

    Returns (result, issues):
      - result is None if the file should be skipped (fatal problem)
      - issues is a list of warning/error strings (may be non-empty even on success)
    """
    issues: list[str] = []

    missing = [k for k in ("x_categories", "y_categories", "matrix") if k not in data]
    if missing:
        return None, [f"MALFORMED — missing keys: {missing}"]

    x_cats = data["x_categories"]
    y_cats = data["y_categories"]
    matrix = data["matrix"]

    for name, val in (("x_categories", x_cats), ("y_categories", y_cats), ("matrix", matrix)):
        if not isinstance(val, list):
            return None, [f"MALFORMED — {name} is not a list (got {type(val).__name__})"]

    if not x_cats or not y_cats:
        return None, ["MALFORMED — x_categories or y_categories is empty"]

    if len(matrix) != len(y_cats):
        issues.append(
            f"WARNING — matrix has {len(matrix)} rows but y_categories has {len(y_cats)} entries; converting available rows"
        )

    data_points = []
    non_numeric = 0

    for ri, row_lbl in enumerate(y_cats):
        if ri >= len(matrix):
            break
        row = matrix[ri]
        if not isinstance(row, list):
            issues.append(f"WARNING — row {ri} ({row_lbl!r}) is not a list, skipped")
            continue
        if len(row) != len(x_cats):
            issues.append(
                f"WARNING — row {ri} has {len(row)} columns but x_categories has {len(x_cats)}; converting available columns"
            )
        for ci, col_lbl in enumerate(x_cats):
            if ci >= len(row):
                break
            vf = _to_float(row[ci])
            if vf is None:
                if row[ci] is not None:
                    non_numeric += 1
                continue
            data_points.append({
                "x_value": _label(col_lbl),
                "y_value": _label(row_lbl),
                "cell_value": round(vf, 4),
            })

    if non_numeric:
        issues.append(f"WARNING — {non_numeric} non-numeric cell value(s) skipped")

    if not data_points:
        return None, issues + ["EMPTY — no valid data_points after conversion (all null/NaN); skipped"]

    values = [dp["cell_value"] for dp in data_points]
    result = {
        "chart_title": data.get("chart_title"),
        "x_axis_label": data.get("x_axis_label"),
        "y_axis_label": data.get("y_axis_label"),
        "x_axis": {"min": None, "max": None, "is_log": False},
        "y_axis": {"min": None, "max": None, "is_log": False},
        "cell_axis": {"min": min(values), "max": max(values), "is_log": False},
        "data_points": data_points,
    }
    return result, issues


def main():
    if not PREDICTIONS_DIR.exists():
        print(f"ERROR: {PREDICTIONS_DIR} not found. Run from the project root.", file=sys.stderr)
        sys.exit(1)

    n_converted = 0
    n_skipped = 0
    failed: list[tuple[Path, list[str]]] = []   # not written
    warned: list[tuple[Path, list[str]]] = []   # written but with warnings

    for model_dir in sorted(PREDICTIONS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_MODELS:
            continue

        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            raw_dir = dataset_dir / "heatmap_raw"
            if not raw_dir.exists():
                continue

            out_dir = dataset_dir / "heatmap"
            out_dir.mkdir(exist_ok=True)

            for raw_file in sorted(raw_dir.glob("*.json")):
                out_file = out_dir / raw_file.name

                if out_file.exists():
                    n_skipped += 1
                    continue

                try:
                    with open(raw_file, encoding="utf-8") as f:
                        data = json.load(f)
                except json.JSONDecodeError as e:
                    failed.append((raw_file, [f"ERROR — invalid JSON: {e}"]))
                    continue
                except OSError as e:
                    failed.append((raw_file, [f"ERROR — {e}"]))
                    continue

                result, issues = convert(data)

                if result is None:
                    failed.append((raw_file, issues))
                    continue

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                n_converted += 1
                if issues:
                    warned.append((raw_file, issues))

    print()
    print("=== Conversion Summary ===")
    print(f"Converted:  {n_converted}")
    print(f"Skipped:    {n_skipped}  (already exist)")
    print(f"Failed:     {len(failed)}  (not written)")
    print(f"Warnings:   {len(warned)}  (written with issues)")

    if failed:
        print()
        print("--- Failed (not written) ---")
        for path, issues in failed:
            for issue in issues:
                print(f"  {path}: {issue}")

    if warned:
        print()
        print("--- Written with warnings ---")
        for path, issues in warned:
            for issue in issues:
                print(f"  {path}: {issue}")


if __name__ == "__main__":
    main()
