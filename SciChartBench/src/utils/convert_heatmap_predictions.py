"""
Convert heatmap predictions (matrix format) to ground truth format (data_points format).

Input  (HEATMAP_SCHEMA):  chart_title, x_axis_label, y_axis_label,
                           x_categories, y_categories, matrix
Output (ground truth):    chart_title, x_axis_label, y_axis_label,
                           x_axis, y_axis, cell_axis, data_points
"""

import json
import math
from pathlib import Path



def convert_heatmap_prediction(pred: dict) -> dict:
    x_cats = pred["x_categories"]
    y_cats = pred["y_categories"]
    matrix = pred["matrix"]

    dp = []
    values = []
    for ri, row_lbl in enumerate(y_cats):
        for ci, col_lbl in enumerate(x_cats):
            val = matrix[ri][ci]
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(fval):
                continue
            dp.append({
                "x_value": col_lbl,
                "y_value": row_lbl,
                "cell_value": round(fval, 4),
            })
            values.append(fval)

    cell_min = round(min(values), 4) if values else None
    cell_max = round(max(values), 4) if values else None

    return {
        "chart_title":  pred.get("chart_title"),
        "x_axis_label": pred.get("x_axis_label"),
        "y_axis_label": pred.get("y_axis_label"),
        "cell_axis":    {"min": cell_min, "max": cell_max, "is_log": False},
        "data_points":  dp,
    }


def convert_directory(src_dir: str | Path, dst_dir: str | Path) -> int:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.json"))
    if not files:
        return 0

    converted = 0
    for src_path in files:
        dst_path = dst_dir / src_path.name
        try:
            with src_path.open() as f:
                pred = json.load(f)
            result = convert_heatmap_prediction(pred)
            with dst_path.open("w") as f:
                json.dump(result, f, indent=2)
            converted += 1
        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
            print(f"    SKIP {src_path.name}: {e}")

    return converted


def convert_all(predictions_root: str | Path, converted_root: str | Path) -> None:
    predictions_root = Path(predictions_root)
    converted_root = Path(converted_root)

    heatmap_dirs = sorted(predictions_root.glob("*/*/heatmap"))
    if not heatmap_dirs:
        print("No heatmap directories found.")
        return

    total = 0
    for src_dir in heatmap_dirs:
        rel = src_dir.relative_to(predictions_root)
        dst_dir = converted_root / rel
        n = convert_directory(src_dir, dst_dir)
        if n:
            print(f"  {rel}: {n} files")
        total += n

    print(f"\nTotal: {total} files converted → {converted_root}")


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    convert_all(
        predictions_root=BASE / "outputs/predictions",
        converted_root=BASE / "outputs/heatmap_predictions_converted",
    )
