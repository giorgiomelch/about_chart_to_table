import json
import statistics
import warnings

from src.config import GROUNDTRUTH_DIR as GROUNDTRUTH_ROOT
from src.config import IMAGES_DIR as IMAGES_ROOT
from src.config import PREDICTIONS_DIR as PREDICTIONS_ROOT
from src.evaluation.evaluate import estrai_basi, get_available_models, load_prediction


def _conta_elementi_pred(pred_data):
    if pred_data is None:
        return 0
    if "matrix" in pred_data and "x_categories" in pred_data:
        return sum(1 for row in pred_data["matrix"] for cell in row if cell is not None)
    return len(pred_data.get("data_points", []))


def compute_coverage_for_class(model_name, dataset_type, chart_class):
    """Restituisce lista di ratio% (pred_count/gt_count*100) per ogni file GT."""
    gt_dir = GROUNDTRUTH_ROOT / dataset_type / chart_class
    pred_dir = PREDICTIONS_ROOT / model_name / dataset_type / chart_class

    if not gt_dir.exists():
        return []

    ratios = []
    for gt_file in sorted(gt_dir.rglob("*.json")):
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except Exception:
            continue

        gt_count = len(gt_data.get("data_points", []))

        if pred_dir.exists():
            rel_path = gt_file.relative_to(gt_dir)
            pred_file = pred_dir / rel_path
            basi_gt = estrai_basi(gt_data)
            pred_data = load_prediction(pred_file, basi_gt)
        else:
            pred_data = None

        pred_count = _conta_elementi_pred(pred_data)
        ratio = (pred_count / gt_count * 100) if gt_count > 0 else 0.0
        ratios.append(ratio)

    return ratios


def stampa_risultati_coverage(dati_coverage, dataset_label):
    if not dati_coverage:
        return

    chart_classes = sorted(dati_coverage.keys())
    model_names = sorted(next(iter(dati_coverage.values())).keys())

    col_w = 10
    label_w = max(20, max(len(cc) for cc in chart_classes + ["Media Globale"]) + 2)

    header = "Classe".ljust(label_w) + "".join(m[:col_w - 1].rjust(col_w) for m in model_names)
    sep = "-" * len(header)

    print(f"\n=== Coverage (pred/GT %) — Dataset: {dataset_label.upper()} ===\n")
    print(header)
    print(sep)

    global_scores = {m: [] for m in model_names}

    for cc in chart_classes:
        row = cc.ljust(label_w)
        for model in model_names:
            ratios = dati_coverage[cc].get(model, [])
            global_scores[model].extend(ratios)
            mean = statistics.mean(ratios) if ratios else 0.0
            row += f"{mean:>{col_w}.1f}"
        print(row)

    print(sep)
    row = "Media Globale".ljust(label_w)
    for model in model_names:
        mean = statistics.mean(global_scores[model]) if global_scores[model] else 0.0
        row += f"{mean:>{col_w}.1f}"
    print(row)
    print()


def run_coverage_analysis(suppress_warnings=True):
    models = get_available_models()
    if not models:
        print("Errore: Nessun modello trovato in outputs/predictions")
        return

    print(f"Modelli rilevati: {', '.join(models)}")

    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")

        for dataset_type in ["arXiv", "PMCharts", "synthetic"]:
            img_base_dir = IMAGES_ROOT / dataset_type
            if not img_base_dir.exists():
                continue

            chart_classes = sorted([d.name for d in img_base_dir.iterdir() if d.is_dir()])
            dati_coverage = {cc: {} for cc in chart_classes}

            for chart_class in chart_classes:
                for model in models:
                    dati_coverage[chart_class][model] = compute_coverage_for_class(
                        model, dataset_type, chart_class
                    )

            stampa_risultati_coverage(dati_coverage, dataset_type)


if __name__ == "__main__":
    run_coverage_analysis()
