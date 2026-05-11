import statistics
import warnings
from pathlib import Path
import json
from typing import Any
from src.evaluation.rms import compute_rms
from src.evaluation.plot_results import salva_grafico_comparativo

# --- CONFIGURAZIONE ---
from src.config import PREDICTIONS_DIR as PREDICTIONS_ROOT
from src.config import GROUNDTRUTH_DIR as GROUNDTRUTH_ROOT
from src.config import IMAGES_DIR as IMAGES_ROOT

# --- UTILS ---

def get_available_models():
    """Rileva dinamicamente i modelli presenti in ordine alfabetico."""
    if not PREDICTIONS_ROOT.exists():
        return []
    return sorted([d.name for d in PREDICTIONS_ROOT.iterdir() if d.is_dir()])

def sottrai_valore_base(valore, base):
    if isinstance(valore, (int, float)) and not isinstance(valore, bool):
        return valore - base
    elif isinstance(valore, dict):
        return {
            chiave: (sub_valore - base if isinstance(sub_valore, (int, float)) and not isinstance(sub_valore, bool) else sub_valore)
            for chiave, sub_valore in valore.items()
        }
    return valore

def estrai_basi(dati_json):
    """Estrae i valori base da un dizionario JSON e li restituisce in un dizionario."""
    return {
        "x_base": dati_json.get("x_base"),
        "y_base": dati_json.get("y_base"),
        "w_base": dati_json.get("w_base"),
        "z_base": dati_json.get("z_base")
    }

def normalizza_valori(dati_json, basi):
    """Applica il dizionario delle basi ai data_points forniti."""
    if "data_points" not in dati_json or not isinstance(dati_json["data_points"], list):
        return dati_json

    for punto in dati_json["data_points"]:
        if basi.get("x_base") is not None and "x_value" in punto:
            punto["x_value"] = sottrai_valore_base(punto["x_value"], basi["x_base"])
            
        if basi.get("y_base") is not None and "y_value" in punto:
            punto["y_value"] = sottrai_valore_base(punto["y_value"], basi["y_base"])
            
        if basi.get("w_base") is not None and "w_value" in punto:
            punto["w_value"] = sottrai_valore_base(punto["w_value"], basi["w_base"])
            
        if basi.get("z_base") is not None and "z_value" in punto:
            punto["z_value"] = sottrai_valore_base(punto["z_value"], basi["z_base"])

    return dati_json

def _merge_chart_list(charts: list) -> dict:
    """
    Converte una lista di chart in un unico dict.
    Se tutti i data_points hanno series_name == 'Main', usa chart_title come series_name.
    Altrimenti restituisce il primo elemento della lista.
    """
    if not charts:
        return {}
    all_main = all(
        dp.get("series_name", "Main") == "Main"
        for chart in charts
        for dp in chart.get("data_points", [])
    )
    if not all_main:
        return charts[0]
    merged_points = []
    for chart in charts:
        title = chart.get("chart_title") or "Main"
        for dp in chart.get("data_points", []):
            new_dp = dict(dp)
            new_dp["series_name"] = title
            merged_points.append(new_dp)
    return {"data_points": merged_points}


def deplot_txt_to_json(txt: str) -> dict:
    """
    Parse a DePlot markdown-table .txt output into a standard chart JSON dict.

    DePlot format
    -------------
    TITLE | <title>
    <x_axis_label> | <series1> | <series2> | ...
    <x_value>      | <y1>      | <y2>      | ...
    ...

    Returns a categorical_x dict compatible with compute_rms.
    Numeric cells are converted to float; non-numeric are kept as strings.
    """
    lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
    if not lines:
        return {"data_points": []}

    chart_title = None
    header_idx = 0

    # Optional TITLE row
    if lines[0].upper().startswith("TITLE"):
        parts = lines[0].split("|", 1)
        raw_title = parts[1].strip() if len(parts) > 1 else ""
        chart_title = raw_title if raw_title else None
        header_idx = 1

    if header_idx >= len(lines):
        return {"data_points": [], **({"chart_title": chart_title} if chart_title else {})}

    # Header row: first cell = x-axis label (discarded), rest = series names
    header_parts = [p.strip() for p in lines[header_idx].split("|")]
    series_names = header_parts[1:]  # may be empty

    data_points = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue
        x_val = parts[0]
        for i, series in enumerate(series_names):
            raw_y = parts[i + 1] if i + 1 < len(parts) else ""
            try:
                y_val: Any = float(raw_y)
            except (ValueError, TypeError):
                y_val = raw_y
            data_points.append({
                "series_name": series,
                "x_value": x_val,
                "y_value": y_val,
            })

    result: dict = {"categorical_axis": "x", "data_points": data_points}
    if chart_title is not None:
        result["chart_title"] = chart_title
    return result


def load_prediction(pred_path, basi_gt):
    """
    Carica la predizione e applica la normalizzazione dei valori base.

    Supporta:
      - .json  → formato chart JSON standard
      - .txt   → formato DePlot markdown table (convertito via deplot_txt_to_json)

    Se pred_path ha estensione .json ma il file non esiste, prova automaticamente
    il corrispondente .txt (utile quando la valutazione itera su file .json del GT).
    """
    # Fallback .json → .txt (DePlot salva in .txt)
    if not pred_path.exists() and pred_path.suffix == '.json':
        pred_path = pred_path.with_suffix('.txt')

    if not pred_path.exists():
        return None

    if pred_path.suffix == '.json':
        try:
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
            if isinstance(pred_data, list):
                pred_data = _merge_chart_list(pred_data)
            if not isinstance(pred_data, dict):
                return None
            return normalizza_valori(pred_data, basi_gt)
        except Exception:
            return None

    if pred_path.suffix == '.txt':
        try:
            txt = pred_path.read_text(encoding='utf-8')
            pred_data = deplot_txt_to_json(txt)
            return normalizza_valori(pred_data, basi_gt)
        except Exception:
            return None

    return None

# --- CORE CALCULATION ---

# DePlot is only trained on bar, line, and pie charts.
# For all other chart types we record explicit F1=0 entries so that the
# global average is computed over the full set of chart types.
_DEPLOT_SUPPORTED = frozenset({"bar", "line", "pie", "histogram"})


def compute_metrics_for_class(model_name, dataset_type, chart_class):
    """Calcola l'F1 per una specifica classe e restituisce tuple: (numero_elementi, f1_score)."""
    gt_dir = GROUNDTRUTH_ROOT / dataset_type / chart_class

    # DePlot: unsupported types → F1=0 for every GT file
    if model_name.lower() == "deplot" and chart_class not in _DEPLOT_SUPPORTED:
        if not gt_dir.exists():
            return []
        zeros = []
        for gt_file in sorted(gt_dir.rglob("*.json")):
            try:
                with open(gt_file, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                n = len(gt_data.get("data_points", []))
            except Exception:
                n = 0
            zeros.append((n, 0.0, 0.0, 0.0))
        return zeros

    pred_dir = PREDICTIONS_ROOT / model_name / dataset_type / chart_class

    if not pred_dir.exists() or not gt_dir.exists():
        return []

    f1_data = []

    for gt_file in gt_dir.rglob("*.json"):
        rel_path = gt_file.relative_to(gt_dir)
        pred_file = pred_dir / rel_path

        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        num_elementi = len(gt_data.get("data_points", []))

        basi_gt = estrai_basi(gt_data)
        gt_data_norm = normalizza_valori(gt_data, basi_gt)

        pred_data = load_prediction(pred_file, basi_gt)
        if pred_data is None:
            f1_data.append((num_elementi, 0.0, 0.0, 0.0))
            continue

        try:
            try_transpose = model_name.lower() == "deplot"
            result = compute_rms(pred_data, gt_data_norm, chart_type=chart_class, try_transpose=try_transpose)
        except Exception as e:
            print(f"[WARN] compute_rms fallito su {gt_file.name}: {e}")
            continue

        f1_data.append((num_elementi, result['f1'] * 100, result['precision'] * 100, result['recall'] * 100))

    return f1_data

def stampa_risultati_f1(dati_f1, dataset_label):
    if not dati_f1:
        return

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    col_w = 10
    label_w = max(20, max(len(cc) for cc in chart_classes + ["Media Globale"]) + 2)

    header = "Classe".ljust(label_w) + "".join(m[:col_w-1].rjust(col_w) for m in model_names)
    sep = "-" * len(header)

    print(f"\n=== F1 Score — Dataset: {dataset_label.upper()} ===\n")
    print(header)
    print(sep)

    global_scores = {m: [] for m in model_names}

    for cc in chart_classes:
        row = cc.ljust(label_w)
        for model in model_names:
            scores = [t[1] for t in dati_f1[cc].get(model, [])]
            global_scores[model].extend(scores)
            mean = statistics.mean(scores) if scores else 0.0
            row += f"{mean:>{col_w}.1f}"
        print(row)

    print(sep)
    row = "Media Globale".ljust(label_w)
    for model in model_names:
        mean = statistics.mean(global_scores[model]) if global_scores[model] else 0.0
        row += f"{mean:>{col_w}.1f}"
    print(row)
    print()

def stampa_risultati_precision(dati_f1, dataset_label):
    if not dati_f1:
        return

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    col_w = 10
    label_w = max(20, max(len(cc) for cc in chart_classes + ["Media Globale"]) + 2)

    header = "Classe".ljust(label_w) + "".join(m[:col_w-1].rjust(col_w) for m in model_names)
    sep = "-" * len(header)

    print(f"\n=== Precision — Dataset: {dataset_label.upper()} ===\n")
    print(header)
    print(sep)

    global_scores = {m: [] for m in model_names}

    for cc in chart_classes:
        row = cc.ljust(label_w)
        for model in model_names:
            scores = [t[2] for t in dati_f1[cc].get(model, [])]
            global_scores[model].extend(scores)
            mean = statistics.mean(scores) if scores else 0.0
            row += f"{mean:>{col_w}.1f}"
        print(row)

    print(sep)
    row = "Media Globale".ljust(label_w)
    for model in model_names:
        mean = statistics.mean(global_scores[model]) if global_scores[model] else 0.0
        row += f"{mean:>{col_w}.1f}"
    print(row)
    print()

def stampa_risultati_recall(dati_f1, dataset_label):
    if not dati_f1:
        return

    chart_classes = sorted(dati_f1.keys())
    model_names = sorted(next(iter(dati_f1.values())).keys())

    col_w = 10
    label_w = max(20, max(len(cc) for cc in chart_classes + ["Media Globale"]) + 2)

    header = "Classe".ljust(label_w) + "".join(m[:col_w-1].rjust(col_w) for m in model_names)
    sep = "-" * len(header)

    print(f"\n=== Recall — Dataset: {dataset_label.upper()} ===\n")
    print(header)
    print(sep)

    global_scores = {m: [] for m in model_names}

    for cc in chart_classes:
        row = cc.ljust(label_w)
        for model in model_names:
            scores = [t[3] for t in dati_f1[cc].get(model, [])]
            global_scores[model].extend(scores)
            mean = statistics.mean(scores) if scores else 0.0
            row += f"{mean:>{col_w}.1f}"
        print(row)

    print(sep)
    row = "Media Globale".ljust(label_w)
    for model in model_names:
        mean = statistics.mean(global_scores[model]) if global_scores[model] else 0.0
        row += f"{mean:>{col_w}.1f}"
    print(row)
    print()

# --- MAIN ---

def run_evaluation(suppress_warnings: bool = True):
    models = get_available_models()
    if not models:
        print("Errore: Nessun modello trovato in outputs/predictions")
        return

    print(f"Modelli rilevati per il benchmark: {', '.join(models)}")

    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")

        for dataset_type in ["arXiv","PMCharts", "synthetic"]:
            print(f"\nAnalisi dataset: {dataset_type.upper()}...")

            img_base_dir = IMAGES_ROOT / dataset_type
            if not img_base_dir.exists():
                print(f"Skip: Directory {img_base_dir} non trovata.")
                continue

            chart_classes = sorted([d.name for d in img_base_dir.iterdir() if d.is_dir()])
            results_f1 = {cc: {} for cc in chart_classes}

            for chart_class in chart_classes:
                for model in models:
                    f1_data = compute_metrics_for_class(model, dataset_type, chart_class)
                    results_f1[chart_class][model] = f1_data

            salva_grafico_comparativo(results_f1, dataset_type)
            stampa_risultati_f1(results_f1, dataset_type)
            stampa_risultati_precision(results_f1, dataset_type)
            stampa_risultati_recall(results_f1, dataset_type)

if __name__ == "__main__":
    run_evaluation()