"""
Loader per annotazioni in formato LabelStudio (export JSON).

Struttura attesa per il ground truth:
  [{
    "data": {"image": "<url>"},
    "annotations": [{"result": [{"value": {"x","y","width","height","rectanglelabels":[...]}, ...}]}]
  }]

Struttura attesa per le predizioni:
  [{
    "data": {"image": "<url>"},
    "predictions": [{"result": [{"value": {"x","y","width","height","rectanglelabels":[...]}, ...}]}]
  }]

Le coordinate sono in percentuale (0–100), formato LabelStudio.
"""
import json
import os
from pathlib import Path
from typing import Dict, List


def _filename_from_url(url: str) -> str:
    """Estrae il nome file dall'URL LabelStudio (gestisce ?d= e path Windows/Linux)."""
    clean = url.split("?d=")[-1]
    return os.path.basename(clean.replace("\\", "/"))


def _parse_result_items(result_items: List[dict]) -> List[dict]:
    """Converte una lista di result items LabelStudio in lista di {bbox, label}."""
    boxes = []
    for item in result_items:
        v = item.get("value", {})
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        boxes.append({
            "bbox":  [v["x"], v["y"], v["width"], v["height"]],
            "label": labels[0],
        })
    return boxes


def load_ground_truth(path: str | Path) -> Dict[str, List[dict]]:
    """
    Carica il ground truth da un file JSON esportato da LabelStudio.

    Returns:
        dict  filename → lista di {"bbox": [x%, y%, w%, h%], "label": str}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, List[dict]] = {}
    for task in data:
        fname = _filename_from_url(task["data"]["image"])
        items = []
        if task.get("annotations"):
            items = task["annotations"][0].get("result", [])
        result[fname] = _parse_result_items(items)
    return result


def load_coco_predictions(
    predictions_path: str | Path,
    gt_path: str | Path,
    score_threshold: float = 0.0,
) -> Dict[str, List[dict]]:
    """
    Carica predizioni in formato COCO (lista di {image_id, category_id, bbox, score})
    e le converte nel formato interno {filename: [{bbox: [x%, y%, w%, h%], label, score}]}.

    Il mapping image_id→filename e le dimensioni immagine vengono estratti dal
    ground truth LabelStudio (usa inner_id come image_id).

    Args:
        predictions_path: file JSON con predizioni COCO
        gt_path:          file JSON ground truth LabelStudio (per il mapping id→filename+dims)
        score_threshold:  filtra predizioni con score < soglia

    Returns:
        dict  filename → lista di {"bbox": [x%, y%, w%, h%], "label": "detection", "score": float}
    """
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    # inner_id → (filename, original_width, original_height)
    id_to_meta: dict = {}
    for task in gt_data:
        fname = _filename_from_url(task["data"]["image"])
        inner_id = task["inner_id"]
        w = h = None
        anns = task.get("annotations", [])
        if anns and anns[0].get("result"):
            r = anns[0]["result"][0]
            w = r.get("original_width")
            h = r.get("original_height")
        id_to_meta[inner_id] = (fname, w, h)

    # inizializza risultato con tutte le immagini GT (anche senza predizioni)
    result: Dict[str, List[dict]] = {meta[0]: [] for meta in id_to_meta.values()}

    with open(predictions_path, encoding="utf-8") as f:
        preds = json.load(f)

    for pred in preds:
        if pred["score"] < score_threshold:
            continue
        image_id = pred["image_id"]
        meta = id_to_meta.get(image_id)
        if meta is None:
            continue
        fname, img_w, img_h = meta
        if img_w is None or img_h is None:
            continue
        x_px, y_px, w_px, h_px = pred["bbox"]
        result[fname].append({
            "bbox":  [x_px / img_w * 100, y_px / img_h * 100,
                      w_px / img_w * 100, h_px / img_h * 100],
            "label": "detection",
            "score": pred["score"],
        })

    return result


def load_predictions(path: str | Path) -> Dict[str, List[dict]]:
    """
    Carica le predizioni da un file JSON in formato LabelStudio.

    Returns:
        dict  filename → lista di {"bbox": [x%, y%, w%, h%], "label": str}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, List[dict]] = {}
    for task in data:
        fname = _filename_from_url(task["data"]["image"])
        items = []
        if task.get("predictions"):
            items = task["predictions"][0].get("result", [])
        result[fname] = _parse_result_items(items)
    return result
