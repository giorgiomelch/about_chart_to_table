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
