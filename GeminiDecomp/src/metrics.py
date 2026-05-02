"""
Calcolo metriche di object detection con sweep della soglia IoU.

Le bbox sono in percentuale LabelStudio (0–100) — IoU è scale-invariante,
quindi non serve convertire in pixel per confrontare box sulla stessa immagine.
"""
from typing import Dict, List, Optional

import pandas as pd

from src.iou_matching import hungarian_match


def _boxes_for_class(
    data: Dict[str, List[dict]],
    target_class: Optional[str],
) -> Dict[str, List[List[float]]]:
    """Filtra le box per classe e restituisce {filename: [[x,y,w,h], ...]}."""
    out: Dict[str, List[List[float]]] = {}
    for fname, boxes in data.items():
        filtered = [
            b["bbox"] for b in boxes
            if target_class is None or b["label"] == target_class
        ]
        out[fname] = filtered
    return out


def compute_metrics_at_iou(
    gt: Dict[str, List[dict]],
    preds: Dict[str, List[dict]],
    iou_threshold: float,
    target_class: Optional[str] = None,
) -> dict:
    """
    Calcola precision, recall e F1 per una singola soglia IoU.

    Args:
        gt:            ground truth da load_ground_truth()
        preds:         predizioni da load_predictions()
        iou_threshold: soglia IoU per considerare un match valido
        target_class:  "chart", "non_chart" o None per tutte le classi

    Returns:
        {"precision": float, "recall": float, "f1": float,
         "tp": int, "fp": int, "fn": int}
    """
    gt_boxes   = _boxes_for_class(gt, target_class)
    pred_boxes = _boxes_for_class(preds, target_class)
    all_files  = set(gt_boxes) | set(pred_boxes)

    tp = fp = fn = 0
    for fname in all_files:
        g = gt_boxes.get(fname, [])
        p = pred_boxes.get(fname, [])
        t, f_p, f_n, _ = hungarian_match(p, g, iou_threshold)
        tp += t
        fp += f_p
        fn += f_n

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def compute_metrics_sweep(
    gt: Dict[str, List[dict]],
    preds: Dict[str, List[dict]],
    iou_thresholds: List[float],
    classes: List[str],
) -> pd.DataFrame:
    """
    Calcola le metriche per ogni combinazione (soglia IoU × classe).

    Returns:
        DataFrame con colonne:
        [iou_threshold, class, precision, recall, f1, tp, fp, fn]
        Ogni soglia IoU produce righe per ogni classe + una riga "all".
    """
    rows = []
    for thr in iou_thresholds:
        for cls in classes:
            m = compute_metrics_at_iou(gt, preds, thr, target_class=cls)
            rows.append({"iou_threshold": thr, "class": cls, **m})
        m_all = compute_metrics_at_iou(gt, preds, thr, target_class=None)
        rows.append({"iou_threshold": thr, "class": "all", **m_all})

    df = pd.DataFrame(rows)
    df = df[["iou_threshold", "class", "precision", "recall", "f1", "tp", "fp", "fn"]]
    return df


def compute_map(
    gt: Dict[str, List[dict]],
    preds: Dict[str, List[dict]],
    iou_thresholds: List[float],
) -> float:
    """
    Mean F1 medio su tutte le classi e soglie IoU fornite.

    Nota: senza confidence score non è possibile calcolare l'AP tradizionale
    (area sotto la curva PR). Questo indice è la media dell'F1 su tutto lo sweep,
    equivalente a un mF1@[iou_range].
    """
    df = compute_metrics_sweep(gt, preds, iou_thresholds, classes=["chart", "non_chart"])
    return float(df[df["class"] != "all"]["f1"].mean())


def get_per_image_errors(
    gt: Dict[str, List[dict]],
    preds: Dict[str, List[dict]],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """
    Restituisce la lista delle immagini con almeno un errore (FP o FN)
    alla soglia IoU specificata.

    Ogni elemento ha:
        filename, tp, fp, fn,
        gt_boxes    (lista di {bbox, label}),
        pred_boxes  (lista di {bbox, label}),
        matched_pred_indices  (set di indici pred che sono TP),
        matched_gt_indices    (set di indici gt che sono TP).
    """
    all_files = set(gt) | set(preds)
    errors = []

    for fname in sorted(all_files):
        gt_items   = gt.get(fname, [])
        pred_items = preds.get(fname, [])

        # Matching globale (tutte le classi insieme)
        g_boxes = [b["bbox"] for b in gt_items]
        p_boxes = [b["bbox"] for b in pred_items]
        tp, fp, fn, matches = hungarian_match(p_boxes, g_boxes, iou_threshold)

        if fp > 0 or fn > 0:
            matched_pred = {m[0] for m in matches}
            matched_gt   = {m[1] for m in matches}
            errors.append({
                "filename":            fname,
                "tp": tp, "fp": fp, "fn": fn,
                "gt_boxes":            gt_items,
                "pred_boxes":          pred_items,
                "matched_pred_indices": matched_pred,
                "matched_gt_indices":   matched_gt,
            })

    return errors


def collect_iou_scores(
    gt: Dict[str, List[dict]],
    preds: Dict[str, List[dict]],
    iou_threshold: float,
    target_class: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Raccoglie gli IoU score di tutti i match TP a una soglia data.

    Returns:
        {"chart": [iou, ...], "non_chart": [iou, ...]}
    """
    scores: Dict[str, List[float]] = {"chart": [], "non_chart": []}
    all_files = set(gt) | set(preds)

    for fname in all_files:
        gt_boxes   = gt.get(fname, [])
        pred_boxes = preds.get(fname, [])

        classes_to_check = [target_class] if target_class else ["chart", "non_chart"]
        for cls in classes_to_check:
            g = [b["bbox"] for b in gt_boxes   if b["label"] == cls]
            p = [b["bbox"] for b in pred_boxes if b["label"] == cls]
            _, _, _, matches = hungarian_match(p, g, iou_threshold)
            for _, _, iou_val in matches:
                scores[cls].append(iou_val)

    return scores
