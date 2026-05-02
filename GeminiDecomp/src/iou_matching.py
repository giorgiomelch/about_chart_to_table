"""
IoU computation and Hungarian matching for object detection evaluation.
Bbox format: [x, y, width, height] (any consistent unit — pixel or percentage).
"""
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection-over-Union for two [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _cost_matrix(pred_boxes: List, gt_boxes: List) -> np.ndarray:
    cost = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            cost[i, j] = 1.0 - compute_iou(p, g)
    return cost


def hungarian_match(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """
    Optimal 1-to-1 matching via Hungarian algorithm.

    Returns (TP, FP, FN, matches) where matches is a list of
    (pred_idx, gt_idx, iou_score) tuples for accepted pairs.
    """
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0, []
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, []

    cost = _cost_matrix(pred_boxes, gt_boxes)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        iou_score = 1.0 - cost[r, c]
        if iou_score >= iou_threshold:
            tp += 1
            matches.append((r, c, iou_score))

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches
