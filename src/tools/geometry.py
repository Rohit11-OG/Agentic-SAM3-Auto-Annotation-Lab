from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np

from src.core.models import MaskRecord


def compute_area(mask: np.ndarray) -> float:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    return float(np.count_nonzero(mask))


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape for IoU.")

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def summarize_masks(masks: Iterable[MaskRecord]) -> Dict[str, Any]:
    masks_list = list(masks)
    by_class = Counter(mask.class_id for mask in masks_list)
    confidences = [mask.confidence for mask in masks_list]
    areas = [mask.area for mask in masks_list]
    return {
        "count_total": len(masks_list),
        "count_by_class": dict(by_class),
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "avg_area": float(np.mean(areas)) if areas else 0.0,
    }


def pairwise_bbox_iou(masks: List[MaskRecord]) -> List[Dict[str, Any]]:
    overlaps: List[Dict[str, Any]] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            iou = bbox_iou(masks[i].bbox, masks[j].bbox)
            overlaps.append(
                {
                    "mask_a": masks[i].mask_id,
                    "mask_b": masks[j].mask_id,
                    "class_a": masks[i].class_id,
                    "class_b": masks[j].class_id,
                    "iou": iou,
                }
            )
    return overlaps
