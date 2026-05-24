import numpy as np

from src.core.models import MaskRecord
from src.tools.geometry import bbox_iou, compute_area, compute_iou, summarize_masks


def test_compute_area() -> None:
    mask = np.array([[0, 1], [1, 1]])
    assert compute_area(mask) == 3.0


def test_compute_iou() -> None:
    a = np.array([[1, 1], [0, 0]])
    b = np.array([[1, 0], [1, 0]])
    assert compute_iou(a, b) == 1 / 3


def test_bbox_iou() -> None:
    assert bbox_iou((0, 0, 10, 10), (5, 5, 10, 10)) > 0
    assert bbox_iou((0, 0, 2, 2), (5, 5, 2, 2)) == 0


def test_summarize_masks() -> None:
    masks = [
        MaskRecord(
            mask_id="m1",
            image_id="img",
            class_id="car",
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
            bbox=(0, 0, 1, 1),
            area=1,
            confidence=0.8,
            source="sam3",
            version=1,
        ),
        MaskRecord(
            mask_id="m2",
            image_id="img",
            class_id="car",
            polygon=[(0, 0), (2, 0), (2, 2), (0, 2)],
            bbox=(0, 0, 2, 2),
            area=4,
            confidence=0.7,
            source="sam3",
            version=1,
        ),
    ]
    summary = summarize_masks(masks)
    assert summary["count_total"] == 2
    assert summary["count_by_class"]["car"] == 2
