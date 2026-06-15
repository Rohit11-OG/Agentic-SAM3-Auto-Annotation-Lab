from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.core.models import AnnotationBundle


def export_yolo(
    bundles: List[AnnotationBundle],
    output_path: Path,
    label_schema: Optional[Sequence[str]] = None,
    segmentation: bool = False,
    force_all: bool = False,
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    labels_dir = output_path / ("yolo_seg_labels" if segmentation else "yolo_labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    if force_all:
        targets = list(bundles)
    else:
        targets = [b for b in bundles if b.status == "ACCEPTED"]

    if label_schema:
        classes = list(label_schema)
    else:
        classes = sorted({mask.class_id for bundle in targets for mask in bundle.masks})
    class_to_id: Dict[str, int] = {name: i for i, name in enumerate(classes)}

    for bundle in targets:
        lines: List[str] = []
        width = max(bundle.image.width, 1)
        height = max(bundle.image.height, 1)
        for mask in bundle.masks:
            if mask.class_id not in class_to_id:
                continue
            cid = class_to_id[mask.class_id]

            if segmentation and mask.polygon and len(mask.polygon) >= 3:
                coords: List[str] = []
                for (px, py) in mask.polygon:
                    nx = min(1.0, max(0.0, px / width))
                    ny = min(1.0, max(0.0, py / height))
                    coords.append(f"{nx:.6f}")
                    coords.append(f"{ny:.6f}")
                lines.append(f"{cid} " + " ".join(coords))
                continue

            x, y, w, h = mask.bbox
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            x_center = min(1.0, max(0.0, (x + (w / 2.0)) / width))
            y_center = min(1.0, max(0.0, (y + (h / 2.0)) / height))
            norm_w = min(1.0, max(0.0, w / width))
            norm_h = min(1.0, max(0.0, h / height))
            lines.append(
                f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        label_file = labels_dir / f"{bundle.image.path.stem}.txt"
        label_file.write_text("\n".join(lines), encoding="utf-8")

    classes_file = output_path / "classes.txt"
    classes_file.write_text("\n".join(classes), encoding="utf-8")
    return labels_dir
