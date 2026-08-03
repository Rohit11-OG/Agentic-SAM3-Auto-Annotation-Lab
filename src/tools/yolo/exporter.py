from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Sequence, Set

from src.core.models import AnnotationBundle

LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def _prune_stale_exports(
    export_dir: Path,
    label_glob: str,
    bundles: List[AnnotationBundle],
    keep: Optional[Set[str]] = None,
) -> None:
    """Drop label/image pairs this exporter wrote for images no longer in the run.

    The exporters copy every image next to its label, so a stem with a copied
    image beside it is one we own and may prune. A label file *without* that
    copy is somebody else's — a hand-made label, or another tool's output — and
    must be left alone, otherwise pointing output at an existing labels folder
    would quietly destroy its contents.
    """
    keep = keep or set()
    current = {b.image.path.stem for b in bundles}
    for label_file in export_dir.glob(label_glob):
        if label_file.name in keep or label_file.stem in current:
            continue
        copies = [export_dir / f"{label_file.stem}{ext}" for ext in IMAGE_EXTS]
        copies = [c for c in copies if c.exists()]
        if not copies:
            continue  # no image copy beside it -> not ours, don't touch
        label_file.unlink()
        for c in copies:
            c.unlink()


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
    dropped: Set[str] = set()

    for bundle in targets:
        stem = bundle.image.path.stem
        # If the source image was deleted from dataset, delete its label/copied image from output
        if not bundle.image.path.exists():
            label_file = labels_dir / f"{stem}.txt"
            if label_file.exists():
                label_file.unlink()
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                copied_img = labels_dir / f"{stem}{ext}"
                if copied_img.exists():
                    copied_img.unlink()
            continue

        # Copy image file to labels directory
        dest_img_path = labels_dir / bundle.image.path.name
        try:
            if not dest_img_path.exists() or dest_img_path.stat().st_size != bundle.image.path.stat().st_size:
                shutil.copy2(bundle.image.path, dest_img_path)
        except Exception:
            pass

        lines: List[str] = []
        width = max(bundle.image.width, 1)
        height = max(bundle.image.height, 1)
        for mask in bundle.masks:
            if mask.class_id not in class_to_id:
                dropped.add(mask.class_id)
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

        label_file = labels_dir / f"{stem}.txt"
        label_file.write_text("\n".join(lines), encoding="utf-8")

    if dropped:
        LOGGER.warning(
            "Dropped masks for %d class(es) missing from label_schema: %s. "
            "Add them to label_schema to export these annotations.",
            len(dropped), ", ".join(sorted(dropped)),
        )

    _prune_stale_exports(labels_dir, "*.txt", bundles, keep={"classes.txt"})

    classes_file = output_path / "classes.txt"
    classes_file.write_text("\n".join(classes), encoding="utf-8")
    return labels_dir
