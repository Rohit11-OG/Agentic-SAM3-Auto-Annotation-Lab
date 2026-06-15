from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import List

from src.core.models import AnnotationBundle


def export_labelme(
    bundles: List[AnnotationBundle],
    output_path: Path,
    force_all: bool = False,
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    labelme_dir = output_path / "labelme_json"
    labelme_dir.mkdir(parents=True, exist_ok=True)

    if force_all:
        targets = list(bundles)
    else:
        targets = [b for b in bundles if b.status == "ACCEPTED"]

    for bundle in targets:
        stem = bundle.image.path.stem
        # If the source image has been deleted from dataset, delete its label/copied image from output
        if not bundle.image.path.exists():
            json_file = labelme_dir / f"{stem}.json"
            if json_file.exists():
                json_file.unlink()
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                copied_img = labelme_dir / f"{stem}{ext}"
                if copied_img.exists():
                    copied_img.unlink()
            continue

        # Copy image file to labelme directory next to JSON
        dest_img_path = labelme_dir / bundle.image.path.name
        try:
            if not dest_img_path.exists() or dest_img_path.stat().st_size != bundle.image.path.stat().st_size:
                shutil.copy2(bundle.image.path, dest_img_path)
        except Exception:
            pass

        shapes = []
        for mask in bundle.masks:
            # Export polygon if available, else bounding box
            if mask.polygon and len(mask.polygon) >= 3:
                points = [[float(px), float(py)] for (px, py) in mask.polygon]
                shape_type = "polygon"
            else:
                x, y, w, h = mask.bbox
                points = [
                    [float(x), float(y)],
                    [float(x + w), float(y + h)]
                ]
                shape_type = "rectangle"

            shapes.append({
                "label": mask.class_id,
                "points": points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
            })

        width = bundle.image.width
        height = bundle.image.height
        
        data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": bundle.image.path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

        json_file = labelme_dir / f"{stem}.json"
        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Cleanup orphaned label files in labelme_dir if the image next to it was deleted
    for json_file in labelme_dir.glob("*.json"):
        stem = json_file.stem
        has_img = False
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            if (labelme_dir / f"{stem}{ext}").exists():
                has_img = True
                break
        if not has_img:
            json_file.unlink()

    return labelme_dir
