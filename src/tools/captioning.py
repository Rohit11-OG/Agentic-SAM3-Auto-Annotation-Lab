from __future__ import annotations

from pathlib import Path


def caption_image(image_path: Path) -> str:
    """Placeholder captioning hook."""
    stem = image_path.stem.replace("_", " ")
    return f"Image likely contains objects related to: {stem}."
