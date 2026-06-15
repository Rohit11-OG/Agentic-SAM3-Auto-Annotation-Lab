"""Extract evenly-spaced frames from a video file.

Uses OpenCV (cv2) under the hood. If cv2 is not installed the public
functions raise ``RuntimeError`` with an install hint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class ExtractionResult:
    """Summary returned after one video is processed."""

    video_path: Path
    output_dir: Path
    total_video_frames: int = 0
    requested_frames: int = 0
    saved_frames: int = 0
    fps: float = 0.0
    duration_secs: float = 0.0
    saved_paths: List[Path] = field(default_factory=list)
    error: Optional[str] = None


ProgressCallback = Callable[[int, int, str], None]
"""(current, total, message) — called from the worker thread."""


def _ensure_cv2() -> Any:
    try:
        import cv2  # type: ignore
        return cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for video frame extraction.\n"
            "Install it with:  pip install opencv-python"
        ) from exc


def extract_frames(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
    *,
    prefix: Optional[str] = None,
    quality: int = 95,
    callback: Optional[ProgressCallback] = None,
) -> ExtractionResult:
    """Extract *num_frames* evenly-spaced frames from *video_path*.

    Frames are saved as ``{prefix}_frame_{index:06d}.jpg`` inside
    *output_dir* (created automatically).

    Parameters
    ----------
    video_path:
        Path to a video file.
    output_dir:
        Destination folder for the extracted JPEG frames.
    num_frames:
        How many frames to extract (evenly distributed across the video).
    prefix:
        File-name prefix; defaults to the video stem.
    quality:
        JPEG quality 1–100.
    callback:
        Optional ``(current, total, message)`` progress reporter.
    """
    cv2 = _ensure_cv2()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = video_path.stem

    result = ExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        requested_frames=num_frames,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error = f"Cannot open video: {video_path}"
        LOGGER.error(result.error)
        return result

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        result.total_video_frames = total_frames
        result.fps = float(fps)
        result.duration_secs = total_frames / fps if fps > 0 else 0.0

        if total_frames <= 0:
            result.error = "Video reports 0 frames."
            LOGGER.error(result.error)
            return result

        # Clamp to actual frame count
        actual = min(num_frames, total_frames)

        # Compute evenly-spaced frame indices
        if actual >= total_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / actual
            indices = [int(i * step) for i in range(actual)]

        LOGGER.info(
            "Extracting %d frames from %s (%d total, %.1f fps, %.1fs)",
            len(indices), video_path.name, total_frames, fps, result.duration_secs,
        )

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

        for count, frame_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                LOGGER.debug("Could not read frame %d", frame_idx)
                continue

            out_name = f"{prefix}_frame_{frame_idx:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame, encode_params)
            result.saved_paths.append(out_path)
            result.saved_frames += 1

            if callback is not None:
                callback(count, len(indices), f"Saved {out_name}")

    finally:
        cap.release()

    LOGGER.info("Saved %d/%d frames to %s", result.saved_frames, num_frames, output_dir)
    return result


def extract_frames_batch(
    jobs: List[Tuple[Path, int]],
    output_dir: Path,
    *,
    quality: int = 95,
    callback: Optional[ProgressCallback] = None,
) -> List[ExtractionResult]:
    """Extract frames from multiple videos into *output_dir*.

    Each video gets a sub-folder named after the video stem.

    Parameters
    ----------
    jobs:
        List of ``(video_path, num_frames)`` tuples.
    output_dir:
        Root output folder.
    quality:
        JPEG quality.
    callback:
        Progress reporter receiving global progress across all videos.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute total frames for global progress
    total_requested = sum(n for _, n in jobs)
    global_done = 0
    results: List[ExtractionResult] = []

    for video_path, num_frames in jobs:
        video_path = Path(video_path)
        sub_dir = output_dir / video_path.stem

        def _relay(cur: int, tot: int, msg: str) -> None:
            if callback is not None:
                callback(global_done + cur, total_requested, msg)

        res = extract_frames(
            video_path,
            sub_dir,
            num_frames,
            quality=quality,
            callback=_relay,
        )
        global_done += res.saved_frames
        results.append(res)

    return results
