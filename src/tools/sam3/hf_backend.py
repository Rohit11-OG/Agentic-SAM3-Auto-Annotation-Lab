"""Local Hugging Face backend for SAM3 (facebook/sam3, video-session API).

Single-image inference treats the image as a 1-frame video.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_CACHE: Dict[str, Tuple[Any, Any, str, Any]] = {}  # repo -> (model, processor, device_str, dtype)

_IMG_LOCK = threading.Lock()
_LAST_IMG: Dict[str, Any] = {}  # path -> PIL.Image (decoded RGB), keep only most recent
_LAST_IMG_PATH: List[str] = []  # FIFO of paths cached; cap at 2


@dataclass
class _Detection:
    polygon: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float


def _resolve_device(prefer: Optional[str]) -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torch not installed.") from exc
    if prefer and prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(repo_id: str, params: Dict[str, Any]) -> Tuple[Any, Any, str, Any]:
    cache_key = f"{repo_id}|{params.get('local_dir') or ''}|{params.get('device') or 'auto'}"
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Install: pip install torch torchvision transformers accelerate"
            ) from exc

        device = _resolve_device(params.get("device"))
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        local_dir = params.get("local_dir")
        source = local_dir if local_dir else repo_id
        LOGGER.info("Loading SAM3 from %s (device=%s, dtype=%s)", source, device, dtype)

        local_only = bool(local_dir)
        # Suppress HF hub pings when running fully offline from a local snapshot
        if local_only:
            import os as _os
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        processor = AutoProcessor.from_pretrained(source, local_files_only=local_only)
        try:
            model = AutoModel.from_pretrained(source, dtype=dtype, local_files_only=local_only)
        except TypeError:
            model = AutoModel.from_pretrained(source, torch_dtype=dtype, local_files_only=local_only)
        model.to(device)
        model.eval()

        _CACHE[cache_key] = (model, processor, device, dtype)
        return _CACHE[cache_key]


def _mask_to_detection(
    mask,
    confidence: float,
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[_Detection]:
    import numpy as np  # type: ignore

    arr = mask
    if hasattr(arr, "detach"):
        try:
            import torch  # type: ignore

            arr = arr.detach().cpu().to(dtype=torch.float32).numpy()
        except Exception:  # noqa: BLE001
            arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr.squeeze(0)

    if target_size is not None:
        target_w, target_h = target_size
        if arr.shape != (target_h, target_w):
            try:
                import cv2  # type: ignore

                arr = cv2.resize(arr.astype("float32"), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            except Exception:  # noqa: BLE001
                from PIL import Image as _Image  # type: ignore

                im_tmp = _Image.fromarray((arr * 255).clip(0, 255).astype("uint8"))
                im_tmp = im_tmp.resize((target_w, target_h), _Image.BILINEAR)
                arr = np.asarray(im_tmp).astype("float32") / 255.0

    if arr.dtype != bool:
        arr = arr > 0.0

    ys, xs = np.where(arr)
    if xs.size == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    area = float(arr.sum())

    polygon: List[Tuple[int, int]] = []
    try:
        import cv2  # type: ignore

        bin_mask = arr.astype("uint8")
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            eps = 0.002 * cv2.arcLength(biggest, True)
            approx = cv2.approxPolyDP(biggest, eps, True)
            polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]
    except Exception:  # noqa: BLE001
        polygon = []

    if len(polygon) < 3:
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    return _Detection(
        polygon=polygon,
        bbox=(x_min, y_min, w, h),
        area=area,
        confidence=float(confidence),
    )


def segment_text_prompts_multi(
    image_path: Path,
    class_prompts: List[Tuple[str, str]],
    model_name: str,
    params: Dict[str, Any],
) -> Dict[str, List[_Detection]]:
    """Run several text prompts on one image inside a single SAM3 session.

    Saves the cost of re-encoding the image per class. Returns a dict
    class_id -> list of detections.
    """
    repo_id = params.get("hf_repo_id") or model_name or "facebook/sam3"
    model, processor, device, dtype = _load_model(repo_id, params)

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL/torch required") from exc

    key = str(image_path)
    with _IMG_LOCK:
        cached = _LAST_IMG.get(key)
    if cached is not None:
        image = cached
    else:
        image = Image.open(image_path).convert("RGB")
        with _IMG_LOCK:
            _LAST_IMG[key] = image
            _LAST_IMG_PATH.append(key)
            while len(_LAST_IMG_PATH) > 2:
                old = _LAST_IMG_PATH.pop(0)
                _LAST_IMG.pop(old, None)

    image_size = image.size
    threshold = float(params.get("hf_score_threshold", 0.4))

    session = processor.init_video_session(
        video=[image],
        inference_device=device,
        dtype=dtype,
    )

    results: Dict[str, List[_Detection]] = {}
    with torch.no_grad():
        for class_id, prompt_text in class_prompts:
            try:
                session.reset_tracking_data()
            except Exception:  # noqa: BLE001
                pass
            processor.add_text_prompt(session, text=str(prompt_text))
            class_dets: List[_Detection] = []
            for out in model.propagate_in_video_iterator(session):
                obj_ids = list(getattr(out, "object_ids", []))
                suppressed = set(getattr(out, "suppressed_obj_ids", []) or [])
                removed = set(getattr(out, "removed_obj_ids", []) or [])
                obj_to_mask = getattr(out, "obj_id_to_mask", {}) or {}
                obj_to_score = getattr(out, "obj_id_to_score", {}) or {}
                for oid in obj_ids:
                    if oid in suppressed or oid in removed:
                        continue
                    score = float(obj_to_score.get(oid, 0.5))
                    if score < threshold:
                        continue
                    mask = obj_to_mask.get(oid)
                    if mask is None:
                        continue
                    det = _mask_to_detection(mask, score, target_size=image_size)
                    if det is not None:
                        class_dets.append(det)
                break  # single-frame only
            results[class_id] = class_dets
    return results


def segment_text_prompt(
    image_path: Path,
    class_name: str,
    model_name: str,
    params: Dict[str, Any],
) -> List[_Detection]:
    repo_id = params.get("hf_repo_id") or model_name or "facebook/sam3"
    model, processor, device, dtype = _load_model(repo_id, params)

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL/torch required") from exc

    # Cache decoded image across rapid same-path calls (e.g. SAM3Agent looping classes)
    key = str(image_path)
    with _IMG_LOCK:
        cached = _LAST_IMG.get(key)
    if cached is not None:
        image = cached
    else:
        image = Image.open(image_path).convert("RGB")
        with _IMG_LOCK:
            _LAST_IMG[key] = image
            _LAST_IMG_PATH.append(key)
            while len(_LAST_IMG_PATH) > 2:
                old = _LAST_IMG_PATH.pop(0)
                _LAST_IMG.pop(old, None)
    image_size = image.size  # (W, H)
    threshold = float(params.get("hf_score_threshold", 0.4))

    session = processor.init_video_session(
        video=[image],
        inference_device=device,
        dtype=dtype,
    )
    processor.add_text_prompt(session, text=str(class_name))

    detections: List[_Detection] = []
    with _INFERENCE_LOCK:
        with torch.no_grad():
            for out in model.propagate_in_video_iterator(session):
                obj_ids = list(getattr(out, "object_ids", []))
                suppressed = set(getattr(out, "suppressed_obj_ids", []) or [])
                removed = set(getattr(out, "removed_obj_ids", []) or [])
                obj_to_mask = getattr(out, "obj_id_to_mask", {}) or {}
                obj_to_score = getattr(out, "obj_id_to_score", {}) or {}

                for oid in obj_ids:
                    if oid in suppressed or oid in removed:
                        continue
                    score = float(obj_to_score.get(oid, 0.5))
                    if score < threshold:
                        continue
                    mask = obj_to_mask.get(oid)
                    if mask is None:
                        continue
                    det = _mask_to_detection(mask, score, target_size=image_size)
                    if det is not None:
                        detections.append(det)
                break  # single-frame only

    return detections
