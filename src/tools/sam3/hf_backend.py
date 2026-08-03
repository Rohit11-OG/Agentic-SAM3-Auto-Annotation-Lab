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

# Image-only model (for box/point prompts) is separate from the video model
_IMAGE_MODEL_LOCK = threading.Lock()
_IMAGE_MODEL_CACHE: Dict[str, Tuple[Any, Any, str, Any]] = {}

# Tracker-video model (for few-shot box propagation) — third flavor of the same checkpoint
_TRACKER_MODEL_LOCK = threading.Lock()
_TRACKER_MODEL_CACHE: Dict[str, Tuple[Any, Any, str, Any]] = {}


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

        attn_impl = "sdpa" if device.startswith("cuda") else None
        kwargs: Dict[str, Any] = {"local_files_only": local_only}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        processor = AutoProcessor.from_pretrained(source, local_files_only=local_only)
        try:
            model = AutoModel.from_pretrained(source, dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModel.from_pretrained(source, torch_dtype=dtype, **kwargs)
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

    Saves the cost of re-encoding the image per class. A prompt may hold
    "|"-separated variants ("car|vehicle|automobile"); they are tried in order
    inside the same session until one yields detections, so falling back costs
    no extra image encoding. Returns a dict class_id -> list of detections.
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
    with _INFERENCE_LOCK:
        with torch.no_grad():
            for class_id, prompt_text in class_prompts:
                variants = [v.strip() for v in str(prompt_text).split("|") if v.strip()]
                if not variants:
                    variants = [str(prompt_text)]
                class_dets: List[_Detection] = []
                for variant in variants:
                    try:
                        session.reset_tracking_data()
                    except Exception:  # noqa: BLE001
                        pass
                    processor.add_text_prompt(session, text=variant)
                    class_dets = []
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
                    if class_dets:
                        break  # this variant matched; skip remaining fallbacks
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


def _load_image_model(repo_id: str, params: Dict[str, Any]) -> Tuple[Any, Any, str, Any]:
    """Load the image-only Sam3 model + processor (separate from video flow)."""
    cache_key = f"{repo_id}|{params.get('local_dir') or ''}|{params.get('device') or 'auto'}"
    cached = _IMAGE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _IMAGE_MODEL_LOCK:
        cached = _IMAGE_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch  # type: ignore
            from transformers.models.sam3.processing_sam3 import Sam3Processor  # type: ignore
            from transformers.models.sam3.modeling_sam3 import Sam3Model  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Install: pip install torch torchvision transformers accelerate"
            ) from exc

        device = _resolve_device(params.get("device"))
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        local_dir = params.get("local_dir")
        source = local_dir if local_dir else repo_id
        if bool(local_dir):
            import os as _os
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        attn_impl = "sdpa" if device.startswith("cuda") else None
        kwargs: Dict[str, Any] = {"local_files_only": bool(local_dir)}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        LOGGER.info("Loading SAM3 image model from %s (device=%s)", source, device)
        processor = Sam3Processor.from_pretrained(source, local_files_only=bool(local_dir))
        try:
            model = Sam3Model.from_pretrained(source, dtype=dtype, **kwargs)
        except TypeError:
            model = Sam3Model.from_pretrained(source, torch_dtype=dtype, **kwargs)
        model.to(device).eval()
        _IMAGE_MODEL_CACHE[cache_key] = (model, processor, device, dtype)
        return _IMAGE_MODEL_CACHE[cache_key]


def _load_tracker_model(repo_id: str, params: Dict[str, Any]) -> Tuple[Any, Any, str, Any]:
    """Load Sam3TrackerVideo model + processor (SAM2-style box/point tracking)."""
    cache_key = f"{repo_id}|{params.get('local_dir') or ''}|{params.get('device') or 'auto'}"
    cached = _TRACKER_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _TRACKER_MODEL_LOCK:
        cached = _TRACKER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch  # type: ignore
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Install: pip install torch torchvision transformers accelerate"
            ) from exc

        device = _resolve_device(params.get("device"))
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        local_dir = params.get("local_dir")
        source = local_dir if local_dir else repo_id
        if bool(local_dir):
            import os as _os
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        attn_impl = "sdpa" if device.startswith("cuda") else None
        kwargs: Dict[str, Any] = {"local_files_only": bool(local_dir)}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        LOGGER.info("Loading SAM3 tracker model from %s (device=%s)", source, device)
        processor = Sam3TrackerVideoProcessor.from_pretrained(source, local_files_only=bool(local_dir))
        try:
            model = Sam3TrackerVideoModel.from_pretrained(source, dtype=dtype, **kwargs)
        except TypeError:
            model = Sam3TrackerVideoModel.from_pretrained(source, torch_dtype=dtype, **kwargs)
        model.to(device).eval()
        _TRACKER_MODEL_CACHE[cache_key] = (model, processor, device, dtype)
        return _TRACKER_MODEL_CACHE[cache_key]


def segment_fewshot(
    ref_data: List[Tuple[Path, List[Tuple[int, int, int, int]]]],
    target_paths: List[Path],
    class_name: str,
    model_name: str,
    params: Dict[str, Any],
) -> Dict[str, List[_Detection]]:
    """Few-shot annotation via SAM3 tracker: reference boxes seed object masklets,
    the tracker propagates them across the remaining frames.

    ref_data: list of (image_path, list_of_bboxes_x1y1x2y2)
    target_paths: list of unannotated image paths
    Returns: {str(target_path): [_Detection, ...]}
    """
    repo_id = params.get("hf_repo_id") or model_name or "facebook/sam3"
    model, processor, device, dtype = _load_tracker_model(repo_id, params)

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL/torch required") from exc

    target_frames = [Image.open(p).convert("RGB") for p in target_paths]

    results: Dict[str, List[_Detection]] = {str(p): [] for p in target_paths}

    # One session per reference frame: the tracker only supports a single
    # conditioning frame per masklet, so multi-ref runs are merged afterwards.
    with _INFERENCE_LOCK:
        for ref_path, bboxes in ref_data:
            if not bboxes:
                continue
            ref_frame = Image.open(ref_path).convert("RGB")
            all_frames = [ref_frame] + target_frames

            session = processor.init_video_session(
                video=all_frames,
                inference_device=device,
                dtype=dtype,
            )

            # All boxes must go in one call: incremental adds leave earlier
            # objects without conditioning memory and propagation fails.
            box_list = [[float(v) for v in bbox] for bbox in bboxes]
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=0,
                obj_ids=list(range(len(box_list))),
                input_boxes=[box_list],
            )

            with torch.no_grad():
                # Build conditioning memory on the prompted frame, then propagate
                model(inference_session=session, frame_idx=0)
                for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
                    frame_idx = int(getattr(out, "frame_idx", -1))
                    target_idx = frame_idx - 1
                    if target_idx < 0 or target_idx >= len(target_paths):
                        continue
                    target_path = target_paths[target_idx]
                    target_img = target_frames[target_idx]

                    pred_masks = getattr(out, "pred_masks", None)
                    if pred_masks is None:
                        continue

                    pm = pred_masks
                    if hasattr(pm, "float"):
                        pm = pm.float()
                    # Shape (num_objs, 1, H, W) or (num_objs, H, W) — logits: >0 means mask
                    for i in range(pm.shape[0]):
                        mask = pm[i]
                        det = _mask_to_detection(mask, confidence=0.75, target_size=target_img.size)
                        if det is not None:
                            results[str(target_path)].append(det)

            del session
            try:
                import torch as _torch
                if device.startswith("cuda"):
                    _torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # Deduplicate overlapping masks from different references (IoU > 0.6 on bboxes)
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    for key, dets in results.items():
        kept: List[_Detection] = []
        for det in sorted(dets, key=lambda d: -d.area):
            if all(_bbox_iou(det.bbox, k.bbox) <= 0.6 for k in kept):
                kept.append(det)
        results[key] = kept

    return results


def segment_box_prompt(
    image_path: Path,
    box: Tuple[int, int, int, int],
    class_name: str,
    model_name: str,
    params: Dict[str, Any],
) -> List[_Detection]:
    """Segment an object inside a user-drawn bounding box. Returns top-1 mask.

    `box` is (x1, y1, x2, y2) in image pixel coordinates.
    `class_name` provides the SAM3 text hint that narrows the proposal set.
    """
    repo_id = params.get("hf_repo_id") or model_name or "facebook/sam3"
    model, processor, device, dtype = _load_image_model(repo_id, params)

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

    image_size = image.size  # (W, H)
    x1, y1, x2, y2 = box
    x1, x2 = sorted([int(x1), int(x2)])
    y1, y2 = sorted([int(y1), int(y2)])

    inputs = processor(
        images=image,
        text=str(class_name or "object"),
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
    ).to(device)
    # Cast float tensors to model dtype (fp16 on CUDA)
    for k in list(inputs.keys()):
        v = inputs[k]
        if hasattr(v, "dtype") and v.dtype == torch.float32:
            inputs[k] = v.to(dtype)

    # Serialised with the other backends: the Labeler runs box prompts from its
    # own worker thread, which can overlap a pipeline run on the same GPU.
    with _INFERENCE_LOCK:
        with torch.no_grad():
            outputs = model(**inputs)

    pred_masks = getattr(outputs, "pred_masks", None)
    if pred_masks is None:
        return []
    # pred_masks shape: (batch, N_proposals, H_mask, W_mask). Pick best by overlap with user box.
    pm = pred_masks[0].float()  # (N, H, W)
    N, H_m, W_m = pm.shape
    W_img, H_img = image_size

    # Project user's box into mask coordinates
    sx = W_m / max(1, W_img)
    sy = H_m / max(1, H_img)
    mx1 = max(0, min(W_m - 1, int(x1 * sx)))
    my1 = max(0, min(H_m - 1, int(y1 * sy)))
    mx2 = max(0, min(W_m, int(x2 * sx)))
    my2 = max(0, min(H_m, int(y2 * sy)))
    if mx2 <= mx1 or my2 <= my1:
        return []

    bin_all = (pm.sigmoid() > 0.5)  # (N, H, W) bool
    in_box = bin_all[:, my1:my2, mx1:mx2]
    overlap_counts = in_box.sum(dim=(-2, -1)).float()
    total_counts = bin_all.sum(dim=(-2, -1)).float().clamp(min=1.0)
    # Precision-style score: how much of the mask falls inside the user's box
    precision = overlap_counts / total_counts
    # Recall-style score: how much of the box the mask fills
    box_area = float((mx2 - mx1) * (my2 - my1))
    recall = overlap_counts / max(1.0, box_area)
    # Balanced F-like score, favoring recall slightly
    score = (precision * recall) / (precision + recall).clamp(min=1e-6)

    # Require a non-trivial number of pixels inside the box
    valid = overlap_counts > 5
    if not bool(valid.any().item()):
        return []
    score = score.masked_fill(~valid, -1.0)
    best_idx = int(torch.argmax(score).item())
    best_mask = bin_all[best_idx]
    conf = float(score[best_idx].item())

    det = _mask_to_detection(best_mask, confidence=conf, target_size=image_size)
    return [det] if det is not None else []
