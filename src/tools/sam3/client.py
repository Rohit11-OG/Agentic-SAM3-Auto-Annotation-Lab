from __future__ import annotations

import base64
import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Tuple
from urllib import request as urlrequest


@dataclass
class RawMask:
    polygon: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float


def _seed_from(image_path: Path, class_name: str) -> int:
    payload = f"{image_path}|{class_name}".encode("utf-8")
    digest = hashlib.md5(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFF


def _mock_text_prompt(
    image_path: Path,
    class_name: str,
    model_name: str,
    params: Dict[str, Any] | None = None,
) -> List[RawMask]:
    params = params or {}
    rng = Random(_seed_from(image_path, class_name) + int(params.get("retry_seed_bump", 0)))
    base_count = 1 + int(rng.random() > 0.55)
    min_instances = int(params.get("min_instances", 0))
    count = max(base_count, min_instances)

    img_w, img_h = _peek_image_size(image_path)
    min_area_rel = float(params.get("min_area", 0.0))
    max_area_rel = float(params.get("max_area", 1.0))
    raise_conf = float(params.get("raise_confidence", 0.0))

    masks: List[RawMask] = []
    attempts = 0
    while len(masks) < count and attempts < count * 5:
        attempts += 1
        x = rng.randint(5, max(6, img_w // 3))
        y = rng.randint(5, max(6, img_h // 3))
        w = rng.randint(24, max(25, img_w // 2))
        h = rng.randint(24, max(25, img_h // 2))
        x2 = min(x + w, img_w - 1)
        y2 = min(y + h, img_h - 1)
        w = max(1, x2 - x)
        h = max(1, y2 - y)
        area = float(w * h)
        rel_area = area / max(1, img_w * img_h)
        if rel_area < min_area_rel:
            # Bump up size to satisfy min_area hint
            target_area = min_area_rel * img_w * img_h * 1.5
            scale = (target_area / area) ** 0.5
            w = min(int(w * scale), img_w - x - 1)
            h = min(int(h * scale), img_h - y - 1)
            area = float(max(1, w * h))
            rel_area = area / max(1, img_w * img_h)
        if rel_area > max_area_rel:
            continue
        conf_bias = 0.0 if model_name else -0.1
        confidence = min(0.99, max(0.05, rng.uniform(0.35, 0.92) + conf_bias + raise_conf))
        polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        masks.append(
            RawMask(
                polygon=polygon,
                bbox=(x, y, w, h),
                area=area,
                confidence=confidence,
            )
        )

    if params.get("dedupe") and len(masks) > 1:
        masks = masks[:1]

    return masks


def _peek_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 1024, 1024


def _mock_exemplar_prompt(exemplar_bbox: Tuple[int, int, int, int], params: Dict[str, Any]) -> List[RawMask]:
    x, y, w, h = exemplar_bbox
    confidence_boost = 0.05 if params.get("retry_mode") else 0.0
    return [
        RawMask(
            polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            bbox=exemplar_bbox,
            area=float(w * h),
            confidence=min(0.99, 0.65 + confidence_boost),
        )
    ]


def _coerce_raw_masks(raw_masks: Any) -> List[RawMask]:
    if isinstance(raw_masks, dict):
        raw_masks = raw_masks.get("masks", [])
    if not isinstance(raw_masks, list):
        raise ValueError("SAM3 API response must be a list or {'masks': [...]} shape.")

    parsed: List[RawMask] = []
    for mask in raw_masks:
        polygon = [tuple(point) for point in mask.get("polygon", [])]
        bbox_tuple = tuple(mask["bbox"])
        if len(bbox_tuple) != 4:
            raise ValueError("Mask bbox must have 4 values.")
        x, y, w, h = bbox_tuple
        area = float(mask.get("area", w * h))
        parsed.append(
            RawMask(
                polygon=polygon,  # type: ignore[arg-type]
                bbox=(int(x), int(y), int(w), int(h)),
                area=area,
                confidence=float(mask.get("confidence", 0.5)),
            )
        )
    return parsed


def _call_sam3_api(payload: Dict[str, Any], params: Dict[str, Any]) -> List[RawMask]:
    api_url = params.get("api_url")
    if not api_url:
        raise ValueError("sam3.api_url must be set when backend='sam3_api'.")

    api_key = params.get("api_key")
    api_key_env = params.get("api_key_env", "SAM3_API_KEY")
    if not api_key:
        api_key = os.getenv(str(api_key_env), "")

    headers = {"Content-Type": "application/json"}
    auth_header = params.get("api_auth_header", "Authorization")
    if api_key:
        headers[auth_header] = f"Bearer {api_key}"
    extra_headers = params.get("api_headers", {})
    if isinstance(extra_headers, dict):
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(str(api_url), data=body, headers=headers, method="POST")
    timeout = float(params.get("api_timeout_s", 60))
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return _coerce_raw_masks(json.loads(raw))


def _build_api_payload(
    image_path: Path,
    model_name: str,
    mode: str,
    params: Dict[str, Any],
    class_name: str | None = None,
    exemplar_bbox: Tuple[int, int, int, int] | None = None,
) -> Dict[str, Any]:
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload: Dict[str, Any] = {
        "model_name": model_name,
        "mode": mode,
        "image_name": image_path.name,
        "image_b64": image_b64,
        "params": params,
    }
    if class_name is not None:
        payload["class_name"] = class_name
    if exemplar_bbox is not None:
        payload["exemplar_bbox"] = list(exemplar_bbox)
    return payload


def sam3_segment_text_prompt(
    image_path: Path,
    class_name: str,
    model_name: str,
    params: Dict[str, Any],
) -> List[RawMask]:
    """SAM3 text segmentation.

    Supported backends:
    - mock (default): deterministic synthetic masks
    - sam3_api: call a hosted API endpoint and parse masks
    - hf_local: load facebook/sam3 via transformers and run locally
    """
    backend = str(params.get("backend", "mock")).lower()
    if backend == "sam3_api":
        payload = _build_api_payload(
            image_path=image_path,
            model_name=model_name,
            mode="text_prompt",
            params=params,
            class_name=class_name,
        )
        try:
            return _call_sam3_api(payload, params)
        except Exception as exc:  # noqa: BLE001
            if params.get("allow_mock_fallback", True):
                warnings.warn(f"SAM3 API call failed; falling back to mock backend. Reason: {exc}", stacklevel=2)
            else:
                raise

    if backend == "hf_local":
        try:
            from src.tools.sam3.hf_backend import segment_text_prompt as _hf_segment

            dets = _hf_segment(image_path, class_name, model_name, params)
            return [
                RawMask(polygon=d.polygon, bbox=d.bbox, area=d.area, confidence=d.confidence)
                for d in dets
            ]
        except Exception as exc:  # noqa: BLE001
            if params.get("allow_mock_fallback", True):
                warnings.warn(
                    f"SAM3 hf_local backend failed; falling back to mock. Reason: {exc}",
                    stacklevel=2,
                )
            else:
                raise

    return _mock_text_prompt(image_path, class_name, model_name, params)


def sam3_segment_exemplar_prompt(
    image_path: Path,
    exemplar_bbox: Tuple[int, int, int, int],
    model_name: str,
    params: Dict[str, Any],
) -> List[RawMask]:
    backend = str(params.get("backend", "mock")).lower()
    if backend == "sam3_api":
        payload = _build_api_payload(
            image_path=image_path,
            model_name=model_name,
            mode="exemplar_prompt",
            params=params,
            exemplar_bbox=exemplar_bbox,
        )
        try:
            return _call_sam3_api(payload, params)
        except Exception as exc:  # noqa: BLE001
            if params.get("allow_mock_fallback", True):
                warnings.warn(f"SAM3 API call failed; falling back to mock backend. Reason: {exc}", stacklevel=2)
            else:
                raise

    if backend == "hf_local":
        if not params.get("allow_mock_fallback", True):
            raise RuntimeError("hf_local backend does not support exemplar (bounding box) prompting.")
        warnings.warn(
            "SAM3 hf_local backend does not support exemplar prompting; falling back to mock backend.",
            stacklevel=2,
        )

    return _mock_exemplar_prompt(exemplar_bbox, params)
