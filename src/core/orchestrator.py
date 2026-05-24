from __future__ import annotations

import json
import logging
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.curation_agent import CurationAgent
from src.agents.sam3_agent import SAM3Agent
from src.core.models import AnnotationBundle, ConversationMessage, ImageRecord, MaskRecord, ProjectConfig, QAResult
from src.tools.captioning import caption_image
from src.tools.yolo import export_yolo

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

try:
    from PIL import Image as _PILImage  # type: ignore

    _PILImage.MAX_IMAGE_PIXELS = None  # disable decompression-bomb cap for trusted local files
except Exception:  # noqa: BLE001
    pass


def _read_image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height). Use Pillow if installed, else parse headers."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass

    suffix = image_path.suffix.lower()
    try:
        with image_path.open("rb") as fh:
            header = fh.read(64)
        if suffix == ".png" and header.startswith(b"\x89PNG\r\n\x1a\n"):
            w, h = struct.unpack(">II", header[16:24])
            return int(w), int(h)
        if suffix == ".bmp" and header.startswith(b"BM"):
            w, h = struct.unpack("<ii", header[18:26])
            return int(abs(w)), int(abs(h))
        if suffix in {".jpg", ".jpeg"}:
            with image_path.open("rb") as fh:
                fh.read(2)
                while True:
                    byte = fh.read(1)
                    while byte == b"\xff":
                        byte = fh.read(1)
                    marker = byte
                    if not marker:
                        break
                    if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3"):
                        fh.read(3)
                        h, w = struct.unpack(">HH", fh.read(4))
                        return int(w), int(h)
                    size = struct.unpack(">H", fh.read(2))[0]
                    fh.read(size - 2)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not read dimensions for %s: %s", image_path, exc)

    LOGGER.warning("Falling back to default 1024x1024 for %s", image_path)
    return 1024, 1024


def list_image_paths(dataset_path: Path, recursive: bool = True) -> List[Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")
    iterator = dataset_path.rglob("*") if recursive else dataset_path.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover_images(dataset_path: Path, recursive: bool = True) -> List[ImageRecord]:
    paths = list_image_paths(dataset_path, recursive=recursive)
    if not paths:
        return []

    if len(paths) > 4:
        workers = min(8, max(2, len(paths) // 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(_read_image_size, paths))
    else:
        sizes = [_read_image_size(p) for p in paths]

    return [
        ImageRecord(
            id=f"img_{idx:05d}",
            path=image_path,
            width=width,
            height=height,
            meta={},
        )
        for idx, (image_path, (width, height)) in enumerate(zip(paths, sizes))
    ]


def _append_message(bundle: AnnotationBundle, message: ConversationMessage) -> None:
    bundle.history.append(message)


def _apply_actions(bundle: AnnotationBundle, message: ConversationMessage) -> None:
    for action in message.actions:
        action_type = action.get("type")

        if action_type == "ANNOTATION_RESULT":
            bundle.masks = [MaskRecord.model_validate(mask) for mask in action.get("masks", [])]
            bundle.status = "ANNOTATED"

        elif action_type == "QA_DECISION":
            bundle.qa_result = QAResult.model_validate(action.get("qa_result", {}))

        elif action_type == "SET_STATUS":
            new_status = action.get("status")
            if new_status in {"NEW", "ANNOTATED", "QA_RETRY", "ACCEPTED", "HUMAN_REVIEW"}:
                bundle.status = new_status
                if new_status == "QA_RETRY":
                    bundle.retry_count += 1


def _contains_action(message: ConversationMessage, action_name: str) -> bool:
    return any(action.get("type") == action_name for action in message.actions)


def run_bundle_conversation(
    bundle: AnnotationBundle,
    config: ProjectConfig,
    coordinator: CoordinatorAgent,
    sam3_agent: SAM3Agent,
    curation_agent: CurationAgent,
    max_turns: int | None = None,
) -> AnnotationBundle:
    if max_turns is None:
        max_turns = config.max_turns_per_bundle
    turns = 0
    while bundle.status not in {"ACCEPTED", "HUMAN_REVIEW"} and turns < max_turns:
        turns += 1

        if coordinator.should_respond(bundle):
            coordinator_msg = coordinator.respond(bundle)
            _append_message(bundle, coordinator_msg)
            _apply_actions(bundle, coordinator_msg)
            LOGGER.debug("Coordinator: %s", coordinator_msg.content)

            if _contains_action(coordinator_msg, "REQUEST_ANNOTATION") or _contains_action(
                coordinator_msg, "REQUEST_RETRY"
            ):
                sam_msg = sam3_agent.respond(bundle)
                _append_message(bundle, sam_msg)
                _apply_actions(bundle, sam_msg)
                LOGGER.debug("SAM3Agent: %s", sam_msg.content)
                continue

            if _contains_action(coordinator_msg, "REQUEST_QA"):
                qa_msg = curation_agent.respond(bundle)
                _append_message(bundle, qa_msg)
                _apply_actions(bundle, qa_msg)
                LOGGER.debug("CurationAgent: %s", qa_msg.content)
                continue

    if turns >= max_turns and bundle.status not in {"ACCEPTED", "HUMAN_REVIEW"}:
        bundle.status = "HUMAN_REVIEW"
        bundle.history.append(
            ConversationMessage(
                image_id=bundle.image.id,
                sender="CoordinatorAgent",
                role="system",
                content="Max turns reached; escalating to HUMAN_REVIEW.",
                actions=[{"type": "SET_STATUS", "image_id": bundle.image.id, "status": "HUMAN_REVIEW"}],
            )
        )
    return bundle


def _slim_message(msg_dict: dict) -> dict:
    """Strip heavy mask polygon data from a message dict (keep counts only)."""
    actions = msg_dict.get("actions", [])
    new_actions = []
    for action in actions:
        if action.get("type") == "ANNOTATION_RESULT" and "masks" in action:
            masks = action["masks"]
            new_actions.append({
                **{k: v for k, v in action.items() if k != "masks"},
                "mask_count": len(masks),
                "mask_classes": [m.get("class_id") for m in masks],
            })
        else:
            new_actions.append(action)
    return {**msg_dict, "actions": new_actions}


def _export_conversation_logs(
    bundles: Iterable[AnnotationBundle],
    output_path: Path,
    slim: bool = True,
) -> Path:
    logs_file = output_path / "conversation_logs.json"
    payload = []
    for bundle in bundles:
        history_dicts = [message.model_dump(mode="json") for message in bundle.history]
        if slim and len(history_dicts) > 1:
            # Keep masks only in the most recent ANNOTATION_RESULT
            slimmed = []
            kept_final = False
            for msg in reversed(history_dicts):
                has_ar = any(a.get("type") == "ANNOTATION_RESULT" for a in msg.get("actions", []))
                if has_ar and not kept_final:
                    slimmed.append(msg)
                    kept_final = True
                else:
                    slimmed.append(_slim_message(msg))
            history_dicts = list(reversed(slimmed))
        payload.append(
            {
                "image_id": bundle.image.id,
                "status": bundle.status,
                "retry_count": bundle.retry_count,
                "qa_result": bundle.qa_result.model_dump() if bundle.qa_result else None,
                "history": history_dicts,
            }
        )
    logs_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return logs_file


_UUID_RE = re.compile(
    r":\s*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    flags=re.IGNORECASE,
)


def _normalize_issue(text: str) -> str:
    # Strip mask UUIDs so similar issues group together
    return _UUID_RE.sub("", text).strip()


def _export_qa_report(
    bundles: List[AnnotationBundle],
    output_path: Path,
    label_schema: List[str],
) -> Path:
    from collections import Counter

    status_counts: Counter = Counter(b.status for b in bundles)
    class_counts: Counter = Counter()
    confidences: List[float] = []
    issues_counter: Counter = Counter()
    retries_total = 0
    accepted_with_retries = 0
    human_review_ids: List[str] = []

    for bundle in bundles:
        retries_total += bundle.retry_count
        if bundle.status == "ACCEPTED" and bundle.retry_count > 0:
            accepted_with_retries += 1
        if bundle.status == "HUMAN_REVIEW":
            human_review_ids.append(bundle.image.id)
        for mask in bundle.masks:
            class_counts[mask.class_id] += 1
            confidences.append(mask.confidence)
        if bundle.qa_result:
            for issue in bundle.qa_result.issues:
                issues_counter[_normalize_issue(issue)] += 1

    total = len(bundles)
    accept_rate = (status_counts.get("ACCEPTED", 0) / total) if total else 0.0
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    report = {
        "total_images": total,
        "status_counts": dict(status_counts),
        "accept_rate": round(accept_rate, 4),
        "retries_total": retries_total,
        "accepted_after_retry": accepted_with_retries,
        "human_review_image_ids": human_review_ids,
        "per_class_mask_counts": {cls: class_counts.get(cls, 0) for cls in label_schema},
        "avg_mask_confidence": round(avg_conf, 4),
        "top_issues": issues_counter.most_common(10),
    }

    report_file = output_path / "qa_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_file


def run_orchestrator(
    config: ProjectConfig,
    cancel_event: Optional[threading.Event] = None,
) -> List[AnnotationBundle]:
    config.output_path.mkdir(parents=True, exist_ok=True)
    images = discover_images(config.dataset_path)
    LOGGER.info("Discovered %d images in dataset (recursive scan).", len(images))
    if not images:
        LOGGER.warning(
            "No images found under %s (supported: %s). "
            "Check the folder path or that files have image extensions.",
            config.dataset_path,
            ", ".join(sorted(IMAGE_SUFFIXES)),
        )

    coordinator = CoordinatorAgent(
        label_schema=config.label_schema,
        max_retries=config.max_retries,
        per_class_prompt=config.per_class_prompt,
    )
    sam3_agent = SAM3Agent(model_name=config.sam3_model_name, default_params=config.sam3_params)
    curation_agent = CurationAgent(
        min_mask_area=config.min_mask_area,
        max_mask_area=config.max_mask_area,
        iou_threshold=config.qa_iou_threshold,
        confidence_threshold=config.qa_confidence_threshold,
        label_schema=config.label_schema,
        enable_captioning=config.enable_captioning,
        caption_fn=caption_image if config.enable_captioning else None,
    )

    bundles: List[AnnotationBundle] = [AnnotationBundle(image=image, status="NEW") for image in images]

    def _process(bundle: AnnotationBundle) -> AnnotationBundle:
        if cancel_event is not None and cancel_event.is_set():
            bundle.status = "HUMAN_REVIEW"
            bundle.history.append(
                ConversationMessage(
                    image_id=bundle.image.id,
                    sender="CoordinatorAgent",
                    role="system",
                    content="Cancelled by user before processing.",
                    actions=[],
                )
            )
            return bundle
        return run_bundle_conversation(bundle, config, coordinator, sam3_agent, curation_agent)

    if config.max_workers > 1 and len(bundles) > 1:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            list(pool.map(_process, bundles))
    else:
        for bundle in bundles:
            if cancel_event is not None and cancel_event.is_set():
                LOGGER.warning("Cancel signal received; skipping remaining bundles.")
                break
            _process(bundle)

    export_yolo(
        bundles,
        config.output_path,
        label_schema=config.label_schema,
        segmentation=config.yolo_segmentation,
    )
    _export_conversation_logs(bundles, config.output_path, slim=config.slim_conversation_logs)
    _export_qa_report(bundles, config.output_path, config.label_schema)
    return bundles
