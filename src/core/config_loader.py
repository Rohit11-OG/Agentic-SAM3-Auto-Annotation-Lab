from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.core.models import HumanReviewPolicy, ProjectConfig


def _resolve_path(path_value: Any, config_path: Path, default: str) -> Path:
    if path_value is None:
        return (config_path.parent / default).resolve()

    path = Path(str(path_value))
    if path.is_absolute():
        return path

    if path.exists():
        return path.resolve()

    return (config_path.parent / path).resolve()


def _require_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def load_project_config(config_path: Path) -> ProjectConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping.")

    sam3_cfg = _require_mapping(raw, "sam3")
    qa_cfg = _require_mapping(raw, "qa")
    llm_cfg = _require_mapping(raw, "llm")
    hr_cfg = _require_mapping(raw, "human_review")

    dataset_path = _resolve_path(raw.get("dataset_path", "./data/images"), config_path, "./data/images")
    output_path = _resolve_path(raw.get("output_path", "./data/annotations_final"), config_path, "./data/annotations_final")

    label_schema = list(raw.get("label_schema", []))
    if not label_schema:
        raise ValueError("Config 'label_schema' must contain at least one class.")

    return ProjectConfig(
        project_name=raw.get("project_name", "sam3_auto_annotation_lab"),
        dataset_path=dataset_path,
        output_path=output_path,
        label_schema=label_schema,
        sam3_model_name=sam3_cfg.get("model_name", "sam3_b"),
        max_retries=int(qa_cfg.get("max_retries", 2)),
        min_mask_area=float(qa_cfg.get("min_mask_area", 0.001)),
        max_mask_area=float(qa_cfg.get("max_mask_area", 0.5)),
        qa_iou_threshold=float(qa_cfg.get("iou_threshold", 0.7)),
        qa_confidence_threshold=float(qa_cfg.get("confidence_threshold", 0.3)),
        enable_captioning=bool(qa_cfg.get("enable_captioning", False)),
        llm_model_name=llm_cfg.get("model_name", "mock-llm"),
        human_review_policy=HumanReviewPolicy(
            enabled=bool(hr_cfg.get("enabled", True)),
            max_retries=int(hr_cfg.get("max_retries", qa_cfg.get("max_retries", 2))),
        ),
        sam3_params=sam3_cfg,
        max_turns_per_bundle=int(raw.get("max_turns_per_bundle", 25)),
        max_workers=max(1, int(raw.get("max_workers", 1))),
        yolo_segmentation=bool(raw.get("yolo_segmentation", False)),
        slim_conversation_logs=bool(raw.get("slim_conversation_logs", True)),
    )
