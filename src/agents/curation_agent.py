from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.core.models import AnnotationBundle, ConversationMessage, QAResult
from src.tools.geometry import pairwise_bbox_iou


class CurationAgent(BaseAgent):
    def __init__(
        self,
        min_mask_area: float,
        max_mask_area: float,
        iou_threshold: float,
        confidence_threshold: float,
        label_schema: Optional[List[str]] = None,
        enable_captioning: bool = False,
        caption_fn: Optional[Callable[..., str]] = None,
        require_all_classes: bool = False,
    ) -> None:
        super().__init__(name="CurationAgent")
        self.require_all_classes = require_all_classes
        self.min_mask_area = min_mask_area
        self.max_mask_area = max_mask_area
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.label_schema = list(label_schema) if label_schema else []
        self.enable_captioning = enable_captioning
        self.caption_fn = caption_fn
        self._caption_cache: Dict[str, str] = {}
        self._caption_lock = threading.Lock()

    def should_respond(self, bundle: AnnotationBundle) -> bool:
        if not bundle.history:
            return False
        last = bundle.history[-1]
        if last.sender != "CoordinatorAgent":
            return False
        return any(action.get("type") == "REQUEST_QA" for action in last.actions)

    def _quality_checks(self, bundle: AnnotationBundle) -> tuple[float, List[str], Dict[str, Any]]:
        issues: List[str] = []
        hints: Dict[str, Any] = {}
        score = 1.0
        image_area = max(1, bundle.image.area_pixels)

        if not bundle.masks:
            issues.append("No masks produced for any class.")
            for class_name in self.label_schema:
                hints[class_name] = {"min_instances": 1}
            return 0.0, issues, hints

        # A class with no masks is only a defect when every image is expected to
        # contain every class, which is rare — most datasets have images where a
        # class simply isn't present. Off by default; the caption cross-check
        # below still catches classes the image is described as containing.
        class_counts = Counter(mask.class_id for mask in bundle.masks)
        if self.require_all_classes:
            for class_name in self.label_schema:
                if class_counts.get(class_name, 0) == 0:
                    issues.append(f"No masks for class '{class_name}'.")
                    hints.setdefault(class_name, {})["min_instances"] = 1
                    score -= 0.1

        for mask in bundle.masks:
            rel_area = mask.area / image_area
            if rel_area < self.min_mask_area:
                issues.append(f"{mask.class_id}:{mask.mask_id} too small ({rel_area:.4f})")
                hints.setdefault(mask.class_id, {})["min_area"] = self.min_mask_area
                score -= 0.12
            if rel_area > self.max_mask_area:
                issues.append(f"{mask.class_id}:{mask.mask_id} too large ({rel_area:.4f})")
                hints.setdefault(mask.class_id, {})["max_area"] = self.max_mask_area
                score -= 0.12
            if mask.confidence < self.confidence_threshold:
                issues.append(f"{mask.class_id}:{mask.mask_id} low confidence ({mask.confidence:.3f})")
                hints.setdefault(mask.class_id, {})["raise_confidence"] = self.confidence_threshold
                score -= 0.1

        for overlap in pairwise_bbox_iou(bundle.masks):
            same_class = overlap["class_a"] == overlap["class_b"]
            if same_class and overlap["iou"] > self.iou_threshold:
                issues.append(
                    f"Duplicate overlap likely between {overlap['mask_a']} and {overlap['mask_b']} "
                    f"(iou={overlap['iou']:.3f})"
                )
                hints.setdefault(overlap["class_a"], {})["dedupe"] = True
                score -= 0.1

        score = max(0.0, min(1.0, score))
        return score, issues, hints

    def _maybe_caption(self, bundle: AnnotationBundle) -> Optional[str]:
        if not self.enable_captioning or self.caption_fn is None:
            return None
        with self._caption_lock:
            cached = self._caption_cache.get(bundle.image.id)
            if cached is not None:
                return cached
        try:
            caption = self.caption_fn(bundle.image.path)
        except Exception:  # noqa: BLE001
            return None
        with self._caption_lock:
            self._caption_cache[bundle.image.id] = caption
        return caption

    def respond(self, bundle: AnnotationBundle) -> ConversationMessage:
        score, issues, hints = self._quality_checks(bundle)

        caption = self._maybe_caption(bundle)
        if caption:
            for class_name in self.label_schema:
                if class_name.lower() in caption.lower():
                    counts = sum(1 for m in bundle.masks if m.class_id == class_name)
                    if counts == 0:
                        issues.append(f"Caption mentions '{class_name}' but no masks present.")
                        hints.setdefault(class_name, {})["min_instances"] = 1
                        score = max(0.0, score - 0.1)

        if score >= 0.75 and not issues:
            decision = "ACCEPT"
        elif bundle.retry_count >= 1 and score < 0.45:
            decision = "HUMAN_REVIEW"
        else:
            decision = "RETRY_WITH_HINTS"

        qa_result = QAResult(
            image_id=bundle.image.id,
            quality_score=score,
            issues=issues,
            decision=decision,
            hints=hints,
        )

        actions: List[Dict[str, Any]] = [
            {
                "type": "QA_DECISION",
                "image_id": bundle.image.id,
                "decision": decision,
                "qa_result": qa_result.model_dump(),
            }
        ]
        if caption:
            actions[0]["caption"] = caption

        return ConversationMessage(
            image_id=bundle.image.id,
            sender=self.name,
            role="agent",
            content=f"QA decision for {bundle.image.id}: {decision} (score={score:.3f})",
            actions=actions,
        )
