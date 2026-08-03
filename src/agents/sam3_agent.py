from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.core.models import AnnotationBundle, ConversationMessage, MaskRecord, new_mask_id
from src.tools.sam3 import (
    sam3_segment_exemplar_prompt,
    sam3_segment_text_prompts_multi,
)


class SAM3Agent(BaseAgent):
    def __init__(self, model_name: str, default_params: Dict[str, Any] | None = None) -> None:
        super().__init__(name="SAM3Agent")
        self.model_name = model_name
        self.default_params = default_params or {}

    def should_respond(self, bundle: AnnotationBundle) -> bool:
        if not bundle.history:
            return bundle.status in ("NEW", "QA_RETRY")
        last = bundle.history[-1]
        if last.sender != "CoordinatorAgent":
            return False
        action_types = {action.get("type") for action in last.actions}
        return "REQUEST_ANNOTATION" in action_types or "REQUEST_RETRY" in action_types

    def _extract_request(self, bundle: AnnotationBundle) -> Dict[str, Any]:
        if not bundle.history:
            return {}
        last = bundle.history[-1]
        for action in last.actions:
            if action.get("type") in {"REQUEST_ANNOTATION", "REQUEST_RETRY"}:
                return action
        return {}

    def _retry_params(self, retry_count: int) -> Dict[str, Any]:
        """Params that make a retry actually differ from the attempt before it.

        The real SAM3 backend reads only ``hf_score_threshold``, so without
        loosening it a retry re-runs the identical prompt on the identical image
        and produces identical masks — burning a full inference pass to fail the
        same QA check again. ``retry_mode``/``retry_seed_bump`` are kept for the
        mock backend, which is the only thing that reads them.
        """
        base = float(self.default_params.get("hf_score_threshold", 0.4))
        return {
            "hf_score_threshold": max(0.05, base - 0.1 * retry_count),
            "retry_mode": True,
            "retry_seed_bump": retry_count,
        }

    def respond(self, bundle: AnnotationBundle) -> ConversationMessage:
        request = self._extract_request(bundle)
        classes = request.get("classes", [])
        hints = request.get("hints", {})
        per_class_prompt: Dict[str, str] = request.get("per_class_prompt", {}) or {}
        version = bundle.retry_count + 1

        # Separate exemplar-mode classes (each needs its own bbox) from text-mode
        text_classes: List[str] = []
        exemplar_classes: List[str] = []
        for class_name in classes:
            class_hints = hints.get(class_name, {})
            mode = class_hints.get("mode", "text_prompt")
            if mode == "exemplar" and class_hints.get("exemplar_bbox"):
                exemplar_classes.append(class_name)
            else:
                text_classes.append(class_name)

        generated_masks: List[MaskRecord] = []
        candidate_counts: Dict[str, int] = {}

        # Multi-class text prompts -> single-session call (saves image encoding).
        # "|"-separated variants ("defect|paint mark|stain") are tried inside that
        # same session by the backend, so fallbacks cost no extra image encoding.
        if text_classes:
            class_prompts = [(cls, str(per_class_prompt.get(cls, cls))) for cls in text_classes]
            merged_params = {**self.default_params}
            if bundle.retry_count > 0:
                merged_params.update(self._retry_params(bundle.retry_count))

            batch = sam3_segment_text_prompts_multi(
                image_path=bundle.image.path,
                class_prompts=class_prompts,
                model_name=self.model_name,
                params=merged_params,
                candidate_counts=candidate_counts,
            )

            for class_name in text_classes:
                for raw in batch.get(class_name, []):
                    generated_masks.append(
                        MaskRecord(
                            mask_id=new_mask_id(),
                            image_id=bundle.image.id,
                            class_id=class_name,
                            polygon=raw.polygon,
                            bbox=raw.bbox,
                            area=raw.area,
                            confidence=raw.confidence,
                            source="sam3",
                            version=version,
                        )
                    )

        # Exemplar prompts: still per-class
        for class_name in exemplar_classes:
            class_hints = hints.get(class_name, {})
            params = {**self.default_params, **class_hints}
            if bundle.retry_count > 0:
                params.update(self._retry_params(bundle.retry_count))
            raw_masks = sam3_segment_exemplar_prompt(
                image_path=bundle.image.path,
                exemplar_bbox=tuple(class_hints["exemplar_bbox"]),
                model_name=self.model_name,
                params=params,
            )
            for raw in raw_masks:
                generated_masks.append(
                    MaskRecord(
                        mask_id=new_mask_id(),
                        image_id=bundle.image.id,
                        class_id=class_name,
                        polygon=raw.polygon,
                        bbox=raw.bbox,
                        area=raw.area,
                        confidence=raw.confidence,
                        source="sam3",
                        version=version,
                    )
                )

        # The model proposed nothing at any score for every class it was asked
        # about, so re-running with a looser threshold has nothing to find. Say
        # so, and QA can escalate instead of paying for an identical pass.
        no_candidates = bool(candidate_counts) and not any(candidate_counts.values())

        detail = ", ".join(f"{c}: {sum(1 for m in generated_masks if m.class_id == c)}" for c in classes)
        return ConversationMessage(
            image_id=bundle.image.id,
            sender=self.name,
            role="agent",
            content=f"Annotated {bundle.image.id}. Mask counts -> {detail}",
            actions=[
                {
                    "type": "ANNOTATION_RESULT",
                    "image_id": bundle.image.id,
                    "masks": [mask.model_dump() for mask in generated_masks],
                    "no_candidates": no_candidates,
                }
            ],
        )
