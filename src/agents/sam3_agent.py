from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.core.models import AnnotationBundle, ConversationMessage, MaskRecord, new_mask_id
from src.tools.sam3 import sam3_segment_exemplar_prompt, sam3_segment_text_prompt


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

    def respond(self, bundle: AnnotationBundle) -> ConversationMessage:
        request = self._extract_request(bundle)
        classes = request.get("classes", [])
        hints = request.get("hints", {})
        per_class_prompt: Dict[str, str] = request.get("per_class_prompt", {}) or {}
        version = bundle.retry_count + 1

        generated_masks: List[MaskRecord] = []
        for class_name in classes:
            class_hints = hints.get(class_name, {})
            params = {**self.default_params, **class_hints}
            if bundle.retry_count > 0:
                params.setdefault("retry_mode", True)
                params.setdefault("retry_seed_bump", bundle.retry_count)
            mode = class_hints.get("mode", "text_prompt")

            if mode == "exemplar" and class_hints.get("exemplar_bbox"):
                raw_masks = sam3_segment_exemplar_prompt(
                    image_path=bundle.image.path,
                    exemplar_bbox=tuple(class_hints["exemplar_bbox"]),
                    model_name=self.model_name,
                    params=params,
                )
            else:
                prompt_text = per_class_prompt.get(class_name, class_name)
                raw_masks = sam3_segment_text_prompt(
                    image_path=bundle.image.path,
                    class_name=prompt_text,
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
                }
            ],
        )
