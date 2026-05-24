from __future__ import annotations

from typing import Dict, List

from src.agents.base_agent import BaseAgent
from src.core.models import AnnotationBundle, ConversationMessage


class CoordinatorAgent(BaseAgent):
    def __init__(
        self,
        label_schema: List[str],
        max_retries: int,
        per_class_prompt: Dict[str, str] | None = None,
    ) -> None:
        super().__init__(name="CoordinatorAgent")
        self.label_schema = label_schema
        self.max_retries = max_retries
        self.per_class_prompt = per_class_prompt or {}

    def should_respond(self, bundle: AnnotationBundle) -> bool:
        if not bundle.history:
            return True
        return bundle.status not in ("ACCEPTED", "HUMAN_REVIEW")

    def _next_for_new(self, bundle: AnnotationBundle) -> ConversationMessage:
        return ConversationMessage(
            image_id=bundle.image.id,
            sender=self.name,
            role="agent",
            content=f"{bundle.image.id} is NEW. Requesting initial annotation.",
            actions=[
                {
                    "type": "REQUEST_ANNOTATION",
                    "image_id": bundle.image.id,
                    "classes": self.label_schema,
                    "mode": "text_prompt",
                    "per_class_prompt": self.per_class_prompt,
                }
            ],
        )

    def _after_annotation(self, bundle: AnnotationBundle) -> ConversationMessage:
        return ConversationMessage(
            image_id=bundle.image.id,
            sender=self.name,
            role="agent",
            content=f"Received annotation for {bundle.image.id}. Requesting QA.",
            actions=[{"type": "REQUEST_QA", "image_id": bundle.image.id}],
        )

    def _after_qa(self, bundle: AnnotationBundle) -> ConversationMessage:
        decision = bundle.qa_result.decision if bundle.qa_result else "RETRY_WITH_HINTS"

        if decision == "ACCEPT":
            return ConversationMessage(
                image_id=bundle.image.id,
                sender=self.name,
                role="agent",
                content=f"{bundle.image.id} accepted.",
                actions=[
                    {"type": "SET_STATUS", "image_id": bundle.image.id, "status": "ACCEPTED"},
                ],
            )

        if decision == "HUMAN_REVIEW" or bundle.retry_count >= self.max_retries:
            return ConversationMessage(
                image_id=bundle.image.id,
                sender=self.name,
                role="agent",
                content=f"{bundle.image.id} escalated to human review.",
                actions=[
                    {"type": "SET_STATUS", "image_id": bundle.image.id, "status": "HUMAN_REVIEW"},
                ],
            )

        return ConversationMessage(
            image_id=bundle.image.id,
            sender=self.name,
            role="agent",
            content=f"{bundle.image.id} needs retry #{bundle.retry_count + 1}.",
            actions=[
                {"type": "SET_STATUS", "image_id": bundle.image.id, "status": "QA_RETRY"},
                {
                    "type": "REQUEST_RETRY",
                    "image_id": bundle.image.id,
                    "classes": self.label_schema,
                    "retry_count": bundle.retry_count + 1,
                    "hints": (bundle.qa_result.hints if bundle.qa_result else {}),
                    "per_class_prompt": self.per_class_prompt,
                },
            ],
        )

    def respond(self, bundle: AnnotationBundle) -> ConversationMessage:
        if bundle.status == "NEW":
            return self._next_for_new(bundle)
        if bundle.status in ("ANNOTATED", "QA_RETRY") and bundle.history:
            last = bundle.history[-1]
            action_types = {a.get("type") for a in last.actions}
            if "ANNOTATION_RESULT" in action_types:
                return self._after_annotation(bundle)
            if "QA_DECISION" in action_types:
                return self._after_qa(bundle)
            if bundle.status == "QA_RETRY":
                return ConversationMessage(
                    image_id=bundle.image.id,
                    sender=self.name,
                    role="agent",
                    content=f"{bundle.image.id} retry requested.",
                    actions=[
                        {
                            "type": "REQUEST_RETRY",
                            "image_id": bundle.image.id,
                            "classes": self.label_schema,
                            "retry_count": bundle.retry_count + 1,
                            "hints": (bundle.qa_result.hints if bundle.qa_result else {}),
                            "per_class_prompt": self.per_class_prompt,
                        }
                    ],
                )

        return ConversationMessage(
            image_id=bundle.image.id,
            sender=self.name,
            role="agent",
            content=f"No action needed for {bundle.image.id}.",
            actions=[],
        )
