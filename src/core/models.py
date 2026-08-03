from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

ImageId = str
MaskId = str
ClassId = str
AgentName = str

BundleStatus = Literal["NEW", "ANNOTATED", "QA_RETRY", "ACCEPTED", "HUMAN_REVIEW"]
QADecision = Literal["ACCEPT", "RETRY_WITH_HINTS", "HUMAN_REVIEW"]
MaskSource = Literal["sam3", "human", "other"]
MessageRole = Literal["system", "agent", "tool"]


class HumanReviewPolicy(BaseModel):
    enabled: bool = True
    max_retries: int = 2


class ProjectConfig(BaseModel):
    project_name: str
    dataset_path: Path
    output_path: Path
    label_schema: List[ClassId]
    sam3_model_name: str = "sam3_b"
    max_retries: int = 2
    min_mask_area: float = 0.001
    max_mask_area: float = 0.5
    qa_iou_threshold: float = 0.7
    qa_confidence_threshold: float = 0.3
    enable_captioning: bool = False
    require_all_classes: bool = False
    llm_model_name: str = "mock-llm"
    human_review_policy: HumanReviewPolicy = Field(default_factory=HumanReviewPolicy)
    sam3_params: Dict[str, Any] = Field(default_factory=dict)
    max_turns_per_bundle: int = 25
    max_workers: int = 1
    user_prompt: Optional[str] = None
    per_class_prompt: Dict[str, str] = Field(default_factory=dict)
    yolo_segmentation: bool = False
    slim_conversation_logs: bool = True


class ImageRecord(BaseModel):
    id: ImageId
    path: Path
    width: int
    height: int
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def area_pixels(self) -> int:
        return self.width * self.height


class MaskRecord(BaseModel):
    mask_id: MaskId
    image_id: ImageId
    class_id: ClassId
    polygon: List[Tuple[int, int]] = Field(default_factory=list)
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float
    source: MaskSource = "sam3"
    version: int = 1


class QAResult(BaseModel):
    image_id: ImageId
    quality_score: float
    issues: List[str] = Field(default_factory=list)
    decision: QADecision
    hints: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    image_id: Optional[ImageId] = None
    sender: AgentName
    role: MessageRole = "agent"
    content: str
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AnnotationBundle(BaseModel):
    image: ImageRecord
    masks: List[MaskRecord] = Field(default_factory=list)
    qa_result: Optional[QAResult] = None
    status: BundleStatus = "NEW"
    history: List[ConversationMessage] = Field(default_factory=list)
    retry_count: int = 0


def new_mask_id() -> MaskId:
    return str(uuid4())
