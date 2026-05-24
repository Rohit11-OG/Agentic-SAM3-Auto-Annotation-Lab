from pathlib import Path

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.curation_agent import CurationAgent
from src.agents.sam3_agent import SAM3Agent
from src.core.models import AnnotationBundle, ImageRecord, ProjectConfig
from src.core.orchestrator import run_bundle_conversation


def _make_config(**overrides) -> ProjectConfig:
    base = dict(
        project_name="test",
        dataset_path=Path("."),
        output_path=Path("."),
        label_schema=["person", "car"],
        sam3_model_name="sam3_b",
        max_retries=2,
        min_mask_area=0.001,
        max_mask_area=0.9,
        qa_iou_threshold=0.95,
        qa_confidence_threshold=0.01,
        enable_captioning=False,
        llm_model_name="mock-llm",
    )
    base.update(overrides)
    return ProjectConfig(**base)


def test_single_bundle_reaches_terminal_state() -> None:
    image = ImageRecord(id="img_1", path=Path("img.jpg"), width=1024, height=1024, meta={})
    bundle = AnnotationBundle(image=image, status="NEW")

    config = _make_config()

    coordinator = CoordinatorAgent(label_schema=config.label_schema, max_retries=config.max_retries)
    sam3 = SAM3Agent(model_name=config.sam3_model_name)
    curation = CurationAgent(
        min_mask_area=config.min_mask_area,
        max_mask_area=config.max_mask_area,
        iou_threshold=config.qa_iou_threshold,
        confidence_threshold=config.qa_confidence_threshold,
    )

    result = run_bundle_conversation(bundle, config, coordinator, sam3, curation, max_turns=20)
    assert result.status in {"ACCEPTED", "HUMAN_REVIEW"}
    assert len(result.history) > 0


def test_curation_flags_empty_masks() -> None:
    image = ImageRecord(id="img_x", path=Path("x.jpg"), width=640, height=480)
    bundle = AnnotationBundle(image=image, status="ANNOTATED", masks=[])

    curation = CurationAgent(
        min_mask_area=0.001,
        max_mask_area=0.9,
        iou_threshold=0.7,
        confidence_threshold=0.3,
        label_schema=["person", "car"],
    )
    score, issues, hints = curation._quality_checks(bundle)
    assert score == 0.0
    assert any("No masks" in issue for issue in issues)
    assert "person" in hints and "car" in hints


def test_curation_caption_memoized() -> None:
    calls = {"n": 0}

    def fake_caption(path):
        calls["n"] += 1
        return "image with person and dog"

    curation = CurationAgent(
        min_mask_area=0.001,
        max_mask_area=0.9,
        iou_threshold=0.7,
        confidence_threshold=0.3,
        label_schema=["person"],
        enable_captioning=True,
        caption_fn=fake_caption,
    )
    image = ImageRecord(id="img_c", path=Path("c.jpg"), width=100, height=100)
    bundle = AnnotationBundle(image=image, status="ANNOTATED", masks=[])

    curation.respond(bundle)
    curation.respond(bundle)
    curation.respond(bundle)
    assert calls["n"] == 1
