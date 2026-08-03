"""Regressions for the issues found in the deep scan."""
from pathlib import Path

import pytest
from PIL import Image

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.curation_agent import CurationAgent
from src.core.config_loader import _resolve_path
from src.core.models import AnnotationBundle, ConversationMessage, ImageRecord, MaskRecord
from src.tools.labelme import export_labelme
from src.tools.yolo import export_yolo


def _bundle(tmp_path, name="a.jpg", cls="cat"):
    img = tmp_path / name
    Image.new("RGB", (50, 50)).save(img)
    rec = ImageRecord(id="i0", path=img, width=50, height=50)
    mask = MaskRecord(
        mask_id="m1", image_id="i0", class_id=cls,
        bbox=(1, 1, 5, 5), area=25, confidence=0.9,
        polygon=[(1, 1), (6, 1), (6, 6)],
    )
    return AnnotationBundle(image=rec, masks=[mask], status="ACCEPTED")


@pytest.mark.parametrize(
    "export, subdir, ext",
    [(export_yolo, "yolo_labels", ".txt"), (export_labelme, "labelme_json", ".json")],
)
def test_export_keeps_foreign_labels_prunes_own_stale(tmp_path, export, subdir, ext):
    """Label files we did not write must survive; our own stale pairs must go."""
    out = tmp_path / "out"
    d = out / subdir
    d.mkdir(parents=True)

    foreign = d / f"handmade{ext}"          # no image copy beside it -> not ours
    foreign.write_text("keep me")
    stale = d / f"old{ext}"                  # label + copied image -> ours, gone from run
    stale.write_text("prune me")
    Image.new("RGB", (50, 50)).save(d / "old.jpg")

    export([_bundle(tmp_path)], out)

    assert foreign.exists(), "a label file we did not write was deleted"
    assert not stale.exists() and not (d / "old.jpg").exists()
    assert (d / f"a{ext}").exists()


def test_export_yolo_warns_on_class_outside_schema(tmp_path, caplog):
    export_yolo([_bundle(tmp_path, cls="ufo")], tmp_path / "out", label_schema=["cat"])
    assert "ufo" in caplog.text and "label_schema" in caplog.text


def test_stalled_conversation_escalates_instead_of_vanishing():
    """A bundle must never end in a non-terminal status: exporters would skip it."""
    from src.core.models import ProjectConfig
    from src.core.orchestrator import run_bundle_conversation

    rec = ImageRecord(id="i0", path=Path("x.jpg"), width=100, height=100)
    bundle = AnnotationBundle(image=rec, status="ANNOTATED")
    bundle.history.append(
        ConversationMessage(image_id="i0", sender="Other", content="noise", actions=[])
    )
    cfg = ProjectConfig(
        project_name="t", dataset_path=Path("."), output_path=Path("."), label_schema=["cat"],
    )

    class _Never:
        def should_respond(self, b):
            return False

        def respond(self, b):  # pragma: no cover - must not be reached
            raise AssertionError("should not be called")

    out = run_bundle_conversation(
        bundle, cfg, CoordinatorAgent(label_schema=["cat"], max_retries=2), _Never(), _Never(),
    )
    assert out.status == "HUMAN_REVIEW"
    # stopped promptly rather than spinning out max_turns with no-op messages
    assert len(out.history) < 5


@pytest.mark.parametrize("require_all, expect_issue", [(False, False), (True, True)])
def test_missing_class_check_is_independent_of_captioning(require_all, expect_issue):
    rec = ImageRecord(id="i0", path=Path("x.jpg"), width=100, height=100)
    mask = MaskRecord(
        mask_id="m1", image_id="i0", class_id="cat",
        bbox=(0, 0, 10, 10), area=100, confidence=0.9,
    )
    bundle = AnnotationBundle(image=rec, masks=[mask], status="ANNOTATED")
    agent = CurationAgent(
        min_mask_area=0.001, max_mask_area=0.5, iou_threshold=0.7, confidence_threshold=0.3,
        label_schema=["cat", "dog"], enable_captioning=False, require_all_classes=require_all,
    )
    _, issues, _ = agent._quality_checks(bundle)
    assert any("dog" in i for i in issues) is expect_issue


def test_resolve_path_prefers_config_relative(tmp_path, monkeypatch):
    """Same config loaded from a different cwd must point at the same folder."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "data").mkdir()
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "data").mkdir(parents=True)

    monkeypatch.chdir(elsewhere)
    resolved = _resolve_path("./data", cfg_dir / "project.yaml", "./default")
    assert resolved == (cfg_dir / "data").resolve()


def test_extract_frames_rejects_zero_without_crashing(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.tools.video_extractor import extract_frames

    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for _ in range(20):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()

    result = extract_frames(video, tmp_path / "out", 0)  # must not ZeroDivisionError
    assert result.saved_frames >= 1


def test_no_candidates_skips_the_futile_retry():
    """Zero pre-threshold candidates means a looser retry finds the same nothing."""
    rec = ImageRecord(id="i0", path=Path("x.jpg"), width=100, height=100)
    agent = CurationAgent(
        min_mask_area=0.001, max_mask_area=0.5, iou_threshold=0.7,
        confidence_threshold=0.3, label_schema=["cat"],
    )

    def _decide(no_candidates):
        bundle = AnnotationBundle(image=rec, masks=[], status="ANNOTATED")
        bundle.history.append(
            ConversationMessage(
                image_id="i0", sender="SAM3Agent", content="none",
                actions=[{
                    "type": "ANNOTATION_RESULT", "image_id": "i0",
                    "masks": [], "no_candidates": no_candidates,
                }],
            )
        )
        return agent.respond(bundle).actions[0]["decision"]

    # nothing proposed at any score -> stop, a second pass is identical
    assert _decide(True) == "HUMAN_REVIEW"
    # candidates existed but scored below threshold -> a looser retry can help
    assert _decide(False) == "RETRY_WITH_HINTS"
