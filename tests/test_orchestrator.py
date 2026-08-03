from pathlib import Path

import pytest
from PIL import Image

from src.core.config_loader import load_project_config
from src.core.orchestrator import _read_image_size, discover_images, run_orchestrator


def test_read_image_size_jpg(tmp_path: Path) -> None:
    p = tmp_path / "a.jpg"
    Image.new("RGB", (321, 234), (10, 20, 30)).save(p)
    assert _read_image_size(p) == (321, 234)


def test_read_image_size_png(tmp_path: Path) -> None:
    p = tmp_path / "b.png"
    Image.new("RGB", (640, 480), (10, 20, 30)).save(p)
    assert _read_image_size(p) == (640, 480)


def test_discover_images_uses_real_dims(tmp_path: Path) -> None:
    Image.new("RGB", (200, 100), (1, 2, 3)).save(tmp_path / "x.jpg")
    Image.new("RGB", (50, 75), (4, 5, 6)).save(tmp_path / "y.png")
    records = discover_images(tmp_path)
    dims = {r.path.name: (r.width, r.height) for r in records}
    assert dims["x.jpg"] == (200, 100)
    assert dims["y.png"] == (50, 75)


def test_config_loader_rejects_empty_label_schema(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "project_name: t\n"
        "dataset_path: ./d\n"
        "output_path: ./o\n"
        "label_schema: []\n"
        "sam3: {model_name: sam3_b}\n"
        "qa: {}\n"
        "llm: {}\n"
        "human_review: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_project_config(cfg)


def test_orchestrator_end_to_end(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (320, 240), (50, 60, 70)).save(images_dir / "img_a.jpg")
    Image.new("RGB", (400, 300), (90, 80, 70)).save(images_dir / "img_b.png")

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        f"project_name: t\n"
        f"dataset_path: {images_dir.as_posix()}\n"
        f"output_path: {(tmp_path / 'out').as_posix()}\n"
        "label_schema: [person, car]\n"
        "sam3: {model_name: sam3_b, backend: mock}\n"
        "qa: {max_retries: 2, min_mask_area: 0.0, max_mask_area: 0.95, "
        "iou_threshold: 0.95, confidence_threshold: 0.0}\n"
        "llm: {}\n"
        "human_review: {}\n"
        "max_workers: 2\n",
        encoding="utf-8",
    )
    config = load_project_config(cfg_file)
    bundles = run_orchestrator(config)

    assert len(bundles) == 2
    classes_file = config.output_path / "classes.txt"
    assert classes_file.exists()
    assert classes_file.read_text(encoding="utf-8").splitlines() == ["person", "car"]


def test_orchestrator_custom_prompt(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (320, 240), (50, 60, 70)).save(images_dir / "img_a.jpg")

    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        f"project_name: t\n"
        f"dataset_path: {images_dir.as_posix()}\n"
        f"output_path: {(tmp_path / 'out').as_posix()}\n"
        "label_schema: [person, car]\n"
        "sam3: {model_name: sam3_b, backend: mock}\n"
        "qa: {max_retries: 2, min_mask_area: 0.0, max_mask_area: 0.95, "
        "iou_threshold: 0.95, confidence_threshold: 0.0}\n"
        "llm: {}\n"
        "human_review: {}\n"
        "max_workers: 1\n",
        encoding="utf-8",
    )
    config = load_project_config(cfg_file)
    from src.tools.prompt_interpreter import interpret_prompt
    plan = interpret_prompt("there is a metal box in silver color so annotate it")
    config.user_prompt = plan.raw_input
    config.label_schema = plan.classes
    config.per_class_prompt = plan.per_class_prompt

    bundles = run_orchestrator(config)
    assert len(bundles) == 1
    assert config.label_schema == ["box"]
    classes_file = config.output_path / "classes.txt"
    assert classes_file.exists()
    assert classes_file.read_text(encoding="utf-8").splitlines() == ["box"]


def test_discover_images_excludes_output_dir(tmp_path):
    """Output nested in the dataset must not be re-annotated on the next run.

    The exporters copy each image next to its label file, so a recursive scan
    would otherwise pick those copies up and multiply the work every run.
    """
    from PIL import Image

    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(tmp_path / name)

    # Simulate a previous run that exported into the dataset folder
    out = tmp_path / "annotations"
    (out / "yolo_seg_labels").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(out / "yolo_seg_labels" / name)

    records = discover_images(tmp_path, exclude=out)
    assert [r.path.name for r in records] == ["a.jpg", "b.jpg"]
    # ids stay positional over the originals only
    assert [r.id for r in records] == ["img_00000", "img_00001"]

    # Export subfolders are skipped even without an explicit exclude, which is
    # what saves the common case of output_path being the dataset folder itself.
    assert [r.path.name for r in discover_images(tmp_path)] == ["a.jpg", "b.jpg"]
    assert [r.path.name for r in discover_images(tmp_path, exclude=tmp_path)] == ["a.jpg", "b.jpg"]
