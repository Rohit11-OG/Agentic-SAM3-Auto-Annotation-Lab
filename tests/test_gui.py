"""GUI deep tests for AnnotatorGUI and LabelerPanel.

Uses a withdrawn Tk root so no window appears. Skips if tk unavailable.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

import pytest
from PIL import Image

from src.ui._helpers import attach_tooltip, color_for, load_json, save_json
from src.ui.labeler import LabelerPanel, _Shape
from src.ui.review_app import AnnotatorGUI, _QueueHandler


@pytest.fixture
def root():
    import gc
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available")
    r.withdraw()
    yield r
    try:
        for child in list(r.children.values()):
            try:
                child.destroy()
            except Exception:
                pass
        r.destroy()
    except Exception:
        pass
    # Reset Tk's default-root cache so the next test can create a fresh Tk()
    try:
        tk._default_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    gc.collect()


@pytest.fixture
def gui(root):
    g = AnnotatorGUI(root)
    root.update_idletasks(); root.update()
    return g


# ---------- helpers ----------
def test_color_for_consistent() -> None:
    assert color_for("car") == color_for("car")
    assert isinstance(color_for("anything"), str)


def test_color_for_stable_across_process_hash_seeds() -> None:
    """color_for must not depend on Python's randomized hash() -- a class's
    color would otherwise shift on every restart of the app."""
    import os, subprocess, sys
    outputs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", "from src.ui._helpers import color_for; print(color_for('car'))"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, env=env,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"color_for('car') varied across hash seeds: {outputs}"


def test_attach_tooltip_no_crash(root) -> None:
    btn = tk.Button(root, text="hi")
    attach_tooltip(btn, "tip")
    btn.event_generate("<Enter>")
    root.update()
    btn.event_generate("<Leave>")
    root.update()


def test_load_save_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    save_json(p, {"a": 1, "b": [1, 2, 3]})
    assert load_json(p) == {"a": 1, "b": [1, 2, 3]}


def test_load_json_missing_returns_empty(tmp_path: Path) -> None:
    assert load_json(tmp_path / "nope.json") == {}


# ---------- AnnotatorGUI structure ----------
def test_gui_builds(gui) -> None:
    assert int(gui.notebook.index("end")) == 7  # Dashboard, Setup, Run Log, Results, Few-Shot, Video→Frames, Labeler
    assert str(gui.run_btn["state"]) == "normal"
    assert str(gui.cancel_btn["state"]) == "disabled"
    assert "labels_listbox" in dir(gui)
    assert "preview_canvas" in dir(gui)


def test_theme_toggle(gui) -> None:
    initial = gui.theme_name.get()
    gui._toggle_theme()
    assert gui.theme_name.get() != initial
    gui._toggle_theme()
    assert gui.theme_name.get() == initial


def test_hotkey_bindings_present(gui) -> None:
    binds = " ".join(gui.root.bind())
    assert "Control-Key-r" in binds or "Control-Key-R" in binds or "<Control-r>" in binds
    assert "Control-Key-t" in binds or "Control-Key-T" in binds or "<Control-t>" in binds
    assert "Key-Escape" in binds or "<Escape>" in binds


def test_menu_bar_present(gui) -> None:
    cfg = gui.root.cget("menu")
    assert cfg  # menu attached


def test_queue_handler_emits() -> None:
    import logging
    import queue

    q: queue.Queue = queue.Queue()
    h = _QueueHandler(q)
    h.setFormatter(logging.Formatter("%(message)s"))
    # Agent tag detected when name OR message contains the agent class name
    rec = logging.LogRecord("x", logging.INFO, "x.py", 1, "SAM3Agent annotated img_1", None, None)
    h.emit(rec)
    level, agent, msg = q.get_nowait()
    assert level == "INFO"
    assert agent == "SAM3Agent"
    assert "SAM3Agent" in msg


def test_queue_handler_bounded() -> None:
    import logging
    import queue

    q: queue.Queue = queue.Queue(maxsize=3)
    h = _QueueHandler(q)
    h.setFormatter(logging.Formatter("%(message)s"))
    for i in range(10):
        rec = logging.LogRecord("x", logging.INFO, "x.py", 1, f"msg {i}", None, None)
        h.emit(rec)
    assert q.qsize() == 3  # bounded


# ---------- LabelerPanel ----------
@pytest.fixture
def labeler(root):
    panel = LabelerPanel(root, get_dataset_path=lambda: "", get_config_path=lambda: "")
    panel.pack()
    root.update_idletasks(); root.update()
    return panel


def test_labeler_builds(labeler) -> None:
    assert labeler.mode.get() == "rectangle"
    assert labeler.current_class.get() == "object"
    assert labeler.shapes == []


def test_labeler_mode_switching(labeler) -> None:
    for m in ("polygon", "edit", "rectangle"):
        labeler._set_mode(m)
        assert labeler.mode.get() == m


def test_labeler_add_class(labeler) -> None:
    labeler.classes.append("car")
    labeler.class_combo.configure(values=labeler.classes)
    labeler.current_class.set("car")
    assert "car" in labeler.classes


def test_shape_labelme_roundtrip() -> None:
    s = _Shape("polygon", "car", [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
    d = s.to_labelme()
    assert d["label"] == "car"
    assert d["shape_type"] == "polygon"
    s2 = _Shape.from_labelme(d)
    assert s2.label == s.label
    assert s2.points == s.points


def test_shape_rectangle_labelme() -> None:
    s = _Shape("rectangle", "tank", [(0.0, 0.0), (10.0, 20.0)])
    d = s.to_labelme()
    assert d["shape_type"] == "rectangle"
    assert len(d["points"]) == 2


def test_labeler_load_image_and_save_json(labeler, tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (200, 100), (40, 80, 120)).save(img_path)
    labeler.image_paths = [img_path]
    labeler.current_idx = 0
    labeler._load_image(img_path)
    assert labeler.current_image_path == img_path
    assert labeler._image_size == (200, 100)

    # Add a shape
    labeler.shapes.append(_Shape("rectangle", "car", [(10.0, 10.0), (50.0, 60.0)]))
    labeler._save_json()
    json_path = img_path.with_suffix(".json")
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["imagePath"] == "sample.jpg"
    assert data["imageWidth"] == 200
    assert data["imageHeight"] == 100
    assert len(data["shapes"]) == 1


def test_labeler_undo(labeler, tmp_path: Path) -> None:
    img = tmp_path / "u.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    labeler._snapshot()
    labeler.shapes.append(_Shape("rectangle", "x", [(0, 0), (10, 10)]))
    assert len(labeler.shapes) == 1
    labeler._undo()
    assert len(labeler.shapes) == 0


def test_labeler_delete_shape(labeler, tmp_path: Path) -> None:
    img = tmp_path / "d.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    labeler.shapes.append(_Shape("polygon", "y", [(0, 0), (1, 0), (1, 1)]))
    labeler._refresh_shape_list()
    labeler.shape_list.selection_set(0)
    labeler._delete_selected_shape()
    assert labeler.shapes == []


def test_labeler_zoom_clamping(labeler, tmp_path: Path) -> None:
    img = tmp_path / "z.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    for _ in range(50):
        labeler._zoom_by(2.0)
    assert labeler._zoom <= 15.0 + 1e-6
    for _ in range(50):
        labeler._zoom_by(0.5)
    assert labeler._zoom >= 0.2 - 1e-6


def test_labeler_load_existing_json(labeler, tmp_path: Path) -> None:
    img = tmp_path / "e.jpg"
    Image.new("RGB", (50, 50)).save(img)
    payload = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [{
            "label": "dog",
            "points": [[5.0, 5.0], [25.0, 25.0]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }],
        "imagePath": "e.jpg",
        "imageData": None,
        "imageHeight": 50,
        "imageWidth": 50,
    }
    img.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].label == "dog"
    assert "dog" in labeler.classes


def test_labeler_load_image_navigation(labeler, tmp_path: Path) -> None:
    for i in range(3):
        Image.new("RGB", (60, 60)).save(tmp_path / f"img_{i}.jpg")
    labeler._populate_files(tmp_path)
    assert len(labeler.image_paths) == 3
    assert labeler.current_idx == 0
    labeler._next_image()
    assert labeler.current_idx == 1
    labeler._next_image()
    assert labeler.current_idx == 2
    labeler._next_image()
    assert labeler.current_idx == 0  # wrap
    labeler._prev_image()
    assert labeler.current_idx == 2  # wrap back


def test_labeler_coord_roundtrip(labeler, tmp_path: Path) -> None:
    img = tmp_path / "c.jpg"
    Image.new("RGB", (400, 300)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.update_idletasks()
    labeler.canvas.update_idletasks()
    cx, cy = labeler._image_to_canvas(200, 150)
    ix, iy = labeler._canvas_to_image(int(cx), int(cy))
    assert abs(ix - 200) < 2
    assert abs(iy - 150) < 2


def test_labeler_finish_polygon_needs_3pts(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("polygon")

    labeler._draft_points = [(10, 10), (20, 20)]
    labeler._finish_polygon()
    assert labeler.shapes == []  # < 3 points, dropped

    labeler._draft_points = [(10, 10), (20, 20), (15, 5)]
    labeler._finish_polygon()
    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].kind == "polygon"


# ---------- AnnotatorGUI label viewer ----------
def test_results_filter_combobox_options(gui) -> None:
    # Should expose All / ACCEPTED / HUMAN_REVIEW
    assert gui.filter_var.get() == "All"


def test_save_settings_writes_file(gui, tmp_path: Path, monkeypatch) -> None:
    fake_settings = tmp_path / "settings.json"
    monkeypatch.setattr("src.ui.review_app.SETTINGS_FILE", fake_settings)
    gui._on_close()
    # _on_close destroys root; recreate for fixture teardown safety
    try:
        assert fake_settings.exists()
        data = json.loads(fake_settings.read_text(encoding="utf-8"))
        assert "theme" in data and "workers" in data
    finally:
        try:
            gui.root.destroy()
        except Exception:
            pass


def test_polygon_classes_in_color_map() -> None:
    # color_for is deterministic and returns from palette
    from src.ui._helpers import PALETTE
    for cls in ("car", "tank", "dog"):
        assert color_for(cls) in PALETTE


def test_load_and_accept_results_flow(gui, tmp_path: Path, monkeypatch) -> None:
    # Set up folders
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Create dummy image
    img_path = dataset_dir / "test_frame.jpg"
    img_path.write_bytes(b"")

    # Create conversation_logs.json
    logs_data = [{
        "image_id": "img_00000",
        "status": "HUMAN_REVIEW",
        "retry_count": 1,
        "qa_result": None,
        "history": [
            {
                "sender": "agent",
                "role": "agent",
                "content": "Requested annotation",
                "actions": [{"type": "REQUEST_ANNOTATION", "classes": ["box"]}],
                "timestamp": "2026-06-15T10:00:00Z"
            },
            {
                "sender": "agent",
                "role": "agent",
                "content": "Result",
                "actions": [{
                    "type": "ANNOTATION_RESULT",
                    "masks": [{
                        "mask_id": "mask-123",
                        "image_id": "img_00000",
                        "class_id": "box",
                        "bbox": [10, 20, 30, 40],
                        "polygon": [[10, 20], [40, 20], [40, 60], [10, 60]],
                        "area": 1200.0,
                        "confidence": 0.95
                    }]
                }],
                "timestamp": "2026-06-15T10:01:00Z"
            }
        ]
    }]
    logs_file = out_dir / "conversation_logs.json"
    logs_file.write_text(json.dumps(logs_data), encoding="utf-8")

    # Set GUI fields
    gui.dataset_path.set(str(dataset_dir))
    gui.output_path.set(str(out_dir))

    # Mock messagebox to avoid popups during test
    mock_shown = []
    def mock_showinfo(title, msg):
        mock_shown.append(("info", title, msg))
    def mock_showwarning(title, msg):
        mock_shown.append(("warning", title, msg))
    def mock_showerror(title, msg):
        mock_shown.append(("error", title, msg))

    monkeypatch.setattr("tkinter.messagebox.showinfo", mock_showinfo)
    monkeypatch.setattr("tkinter.messagebox.showwarning", mock_showwarning)
    monkeypatch.setattr("tkinter.messagebox.showerror", mock_showerror)

    # 1. Load results
    gui._load_previous_results()

    assert gui.last_bundles is not None
    assert len(gui.last_bundles) == 1
    assert gui.last_bundles[0].status == "HUMAN_REVIEW"
    assert gui._bundle_status["test_frame"] == "HUMAN_REVIEW"
    assert gui._classes == ["box"]

    # 2. Verify listbox populates and selects
    assert gui.labels_listbox.size() == 1
    assert "test_frame" in gui.labels_listbox.get(0)
    gui.labels_listbox.selection_set(0)

    # Trigger select callback manually
    gui._on_label_select()
    # Accept button should be enabled since status is HUMAN_REVIEW
    assert str(gui.accept_btn.cget("state")) == "normal"

    # 3. Accept annotation
    gui._accept_selected_annotation()

    # Should update status to ACCEPTED
    assert gui.last_bundles[0].status == "ACCEPTED"
    assert gui._bundle_status["test_frame"] == "ACCEPTED"

    # Should rewrite logs
    logs_text = logs_file.read_text(encoding="utf-8")
    updated_logs = json.loads(logs_text)
    assert updated_logs[0]["status"] == "ACCEPTED"

    # Check that YOLO label file was updated/written
    yolo_file = out_dir / "yolo_seg_labels" / "test_frame.txt"
    assert yolo_file.exists()
    assert yolo_file.read_text(encoding="utf-8").strip() != ""

    # Check that LabelMe JSON was written
    labelme_file = out_dir / "labelme_json" / "test_frame.json"
    assert labelme_file.exists()
    lm_data = json.loads(labelme_file.read_text(encoding="utf-8"))
    assert lm_data["version"] == "5.4.1"
    assert len(lm_data["shapes"]) == 1
    assert lm_data["shapes"][0]["label"] == "box"
    assert lm_data["shapes"][0]["shape_type"] == "polygon"
    assert lm_data["imagePath"] == "test_frame.jpg"

    # Check that image files are co-located next to label files in output
    yolo_img = out_dir / "yolo_seg_labels" / "test_frame.jpg"
    labelme_img = out_dir / "labelme_json" / "test_frame.jpg"
    assert yolo_img.exists()
    assert labelme_img.exists()

    # Simulate deleting the source image from dataset, and verify label/copied image are deleted on next sync
    img_path.unlink()
    from src.tools.labelme import export_labelme
    export_labelme(gui.last_bundles, out_dir, force_all=True)
    
    # The label JSON file and the copied image should now be deleted because the source image was deleted
    assert not labelme_file.exists()
    assert not labelme_img.exists()


def test_load_and_accept_all_warnings_flow(gui, tmp_path: Path, monkeypatch) -> None:
    # Set up folders
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Create dummy image
    img_path = dataset_dir / "test_frame.jpg"
    img_path.write_bytes(b"")

    # Create conversation_logs.json
    logs_data = [{
        "image_id": "img_00000",
        "status": "HUMAN_REVIEW",
        "retry_count": 1,
        "qa_result": None,
        "history": [
            {
                "sender": "agent",
                "role": "agent",
                "content": "Requested annotation",
                "actions": [{"type": "REQUEST_ANNOTATION", "classes": ["box"]}],
                "timestamp": "2026-06-15T10:00:00Z"
            },
            {
                "sender": "agent",
                "role": "agent",
                "content": "Result",
                "actions": [{
                    "type": "ANNOTATION_RESULT",
                    "masks": [{
                        "mask_id": "mask-123",
                        "image_id": "img_00000",
                        "class_id": "box",
                        "bbox": [10, 20, 30, 40],
                        "polygon": [[10, 20], [40, 20], [40, 60], [10, 60]],
                        "area": 1200.0,
                        "confidence": 0.95
                    }]
                }],
                "timestamp": "2026-06-15T10:01:00Z"
            }
        ]
    }]
    logs_file = out_dir / "conversation_logs.json"
    logs_file.write_text(json.dumps(logs_data), encoding="utf-8")

    # Set GUI fields
    gui.dataset_path.set(str(dataset_dir))
    gui.output_path.set(str(out_dir))

    # Mock messagebox to avoid popups during test
    mock_shown = []
    def mock_showinfo(title, msg):
        mock_shown.append(("info", title, msg))
    def mock_showwarning(title, msg):
        mock_shown.append(("warning", title, msg))
    def mock_showerror(title, msg):
        mock_shown.append(("error", title, msg))
    def mock_askyesno(title, msg):
        mock_shown.append(("askyesno", title, msg))
        return True

    monkeypatch.setattr("tkinter.messagebox.showinfo", mock_showinfo)
    monkeypatch.setattr("tkinter.messagebox.showwarning", mock_showwarning)
    monkeypatch.setattr("tkinter.messagebox.showerror", mock_showerror)
    monkeypatch.setattr("tkinter.messagebox.askyesno", mock_askyesno)

    # 1. Load results
    gui._load_previous_results()

    assert gui.last_bundles is not None
    assert len(gui.last_bundles) == 1
    assert gui.last_bundles[0].status == "HUMAN_REVIEW"
    assert gui._bundle_status["test_frame"] == "HUMAN_REVIEW"
    assert gui._classes == ["box"]

    # Accept All button should be enabled since status has HUMAN_REVIEW
    assert str(gui.accept_all_btn.cget("state")) == "normal"

    # 2. Bulk Accept annotation
    gui._accept_all_warnings()

    # Should update status to ACCEPTED
    assert gui.last_bundles[0].status == "ACCEPTED"
    assert gui._bundle_status["test_frame"] == "ACCEPTED"

    # Accept All button should now be disabled since no more warnings
    assert str(gui.accept_all_btn.cget("state")) == "disabled"

    # Should rewrite logs
    logs_text = logs_file.read_text(encoding="utf-8")
    updated_logs = json.loads(logs_text)
    assert updated_logs[0]["status"] == "ACCEPTED"

    # Check that YOLO label file was updated/written
    yolo_file = out_dir / "yolo_seg_labels" / "test_frame.txt"
    assert yolo_file.exists()
    assert yolo_file.read_text(encoding="utf-8").strip() != ""

    # Check that LabelMe JSON was written
    labelme_file = out_dir / "labelme_json" / "test_frame.json"
    assert labelme_file.exists()
    lm_data = json.loads(labelme_file.read_text(encoding="utf-8"))
    assert lm_data["version"] == "5.4.1"
    assert len(lm_data["shapes"]) == 1
    assert lm_data["shapes"][0]["label"] == "box"
    assert lm_data["shapes"][0]["shape_type"] == "polygon"
    assert lm_data["imagePath"] == "test_frame.jpg"


def test_accept_all_refreshes_qa_report_and_dashboard(gui, tmp_path: Path, monkeypatch) -> None:
    """Accepting HUMAN_REVIEW images must not leave qa_report.json -- and
    therefore the Dashboard tab -- showing the pre-accept counts forever."""
    from src.core.models import AnnotationBundle, ImageRecord, MaskRecord

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    img = dataset_dir / "a.jpg"
    Image.new("RGB", (50, 50)).save(img)

    (out_dir / "qa_report.json").write_text(json.dumps({
        "total_images": 1, "status_counts": {"HUMAN_REVIEW": 1}, "accept_rate": 0.0,
        "retries_total": 1, "accepted_after_retry": 0, "human_review_image_ids": ["img_00000"],
        "per_class_mask_counts": {"car": 0}, "avg_mask_confidence": 0.0, "top_issues": [],
    }), encoding="utf-8")

    gui.dataset_path.set(str(dataset_dir))
    gui.output_path.set(str(out_dir))
    gui.last_output_path = out_dir
    rec = ImageRecord(id="img_00000", path=img, width=50, height=50)
    mask = MaskRecord(mask_id="m1", image_id="img_00000", class_id="car", bbox=(1, 1, 5, 5), area=25, confidence=0.9)
    gui.last_bundles = [AnnotationBundle(image=rec, masks=[mask], status="HUMAN_REVIEW")]
    gui._bundle_status = {"a": "HUMAN_REVIEW"}

    gui._refresh_dashboard()
    assert gui.dash_card_vars["human_review"].get() == "1"

    monkeypatch.setattr("tkinter.messagebox.askyesno", lambda *a, **k: True)
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *a, **k: None)
    gui._accept_all_warnings()

    assert gui.dash_card_vars["human_review"].get() == "0", "Dashboard must refresh, not stay stale"
    assert gui.dash_card_vars["accept_rate"].get() == "100%"
    on_disk = json.loads((out_dir / "qa_report.json").read_text())
    assert on_disk["status_counts"] == {"ACCEPTED": 1}


def test_load_results_fallback_yolo_segmentation_detection(gui, tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Case A: yolo_labels directory exists (bounding-box format)
    (out_dir / "yolo_labels").mkdir()
    
    logs_file = out_dir / "conversation_logs.json"
    logs_file.write_text(json.dumps([]), encoding="utf-8")
    
    gui.dataset_path.set(str(dataset_dir))
    gui.output_path.set(str(out_dir))
    gui.config_path.set("nonexistent_config.yaml") # force fallback config
    
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda a, b: None)
    
    gui._load_previous_results()
    assert gui.last_config is not None
    assert gui.last_config.yolo_segmentation is False

    # Case B: yolo_seg_labels directory exists
    (out_dir / "yolo_labels").rmdir()
    (out_dir / "yolo_seg_labels").mkdir()
    
    gui._load_previous_results()
    assert gui.last_config.yolo_segmentation is True


def test_gui_custom_prompts_blank_class_label(gui, tmp_path: Path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    # Create dummy image
    img = dataset_dir / "a.jpg"
    img.write_bytes(b"")
    
    # Write a basic config
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        f"project_name: t\n"
        f"dataset_path: {dataset_dir.as_posix()}\n"
        f"output_path: {out_dir.as_posix()}\n"
        "label_schema: [person, car]\n"
        "sam3: {model_name: sam3_b, backend: mock}\n"
        "qa: {max_retries: 2, min_mask_area: 0.0, max_mask_area: 0.95, "
        "iou_threshold: 0.95, confidence_threshold: 0.0}\n"
        "llm: {}\n"
        "human_review: {}\n"
        "max_workers: 1\n",
        encoding="utf-8",
    )
    
    gui.config_path.set(str(cfg_file))
    gui.dataset_path.set(str(dataset_dir))
    gui.output_path.set(str(out_dir))
    
    # Empty Class Label but custom SAM3 prompt
    gui.class_label_var.set("")
    gui.sam3_prompt_var.set("police officer, red vehicle")
    
    # Run _do_run directly on current thread
    gui._do_run()
    
    # Verify custom prompts mapped to default classes
    assert gui.last_config.label_schema == ["person", "car"]
    assert gui.last_config.per_class_prompt["person"] == "police officer"
    assert gui.last_config.per_class_prompt["car"] == "red vehicle"



# ---------- Dashboard tab ----------
def test_dashboard_tab_present(gui) -> None:
    assert hasattr(gui, "dashboard_tab")
    assert hasattr(gui, "dash_card_vars")
    assert "total" in gui.dash_card_vars
    assert "accept_rate" in gui.dash_card_vars


def test_dashboard_refresh_no_output(gui) -> None:
    gui.last_output_path = None
    gui.output_path.set("")
    gui._refresh_dashboard()
    # All cards should reset to placeholder
    for v in gui.dash_card_vars.values():
        assert v.get() == "—"


def test_dashboard_refresh_with_qa_report(gui, tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "qa_report.json").write_text(json.dumps({
        "total_images": 10,
        "status_counts": {"ACCEPTED": 8, "HUMAN_REVIEW": 2},
        "accept_rate": 0.8,
        "retries_total": 3,
        "avg_mask_confidence": 0.74,
        "per_class_mask_counts": {"car": 12, "person": 5, "dog": 2},
        "top_issues": [["car too small", 4], ["person low confidence", 2]],
    }), encoding="utf-8")
    gui.last_output_path = out
    gui._refresh_dashboard()
    assert gui.dash_card_vars["total"].get() == "10"
    assert gui.dash_card_vars["accepted"].get() == "8"
    assert gui.dash_card_vars["human_review"].get() == "2"
    assert gui.dash_card_vars["accept_rate"].get() == "80%"
    assert gui.dash_card_vars["avg_conf"].get() == "0.74"
    assert gui.dash_card_vars["retries"].get() == "3"
    chart_text = gui.dash_chart.get("1.0", "end")
    assert "car" in chart_text and "person" in chart_text
    issues_text = gui.dash_issues.get("1.0", "end")
    assert "car too small" in issues_text


# ---------- Global toolbar ----------
def test_global_toolbar_buttons_present(gui) -> None:
    assert hasattr(gui, "tb_run_btn")
    assert hasattr(gui, "tb_cancel_btn")
    assert str(gui.tb_run_btn["state"]) == "normal"
    assert str(gui.tb_cancel_btn["state"]) == "disabled"


def test_set_running_toggles_both_toolbars(gui) -> None:
    gui._set_running(True)
    assert str(gui.run_btn["state"]) == "disabled"
    assert str(gui.cancel_btn["state"]) == "normal"
    assert str(gui.tb_run_btn["state"]) == "disabled"
    assert str(gui.tb_cancel_btn["state"]) == "normal"
    gui._set_running(False)
    assert str(gui.run_btn["state"]) == "normal"
    assert str(gui.cancel_btn["state"]) == "disabled"
    assert str(gui.tb_run_btn["state"]) == "normal"
    assert str(gui.tb_cancel_btn["state"]) == "disabled"


# ---------- _resolve_output_dir fallback chain ----------
def test_resolve_output_dir_prefers_last(gui, tmp_path: Path) -> None:
    p = tmp_path / "out"
    p.mkdir()
    gui.last_output_path = p
    gui.output_path.set("")
    assert gui._resolve_output_dir() == p


def test_resolve_output_dir_falls_back_to_field(gui, tmp_path: Path) -> None:
    p = tmp_path / "field_out"
    p.mkdir()
    gui.last_output_path = None
    gui.output_path.set(str(p))
    assert gui._resolve_output_dir() == p


def test_resolve_output_dir_none_when_nothing(gui) -> None:
    gui.last_output_path = None
    gui.output_path.set("")
    assert gui._resolve_output_dir() is None


# ---------- GPU widget ----------
def test_gpu_widget_string_set(gui) -> None:
    # Should be either GPU info or "CPU only" — never empty after first refresh
    val = gui.gpu_var.get()
    assert val == "CPU only" or val.startswith("GPU") or val == ""


# ---------- Hotkey help dialog ----------
def test_hotkey_help_opens_and_closes(gui) -> None:
    gui._show_hotkey_help()
    gui.root.update()
    # Find the Toplevel just created
    tops = [w for w in gui.root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert tops, "help dialog Toplevel not found"
    tops[-1].destroy()


def test_clear_session_resets_state(gui, tmp_path: Path) -> None:
    # Populate some state
    gui.dataset_path.set(str(tmp_path))
    gui.output_path.set(str(tmp_path / "out"))
    gui.class_label_var.set("car, truck")
    gui.sam3_prompt_var.set("vehicle")
    gui.last_output_path = tmp_path
    gui._bundle_status = {"img_a": "ACCEPTED"}
    gui._label_files = {"img_a": tmp_path / "img_a.txt"}
    gui.dash_card_vars["total"].set("42")
    gui.preview_text.insert("end", "junk")
    gui.log_text.insert("end", "junk")
    gui.summary_text.insert("end", "junk")

    # Confirm-skip
    gui._clear_session(confirm=False)
    gui.root.update_idletasks()

    assert gui.dataset_path.get() == ""
    assert gui.output_path.get() == ""
    assert gui.class_label_var.get() == ""
    assert gui.sam3_prompt_var.get() == ""
    assert gui.last_output_path is None
    assert gui._bundle_status == {}
    assert gui._label_files == {}
    assert gui.dash_card_vars["total"].get() == "—"
    assert gui.preview_text.get("1.0", "end").strip() == ""
    assert gui.log_text.get("1.0", "end").strip() == ""
    assert gui.summary_text.get("1.0", "end").strip() == ""
    assert gui.status_var.get() == "Session cleared."


def test_labeler_middle_button_pan(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    class FakeEvent:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    labeler._on_pan_start(FakeEvent(10, 10))
    assert labeler._dragging_canvas is True
    assert labeler._drag_anchor == (10, 10, 0, 0)

    labeler._on_pan_drag(FakeEvent(25, 30))
    assert labeler._pan == [15, 20]

    labeler._on_pan_end(FakeEvent(25, 30))
    assert labeler._dragging_canvas is False


def test_labeler_autosave_toggle(labeler, tmp_path: Path) -> None:
    from src.ui.labeler import _Shape
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (100, 100)).save(img1)
    Image.new("RGB", (100, 100)).save(img2)
    labeler.image_paths = [img1, img2]
    labeler.current_idx = 0
    labeler._load_image(img1)

    labeler.shapes = [_Shape("rectangle", "test", [(10, 10), (20, 20)])]
    labeler.dirty = True

    assert labeler.autosave_var.get() is True
    assert labeler._confirm_unsaved() is True
    saved_json = tmp_path / "img1.json"
    assert saved_json.exists()


def test_labeler_insert_vertex(labeler, tmp_path: Path) -> None:
    from src.ui.labeler import _Shape
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("edit")

    labeler.shapes = [_Shape("polygon", "box", [(10, 10), (50, 10), (50, 50), (10, 50)])]
    cx, cy = labeler._image_to_canvas(30, 10)

    class FakeEvent:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    labeler._on_double_click(FakeEvent(cx, cy))
    assert len(labeler.shapes[0].points) == 5
    assert labeler.shapes[0].points[1] == (30.0, 10.0)


def test_labeler_delete_vertex(labeler, tmp_path: Path) -> None:
    from src.ui.labeler import _Shape
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("edit")

    labeler.shapes = [_Shape("polygon", "box", [(10, 10), (50, 10), (50, 50), (10, 50)])]
    cx, cy = labeler._image_to_canvas(50, 50)

    class FakeEvent:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    labeler._on_right_click(FakeEvent(cx, cy))
    assert len(labeler.shapes[0].points) == 3
    assert (50.0, 50.0) not in labeler.shapes[0].points

    cx2, cy2 = labeler._image_to_canvas(10, 10)
    labeler._on_right_click(FakeEvent(cx2, cy2))
    assert len(labeler.shapes[0].points) == 3


def test_labeler_labelme_navigation_hotkeys(labeler, tmp_path: Path) -> None:
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (100, 100)).save(img1)
    Image.new("RGB", (100, 100)).save(img2)
    labeler.image_paths = [img1, img2]
    labeler.current_idx = 0
    labeler._load_image(img1)

    binds = labeler.canvas.bind()
    assert "a" in binds
    assert "d" in binds

    labeler._next_image()
    assert labeler.current_idx == 1

    labeler._prev_image()
    assert labeler.current_idx == 0


def test_labeler_labelme_tool_hotkeys(labeler) -> None:
    binds = " ".join(labeler.canvas.bind())
    assert "Control-Key-n" in binds or "Control-Key-N" in binds or "<Control-n>" in binds
    assert "Control-Key-r" in binds or "Control-Key-R" in binds or "<Control-r>" in binds
    assert "Control-Key-e" in binds or "Control-Key-E" in binds or "<Control-e>" in binds


def test_labeler_polygon_start_vertex_closure(labeler, tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("polygon")

    labeler._draft_points = [(10, 10), (50, 10), (50, 50)]
    cx0, cy0 = labeler._image_to_canvas(10, 10)

    class FakeEvent:
        def __init__(self, x, y, state=0):
            self.x = x
            self.y = y
            self.state = state

    monkeypatch.setattr(labeler, "_prompt_label", lambda default: "triangle")

    labeler._on_click(FakeEvent(cx0, cy0))
    
    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].kind == "polygon"
    assert labeler.shapes[0].label == "triangle"
    assert len(labeler._draft_points) == 0


def test_labeler_label_prompt_dialog(labeler, tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    monkeypatch.setattr(labeler, "_prompt_label", lambda default_label: "custom_label")
    
    labeler.mode.set("rectangle")
    sx, sy = labeler._image_to_canvas(10, 10)
    ex, ey = labeler._image_to_canvas(50, 50)
    
    class FakeEvent:
        def __init__(self, x, y, state=0):
            self.x = x
            self.y = y
            self.state = state

    labeler._on_click(FakeEvent(sx, sy))
    labeler._on_release(FakeEvent(ex, ey))

    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].label == "custom_label"
    assert "custom_label" in labeler.classes


def test_fewshot_scan_ignores_output_dir_pollution(gui, tmp_path: Path) -> None:
    """Scanning refs/targets must not re-ingest the pipeline's own prior output.

    Reproduces the layout a real run leaves when output_path sits inside (or
    equals) dataset_path: the exporters copy every annotated image next to its
    label, and the LabelMe exporter writes a JSON beside that copy. Without
    filtering, a target scan would offer those copies as extra images to
    annotate, and a ref scan would pick the pipeline's own predictions back up
    as if a human had drawn them in the Labeler.
    """
    dataset_dir = tmp_path / "cars"
    dataset_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (40, 40)).save(dataset_dir / name)

    # A genuine Labeler-authored reference directly in the dataset root.
    (dataset_dir / "c.json").write_text(json.dumps({
        "version": "5.4.1", "flags": {}, "imagePath": "c.jpg",
        "imageHeight": 40, "imageWidth": 40,
        "shapes": [{
            "label": "car", "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
            "group_id": None, "shape_type": "polygon", "flags": {},
        }],
    }), encoding="utf-8")

    # Simulate a prior pipeline run with output_path == dataset_path: the
    # exporters copy 'a' next to a JSON of their own, unrelated to the human's.
    export_dir = dataset_dir / "labelme_json"
    export_dir.mkdir()
    Image.new("RGB", (40, 40)).save(export_dir / "a.jpg")
    (export_dir / "a.json").write_text(json.dumps({
        "version": "5.4.1", "flags": {}, "imagePath": "a.jpg",
        "imageHeight": 40, "imageWidth": 40,
        "shapes": [{
            "label": "car", "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
            "group_id": None, "shape_type": "polygon", "flags": {},
        }],
    }), encoding="utf-8")

    gui.dataset_path.set(str(dataset_dir))
    gui.output_path.set(str(dataset_dir))
    gui._fs_class_var.set("object")

    gui._fewshot_scan_refs()

    ref_names = {p.name for p, _ in gui._fs_refs}
    assert ref_names == {"c.jpg"}, f"pipeline's own export leaked into refs: {ref_names}"

    target_names = sorted(p.name for p in gui._fs_targets)
    assert target_names == ["a.jpg", "b.jpg"], f"export copy leaked into targets: {target_names}"


def test_labeler_load_paths_bridges_prefiltered_list(labeler, tmp_path: Path) -> None:
    """load_paths must show exactly the given images, in name order, with a note."""
    a = tmp_path / "a.jpg"; b = tmp_path / "b.jpg"; c = tmp_path / "c.jpg"
    for p in (a, b, c):
        Image.new("RGB", (20, 20)).save(p)

    labeler.load_paths([c, a], note="Fixing 2 image(s) that need human review.")

    assert [p.name for p in labeler.image_paths] == ["a.jpg", "c.jpg"]
    assert "b.jpg" not in [p.name for p in labeler.image_paths]
    assert labeler.current_image_path == a
    assert labeler.status_var.get() == "Fixing 2 image(s) that need human review."
    assert labeler.progress_var.get() == "labeled 0/2"


def test_labeler_skip_labeled_navigation(labeler, tmp_path: Path) -> None:
    """With Skip labeled on, Next must jump over already-annotated images."""
    a = tmp_path / "a.jpg"; b = tmp_path / "b.jpg"; c = tmp_path / "c.jpg"
    for p in (a, b, c):
        Image.new("RGB", (20, 20)).save(p)
    # b already has a saved annotation; a and c do not
    b.with_suffix(".json").write_text(json.dumps({
        "version": "5.4.1", "flags": {}, "shapes": [], "imagePath": "b.jpg",
        "imageData": None, "imageHeight": 20, "imageWidth": 20,
    }), encoding="utf-8")

    labeler.load_paths([a, b, c])
    assert labeler.progress_var.get() == "labeled 1/3"
    assert labeler.current_image_path == a

    labeler.skip_labeled_var.set(True)
    labeler._next_image()
    assert labeler.current_image_path == c, "should skip already-labeled b.jpg"

    labeler._next_image()
    assert labeler.current_image_path == a, "wraps back to the only other unlabeled image"


def test_labeler_next_image_updates_progress_after_save(labeler, tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"; b = tmp_path / "b.jpg"
    for p in (a, b):
        Image.new("RGB", (20, 20)).save(p)

    labeler.load_paths([a, b])
    labeler.mode.set("rectangle")
    sx, sy = labeler._image_to_canvas(2, 2)
    ex, ey = labeler._image_to_canvas(10, 10)

    class FakeEvent:
        def __init__(self, x, y):
            self.x, self.y, self.state = x, y, 0

    labeler._on_click(FakeEvent(sx, sy))
    labeler._on_release(FakeEvent(ex, ey))
    assert labeler.dirty is True

    labeler._save_json()
    assert labeler.progress_var.get() == "labeled 1/2"


def test_gui_fix_in_labeler_button_sends_human_review_images(gui, tmp_path: Path, monkeypatch) -> None:
    from src.core.models import AnnotationBundle, ImageRecord

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    ok_img = dataset_dir / "ok.jpg"
    bad_img = dataset_dir / "bad.jpg"
    for p in (ok_img, bad_img):
        Image.new("RGB", (20, 20)).save(p)

    gui.last_bundles = [
        AnnotationBundle(
            image=ImageRecord(id="i0", path=ok_img, width=20, height=20),
            status="ACCEPTED",
        ),
        AnnotationBundle(
            image=ImageRecord(id="i1", path=bad_img, width=20, height=20),
            status="HUMAN_REVIEW",
        ),
    ]
    gui._update_accept_all_btn_state()
    assert str(gui.fix_in_labeler_btn["state"]) == "normal"

    gui._send_human_review_to_labeler()

    assert [p.name for p in gui.labeler_panel.image_paths] == ["bad.jpg"]
    assert gui.notebook.select() == str(gui.labeler_tab)


# ---------- LabelerPanel: remaining functionality coverage ----------

class _SyncThread:
    """Runs the target immediately instead of on a background thread, so a
    test can call root.update() right after to flush the self.after(0, ...)
    callback the worker schedules -- no sleeps, no races."""
    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _fake_event(x, y, state=0):
    class E:
        pass
    e = E()
    e.x, e.y, e.state = x, y, state
    return e


def test_labeler_edit_mode_drag_vertex_moves_polygon_point(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("edit")
    labeler.shapes = [_Shape("polygon", "box", [(10, 10), (50, 10), (50, 50), (10, 50)])]

    cx, cy = labeler._image_to_canvas(10, 10)
    labeler._on_click(_fake_event(cx, cy))  # picks the shape (no vertex selected yet)
    assert labeler._sel_shape_idx == 0
    labeler._on_click(_fake_event(cx, cy))  # now selects that vertex and starts dragging
    assert labeler._sel_vertex_idx == 0
    assert labeler._dragging_vertex is True

    nx, ny = labeler._image_to_canvas(25, 30)
    labeler._on_drag(_fake_event(nx, ny))
    assert labeler.shapes[0].points[0] == (25.0, 30.0)

    labeler._on_release(_fake_event(nx, ny))
    assert labeler._dragging_vertex is False
    assert labeler.dirty is True


def test_labeler_edit_mode_drag_rectangle_corner_keeps_opposite_fixed(labeler, tmp_path: Path) -> None:
    img = tmp_path / "r.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("edit")
    labeler.shapes = [_Shape("rectangle", "box", [(10, 10), (60, 60)])]

    cx, cy = labeler._image_to_canvas(10, 10)
    labeler._on_click(_fake_event(cx, cy))  # pick shape
    labeler._on_click(_fake_event(cx, cy))  # select top-left corner, begin drag

    nx, ny = labeler._image_to_canvas(0, 0)
    labeler._on_drag(_fake_event(nx, ny))
    assert labeler.shapes[0].points[0] == (0.0, 0.0)
    assert labeler.shapes[0].points[1] == (60.0, 60.0), "opposite corner must stay put"


def test_labeler_escape_cancels_draft(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("polygon")

    labeler._draft_points = [(10, 10), (20, 20)]
    labeler._cancel_draft()
    assert labeler._draft_points == []
    assert labeler.shapes == []


def test_labeler_rectangle_micro_drag_creates_nothing(labeler, tmp_path: Path) -> None:
    """A near-zero drag (misclick) must not create a degenerate rectangle."""
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("rectangle")

    sx, sy = labeler._image_to_canvas(30, 30)
    labeler._on_click(_fake_event(sx, sy))
    labeler._on_release(_fake_event(sx + 1, sy + 1))
    assert labeler.shapes == []


def test_labeler_wheel_zoom_in_and_out(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    start = labeler._zoom

    class WheelUp:
        delta = 120
    labeler._on_wheel(WheelUp())
    assert labeler._zoom > start

    zoomed = labeler._zoom
    class WheelDown:
        delta = -120
    labeler._on_wheel(WheelDown())
    assert labeler._zoom < zoomed


def test_labeler_fit_view_resets_zoom_and_pan(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    labeler._zoom_by(2.0)
    labeler._pan = [40, -15]
    labeler._fit_view()
    assert labeler._zoom == 1.0
    assert labeler._pan == [0, 0]


def test_labeler_label_list_pick_sets_current_class(labeler) -> None:
    labeler.classes = ["object", "car", "dog"]
    labeler._refresh_label_list()
    labeler.label_list.selection_set(1)
    labeler._on_label_pick()
    assert labeler.current_class.get() == "car"


def test_labeler_apply_theme_no_crash(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.apply_theme({
        "panel": "#222222", "fg": "#eeeeee", "accent": "#4477ff", "preview_bg": "#111111",
    })
    assert str(labeler.canvas.cget("bg")) in ("#111111",)


def test_labeler_confirm_unsaved_dialog_paths(labeler, tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.autosave_var.set(False)
    labeler.dirty = True

    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda t, m: None)
    assert labeler._confirm_unsaved() is False, "Cancel must block navigation"
    assert not (tmp_path / "p.json").exists()

    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda t, m: False)
    assert labeler._confirm_unsaved() is True, "No must allow navigation without saving"
    assert not (tmp_path / "p.json").exists()

    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda t, m: True)
    assert labeler._confirm_unsaved() is True
    assert (tmp_path / "p.json").exists(), "Yes must save before navigating"


def test_labeler_ctrl_drag_pans_canvas(labeler, tmp_path: Path) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("rectangle")

    labeler._on_click(_fake_event(10, 10, state=0x0004))  # Ctrl held
    assert labeler._dragging_canvas is True
    labeler._on_drag(_fake_event(25, 40, state=0x0004))
    assert labeler._pan == [15, 30]
    assert labeler.shapes == [], "Ctrl-drag must pan, not draw a rectangle"


def test_labeler_delete_key_bound(labeler) -> None:
    binds = " ".join(labeler.canvas.bind())
    assert "Delete" in binds


def test_labeler_populate_files_badges_existing_json(labeler, tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"; b = tmp_path / "b.jpg"
    Image.new("RGB", (20, 20)).save(a)
    Image.new("RGB", (20, 20)).save(b)
    a.with_suffix(".json").write_text(json.dumps({
        "version": "5.4.1", "flags": {}, "shapes": [], "imagePath": "a.jpg",
        "imageData": None, "imageHeight": 20, "imageWidth": 20,
    }), encoding="utf-8")

    labeler._populate_files(tmp_path)
    entries = [labeler.file_list.get(i) for i in range(labeler.file_list.size())]
    assert entries[0].startswith("● ")
    assert entries[1].startswith("  ")
    assert labeler.progress_var.get() == "labeled 1/2"


def test_labeler_right_click_finishes_polygon_in_draw_mode(labeler, tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.mode.set("polygon")
    monkeypatch.setattr(labeler, "_prompt_label", lambda default: "blob")

    labeler._draft_points = [(10, 10), (50, 10), (50, 50)]
    labeler._on_right_click(None)
    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].label == "blob"


def test_labeler_auto_annotate_adds_shapes_from_sam3(labeler, tmp_path: Path, monkeypatch) -> None:
    from src.tools.sam3.client import RawMask
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.current_class.set("car")

    monkeypatch.setattr("src.ui.labeler.threading.Thread", _SyncThread)
    fake = RawMask(polygon=[(1, 1), (5, 1), (5, 5)], bbox=(1, 1, 4, 4), area=16.0, confidence=0.9)
    monkeypatch.setattr("src.tools.sam3.sam3_segment_text_prompt", lambda **kw: [fake])

    labeler._auto_annotate()
    labeler.update()

    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].label == "car"
    assert labeler.dirty is True


def test_labeler_auto_annotate_no_result_leaves_shapes_empty(labeler, tmp_path: Path, monkeypatch) -> None:
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)

    monkeypatch.setattr("src.ui.labeler.threading.Thread", _SyncThread)
    monkeypatch.setattr("src.tools.sam3.sam3_segment_text_prompt", lambda **kw: [])

    labeler._auto_annotate()
    labeler.update()

    assert labeler.shapes == []
    assert "nothing" in labeler.status_var.get().lower()


def test_labeler_sam_box_assist_adds_shape(labeler, tmp_path: Path, monkeypatch) -> None:
    from src.tools.sam3.client import RawMask
    img = tmp_path / "p.jpg"
    Image.new("RGB", (100, 100)).save(img)
    labeler.image_paths = [img]
    labeler.current_idx = 0
    labeler._load_image(img)
    labeler.current_class.set("car")

    monkeypatch.setattr("src.ui.labeler.threading.Thread", _SyncThread)
    fake = RawMask(polygon=[], bbox=(5, 5, 20, 20), area=400.0, confidence=0.8)
    monkeypatch.setattr("src.tools.sam3.sam3_segment_box_prompt", lambda **kw: [fake])

    labeler._run_sam_box(5, 5, 25, 25)
    labeler.update()

    assert len(labeler.shapes) == 1
    assert labeler.shapes[0].kind == "rectangle"
    assert labeler.shapes[0].points == [(5.0, 5.0), (25.0, 25.0)]


def test_labeler_sam_box_result_discarded_if_image_changed(labeler, tmp_path: Path, monkeypatch) -> None:
    """A slow SAM3 call must not paint shapes onto whatever image is open when
    it finally returns -- the user may have already moved on to the next one."""
    from src.tools.sam3.client import RawMask
    img1 = tmp_path / "a.jpg"; img2 = tmp_path / "b.jpg"
    Image.new("RGB", (100, 100)).save(img1)
    Image.new("RGB", (100, 100)).save(img2)
    labeler.image_paths = [img1, img2]
    labeler.current_idx = 0
    labeler._load_image(img1)

    monkeypatch.setattr("src.ui.labeler.threading.Thread", _SyncThread)
    fake = RawMask(polygon=[], bbox=(1, 1, 4, 4), area=16.0, confidence=0.9)

    def _slow_call(**kw):
        labeler.current_idx = 1
        labeler._load_image(img2)  # simulate user navigating away mid-request
        return [fake]

    monkeypatch.setattr("src.tools.sam3.sam3_segment_box_prompt", _slow_call)
    labeler._run_sam_box(1, 1, 5, 5)
    labeler.update()

    assert labeler.shapes == [], "result for the old image must not land on the new one"
    assert "discarded" in labeler.status_var.get().lower()
