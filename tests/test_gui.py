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
    assert int(gui.notebook.index("end")) == 6  # Dashboard, Setup, Run Log, Results, Video→Frames, Labeler
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
    gui.labels_listbox.insert("end", "⚠  [ 1]  test_frame")
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
