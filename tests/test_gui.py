"""GUI deep tests for AnnotatorGUI and LabelerPanel.

Uses a withdrawn Tk root so no window appears. Skips if tk unavailable.
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from pathlib import Path

import pytest
from PIL import Image

from src.ui._helpers import attach_tooltip, color_for, load_json, save_json
from src.ui.labeler import LabelerPanel, _Shape
from src.ui.review_app import AnnotatorGUI, _QueueHandler


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except Exception:
        pass


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
    assert int(gui.notebook.index("end")) == 4  # 4 tabs
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
