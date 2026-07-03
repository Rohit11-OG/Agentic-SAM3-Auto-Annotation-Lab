"""LabelMe-style manual annotator embedded as a tab.

Tries to mirror the LabelMe layout and interactions:
- Top toolbar with grouped tool buttons
- Left panel: file list with status icons
- Center: image canvas with crosshair cursor and zoom/pan
- Right panel: polygon list + label list
- Bottom status bar: mouse coords / zoom / shape count

Compatible with LabelMe `<image>.json` shape files.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

from src.ui._helpers import IMAGE_EXTS, color_for as _color_for

VERTEX_RADIUS = 5
SELECT_RADIUS = 8


class _Shape:
    __slots__ = ("kind", "label", "points")

    def __init__(self, kind: str, label: str, points: List[Tuple[float, float]]) -> None:
        self.kind = kind  # "rectangle" or "polygon"
        self.label = label
        self.points = points  # image coordinates

    def clone(self) -> "_Shape":
        return _Shape(self.kind, self.label, list(self.points))

    def to_labelme(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "points": [[float(x), float(y)] for x, y in self.points],
            "group_id": None,
            "shape_type": "rectangle" if self.kind == "rectangle" else "polygon",
            "flags": {},
        }

    @classmethod
    def from_labelme(cls, d: Dict[str, Any]) -> "_Shape":
        kind = "rectangle" if d.get("shape_type") == "rectangle" else "polygon"
        pts = [(float(p[0]), float(p[1])) for p in d.get("points", [])]
        return cls(kind=kind, label=str(d.get("label", "object")), points=pts)


class LabelerPanel(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        get_dataset_path: callable,
        get_config_path: callable,
    ) -> None:
        super().__init__(parent)
        self._get_dataset = get_dataset_path
        self._get_config = get_config_path

        self.image_paths: List[Path] = []
        self.current_idx: int = -1
        self.current_image_path: Optional[Path] = None
        self.shapes: List[_Shape] = []
        self.classes: List[str] = ["object"]
        self.current_class = tk.StringVar(value="object")
        self.mode = tk.StringVar(value="rectangle")  # rectangle | polygon | edit
        self.dirty = False

        # History (undo)
        self._history: List[List[_Shape]] = []
        self._history_limit = 50

        # Drawing state
        self._draft_points: List[Tuple[float, float]] = []
        self._rect_start_canvas: Optional[Tuple[int, int]] = None

        # Edit state
        self._sel_shape_idx: Optional[int] = None
        self._sel_vertex_idx: Optional[int] = None
        self._dragging_vertex = False

        # View state
        self._zoom: float = 1.0
        self._pan: List[int] = [0, 0]
        self._drag_anchor: Optional[Tuple[int, int, int, int]] = None
        self._dragging_canvas = False
        self._photo = None
        self._image_size: Tuple[int, int] = (0, 0)
        self._image_obj = None
        self._photo_state: Optional[Tuple] = None  # (image_id, w, h)

        # Status state
        self.status_var = tk.StringVar(value="Ready.")
        self.coords_var = tk.StringVar(value="x=0 y=0")
        self.zoom_var = tk.StringVar(value="100%")
        self.count_var = tk.StringVar(value="shapes: 0")
        self.autosave_var = tk.BooleanVar(value=True)

        self._build()
        self._bind_keys()
        self._update_mode_buttons()

    # ---------- layout ----------
    def _build(self) -> None:
        # ===== Top toolbar =====
        toolbar = ttk.Frame(self, padding=4)
        toolbar.pack(fill="x")

        self.btn_open = ttk.Button(toolbar, text="📁 Open", width=10, command=self._load_folder)
        self.btn_open.pack(side="left", padx=2)
        self.btn_save = ttk.Button(toolbar, text="💾 Save", width=10, command=self._save_json)
        self.btn_save.pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        self.btn_prev = ttk.Button(toolbar, text="⟨ Prev", width=8, command=self._prev_image)
        self.btn_prev.pack(side="left", padx=2)
        self.btn_next = ttk.Button(toolbar, text="Next ⟩", width=8, command=self._next_image)
        self.btn_next.pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        # Tool mode buttons (radio-like)
        self.btn_rect = ttk.Button(toolbar, text="▭ Rect", width=9,
                                   command=lambda: self._set_mode("rectangle"))
        self.btn_rect.pack(side="left", padx=2)
        self.btn_poly = ttk.Button(toolbar, text="⬡ Polygon", width=10,
                                   command=lambda: self._set_mode("polygon"))
        self.btn_poly.pack(side="left", padx=2)
        self.btn_edit = ttk.Button(toolbar, text="✥ Edit", width=8,
                                   command=lambda: self._set_mode("edit"))
        self.btn_edit.pack(side="left", padx=2)
        self.btn_sam = ttk.Button(toolbar, text="🎯 SAM", width=8,
                                  command=lambda: self._set_mode("sam_box"))
        self.btn_sam.pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(toolbar, text="↶ Undo", width=8, command=self._undo).pack(side="left", padx=2)
        ttk.Button(toolbar, text="✕ Delete", width=10, command=self._delete_selected_shape).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(toolbar, text="🔍+", width=4, command=lambda: self._zoom_by(1.25)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🔍-", width=4, command=lambda: self._zoom_by(0.8)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="⛶ Fit", width=6, command=self._fit_view).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        # Class controls
        ttk.Label(toolbar, text="Class:").pack(side="left")
        self.class_combo = ttk.Combobox(
            toolbar, textvariable=self.current_class, values=self.classes, width=16,
        )
        self.class_combo.pack(side="left", padx=4)
        ttk.Button(toolbar, text="+", width=3, command=self._add_class).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(toolbar, text="🤖 Auto (SAM3)", command=self._auto_annotate).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Checkbutton(toolbar, text="Auto-save", variable=self.autosave_var).pack(side="left", padx=2)

        # ===== Body: 3 columns =====
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # Left: file list
        left = ttk.LabelFrame(body, text="Files", padding=4, width=220)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.file_list = tk.Listbox(left, font=("Consolas", 9), activestyle="dotbox")
        self.file_list.pack(fill="both", expand=True)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)

        # Center: canvas with scrollbars
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True, padx=8)
        self.canvas = tk.Canvas(center, bg="#1e1e1e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self.canvas.delete("crosshair"))
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # Right: shape + label panels
        right = ttk.Frame(body, width=240)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        shapes_frm = ttk.LabelFrame(right, text="Polygon Labels", padding=4)
        shapes_frm.pack(fill="both", expand=True)
        self.shape_list = tk.Listbox(shapes_frm, font=("Consolas", 9), activestyle="dotbox")
        self.shape_list.pack(fill="both", expand=True)
        self.shape_list.bind("<<ListboxSelect>>", self._on_shape_pick)

        labels_frm = ttk.LabelFrame(right, text="Label List", padding=4, height=180)
        labels_frm.pack(fill="x", pady=(6, 0))
        labels_frm.pack_propagate(False)
        self.label_list = tk.Listbox(labels_frm, font=("Consolas", 9))
        self.label_list.pack(fill="both", expand=True)
        self.label_list.bind("<<ListboxSelect>>", self._on_label_pick)
        self._refresh_label_list()

        # ===== Status bar =====
        status = ttk.Frame(self, relief="sunken", padding=(6, 2))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Label(status, textvariable=self.coords_var, width=14, anchor="e").pack(side="right", padx=4)
        ttk.Label(status, textvariable=self.zoom_var, width=8, anchor="e").pack(side="right", padx=4)
        ttk.Label(status, textvariable=self.count_var, width=14, anchor="e").pack(side="right", padx=4)

    def _bind_keys(self) -> None:
        for w in (self, self.canvas):
            w.bind("<n>", lambda e: self._next_image())
            w.bind("<N>", lambda e: self._next_image())
            w.bind("<p>", lambda e: self._prev_image())
            w.bind("<P>", lambda e: self._prev_image())
            w.bind("<w>", lambda e: self._set_mode("rectangle"))
            w.bind("<y>", lambda e: self._set_mode("polygon"))
            w.bind("<e>", lambda e: self._set_mode("edit"))
            w.bind("<b>", lambda e: self._set_mode("sam_box"))
            w.bind("<Escape>", lambda e: self._cancel_draft())
            w.bind("<Delete>", lambda e: self._delete_selected_shape())
            w.bind("<Control-s>", lambda e: self._save_json())
            w.bind("<Control-S>", lambda e: self._save_json())
            w.bind("<Control-z>", lambda e: self._undo())
            w.bind("<Control-Z>", lambda e: self._undo())

    def apply_theme(self, p: dict) -> None:
        """Apply light/dark colors to standard Tkinter listboxes and canvas."""
        for lb in (self.file_list, self.shape_list, self.label_list):
            try:
                lb.configure(
                    bg=p["panel"],
                    fg=p["fg"],
                    selectbackground=p["accent"],
                    selectforeground="#ffffff",
                )
            except Exception:
                pass
        try:
            self.canvas.configure(bg=p["preview_bg"])
        except Exception:
            pass
        self._refresh_label_list()
        self._render()

    # ---------- modes ----------
    def _set_mode(self, m: str) -> None:
        self.mode.set(m)
        self._cancel_draft()
        self._sel_shape_idx = None
        self._sel_vertex_idx = None
        cursor_map = {
            "rectangle": "crosshair",
            "polygon": "crosshair",
            "edit": "hand2",
            "sam_box": "plus",
        }
        self.canvas.configure(cursor=cursor_map.get(m, "arrow"))
        self._update_mode_buttons()
        self._render()
        hint = "drag box → SAM3 returns a mask" if m == "sam_box" else f"Mode: {m}"
        self.status_var.set(hint)

    def _update_mode_buttons(self) -> None:
        # Visual hint by relief
        for btn, m in (
            (self.btn_rect, "rectangle"),
            (self.btn_poly, "polygon"),
            (self.btn_edit, "edit"),
            (self.btn_sam, "sam_box"),
        ):
            try:
                if self.mode.get() == m:
                    btn.state(["pressed"])
                else:
                    btn.state(["!pressed"])
            except tk.TclError:
                pass

    # ---------- history ----------
    def _snapshot(self) -> None:
        self._history.append([s.clone() for s in self.shapes])
        if len(self._history) > self._history_limit:
            self._history.pop(0)

    def _undo(self) -> None:
        if not self._history:
            self.status_var.set("Nothing to undo.")
            return
        self.shapes = self._history.pop()
        self.dirty = True
        self._refresh_shape_list()
        self._render()
        self.status_var.set("Undo.")

    # ---------- file ops ----------
    def _load_folder(self) -> None:
        default = self._get_dataset() or ""
        folder = filedialog.askdirectory(title="Pick images folder", initialdir=default or None)
        if not folder:
            return
        self._populate_files(Path(folder))

    def _populate_files(self, folder: Path) -> None:
        self.image_paths = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        self.file_list.delete(0, "end")
        for p in self.image_paths:
            has_json = p.with_suffix(".json").exists()
            badge = "● " if has_json else "  "
            self.file_list.insert("end", f"{badge}{p.name}")
        if self.image_paths:
            self.current_idx = 0
            self.file_list.selection_set(0)
            self._load_image(self.image_paths[0])
        else:
            messagebox.showwarning("Empty folder", "No images at top level.")

    def _on_file_select(self, _e=None) -> None:
        sel = self.file_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self.current_idx:
            return
        if not self._confirm_unsaved():
            self.file_list.selection_clear(0, "end")
            self.file_list.selection_set(self.current_idx)
            return
        if idx >= len(self.image_paths):
            return
        self.current_idx = idx
        self._load_image(self.image_paths[idx])

    def _next_image(self) -> None:
        if not self.image_paths:
            return
        if not self._confirm_unsaved():
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self._sync_file_list_selection()
        self._load_image(self.image_paths[self.current_idx])

    def _prev_image(self) -> None:
        if not self.image_paths:
            return
        if not self._confirm_unsaved():
            return
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self._sync_file_list_selection()
        self._load_image(self.image_paths[self.current_idx])

    def _sync_file_list_selection(self) -> None:
        self.file_list.selection_clear(0, "end")
        self.file_list.selection_set(self.current_idx)
        self.file_list.see(self.current_idx)

    def _confirm_unsaved(self) -> bool:
        if not self.dirty:
            return True
        if self.autosave_var.get():
            self._save_json()
            return True
        ans = messagebox.askyesnocancel("Unsaved", "Save annotations before switching?")
        if ans is None:
            return False
        if ans:
            self._save_json()
        return True

    def _load_image(self, path: Path) -> None:
        try:
            from PIL import Image  # type: ignore
        except Exception:  # noqa: BLE001
            messagebox.showerror("Pillow missing", "Install Pillow to use the labeler.")
            return
        try:
            self._image_obj = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open failed", str(exc))
            return
        self.current_image_path = path
        self._image_size = self._image_obj.size
        self.shapes = []
        self._history = []
        self._cancel_draft()
        self._zoom = 1.0
        self._pan = [0, 0]
        self.dirty = False

        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for sh in data.get("shapes", []):
                    self.shapes.append(_Shape.from_labelme(sh))
                for s in self.shapes:
                    if s.label not in self.classes:
                        self.classes.append(s.label)
                self.class_combo.configure(values=self.classes)
                self._refresh_label_list()
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning("Load JSON", f"Could not parse {json_path.name}: {exc}")

        self._refresh_shape_list()
        self._render()
        self.status_var.set(f"{path.name}  ({self._image_size[0]}x{self._image_size[1]})")
        self._update_counts()

    # ---------- class mgmt ----------
    def _add_class(self) -> None:
        from tkinter.simpledialog import askstring

        name = askstring("New class", "Class name:")
        if not name:
            return
        name = name.strip()
        if name and name not in self.classes:
            self.classes.append(name)
            self.class_combo.configure(values=self.classes)
            self.current_class.set(name)
            self._refresh_label_list()

    def _refresh_label_list(self) -> None:
        self.label_list.delete(0, "end")
        for cls in self.classes:
            color = _color_for(cls)
            self.label_list.insert("end", f"■  {cls}")
            try:
                self.label_list.itemconfigure("end", foreground=color)
            except tk.TclError:
                pass

    def _on_label_pick(self, _e=None) -> None:
        sel = self.label_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.classes):
            self.current_class.set(self.classes[idx])

    # ---------- coord transforms ----------
    def _fit_scale(self) -> float:
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        W, H = self._image_size
        if W == 0 or H == 0:
            return 1.0
        return min(cw / W, ch / H)

    def _image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        W, H = self._image_size
        if W == 0 or H == 0:
            return 0, 0
        scale = self._fit_scale() * self._zoom
        cx = cw / 2 + self._pan[0]
        cy = ch / 2 + self._pan[1]
        return cx + (x - W / 2) * scale, cy + (y - H / 2) * scale

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[float, float]:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        W, H = self._image_size
        if W == 0 or H == 0:
            return 0, 0
        scale = self._fit_scale() * self._zoom
        ox = cw / 2 + self._pan[0]
        oy = ch / 2 + self._pan[1]
        return (cx - ox) / scale + W / 2, (cy - oy) / scale + H / 2

    # ---------- rendering ----------
    def _render(self) -> None:
        self.canvas.delete("all")
        if self._image_obj is None:
            return
        try:
            from PIL import Image, ImageTk  # type: ignore
        except Exception:  # noqa: BLE001
            return

        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        W, H = self._image_size
        scale = self._fit_scale() * self._zoom
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))

        # Cache the resized PhotoImage; rebuild only when zoom/canvas changes
        path_key = str(self.current_image_path) if self.current_image_path else ""
        state = (path_key, new_w, new_h)
        if state != self._photo_state or self._photo is None:
            try:
                resample_filter = Image.Resampling.BILINEAR
            except AttributeError:
                resample_filter = Image.BILINEAR
            im_resized = self._image_obj.resize((new_w, new_h), resample=resample_filter)
            self._photo = ImageTk.PhotoImage(im_resized)
            self._photo_state = state

        px = cw / 2 + self._pan[0]
        py = ch / 2 + self._pan[1]
        self.canvas.create_image(px, py, image=self._photo)

        for idx, shape in enumerate(self.shapes):
            color = _color_for(shape.label)
            selected = (idx == self._sel_shape_idx)
            self._draw_shape(shape, color, selected)

        # Draft preview
        if self._draft_points:
            color = _color_for(self.current_class.get())
            canvas_pts = [self._image_to_canvas(x, y) for (x, y) in self._draft_points]
            flat = [v for pt in canvas_pts for v in pt]
            if len(canvas_pts) >= 2 and self.mode.get() == "polygon":
                self.canvas.create_line(*flat, fill=color, width=2, dash=(3, 2))
            for (cxp, cyp) in canvas_pts:
                self.canvas.create_oval(cxp - 4, cyp - 4, cxp + 4, cyp + 4, outline=color, width=2, fill="")

        self.zoom_var.set(f"{int(self._zoom * 100)}%")

    def _draw_shape(self, shape: _Shape, color: str, selected: bool) -> None:
        canvas_pts = [self._image_to_canvas(x, y) for (x, y) in shape.points]
        w = 3 if selected else 2
        if shape.kind == "rectangle" and len(canvas_pts) == 2:
            x1, y1 = canvas_pts[0]
            x2, y2 = canvas_pts[1]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=w)
            self.canvas.create_text(x1 + 4, y1 - 8, anchor="nw", text=shape.label, fill=color)
            if selected:
                for (cx, cy) in canvas_pts:
                    self.canvas.create_rectangle(
                        cx - VERTEX_RADIUS, cy - VERTEX_RADIUS, cx + VERTEX_RADIUS, cy + VERTEX_RADIUS,
                        outline=color, fill="#ffffff",
                    )
        elif shape.kind == "polygon" and len(canvas_pts) >= 3:
            flat = [v for pt in canvas_pts for v in pt]
            self.canvas.create_polygon(*flat, outline=color, fill="", width=w)
            cx0, cy0 = canvas_pts[0]
            self.canvas.create_text(cx0 + 4, cy0 - 8, anchor="nw", text=shape.label, fill=color)
            if selected:
                for (cx, cy) in canvas_pts:
                    self.canvas.create_oval(
                        cx - VERTEX_RADIUS, cy - VERTEX_RADIUS, cx + VERTEX_RADIUS, cy + VERTEX_RADIUS,
                        outline=color, fill="#ffffff",
                    )

    # ---------- canvas events ----------
    def _on_motion(self, e) -> None:
        # Crosshair
        self.canvas.delete("crosshair")
        if self.mode.get() in ("rectangle", "polygon") and self._image_obj is not None:
            self.canvas.create_line(0, e.y, self.canvas.winfo_width(), e.y,
                                    fill="#7a7a7a", dash=(2, 2), tags="crosshair")
            self.canvas.create_line(e.x, 0, e.x, self.canvas.winfo_height(),
                                    fill="#7a7a7a", dash=(2, 2), tags="crosshair")

        if self._image_obj is not None:
            ix, iy = self._canvas_to_image(e.x, e.y)
            self.coords_var.set(f"x={int(ix)} y={int(iy)}")

        # Polygon rubber-band
        if self.mode.get() == "polygon" and self._draft_points:
            self.canvas.delete("rubber")
            color = _color_for(self.current_class.get())
            last_x, last_y = self._image_to_canvas(*self._draft_points[-1])
            self.canvas.create_line(last_x, last_y, e.x, e.y, fill=color, width=1, dash=(2, 2), tags="rubber")

    def _on_click(self, e) -> None:
        if self._image_obj is None:
            return
        # Ctrl-drag => pan
        if (e.state & 0x0004) != 0:
            self._drag_anchor = (e.x, e.y, self._pan[0], self._pan[1])
            self._dragging_canvas = True
            return

        if self.mode.get() == "edit":
            self._begin_edit_pick(e.x, e.y)
            return

        img_x, img_y = self._canvas_to_image(e.x, e.y)
        if self.mode.get() in ("rectangle", "sam_box"):
            self._draft_points = [(img_x, img_y)]
            self._rect_start_canvas = (e.x, e.y)
        else:  # polygon
            self._draft_points.append((img_x, img_y))
            self._render()

    def _on_drag(self, e) -> None:
        if self._dragging_canvas and self._drag_anchor is not None:
            sx, sy, px0, py0 = self._drag_anchor
            self._pan = [px0 + (e.x - sx), py0 + (e.y - sy)]
            self._render()
            return
        if self.mode.get() == "edit" and self._dragging_vertex:
            self._update_dragged_vertex(e.x, e.y)
            return
        if self.mode.get() in ("rectangle", "sam_box") and self._rect_start_canvas is not None:
            self.canvas.delete("rect_preview")
            sx, sy = self._rect_start_canvas
            color = _color_for(self.current_class.get())
            self.canvas.create_rectangle(sx, sy, e.x, e.y, outline=color, width=2, dash=(3, 2), tags="rect_preview")

    def _on_release(self, e) -> None:
        if self._dragging_canvas:
            self._dragging_canvas = False
            self._drag_anchor = None
            return
        if self.mode.get() == "edit" and self._dragging_vertex:
            self._dragging_vertex = False
            self.dirty = True
            return
        if self.mode.get() in ("rectangle", "sam_box") and self._rect_start_canvas is not None:
            sx, sy = self._rect_start_canvas
            if abs(e.x - sx) < 3 and abs(e.y - sy) < 3:
                self._rect_start_canvas = None
                self._draft_points = []
                self._render()
                return
            ix1, iy1 = self._canvas_to_image(sx, sy)
            ix2, iy2 = self._canvas_to_image(e.x, e.y)
            x_min, x_max = sorted([ix1, ix2])
            y_min, y_max = sorted([iy1, iy2])

            if self.mode.get() == "sam_box":
                self._run_sam_box(int(x_min), int(y_min), int(x_max), int(y_max))
            else:
                self._snapshot()
                self.shapes.append(_Shape("rectangle", self.current_class.get(),
                                         [(x_min, y_min), (x_max, y_max)]))
                self._refresh_shape_list()
                self.dirty = True
                self._update_counts()

            self._rect_start_canvas = None
            self._draft_points = []
            self._render()

    def _on_right_click(self, e=None) -> None:
        if e is not None and self.mode.get() == "edit":
            if self._delete_vertex_at(e.x, e.y):
                return
        self._finish_polygon()

    def _on_double_click(self, e=None) -> None:
        if e is not None and self.mode.get() == "edit":
            if self._insert_vertex_on_segment(e.x, e.y):
                return
        self._finish_polygon()

    def _point_to_segment_distance(self, px: float, py: float, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return ((px - ax)**2 + (py - ay)**2)**0.5
            
        t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return ((px - proj_x)**2 + (py - proj_y)**2)**0.5

    def _insert_vertex_on_segment(self, cx: int, cy: int) -> bool:
        if self._image_obj is None:
            return False
        img_x, img_y = self._canvas_to_image(cx, cy)
        scale = self._fit_scale() * self._zoom
        if scale == 0:
            return False
        best_shape_idx = None
        best_vertex_idx = None
        best_dist = 8.0 / scale
        
        for si, sh in enumerate(self.shapes):
            if sh.kind != "polygon":
                continue
            n = len(sh.points)
            for i in range(n):
                p1 = sh.points[i]
                p2 = sh.points[(i + 1) % n]
                d = self._point_to_segment_distance(img_x, img_y, p1, p2)
                if d < best_dist:
                    best_dist = d
                    best_shape_idx = si
                    best_vertex_idx = (i + 1)
                    
        if best_shape_idx is not None and best_vertex_idx is not None:
            self._snapshot()
            self.shapes[best_shape_idx].points.insert(best_vertex_idx, (img_x, img_y))
            self._sel_shape_idx = best_shape_idx
            self._sel_vertex_idx = best_vertex_idx
            self.dirty = True
            self._refresh_shape_list()
            self._render()
            self.status_var.set("Inserted new vertex on segment.")
            return True
        return False

    def _delete_vertex_at(self, cx: int, cy: int) -> bool:
        if self._image_obj is None:
            return False
        for si, sh in enumerate(self.shapes):
            if sh.kind != "polygon":
                continue
            for vi, (px, py) in enumerate(sh.points):
                cxp, cyp = self._image_to_canvas(px, py)
                if abs(cxp - cx) <= SELECT_RADIUS and abs(cyp - cy) <= SELECT_RADIUS:
                    if len(sh.points) > 3:
                        self._snapshot()
                        sh.points.pop(vi)
                        self._sel_shape_idx = None
                        self._sel_vertex_idx = None
                        self.dirty = True
                        self._refresh_shape_list()
                        self._render()
                        self.status_var.set("Deleted vertex.")
                        return True
                    else:
                        self.status_var.set("Cannot delete vertex: polygon needs at least 3 vertices.")
                        return False
        return False

    def _on_pan_start(self, e) -> None:
        if self._image_obj is None:
            return
        self._drag_anchor = (e.x, e.y, self._pan[0], self._pan[1])
        self._dragging_canvas = True
        self.canvas.configure(cursor="fleur")

    def _on_pan_drag(self, e) -> None:
        if self._dragging_canvas and self._drag_anchor is not None:
            sx, sy, px0, py0 = self._drag_anchor
            self._pan = [px0 + (e.x - sx), py0 + (e.y - sy)]
            self._render()

    def _on_pan_end(self, e) -> None:
        self._dragging_canvas = False
        self._drag_anchor = None
        cursor_map = {
            "rectangle": "crosshair",
            "polygon": "crosshair",
            "edit": "hand2",
            "sam_box": "plus",
        }
        self.canvas.configure(cursor=cursor_map.get(self.mode.get(), "arrow"))

    def _finish_polygon(self) -> None:
        if self.mode.get() == "polygon" and len(self._draft_points) >= 3:
            self._snapshot()
            self.shapes.append(_Shape("polygon", self.current_class.get(), list(self._draft_points)))
            self.dirty = True
            self._refresh_shape_list()
            self._update_counts()
        self._draft_points = []
        self._render()

    def _cancel_draft(self) -> None:
        self._draft_points = []
        self._rect_start_canvas = None
        self.canvas.delete("rubber")
        self._render()

    # ---------- edit ----------
    def _begin_edit_pick(self, cx: int, cy: int) -> None:
        # 1) vertex of selected shape?
        if self._sel_shape_idx is not None and 0 <= self._sel_shape_idx < len(self.shapes):
            sh = self.shapes[self._sel_shape_idx]
            for vi, (px, py) in enumerate(sh.points):
                cxp, cyp = self._image_to_canvas(px, py)
                if abs(cxp - cx) <= SELECT_RADIUS and abs(cyp - cy) <= SELECT_RADIUS:
                    self._sel_vertex_idx = vi
                    self._dragging_vertex = True
                    self._snapshot()
                    return
        # 2) pick a shape by clicking near a vertex of any shape
        best = None
        best_d = SELECT_RADIUS + 1
        for si, sh in enumerate(self.shapes):
            for vi, (px, py) in enumerate(sh.points):
                cxp, cyp = self._image_to_canvas(px, py)
                d = max(abs(cxp - cx), abs(cyp - cy))
                if d < best_d:
                    best_d = d
                    best = (si, vi)
        if best:
            self._sel_shape_idx, self._sel_vertex_idx = best
            self._render()
            return
        # 3) deselect
        self._sel_shape_idx = None
        self._sel_vertex_idx = None
        self._render()

    def _update_dragged_vertex(self, cx: int, cy: int) -> None:
        if self._sel_shape_idx is None or self._sel_vertex_idx is None:
            return
        sh = self.shapes[self._sel_shape_idx]
        img_x, img_y = self._canvas_to_image(cx, cy)
        if sh.kind == "rectangle" and len(sh.points) == 2:
            # adjust corner; recompute opposite stays
            other_idx = 1 - self._sel_vertex_idx
            other = sh.points[other_idx]
            sh.points = [(img_x, img_y), other] if self._sel_vertex_idx == 0 else [other, (img_x, img_y)]
        else:
            sh.points[self._sel_vertex_idx] = (img_x, img_y)
        self._render()

    # ---------- view ----------
    def _on_wheel(self, e) -> None:
        if self._image_obj is None:
            return
        delta = 0
        if hasattr(e, "delta") and e.delta:
            delta = 1 if e.delta > 0 else -1
        elif getattr(e, "num", None) == 4:
            delta = 1
        elif getattr(e, "num", None) == 5:
            delta = -1
        self._zoom_by(1.15 if delta > 0 else (1 / 1.15))

    def _zoom_by(self, factor: float) -> None:
        self._zoom = max(0.2, min(15.0, self._zoom * factor))
        self._render()

    def _fit_view(self) -> None:
        self._zoom = 1.0
        self._pan = [0, 0]
        self._render()

    # ---------- shape list ----------
    def _refresh_shape_list(self) -> None:
        self.shape_list.delete(0, "end")
        for i, s in enumerate(self.shapes):
            self.shape_list.insert("end", f"[{i}] {s.kind[:4]:<4} {s.label}")
            try:
                self.shape_list.itemconfigure("end", foreground=_color_for(s.label))
            except tk.TclError:
                pass

    def _on_shape_pick(self, _e=None) -> None:
        sel = self.shape_list.curselection()
        if not sel:
            self._sel_shape_idx = None
        else:
            self._sel_shape_idx = sel[0]
        self._sel_vertex_idx = None
        self._render()

    def _delete_selected_shape(self) -> None:
        sel = self.shape_list.curselection()
        if sel:
            idx = sel[0]
        elif self._sel_shape_idx is not None:
            idx = self._sel_shape_idx
        else:
            return
        if 0 <= idx < len(self.shapes):
            self._snapshot()
            del self.shapes[idx]
            self._sel_shape_idx = None
            self.dirty = True
            self._refresh_shape_list()
            self._render()
            self._update_counts()

    def _update_counts(self) -> None:
        self.count_var.set(f"shapes: {len(self.shapes)}")

    # ---------- save / load JSON ----------
    def _save_json(self) -> None:
        if self.current_image_path is None or self._image_obj is None:
            return
        json_path = self.current_image_path.with_suffix(".json")
        try:
            buf = io.BytesIO()
            self._image_obj.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001
            img_b64 = None

        data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [s.to_labelme() for s in self.shapes],
            "imagePath": self.current_image_path.name,
            "imageData": img_b64,
            "imageHeight": self._image_size[1],
            "imageWidth": self._image_size[0],
        }
        try:
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.dirty = False
            self.status_var.set(f"Saved → {json_path.name}  ({len(self.shapes)} shape(s))")
            # Refresh file list badge
            if self.current_idx >= 0:
                self.file_list.delete(self.current_idx)
                self.file_list.insert(self.current_idx, f"● {self.current_image_path.name}")
                self._sync_file_list_selection()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", str(exc))

    # ---------- auto-annotate ----------
    # ---------- SAM-assist (box prompt) ----------
    def _run_sam_box(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self.current_image_path is None:
            return
        cls = self.current_class.get().strip() or "object"
        self.status_var.set(f"SAM3 box-prompt for '{cls}' on box ({x1},{y1})-({x2},{y2})...")
        cfg_path = self._get_config()
        target_image = self.current_image_path

        def _worker() -> None:
            try:
                from src.core.config_loader import load_project_config
                from src.tools.sam3 import sam3_segment_box_prompt

                params: Dict[str, Any] = {}
                try:
                    if cfg_path and Path(cfg_path).exists():
                        cfg = load_project_config(Path(cfg_path))
                        params = dict(cfg.sam3_params)
                except Exception:  # noqa: BLE001
                    pass

                model_name = params.get("model_name", "sam3_b")
                raw = sam3_segment_box_prompt(
                    image_path=target_image,
                    box=(x1, y1, x2, y2),
                    class_name=cls,
                    model_name=model_name,
                    params=params,
                )

                new_shapes: List[_Shape] = []
                for r in raw:
                    if r.polygon and len(r.polygon) >= 3:
                        new_shapes.append(_Shape("polygon", cls,
                                                 [(float(x), float(y)) for x, y in r.polygon]))
                    else:
                        bx, by, bw, bh = r.bbox
                        new_shapes.append(_Shape("rectangle", cls,
                                                 [(float(bx), float(by)),
                                                  (float(bx + bw), float(by + bh))]))

                def _apply() -> None:
                    if self.current_image_path != target_image:
                        self.status_var.set("SAM3 box result discarded (image changed).")
                        return
                    if new_shapes:
                        self._snapshot()
                        self.shapes.extend(new_shapes)
                        if cls not in self.classes:
                            self.classes.append(cls)
                            self.class_combo.configure(values=self.classes)
                            self._refresh_label_list()
                        self.dirty = True
                        self._refresh_shape_list()
                        self._render()
                        self._update_counts()
                        self.status_var.set(
                            f"SAM3 added {len(new_shapes)} shape(s) for '{cls}' (mode: SAM box)."
                        )
                    else:
                        self.status_var.set("SAM3 returned no mask for that box.")

                self.after(0, _apply)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.after(0, lambda m=msg: messagebox.showerror("SAM3 box-prompt failed", m))

        threading.Thread(target=_worker, daemon=True).start()

    def _auto_annotate(self) -> None:
        if self.current_image_path is None:
            return
        cls = self.current_class.get().strip() or "object"
        self.status_var.set(f"SAM3 running for '{cls}'...")
        cfg_path = self._get_config()
        target_image = self.current_image_path  # capture for race-safety

        def _worker() -> None:
            try:
                from src.core.config_loader import load_project_config
                from src.tools.sam3 import sam3_segment_text_prompt

                params: Dict[str, Any] = {}
                try:
                    if cfg_path and Path(cfg_path).exists():
                        cfg = load_project_config(Path(cfg_path))
                        params = dict(cfg.sam3_params)
                except Exception:  # noqa: BLE001
                    pass

                model_name = params.get("model_name", "sam3_b")
                raw = sam3_segment_text_prompt(
                    image_path=target_image,
                    class_name=cls,
                    model_name=model_name,
                    params=params,
                )

                new_shapes: List[_Shape] = []
                for r in raw:
                    if r.polygon and len(r.polygon) >= 3:
                        new_shapes.append(_Shape("polygon", cls, [(float(x), float(y)) for x, y in r.polygon]))
                    else:
                        x, y, w, h = r.bbox
                        new_shapes.append(_Shape("rectangle", cls,
                                                 [(float(x), float(y)), (float(x + w), float(y + h))]))

                def _apply() -> None:
                    if self.current_image_path != target_image:
                        self.status_var.set("SAM3 result discarded (image changed).")
                        return
                    if new_shapes:
                        self._snapshot()
                        self.shapes.extend(new_shapes)
                        if cls not in self.classes:
                            self.classes.append(cls)
                            self.class_combo.configure(values=self.classes)
                            self._refresh_label_list()
                        self.dirty = True
                        self._refresh_shape_list()
                        self._render()
                        self._update_counts()
                        self.status_var.set(f"SAM3 added {len(new_shapes)} shape(s) for '{cls}'.")
                    else:
                        self.status_var.set(f"SAM3 found nothing for '{cls}'.")

                self.after(0, _apply)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.after(0, lambda m=msg: messagebox.showerror("Auto-annotate failed", m))

        threading.Thread(target=_worker, daemon=True).start()
