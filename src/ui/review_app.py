"""Tkinter GUI for the Agentic SAM3 Auto-Annotation Lab.

Run: python -m src.ui.review_app
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import TYPE_CHECKING, Dict, List, Optional

from src.core.config_loader import load_project_config
from src.core.logging_utils import setup_logging
from src.core.orchestrator import list_image_paths, run_orchestrator
from src.tools.video_extractor import VIDEO_SUFFIXES as _VIDEO_EXTS, extract_frames

if TYPE_CHECKING:
    from src.core.models import AnnotationBundle, ProjectConfig
from src.ui._helpers import (
    IMAGE_EXTS,
    attach_tooltip,
    load_json,
    open_in_explorer,
    project_root,
    resolve_path,
    save_json,
)
from src.ui.labeler import LabelerPanel

RECENT_FILE = Path.home() / ".sam3_annotator_recent.json"
SETTINGS_FILE = Path.home() / ".sam3_annotator_settings.json"
PROMPT_HISTORY_FILE = Path.home() / ".sam3_annotator_prompts.json"

# Color themes
LIGHT_THEME = {
    "bg": "#ffffff",
    "fg": "#1a1a1a",
    "panel": "#f5f6f8",
    "muted": "#586069",
    "accent": "#0366d6",
    "ok": "#22863a",
    "warn": "#b08800",
    "err": "#d73a49",
    "log_bg": "#fafbfc",
    "preview_bg": "#1e1e1e",
}
DARK_THEME = {
    "bg": "#1e1f22",
    "fg": "#e6e6e6",
    "panel": "#2b2d31",
    "muted": "#9ba0a8",
    "accent": "#58a6ff",
    "ok": "#3fb950",
    "warn": "#d29922",
    "err": "#f85149",
    "log_bg": "#161618",
    "preview_bg": "#0f1014",
}


def _load_recent() -> List[str]:
    data = load_json(RECENT_FILE)
    if isinstance(data, list):
        return data[:8]
    return []


def _save_recent(items: List[str]) -> None:
    save_json(RECENT_FILE, items[:8])


class _QueueHandler(logging.Handler):
    def __init__(self, q: "queue.Queue[tuple]") -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        level = record.levelname
        msg = self.format(record)
        agent = "SYSTEM"
        for tag in ("CoordinatorAgent", "SAM3Agent", "CurationAgent"):
            if tag in record.name or tag in msg:
                agent = tag
                break
        try:
            self.q.put_nowait((level, agent, msg))
        except queue.Full:
            # Drop oldest to avoid stalling the producer
            try:
                self.q.get_nowait()
                self.q.put_nowait((level, agent, msg))
            except queue.Empty:
                pass


class AnnotatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.settings = load_json(SETTINGS_FILE)

        root.title("Agentic SAM3 Auto-Annotation Lab")
        root.geometry(self.settings.get("geometry", "1040x760"))
        root.minsize(900, 600)

        self.config_path = tk.StringVar(
            value=self.settings.get(
                "config_path", str(project_root() / "config" / "project_example.yaml")
            )
        )
        self.dataset_path = tk.StringVar(value=self.settings.get("dataset_path", ""))
        self.output_path = tk.StringVar(value=self.settings.get("output_path", ""))
        self.class_label_var = tk.StringVar(value=self.settings.get("class_label", ""))
        self.sam3_prompt_var = tk.StringVar(value=self.settings.get("sam3_prompt", ""))
        self.workers_var = tk.IntVar(value=int(self.settings.get("workers", 1)))
        self.max_retries_var = tk.IntVar(value=int(self.settings.get("max_retries", 2)))
        self.theme_name = tk.StringVar(value=self.settings.get("theme", "light"))
        self.filter_var = tk.StringVar(value="All")
        self.search_var = tk.StringVar(value="")
        self.recent: List[str] = _load_recent()

        self.log_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=5000)
        self.worker_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.total_images = 0
        self.processed = 0
        self.last_output_path: Optional[Path] = None
        self.last_bundles: Optional[List[AnnotationBundle]] = None
        self.last_config: Optional[ProjectConfig] = None
        self._bundle_status: Dict[str, str] = {}
        self._label_files: Dict[str, Path] = {}
        self._classes: List[str] = []
        self._preview_imgref = None
        self._run_started_at: Optional[float] = None
        self._timer_job: Optional[str] = None
        self._zoom: float = 1.0
        self._pan = (0, 0)
        self._drag_start = None
        self._current_preview_image: Optional[Path] = None
        self._current_label_text: str = ""

        self._build_menu()
        self._apply_theme()
        self._build_ui()
        self._bind_hotkeys()
        self._poll_log_queue()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- theme ----------
    def _palette(self) -> dict:
        return DARK_THEME if self.theme_name.get() == "dark" else LIGHT_THEME

    def _apply_theme(self) -> None:
        p = self._palette()
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=p["bg"], foreground=p["fg"], font=("Segoe UI", 10))
        style.configure("TFrame", background=p["bg"])
        style.configure("TLabel", background=p["bg"], foreground=p["fg"])
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=p["fg"])
        style.configure("Muted.TLabel", foreground=p["muted"])
        style.configure("TLabelframe", background=p["bg"], foreground=p["fg"])
        style.configure("TLabelframe.Label", background=p["bg"], foreground=p["fg"])
        style.configure("TNotebook", background=p["bg"])
        style.configure("TNotebook.Tab", padding=(12, 6), font=("Segoe UI", 10))
        style.configure("TButton", padding=(8, 4))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground=p["panel"], foreground=p["fg"])
        style.configure("TCombobox", fieldbackground=p["panel"], foreground=p["fg"])
        style.configure("TSpinbox", fieldbackground=p["panel"], foreground=p["fg"])
        self.root.configure(bg=p["bg"])

        # Reconfigure already-built text widgets if any
        for widget_name in ("log_text", "summary_text", "preview_text", "label_content"):
            w = getattr(self, widget_name, None)
            if w is not None:
                w.configure(bg=p["log_bg"], fg=p["fg"], insertbackground=p["fg"])
        lb = getattr(self, "labels_listbox", None)
        if lb is not None:
            lb.configure(bg=p["panel"], fg=p["fg"], selectbackground=p["accent"], selectforeground="#ffffff")
        canvas = getattr(self, "preview_canvas", None)
        if canvas is not None:
            canvas.configure(bg=p["preview_bg"])
        self._reconfigure_log_tags()

    def _reconfigure_log_tags(self) -> None:
        log = getattr(self, "log_text", None)
        if log is None:
            return
        p = self._palette()
        log.tag_configure("CoordinatorAgent", foreground=p["accent"])
        log.tag_configure("SAM3Agent", foreground=p["ok"])
        log.tag_configure("CurationAgent", foreground=p["warn"])
        log.tag_configure("SYSTEM", foreground=p["muted"])
        log.tag_configure("ERROR", foreground=p["err"], font=("Consolas", 9, "bold"))
        log.tag_configure("WARNING", foreground=p["warn"])

    def _toggle_theme(self) -> None:
        self.theme_name.set("dark" if self.theme_name.get() == "light" else "light")
        self._apply_theme()

    # ---------- menu / hotkeys ----------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open Config...", accelerator="Ctrl+O", command=self._pick_config)
        filem.add_command(label="Open Dataset Folder...", accelerator="Ctrl+B", command=self._pick_dataset)
        filem.add_separator()
        filem.add_command(label="Run Pipeline", accelerator="Ctrl+R", command=self._run_pipeline)
        filem.add_command(label="Cancel", accelerator="Esc", command=self._cancel_pipeline)
        filem.add_separator()
        filem.add_command(label="Quit", accelerator="Ctrl+Q", command=self._on_close)
        menubar.add_cascade(label="File", menu=filem)

        viewm = tk.Menu(menubar, tearoff=0)
        viewm.add_command(label="Toggle Dark/Light", accelerator="Ctrl+T", command=self._toggle_theme)
        viewm.add_command(label="Open Output Folder", command=self._open_output)
        viewm.add_command(label="Open YOLO Labels", command=self._open_yolo)
        menubar.add_cascade(label="View", menu=viewm)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=helpm)

        self.root.config(menu=menubar)

    def _bind_hotkeys(self) -> None:
        self.root.bind("<Control-r>", lambda e: self._run_pipeline())
        self.root.bind("<Control-R>", lambda e: self._run_pipeline())
        self.root.bind("<Control-b>", lambda e: self._pick_dataset())
        self.root.bind("<Control-B>", lambda e: self._pick_dataset())
        self.root.bind("<Control-o>", lambda e: self._pick_config())
        self.root.bind("<Control-O>", lambda e: self._pick_config())
        self.root.bind("<Control-t>", lambda e: self._toggle_theme())
        self.root.bind("<Control-T>", lambda e: self._toggle_theme())
        self.root.bind("<Control-q>", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._cancel_pipeline())
        self.root.bind("<F5>", lambda e: self._run_pipeline())

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Agentic SAM3 Auto-Annotation Lab\n\n"
            "Multi-agent auto-labeling with SAM3 + QA.\n\n"
            "Hotkeys: Ctrl+R run | Ctrl+B browse | Ctrl+T theme | Esc cancel",
        )

    # ---------- layout ----------
    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.setup_tab = ttk.Frame(notebook)
        self.log_tab = ttk.Frame(notebook)
        self.results_tab = ttk.Frame(notebook)
        self.video_tab = ttk.Frame(notebook)
        self.labeler_tab = ttk.Frame(notebook)
        notebook.add(self.setup_tab, text="Setup")
        notebook.add(self.log_tab, text="Run Log")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.video_tab, text="Video \u2192 Frames")
        notebook.add(self.labeler_tab, text="Labeler")
        self.notebook = notebook

        self._build_setup_tab()
        self._build_log_tab()
        self._build_results_tab()
        self._build_video_tab()
        self._build_labeler_tab()

        # Status bar
        status = ttk.Frame(self.root)
        status.pack(fill="x", padx=8, pady=(0, 6))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.elapsed_var, style="Muted.TLabel").pack(side="right", padx=(0, 8))
        self.progress = ttk.Progressbar(status, mode="determinate", length=260)
        self.progress.pack(side="right")

        # Apply theme to newly built widgets
        self._apply_theme()

    def _build_setup_tab(self) -> None:
        frm = ttk.Frame(self.setup_tab, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Project Setup", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        row = 1
        ttk.Label(frm, text="Config (YAML):").grid(row=row, column=0, sticky="w", pady=4)
        cfg_entry = ttk.Entry(frm, textvariable=self.config_path, width=78)
        cfg_entry.grid(row=row, column=1, sticky="we")
        b = ttk.Button(frm, text="Browse...", command=self._pick_config)
        b.grid(row=row, column=2, padx=6)
        attach_tooltip(b, "Ctrl+O — Choose YAML config")

        row += 1
        ttk.Label(frm, text="Images folder:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self.dataset_path, width=78).grid(row=row, column=1, sticky="we")
        ds_btns = ttk.Frame(frm)
        ds_btns.grid(row=row, column=2, padx=6)
        b1 = ttk.Button(ds_btns, text="Browse...", command=self._pick_dataset)
        b1.pack(side="left")
        attach_tooltip(b1, "Ctrl+B — Pick folder (top-level preview shown)")
        b2 = ttk.Button(ds_btns, text="Scan", command=self._scan_recursive)
        b2.pack(side="left", padx=(4, 0))
        attach_tooltip(b2, "Recursive scan for images (background)")

        row += 1
        if self.recent:
            ttk.Label(frm, text="Recent:").grid(row=row, column=0, sticky="w")
            self.recent_var = tk.StringVar()
            recent_box = ttk.Combobox(
                frm, textvariable=self.recent_var, values=self.recent, width=76, state="readonly"
            )
            recent_box.grid(row=row, column=1, sticky="we")
            recent_box.bind("<<ComboboxSelected>>", lambda e: self.dataset_path.set(self.recent_var.get()))
            row += 1

        ttk.Label(frm, text="Output folder:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self.output_path, width=78).grid(row=row, column=1, sticky="we")
        ttk.Button(frm, text="Browse...", command=self._pick_output).grid(row=row, column=2, padx=6)



        row += 1
        ttk.Label(frm, text="Class Label:").grid(row=row, column=0, sticky="w", pady=4)
        class_entry = ttk.Entry(frm, textvariable=self.class_label_var, width=78)
        class_entry.grid(row=row, column=1, sticky="we")
        attach_tooltip(class_entry, "Target classes (e.g. box or person, car)")

        row += 1
        ttk.Label(frm, text="SAM3 Text Prompt:").grid(row=row, column=0, sticky="w", pady=4)
        sam3_entry = ttk.Entry(frm, textvariable=self.sam3_prompt_var, width=78)
        sam3_entry.grid(row=row, column=1, sticky="we")
        attach_tooltip(sam3_entry, "Text prompt for SAM3 (e.g. metal silver color box)")

        row += 1
        opts = ttk.LabelFrame(frm, text="Pipeline options", padding=8)
        opts.grid(row=row, column=0, columnspan=3, sticky="we", pady=12)
        ttk.Label(opts, text="Workers:").pack(side="left")
        ws = ttk.Spinbox(opts, from_=1, to=16, textvariable=self.workers_var, width=5)
        ws.pack(side="left", padx=(4, 16))
        attach_tooltip(ws, "Parallel image workers (useful for sam3_api backend)")
        ttk.Label(opts, text="Max retries:").pack(side="left")
        rs = ttk.Spinbox(opts, from_=0, to=10, textvariable=self.max_retries_var, width=5)
        rs.pack(side="left", padx=(4, 16))
        attach_tooltip(rs, "Retries before HUMAN_REVIEW")

        self.run_btn = ttk.Button(opts, text="▶ Run pipeline (Ctrl+R)", style="Accent.TButton", command=self._run_pipeline)
        self.run_btn.pack(side="right")
        self.cancel_btn = ttk.Button(opts, text="Cancel (Esc)", command=self._cancel_pipeline, state="disabled")
        self.cancel_btn.pack(side="right", padx=(0, 6))

        row += 1
        preview = ttk.LabelFrame(frm, text="Folder preview", padding=4)
        preview.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(0, 4))
        self.preview_text = scrolledtext.ScrolledText(preview, wrap="word", height=14, font=("Consolas", 9))
        self.preview_text.pack(fill="both", expand=True)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

    def _build_labeler_tab(self) -> None:
        panel = LabelerPanel(
            self.labeler_tab,
            get_dataset_path=lambda: self.dataset_path.get().strip(),
            get_config_path=lambda: self.config_path.get().strip(),
        )
        panel.pack(fill="both", expand=True)
        self.labeler_panel = panel

    def _build_log_tab(self) -> None:
        frm = ttk.Frame(self.log_tab, padding=8)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")
        ttk.Label(top, text="Agent activity (live)", style="Header.TLabel").pack(side="left")
        ttk.Button(top, text="Clear log", command=lambda: self.log_text.delete("1.0", "end")).pack(side="right")

        self.log_text = scrolledtext.ScrolledText(frm, wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, pady=(6, 0))

    def _build_results_tab(self) -> None:
        frm = ttk.Frame(self.results_tab, padding=8)
        frm.pack(fill="both", expand=True)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(0, 6))
        self.open_out_btn = ttk.Button(btns, text="Open output folder", command=self._open_output, state="disabled")
        self.open_out_btn.pack(side="left")
        self.load_results_btn = ttk.Button(btns, text="Load Results", command=self._load_previous_results, state="normal")
        self.load_results_btn.pack(side="left", padx=6)
        self.open_yolo_btn = ttk.Button(btns, text="Open YOLO labels", command=self._open_yolo, state="disabled")
        self.open_yolo_btn.pack(side="left", padx=6)
        self.save_preview_btn = ttk.Button(btns, text="Save preview PNG", command=self._save_preview, state="disabled")
        self.save_preview_btn.pack(side="left", padx=6)
        self.export_all_btn = ttk.Button(btns, text="Export all previews", command=self._export_all_previews, state="disabled")
        self.export_all_btn.pack(side="left", padx=6)
        self.export_yolo_all_btn = ttk.Button(btns, text="Force export all to YOLO", command=self._export_all_to_yolo, state="normal")
        self.export_yolo_all_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Toggle theme", command=self._toggle_theme).pack(side="right")

        paned = ttk.PanedWindow(frm, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.LabelFrame(paned, text="QA Summary", padding=4)
        self.summary_text = scrolledtext.ScrolledText(left, wrap="word", font=("Consolas", 9))
        self.summary_text.pack(fill="both", expand=True)
        paned.add(left, weight=1)

        right = ttk.LabelFrame(paned, text="YOLO Labels Viewer", padding=4)
        paned.add(right, weight=2)

        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Filter:").pack(side="left")
        filter_cb = ttk.Combobox(
            controls, textvariable=self.filter_var, width=14, state="readonly",
            values=["All", "ACCEPTED", "HUMAN_REVIEW"],
        )
        filter_cb.pack(side="left", padx=(4, 12))
        filter_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_labels_list())
        ttk.Label(controls, text="Search:").pack(side="left")
        search_entry = ttk.Entry(controls, textvariable=self.search_var, width=20)
        search_entry.pack(side="left", padx=(4, 0))
        search_entry.bind("<KeyRelease>", lambda e: self._refresh_labels_list())
        self.accept_btn = ttk.Button(controls, text="✓ Accept Annotation", command=self._accept_selected_annotation, state="disabled")
        self.accept_btn.pack(side="left", padx=(12, 0))
        self.accept_all_btn = ttk.Button(controls, text="✓ Accept All Warnings", command=self._accept_all_warnings, state="disabled")
        self.accept_all_btn.pack(side="left", padx=(6, 0))

        labels_row = ttk.Frame(right)
        labels_row.pack(fill="both", expand=True)

        list_frm = ttk.Frame(labels_row)
        list_frm.pack(side="left", fill="y")
        ttk.Label(list_frm, text="Images:").pack(anchor="w")
        self.labels_listbox = tk.Listbox(list_frm, width=32, font=("Consolas", 9))
        self.labels_listbox.pack(fill="y", expand=True)
        self.labels_listbox.bind("<<ListboxSelect>>", self._on_label_select)

        detail_frm = ttk.Frame(labels_row)
        detail_frm.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Label(detail_frm, text="Label content:").pack(anchor="w")
        self.label_content = scrolledtext.ScrolledText(detail_frm, height=8, font=("Consolas", 9))
        self.label_content.pack(fill="x")
        ttk.Label(
            detail_frm,
            text="Preview (scroll = zoom, drag = pan, double-click = reset)",
        ).pack(anchor="w", pady=(6, 0))
        self.preview_canvas = tk.Canvas(detail_frm, bg="#1e1e1e", height=320, highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<MouseWheel>", self._on_canvas_wheel)
        self.preview_canvas.bind("<Button-4>", self._on_canvas_wheel)  # linux scroll up
        self.preview_canvas.bind("<Button-5>", self._on_canvas_wheel)  # linux scroll down
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.preview_canvas.bind("<Double-Button-1>", lambda e: self._reset_view())

    # ---------- pickers ----------
    def _pick_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select config YAML",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if path:
            self.config_path.set(path)

    def _pick_dataset(self) -> None:
        path = filedialog.askdirectory(title="Select images folder")
        if not path:
            return
        self.dataset_path.set(path)
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("end", f"Selected: {path}\nListing top-level entries...\n")
        self.status_var.set("Listing top-level...")

        def _worker() -> None:
            try:
                top_entries = list(Path(path).iterdir())
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.root.after(0, lambda m=msg: messagebox.showerror("Folder error", m))
                return
            folders = sorted([p for p in top_entries if p.is_dir()], key=lambda p: p.name.lower())
            files = sorted([p for p in top_entries if p.is_file()], key=lambda p: p.name.lower())
            top_imgs = [f for f in files if f.suffix.lower() in IMAGE_EXTS]
            lines = [f"Top-level contents of {path}:"]
            for d in folders[:200]:
                lines.append(f"  [DIR]  {d.name}/")
            if len(folders) > 200:
                lines.append(f"  ... +{len(folders) - 200} more folders")
            for f in files[:200]:
                tag = "IMG " if f.suffix.lower() in IMAGE_EXTS else "FILE"
                lines.append(f"  [{tag}] {f.name}")
            if len(files) > 200:
                lines.append(f"  ... +{len(files) - 200} more files")
            text = "\n".join(lines)

            def _apply() -> None:
                # Race guard: skip if user picked a different folder while we scanned
                if self.dataset_path.get().strip() != path:
                    return
                self.preview_text.delete("1.0", "end")
                self.preview_text.insert("end", text)
                self.status_var.set(
                    f"Top: {len(folders)} folder(s), {len(files)} file(s), {len(top_imgs)} image(s). "
                    "Click 'Scan' for recursive."
                )
                if path not in self.recent:
                    self.recent.insert(0, path)
                    _save_recent(self.recent)

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def _scan_recursive(self) -> None:
        folder = self.dataset_path.get().strip()
        if not folder:
            messagebox.showwarning("No folder", "Pick an images folder first.")
            return
        self.status_var.set(f"Recursive scanning {folder} ...")
        self.preview_text.insert("end", "\n\nRecursive scan running...\n")
        self.preview_text.see("end")

        cancel = threading.Event()

        def _worker() -> None:
            paths: List[Path] = []
            seen = 0
            try:
                for p in Path(folder).rglob("*"):
                    if cancel.is_set():
                        break
                    seen += 1
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                        paths.append(p)
                    if seen % 2000 == 0:
                        self.root.after(
                            0,
                            lambda s=seen, n=len(paths): self.status_var.set(
                                f"Scanning... {s:,} entries, {n} image(s)"
                            ),
                        )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.root.after(0, lambda m=msg: messagebox.showerror("Scan error", m))
                return
            paths.sort()
            self.root.after(0, lambda: self._after_recursive(folder, paths))

        threading.Thread(target=_worker, daemon=True).start()

    def _after_recursive(self, folder: str, paths: list) -> None:
        if self.dataset_path.get().strip() != folder:
            return  # user moved on
        self.status_var.set(f"Recursive scan: {len(paths)} image(s) under {folder}")
        block = [f"\nRecursive image count: {len(paths)}"]
        block.extend(f"  {p}" for p in paths[:30])
        if len(paths) > 30:
            block.append(f"  ... +{len(paths) - 30} more images")
        self.preview_text.insert("end", "\n".join(block) + "\n")
        self.preview_text.see("end")
        if not paths:
            messagebox.showwarning("No images", "No images found recursively.")

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_path.set(path)

    # ---------- log ----------
    def _append_log(self, agent: str, msg: str, level: str = "INFO") -> None:
        tag = level if level in {"ERROR", "WARNING"} else agent
        self.log_text.insert("end", f"[{agent}] {msg}\n", tag)
        self.log_text.see("end")

    def _poll_log_queue(self) -> None:
        drained = 0
        while drained < 50:
            try:
                level, agent, msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(agent, msg, level)
            if "Discovered" in msg:
                try:
                    n = int(msg.split("Discovered", 1)[1].split()[0])
                    self.total_images = n
                    self.processed = 0
                    self.progress.configure(maximum=max(n, 1), value=0)
                except Exception:  # noqa: BLE001
                    pass
            elif "Done " in msg and "]" in msg:
                try:
                    # Parse current index from "[cur/total] Done img_..."
                    parts = msg.split("]", 1)[0]
                    if parts.startswith("[") and "/" in parts:
                        cur = int(parts[1:].split("/")[0])
                        self.processed = cur
                        self.progress.configure(value=cur)
                        self.status_var.set(f"Running... ({cur}/{self.total_images})")
                except Exception:  # noqa: BLE001
                    pass
            drained += 1
        self.root.after(120, self._poll_log_queue)

    # ---------- timer ----------
    def _start_timer(self) -> None:
        self._run_started_at = time.time()
        self._tick_timer()

    def _tick_timer(self) -> None:
        if self._run_started_at is None:
            return
        elapsed = int(time.time() - self._run_started_at)
        mins, secs = divmod(elapsed, 60)
        self.elapsed_var.set(f"⏱ {mins:02d}:{secs:02d}")
        self._timer_job = self.root.after(500, self._tick_timer)

    def _stop_timer(self) -> None:
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        self._run_started_at = None

    # ---------- run ----------
    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state=("disabled" if running else "normal"))
        self.cancel_btn.configure(state=("normal" if running else "disabled"))
        self.status_var.set("Running..." if running else "Done.")
        if not running:
            self.progress.configure(value=self.progress["maximum"])
            self._stop_timer()

    def _cancel_pipeline(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.cancel_event.set()
        self.status_var.set("Cancel requested. Finishing current image...")

    def _run_pipeline(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "Pipeline already running.")
            return
        self.log_text.delete("1.0", "end")
        self.summary_text.delete("1.0", "end")
        self.cancel_event.clear()
        self._set_running(True)
        self._start_timer()
        self.notebook.select(self.log_tab)
        self.worker_thread = threading.Thread(target=self._do_run, daemon=True)
        self.worker_thread.start()

    def _do_run(self) -> None:
        cfg_path = resolve_path(Path(self.config_path.get()))
        dataset_override = self.dataset_path.get().strip()
        output_override = self.output_path.get().strip()
        workers = max(1, int(self.workers_var.get()))
        max_retries = max(0, int(self.max_retries_var.get()))

        try:
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file not found: {cfg_path}")
            config = load_project_config(cfg_path)

            if dataset_override:
                config.dataset_path = Path(dataset_override)
            else:
                config.dataset_path = resolve_path(config.dataset_path)
            if output_override:
                config.output_path = Path(output_override)
            else:
                config.output_path = resolve_path(config.output_path)
            config.max_workers = workers
            config.max_retries = max_retries

            if not config.dataset_path.exists():
                raise FileNotFoundError(f"Images folder not found: {config.dataset_path}")
            preview_paths = list_image_paths(config.dataset_path)
            self.log_queue.put(("INFO", "SYSTEM", f"Found {len(preview_paths)} image(s) in {config.dataset_path}"))
            if not preview_paths:
                raise RuntimeError("No images found. Supported: .jpg .jpeg .png .bmp .webp")

            classes_raw = self.class_label_var.get().strip()
            prompts_raw = self.sam3_prompt_var.get().strip()

            if classes_raw:
                import re
                manual_classes = [c.strip() for c in re.split(r",", classes_raw) if c.strip()]
                manual_prompts = [p.strip() for p in re.split(r",", prompts_raw) if p.strip()]
                
                config.label_schema = manual_classes
                config.per_class_prompt = {}
                for i, cls in enumerate(manual_classes):
                    if i < len(manual_prompts):
                        config.per_class_prompt[cls] = manual_prompts[i]
                    else:
                        config.per_class_prompt[cls] = cls
                config.user_prompt = f"Manual: {classes_raw}"
            else:
                # Fallback to config schema if empty
                config.user_prompt = "Default config schema"
                if prompts_raw:
                    import re
                    manual_prompts = [p.strip() for p in re.split(r",", prompts_raw) if p.strip()]
                    config.per_class_prompt = {}
                    for i, cls in enumerate(config.label_schema):
                        if i < len(manual_prompts):
                            config.per_class_prompt[cls] = manual_prompts[i]
                        else:
                            config.per_class_prompt[cls] = cls

            import json
            self.log_queue.put(("INFO", "SYSTEM", f"Class Label: {json.dumps(config.label_schema)}"))
            for cls in config.label_schema:
                prompt_text = config.per_class_prompt.get(cls, cls)
                self.log_queue.put(("INFO", "SYSTEM", f"SAM3 Text Prompt: \"{prompt_text}\""))

            config.output_path.mkdir(parents=True, exist_ok=True)
            setup_logging(config.output_path / "logs", level=logging.INFO)
            self.last_output_path = config.output_path

            root_logger = logging.getLogger()
            handler = _QueueHandler(self.log_queue)
            handler.setFormatter(logging.Formatter("%(message)s"))
            root_logger.addHandler(handler)

            try:
                bundles = run_orchestrator(config, cancel_event=self.cancel_event)
                self.last_bundles = bundles
                self.last_config = config
                accepted = sum(1 for b in bundles if b.status == "ACCEPTED")
                hr = sum(1 for b in bundles if b.status == "HUMAN_REVIEW")
                self._bundle_status = {b.image.path.stem: b.status for b in bundles}
                self.log_queue.put(
                    ("INFO", "SYSTEM", f"Done. total={len(bundles)} accepted={accepted} human_review={hr}")
                )
                qa_file = config.output_path / "qa_report.json"
                summary_text = ""
                if qa_file.exists():
                    summary_text = json.dumps(json.loads(qa_file.read_text(encoding="utf-8")), indent=2)
                self.root.after(0, lambda: self._finish_ok(summary_text))
            finally:
                root_logger.removeHandler(handler)

        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            self.log_queue.put(("ERROR", "SYSTEM", f"ERROR: {msg}"))
            self.root.after(0, lambda m=msg: messagebox.showerror("Pipeline failed", m))
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _finish_ok(self, summary: str) -> None:
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", summary or "(no qa_report.json produced)")
        if self.last_output_path:
            self.open_out_btn.configure(state="normal")
            self.open_yolo_btn.configure(state="normal")
            self.export_all_btn.configure(state="normal")
            self.export_yolo_all_btn.configure(state="normal")
            self._load_labels_list()
            self._update_accept_all_btn_state()
        self.notebook.select(self.results_tab)

    # ---------- labels viewer ----------
    def _load_labels_list(self) -> None:
        self._label_files = {}
        if not self.last_output_path:
            self._classes = []
            return
        classes_file = self.last_output_path / "classes.txt"
        if classes_file.exists():
            self._classes = classes_file.read_text(encoding="utf-8").splitlines()
        elif self.last_config and self.last_config.label_schema:
            self._classes = list(self.last_config.label_schema)

        labels_dir = None
        for cand in ("yolo_seg_labels", "yolo_labels"):
            d = self.last_output_path / cand
            if d.exists():
                labels_dir = d
                break
        if labels_dir is None:
            return
        for lf in sorted(labels_dir.glob("*.txt")):
            self._label_files[lf.stem] = lf
        self._refresh_labels_list()

    def _refresh_labels_list(self) -> None:
        self.labels_listbox.delete(0, "end")
        flt = self.filter_var.get()
        search = self.search_var.get().lower()
        for stem, lf in sorted(self._label_files.items()):
            status = self._bundle_status.get(stem, "?")
            if flt != "All" and status != flt:
                continue
            if search and search not in stem.lower():
                continue
            badge = "✓" if status == "ACCEPTED" else ("⚠" if status == "HUMAN_REVIEW" else "·")
            try:
                lines = [l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
                count = len(lines)
            except Exception:  # noqa: BLE001
                count = 0
            self.labels_listbox.insert("end", f"{badge}  [{count:>2}]  {stem}")

    def _on_label_select(self, event=None) -> None:  # noqa: ANN001
        sel = self.labels_listbox.curselection()
        if not sel:
            return
        entry = self.labels_listbox.get(sel[0])
        # strip "badge  [N]  " prefix
        stem = entry.split("  ", 2)[-1].strip()

        # Enable or disable accept button depending on status
        status = self._bundle_status.get(stem, "?")
        if status == "HUMAN_REVIEW":
            self.accept_btn.configure(state="normal")
        else:
            self.accept_btn.configure(state="disabled")

        lf = self._label_files.get(stem)
        if not lf:
            return

        content = lf.read_text(encoding="utf-8")
        self.label_content.delete("1.0", "end")
        self.label_content.insert("end", content if content.strip() else "(empty)")

        dataset_dir = Path(self.dataset_path.get().strip() or "")
        image_path = None
        if dataset_dir.exists():
            for ext in IMAGE_EXTS:
                candidates = list(dataset_dir.rglob(f"{stem}{ext}"))
                if candidates:
                    image_path = candidates[0]
                    break

        self.preview_canvas.delete("all")
        self._current_preview_image = image_path
        self._current_label_text = content
        self._zoom = 1.0
        self._pan = (0, 0)
        if not image_path:
            self.preview_canvas.create_text(
                10, 10, anchor="nw", fill="#cccccc",
                text=f"(image '{stem}' not found in dataset)",
            )
            self.save_preview_btn.configure(state="disabled")
            return
        self.save_preview_btn.configure(state="normal")
        self._draw_preview(image_path, content)

    def _draw_preview(self, image_path: Path, label_text: str) -> None:
        try:
            from PIL import Image, ImageDraw, ImageTk  # type: ignore
        except Exception:
            self.preview_canvas.create_text(10, 10, anchor="nw", fill="#ccc", text="Pillow not installed.")
            return
        try:
            base = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self.preview_canvas.create_text(10, 10, anchor="nw", fill="#ff6666", text=f"Open error: {exc}")
            return

        W, H = base.size
        palette_rgb = [
            (255, 77, 77), (77, 255, 136), (77, 136, 255),
            (255, 210, 77), (255, 77, 255), (77, 255, 255),
        ]
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        for idx, line in enumerate(label_text.splitlines()):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            color = palette_rgb[idx % len(palette_rgb)]
            cls_name = self._classes[cls] if 0 <= cls < len(self._classes) else f"cls{cls}"
            if len(parts) == 5:
                try:
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    continue
                x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
                odraw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
                odraw.text((x1 + 4, max(0, y1 - 14)), cls_name, fill=color + (255,))
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    continue
                pts = [(int(coords[i] * W), int(coords[i + 1] * H)) for i in range(0, len(coords), 2)]
                if len(pts) >= 3:
                    odraw.polygon(pts, fill=color + (90,), outline=color + (255,))
                    odraw.text((pts[0][0] + 4, max(0, pts[0][1] - 14)), cls_name, fill=color + (255,))

        out_im = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        self._composed_preview = out_im  # for save
        self.preview_canvas.update_idletasks()
        cw = max(self.preview_canvas.winfo_width(), 400)
        ch = max(self.preview_canvas.winfo_height(), 300)
        fit_scale = min(cw / W, ch / H, 1.0)
        scale = fit_scale * self._zoom
        new_size = (max(1, int(W * scale)), max(1, int(H * scale)))
        im_resized = out_im.resize(new_size)
        self._preview_imgref = ImageTk.PhotoImage(im_resized)
        px = cw // 2 + self._pan[0]
        py = ch // 2 + self._pan[1]
        self.preview_canvas.create_image(px, py, image=self._preview_imgref)

    # ---------- zoom / pan ----------
    def _on_canvas_wheel(self, e) -> None:  # noqa: ANN001
        if self._current_preview_image is None:
            return
        delta = 0
        if hasattr(e, "delta") and e.delta:
            delta = 1 if e.delta > 0 else -1
        elif getattr(e, "num", None) == 4:
            delta = 1
        elif getattr(e, "num", None) == 5:
            delta = -1
        factor = 1.15 if delta > 0 else (1 / 1.15)
        self._zoom = max(0.2, min(8.0, self._zoom * factor))
        self._draw_preview(self._current_preview_image, self._current_label_text)

    def _on_canvas_press(self, e) -> None:  # noqa: ANN001
        self._drag_start = (e.x, e.y, self._pan[0], self._pan[1])

    def _on_canvas_drag(self, e) -> None:  # noqa: ANN001
        if self._drag_start is None or self._current_preview_image is None:
            return
        sx, sy, px0, py0 = self._drag_start
        self._pan = (px0 + (e.x - sx), py0 + (e.y - sy))
        self._draw_preview(self._current_preview_image, self._current_label_text)

    def _on_canvas_release(self, _e=None) -> None:
        self._drag_start = None

    def _reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = (0, 0)
        if self._current_preview_image is not None:
            self._draw_preview(self._current_preview_image, self._current_label_text)

    # ---------- save / export ----------
    def _save_preview(self) -> None:
        im = getattr(self, "_composed_preview", None)
        if im is None or self._current_preview_image is None:
            return
        default = self._current_preview_image.stem + ".preview.png"
        path = filedialog.asksaveasfilename(
            title="Save preview as PNG",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not path:
            return
        try:
            im.save(path)
            self.status_var.set(f"Saved preview to {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", str(exc))

    def _export_all_previews(self) -> None:
        if not self.last_output_path or not self._label_files:
            return
        out_dir = self.last_output_path / "previews"
        out_dir.mkdir(exist_ok=True)
        dataset_dir = Path(self.dataset_path.get().strip() or "")
        if not dataset_dir.exists():
            messagebox.showerror("Export error", "Dataset folder no longer exists.")
            return

        try:
            from PIL import Image, ImageDraw  # type: ignore
        except Exception:  # noqa: BLE001
            messagebox.showerror("Export error", "Pillow not installed.")
            return

        palette_rgb = [
            (255, 77, 77), (77, 255, 136), (77, 136, 255),
            (255, 210, 77), (255, 77, 255), (77, 255, 255),
        ]
        count = 0
        for stem, lf in self._label_files.items():
            img = None
            for ext in IMAGE_EXTS:
                cand = list(dataset_dir.rglob(f"{stem}{ext}"))
                if cand:
                    img = cand[0]
                    break
            if not img:
                continue
            try:
                base = Image.open(img).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            W, H = base.size
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            for idx, line in enumerate(lf.read_text(encoding="utf-8").splitlines()):
                p = line.split()
                if not p:
                    continue
                try:
                    c = int(p[0])
                except ValueError:
                    continue
                color = palette_rgb[idx % len(palette_rgb)]
                cls_name = self._classes[c] if 0 <= c < len(self._classes) else f"cls{c}"
                if len(p) == 5:
                    cx, cy, w, h = map(float, p[1:])
                    x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
                    od.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
                    od.text((x1 + 4, max(0, y1 - 14)), cls_name, fill=color + (255,))
                elif len(p) >= 7 and len(p) % 2 == 1:
                    coords = list(map(float, p[1:]))
                    pts = [(int(coords[i] * W), int(coords[i + 1] * H)) for i in range(0, len(coords), 2)]
                    if len(pts) >= 3:
                        od.polygon(pts, fill=color + (90,), outline=color + (255,))
                        od.text((pts[0][0] + 4, max(0, pts[0][1] - 14)), cls_name, fill=color + (255,))
            out_im = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
            out_im.save(out_dir / f"{stem}.preview.jpg", quality=88)
            count += 1
        self.status_var.set(f"Exported {count} preview image(s) to {out_dir}")
        open_in_explorer(out_dir)

    def _rebuild_bundles_from_logs(self, out_dir: Path, dataset_dir: Path) -> tuple[List[AnnotationBundle], Optional[List[str]]]:
        from src.core.models import AnnotationBundle, MaskRecord, ConversationMessage, QAResult
        from src.core.orchestrator import discover_images

        logs_file = out_dir / "conversation_logs.json"
        if not logs_file.exists():
            raise FileNotFoundError(f"No conversation_logs.json found in output folder:\n{out_dir}")

        images = discover_images(dataset_dir)
        img_map = {im.id: im for im in images}
        logs_data = json.loads(logs_file.read_text(encoding="utf-8"))

        log_label_schema = None
        try:
            for entry in logs_data:
                for msg in entry.get("history", []):
                    for action in msg.get("actions", []):
                        if action.get("type") == "REQUEST_ANNOTATION" and "classes" in action:
                            log_label_schema = action["classes"]
                            break
                    if log_label_schema:
                        break
                if log_label_schema:
                    break
        except Exception:
            pass

        rebuilt_bundles = []
        for entry in logs_data:
            img_id = entry.get("image_id")
            status = entry.get("status", "HUMAN_REVIEW")
            retry_count = entry.get("retry_count", 0)

            im_record = img_map.get(img_id)
            if not im_record:
                continue

            qa_res = None
            if entry.get("qa_result"):
                try:
                    qa_res = QAResult.model_validate(entry["qa_result"])
                except Exception:
                    pass

            bundle = AnnotationBundle(
                image=im_record,
                status=status,
                retry_count=retry_count,
                qa_result=qa_res
            )

            # Reconstruct history
            history_messages = []
            for h_dict in entry.get("history", []):
                try:
                    history_messages.append(ConversationMessage.model_validate(h_dict))
                except Exception:
                    pass
            bundle.history = history_messages

            # Reconstruct masks from history
            masks = []
            history = entry.get("history", [])
            for msg in reversed(history):
                actions = msg.get("actions", [])
                ar_action = next((a for a in actions if a.get("type") == "ANNOTATION_RESULT"), None)
                if ar_action and "masks" in ar_action:
                    for m_dict in ar_action["masks"]:
                        try:
                            masks.append(MaskRecord.model_validate(m_dict))
                        except Exception:
                            pass
                    break
            bundle.masks = masks
            rebuilt_bundles.append(bundle)

        return rebuilt_bundles, log_label_schema

    def _load_previous_results(self) -> None:
        out_dir = Path(self.output_path.get().strip() or "")
        dataset_dir = Path(self.dataset_path.get().strip() or "")

        if not out_dir.exists():
            messagebox.showerror("Folder not found", "Output folder does not exist.")
            return
        if not dataset_dir.exists():
            messagebox.showerror("Folder not found", "Dataset folder does not exist.")
            return

        logs_file = out_dir / "conversation_logs.json"
        if not logs_file.exists():
            messagebox.showwarning("No logs", f"No conversation_logs.json found in output folder:\n{out_dir}")
            return

        self.status_var.set("Loading previous results from conversation_logs.json...")
        self.root.update_idletasks()

        try:
            bundles, log_label_schema = self._rebuild_bundles_from_logs(out_dir, dataset_dir)
            self.last_bundles = bundles
            self._bundle_status = {b.image.path.stem: b.status for b in bundles}
            self.last_output_path = out_dir

            if log_label_schema:
                self._classes = log_label_schema

            # Load project config fallback/load
            cfg_path = Path(self.config_path.get().strip())
            try:
                config = load_project_config(resolve_path(cfg_path))
            except Exception:
                from src.core.models import ProjectConfig
                yolo_seg_exists = (out_dir / "yolo_seg_labels").exists()
                config = ProjectConfig(
                    project_name="sam3_auto_annotation_lab",
                    dataset_path=dataset_dir,
                    output_path=out_dir,
                    label_schema=self._classes or ["box"],
                    yolo_segmentation=yolo_seg_exists
                )
            if log_label_schema:
                config.label_schema = log_label_schema
            self.last_config = config

            # Enable buttons since we loaded the results
            self.open_out_btn.configure(state="normal")
            self.open_yolo_btn.configure(state="normal")
            self.export_all_btn.configure(state="normal")
            self.export_yolo_all_btn.configure(state="normal")

            # Load labels list
            self._load_labels_list()
            self._update_accept_all_btn_state()

            # Read qa summary report if exists
            qa_file = out_dir / "qa_report.json"
            summary_text = ""
            if qa_file.exists():
                try:
                    summary_text = json.dumps(json.loads(qa_file.read_text(encoding="utf-8")), indent=2)
                except Exception:
                    pass
            self.summary_text.delete("1.0", "end")
            if summary_text:
                self.summary_text.insert("end", summary_text)
            else:
                accepted = sum(1 for b in bundles if b.status == "ACCEPTED")
                hr = sum(1 for b in bundles if b.status == "HUMAN_REVIEW")
                self.summary_text.insert("end", f"Loaded from logs:\nTotal: {len(bundles)}\nAccepted: {accepted}\nHuman Review: {hr}")

            self.status_var.set(f"Successfully loaded {len(bundles)} results from logs.")
            messagebox.showinfo("Success", f"Successfully loaded {len(bundles)} images from logs.")
        except Exception as exc:
            self.status_var.set(f"Failed to load results: {exc}")
            messagebox.showerror("Error loading results", f"Could not parse log file: {exc}")

    def _accept_selected_annotation(self) -> None:
        sel = self.labels_listbox.curselection()
        if not sel:
            return
        entry = self.labels_listbox.get(sel[0])
        stem = entry.split("  ", 2)[-1].strip()

        if not self.last_bundles:
            return

        bundle = next((b for b in self.last_bundles if b.image.path.stem == stem), None)
        if not bundle:
            messagebox.showerror("Error", f"Could not find matching image bundle for '{stem}'.")
            return

        # Change status to ACCEPTED
        bundle.status = "ACCEPTED"
        self._bundle_status[stem] = "ACCEPTED"

        # Save conversation_logs.json
        if self.last_output_path:
            try:
                from src.core.orchestrator import _export_conversation_logs
                slim = True
                if self.last_config:
                    slim = self.last_config.slim_conversation_logs
                _export_conversation_logs(self.last_bundles, self.last_output_path, slim=slim)
            except Exception as exc:
                messagebox.showerror("Error updating logs", f"Could not save changes to logs: {exc}")
                return

            # quiet YOLO and LabelMe export to keep output directory in sync
            try:
                from src.tools.yolo.exporter import export_yolo
                from src.tools.labelme import export_labelme
                label_schema = self._classes
                if self.last_config:
                    label_schema = self.last_config.label_schema
                
                export_yolo(
                    self.last_bundles,
                    self.last_output_path,
                    label_schema=label_schema,
                    segmentation=self.last_config.yolo_segmentation if self.last_config else True,
                    force_all=True,
                )
                export_labelme(
                    self.last_bundles,
                    self.last_output_path,
                    force_all=True,
                )
            except Exception as exc:
                messagebox.showwarning("Warning", f"Could not export updated labels to YOLO files: {exc}")

        # Refresh listbox and maintain selection
        self._refresh_labels_list()
        self._update_accept_all_btn_state()

        # Find the new index of the stem in the listbox and reselect it
        for i in range(self.labels_listbox.size()):
            lbl_entry = self.labels_listbox.get(i)
            lbl_stem = lbl_entry.split("  ", 2)[-1].strip()
            if lbl_stem == stem:
                self.labels_listbox.selection_clear(0, "end")
                self.labels_listbox.selection_set(i)
                self.labels_listbox.activate(i)
                self._on_label_select()
                break

    def _update_accept_all_btn_state(self) -> None:
        has_hr = False
        if self.last_bundles:
            has_hr = any(b.status == "HUMAN_REVIEW" for b in self.last_bundles)
        if has_hr:
            self.accept_all_btn.configure(state="normal")
        else:
            self.accept_all_btn.configure(state="disabled")

    def _accept_all_warnings(self) -> None:
        if not self.last_bundles:
            return

        hr_bundles = [b for b in self.last_bundles if b.status == "HUMAN_REVIEW"]
        if not hr_bundles:
            messagebox.showinfo("No warnings", "No images with HUMAN_REVIEW status found.")
            return

        if not messagebox.askyesno(
            "Confirm Accept All",
            f"Are you sure you want to accept all {len(hr_bundles)} human review annotations?"
        ):
            return

        # Update status of all human review bundles to ACCEPTED
        for bundle in hr_bundles:
            bundle.status = "ACCEPTED"
            self._bundle_status[bundle.image.path.stem] = "ACCEPTED"

        # Save conversation_logs.json
        if self.last_output_path:
            try:
                from src.core.orchestrator import _export_conversation_logs
                slim = True
                if self.last_config:
                    slim = self.last_config.slim_conversation_logs
                _export_conversation_logs(self.last_bundles, self.last_output_path, slim=slim)
            except Exception as exc:
                messagebox.showerror("Error updating logs", f"Could not save changes to logs: {exc}")
                return

            # quiet YOLO and LabelMe export to keep output directory in sync
            try:
                from src.tools.yolo.exporter import export_yolo
                from src.tools.labelme import export_labelme
                label_schema = self._classes
                if self.last_config:
                    label_schema = self.last_config.label_schema

                export_yolo(
                    self.last_bundles,
                    self.last_output_path,
                    label_schema=label_schema,
                    segmentation=self.last_config.yolo_segmentation if self.last_config else True,
                    force_all=True,
                )
                export_labelme(
                    self.last_bundles,
                    self.last_output_path,
                    force_all=True,
                )
            except Exception as exc:
                messagebox.showwarning("Warning", f"Could not export updated labels to YOLO files: {exc}")

        # Refresh listbox and maintain selection
        self._refresh_labels_list()
        self._update_accept_all_btn_state()

        sel = self.labels_listbox.curselection()
        if sel:
            self._on_label_select()

        messagebox.showinfo("Success", f"Successfully accepted all {len(hr_bundles)} annotations.")

    def _export_all_to_yolo(self) -> None:
        cfg_path = Path(self.config_path.get().strip())
        out_dir = Path(self.output_path.get().strip() or "")
        dataset_dir = Path(self.dataset_path.get().strip() or "")

        if not out_dir.exists() or not dataset_dir.exists():
            messagebox.showerror("Folder not found", "Dataset or Output folder does not exist.")
            return

        try:
            config = load_project_config(resolve_path(cfg_path))
        except Exception:
            from src.core.models import ProjectConfig
            config = ProjectConfig(
                project_name="sam3_auto_annotation_lab",
                dataset_path=dataset_dir,
                output_path=out_dir,
                label_schema=self._classes or ["box"],
                yolo_segmentation=True
            )

        bundles = None
        log_label_schema = None
        if self.last_bundles:
            bundles = self.last_bundles
            if self.last_config:
                log_label_schema = self.last_config.label_schema
        else:
            logs_file = out_dir / "conversation_logs.json"
            if not logs_file.exists():
                messagebox.showwarning("No logs", f"No conversation_logs.json found in output folder:\n{out_dir}")
                return

            self.status_var.set("Rebuilding bundles from conversation_logs.json...")
            self.root.update_idletasks()

            try:
                bundles, log_label_schema = self._rebuild_bundles_from_logs(out_dir, dataset_dir)
            except Exception as exc:
                messagebox.showerror("Error reading logs", f"Could not parse log file: {exc}")
                return

        if log_label_schema:
            config.label_schema = log_label_schema

        if not bundles:
            messagebox.showwarning("No annotations", "No images have generated masks to export.")
            return

        annotated_bundles = [b for b in bundles if b.masks]
        if not annotated_bundles:
            messagebox.showwarning("No annotations", "No images have generated masks to export.")
            return

        try:
            from src.tools.yolo.exporter import export_yolo
            from src.tools.labelme import export_labelme
            labels_dir = export_yolo(
                bundles,
                out_dir,
                label_schema=config.label_schema,
                segmentation=config.yolo_segmentation,
                force_all=True,
            )
            export_labelme(
                bundles,
                out_dir,
                force_all=True,
            )
            self.last_output_path = out_dir
            self.open_out_btn.configure(state="normal")
            self.open_yolo_btn.configure(state="normal")
            self.export_all_btn.configure(state="normal")
            self._load_labels_list()
            self.status_var.set(f"Exported {len(annotated_bundles)} annotated image(s) to YOLO format.")
            messagebox.showinfo(
                "Export complete",
                f"Successfully exported {len(annotated_bundles)} annotated image(s) "
                f"(including HUMAN_REVIEW) to:\n{labels_dir}"
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    # ---------- shortcuts ----------
    def _open_output(self) -> None:
        if self.last_output_path:
            open_in_explorer(self.last_output_path)

    def _open_yolo(self) -> None:
        if not self.last_output_path:
            return
        for cand in ("yolo_seg_labels", "yolo_labels"):
            d = self.last_output_path / cand
            if d.exists():
                open_in_explorer(d)
                return
        open_in_explorer(self.last_output_path)

    # ---------- Video -> Frames tab ----------
    def _build_video_tab(self) -> None:
        frm = ttk.Frame(self.video_tab, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Video \u2192 Frames Extractor", style="Header.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 4)
        )
        ttk.Label(
            frm,
            text="Load up to 3 videos, set how many frames you want from each, and extract.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 12))

        # --- Video slots ---
        self._video_slots: list[dict] = []
        video_exts_str = " ".join(f"*{e}" for e in sorted(_VIDEO_EXTS))
        for slot_idx in range(3):
            row = 2 + slot_idx * 2
            ttk.Label(frm, text=f"Video {slot_idx + 1}:").grid(row=row, column=0, sticky="w", pady=4)
            path_var = tk.StringVar()
            path_entry = ttk.Entry(frm, textvariable=path_var, width=62)
            path_entry.grid(row=row, column=1, sticky="we", padx=(4, 0))

            def _browse(var=path_var) -> None:
                p = filedialog.askopenfilename(
                    title="Select video file",
                    filetypes=[("Video files", video_exts_str), ("All", "*.*")],
                )
                if p:
                    var.set(p)

            ttk.Button(frm, text="Browse...", command=_browse).grid(row=row, column=2, padx=6)

            ttk.Label(frm, text="Frames:").grid(row=row, column=3, sticky="w", padx=(8, 0))
            frames_var = tk.IntVar(value=100)
            frames_spin = ttk.Spinbox(frm, from_=1, to=99999, textvariable=frames_var, width=8)
            frames_spin.grid(row=row, column=4, padx=(4, 0))
            attach_tooltip(frames_spin, "Number of evenly-spaced frames to extract")

            # Info label below each slot
            info_var = tk.StringVar(value="")
            ttk.Label(frm, textvariable=info_var, style="Muted.TLabel").grid(
                row=row + 1, column=1, columnspan=4, sticky="w", padx=(4, 0)
            )

            self._video_slots.append({
                "path_var": path_var,
                "frames_var": frames_var,
                "info_var": info_var,
            })

        # --- Output folder ---
        out_row = 2 + 3 * 2
        ttk.Label(frm, text="Output folder:").grid(row=out_row, column=0, sticky="w", pady=(12, 4))
        self._video_output_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._video_output_var, width=62).grid(
            row=out_row, column=1, sticky="we", padx=(4, 0), pady=(12, 4)
        )
        ttk.Button(frm, text="Browse...", command=self._pick_video_output).grid(
            row=out_row, column=2, padx=6, pady=(12, 4)
        )

        # --- Actions row ---
        action_row = out_row + 1
        actions = ttk.Frame(frm)
        actions.grid(row=action_row, column=0, columnspan=5, sticky="we", pady=(12, 8))

        self._video_extract_btn = ttk.Button(
            actions, text="\u25B6 Extract Frames", style="Accent.TButton",
            command=self._run_video_extraction,
        )
        self._video_extract_btn.pack(side="left")

        self._video_open_btn = ttk.Button(
            actions, text="Open output folder", command=self._open_video_output, state="disabled",
        )
        self._video_open_btn.pack(side="left", padx=(8, 0))

        self._video_use_btn = ttk.Button(
            actions, text="Use as dataset \u2192 Setup tab",
            command=self._use_video_frames_as_dataset, state="disabled",
        )
        self._video_use_btn.pack(side="left", padx=(8, 0))

        # --- Progress ---
        prog_row = action_row + 1
        self._video_progress = ttk.Progressbar(frm, mode="determinate", length=400)
        self._video_progress.grid(row=prog_row, column=0, columnspan=5, sticky="we", pady=(4, 4))

        self._video_status_var = tk.StringVar(value="Ready. Add videos and click Extract.")
        ttk.Label(frm, textvariable=self._video_status_var).grid(
            row=prog_row + 1, column=0, columnspan=5, sticky="w"
        )

        # --- Extraction log ---
        log_row = prog_row + 2
        ttk.Label(frm, text="Extraction log:", style="Muted.TLabel").grid(
            row=log_row, column=0, columnspan=5, sticky="w", pady=(8, 0)
        )
        self._video_log = scrolledtext.ScrolledText(frm, wrap="word", height=10, font=("Consolas", 9))
        self._video_log.grid(row=log_row + 1, column=0, columnspan=5, sticky="nsew", pady=(2, 0))

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(log_row + 1, weight=1)

        self._video_worker: Optional[threading.Thread] = None

    def _pick_video_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder for extracted frames")
        if path:
            self._video_output_var.set(path)

    def _open_video_output(self) -> None:
        folder = self._video_output_var.get().strip()
        if folder and Path(folder).exists():
            open_in_explorer(Path(folder))

    def _use_video_frames_as_dataset(self) -> None:
        folder = self._video_output_var.get().strip()
        if folder:
            self.dataset_path.set(folder)
            self.notebook.select(self.setup_tab)
            self.status_var.set(f"Dataset path set to extracted frames: {folder}")

    def _video_log_append(self, msg: str) -> None:
        self._video_log.insert("end", msg + "\n")
        self._video_log.see("end")

    def _run_video_extraction(self) -> None:
        if self._video_worker is not None and self._video_worker.is_alive():
            messagebox.showwarning("Busy", "Extraction already running.")
            return

        # Collect jobs
        jobs: list[tuple[Path, int]] = []
        for slot in self._video_slots:
            vpath = slot["path_var"].get().strip()
            if not vpath:
                continue
            p = Path(vpath)
            if not p.exists():
                messagebox.showerror("File not found", f"Video not found: {vpath}")
                return
            if p.suffix.lower() not in _VIDEO_EXTS:
                messagebox.showwarning(
                    "Unsupported format",
                    f"{p.name} is not a recognized video format.\n"
                    f"Supported: {', '.join(sorted(_VIDEO_EXTS))}",
                )
                return
            try:
                n = int(slot["frames_var"].get())
            except (ValueError, tk.TclError):
                n = 100
            if n < 1:
                n = 1
            jobs.append((p, n))

        if not jobs:
            messagebox.showwarning("No videos", "Add at least one video file.")
            return

        out = self._video_output_var.get().strip()
        if not out:
            messagebox.showwarning("No output", "Pick an output folder first.")
            return
        output_dir = Path(out)

        # Reset UI
        self._video_log.delete("1.0", "end")
        total_frames = sum(n for _, n in jobs)
        self._video_progress.configure(maximum=max(total_frames, 1), value=0)
        self._video_extract_btn.configure(state="disabled")
        self._video_status_var.set("Extracting...")

        for slot in self._video_slots:
            slot["info_var"].set("")

        def _worker() -> None:
            global_done = 0
            all_saved = 0
            try:
                for job_idx, (vpath, num) in enumerate(jobs):
                    sub_dir = output_dir / vpath.stem if len(jobs) > 1 else output_dir

                    def _progress(cur: int, tot: int, msg: str) -> None:
                        val = global_done + cur
                        self.root.after(0, lambda v=val, m=msg: (
                            self._video_progress.configure(value=v),
                            self._video_status_var.set(m),
                        ))

                    self.root.after(0, lambda idx=job_idx, v=vpath: self._video_log_append(
                        f"\n{'='*50}\nProcessing: {v.name}  ({num} frames requested)\n{'='*50}"
                    ))

                    result = extract_frames(
                        vpath, sub_dir, num,
                        callback=_progress,
                    )

                    global_done += result.saved_frames
                    all_saved += result.saved_frames

                    # Update slot info
                    info = (
                        f"\u2713 {result.saved_frames} frames saved | "
                        f"Video: {result.total_video_frames} frames, "
                        f"{result.fps:.1f} fps, {result.duration_secs:.1f}s"
                    )
                    if result.error:
                        info = f"\u2717 Error: {result.error}"

                    self.root.after(0, lambda idx=job_idx, i=info: (
                        self._video_slots[idx]["info_var"].set(i),
                        self._video_log_append(i),
                    ))

                summary = f"Done! Extracted {all_saved} total frames to {output_dir}"
                self.root.after(0, lambda: (
                    self._video_status_var.set(summary),
                    self._video_log_append(f"\n{summary}"),
                    self._video_progress.configure(value=self._video_progress["maximum"]),
                    self._video_extract_btn.configure(state="normal"),
                    self._video_open_btn.configure(state="normal"),
                    self._video_use_btn.configure(state="normal"),
                ))

            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.root.after(0, lambda m=msg: (
                    messagebox.showerror("Extraction failed", m),
                    self._video_status_var.set(f"Error: {m}"),
                    self._video_log_append(f"ERROR: {m}"),
                    self._video_extract_btn.configure(state="normal"),
                ))

        self._video_worker = threading.Thread(target=_worker, daemon=True)
        self._video_worker.start()

    def _on_close(self) -> None:
        # Persist settings
        try:
            data = {
                "config_path": self.config_path.get(),
                "dataset_path": self.dataset_path.get(),
                "output_path": self.output_path.get(),
                "class_label": self.class_label_var.get(),
                "sam3_prompt": self.sam3_prompt_var.get(),
                "workers": self.workers_var.get(),
                "max_retries": self.max_retries_var.get(),
                "theme": self.theme_name.get(),
                "geometry": self.root.geometry(),
            }
            save_json(SETTINGS_FILE, data)
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def _install_safety_guards() -> None:
    """Lift PIL bomb limit (we trust local images) and install Tk error trap."""
    try:
        from PIL import Image as _PILImage  # type: ignore

        _PILImage.MAX_IMAGE_PIXELS = None
    except Exception:  # noqa: BLE001
        pass


def _install_tk_error_handler(root: tk.Tk) -> None:
    import traceback
    log = logging.getLogger(__name__)

    def _report(_exc, _val, _tb):  # noqa: ANN001
        log.exception("Uncaught Tk callback error")
        try:
            messagebox.showerror(
                "Unexpected error",
                "An error occurred in the UI. Check console for details.\n\n"
                + "".join(traceback.format_exception(_exc, _val, _tb))[:1500],
            )
        except Exception:  # noqa: BLE001
            pass

    root.report_callback_exception = _report  # type: ignore[assignment]


def launch_review_app() -> None:
    _install_safety_guards()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = tk.Tk()
    _install_tk_error_handler(root)
    AnnotatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_review_app()
