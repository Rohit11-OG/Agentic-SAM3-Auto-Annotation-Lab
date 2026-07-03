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
        filem.add_command(label="Clear Session", accelerator="Ctrl+Shift+C", command=self._clear_session)
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
        self.root.bind("<F1>", lambda e: self._show_hotkey_help())
        self.root.bind("<Control-slash>", lambda e: self._show_hotkey_help())
        self.root.bind("<Control-Shift-C>", lambda e: self._clear_session())
        self.root.bind("<Control-Shift-c>", lambda e: self._clear_session())

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Agentic SAM3 Auto-Annotation Lab\n\n"
            "Multi-agent auto-labeling with SAM3 + QA.\n\n"
            "Hotkeys: Ctrl+R run | Ctrl+B browse | Ctrl+T theme | Esc cancel",
        )

    def _clear_session(self, confirm: bool = True) -> None:
        """Reset GUI to fresh state: clear paths, logs, previews, results, labeler.

        Does NOT delete files on disk. Does NOT touch persisted settings file
        (those reload on next launch unless user explicitly overwrites).
        """
        if confirm and not messagebox.askyesno(
            "Clear session",
            "Clear all in-GUI state?\n\n"
            "• Resets dataset / output / prompt fields\n"
            "• Clears run log, results, previews, labeler\n"
            "• Does NOT delete any files on disk",
        ):
            return

        # Stop any running pipeline first
        try:
            if self.worker_thread and self.worker_thread.is_alive():
                self.cancel_event.set()
        except Exception:  # noqa: BLE001
            pass

        # Path / config fields
        self.dataset_path.set("")
        self.output_path.set("")
        for vname in ("prompt_var", "class_label_var", "sam3_prompt_var"):
            v = getattr(self, vname, None)
            if v is not None:
                try:
                    v.set("")
                except Exception:  # noqa: BLE001
                    pass
        self.filter_var.set("All")
        self.search_var.set("")

        # State
        self.last_output_path = None
        self.last_bundles = None
        self.last_config = None
        self._bundle_status = {}
        self._label_files = {}
        self._classes = []
        self._preview_imgref = None
        self._current_preview_image = None
        self._current_label_text = ""
        self._zoom = 1.0
        self._pan = (0, 0)
        self.total_images = 0
        self.processed = 0

        # Text widgets
        for w_name in ("preview_text", "log_text", "summary_text", "label_content"):
            w = getattr(self, w_name, None)
            if w is not None:
                try:
                    w.configure(state="normal")
                    w.delete("1.0", "end")
                except tk.TclError:
                    pass

        # Labels list / preview canvas
        try:
            self.labels_listbox.delete(0, "end")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.preview_canvas.delete("all")
        except Exception:  # noqa: BLE001
            pass

        # Dashboard
        for v in self.dash_card_vars.values():
            v.set("—")
        try:
            self.dash_chart.configure(state="normal")
            self.dash_chart.delete("1.0", "end")
            self.dash_chart.configure(state="disabled")
            self.dash_issues.configure(state="normal")
            self.dash_issues.delete("1.0", "end")
            self.dash_issues.configure(state="disabled")
        except Exception:  # noqa: BLE001
            pass

        # Results action buttons -> disabled
        for btn_name in ("open_out_btn", "open_yolo_btn", "export_all_btn",
                         "export_yolo_all_btn", "save_preview_btn"):
            b = getattr(self, btn_name, None)
            if b is not None:
                try:
                    b.configure(state="disabled")
                except tk.TclError:
                    pass

        # Video tab reset
        try:
            for slot in getattr(self, "_video_slots", []):
                slot["path_var"].set("")
                slot["info_var"].set("")
                slot["frames_var"].set(100)
            self._video_output_var.set("")
            self._video_log.delete("1.0", "end")
            self._video_progress.configure(value=0)
            self._video_status_var.set("Ready. Add videos and click Extract.")
        except Exception:  # noqa: BLE001
            pass

        # Labeler tab reset
        try:
            lp = self.labeler_panel
            lp.image_paths = []
            lp.current_idx = -1
            lp.current_image_path = None
            lp.shapes = []
            lp._history = []
            lp._cancel_draft()
            lp._image_obj = None
            lp._photo = None
            lp._photo_state = None
            lp.dirty = False
            lp.file_list.delete(0, "end")
            lp.shape_list.delete(0, "end")
            lp.canvas.delete("all")
            lp.status_var.set("Cleared.")
            lp.coords_var.set("x=0 y=0")
            lp.zoom_var.set("100%")
            lp.count_var.set("shapes: 0")
        except Exception:  # noqa: BLE001
            pass

        # Status / progress
        self.progress.configure(value=0, maximum=100)
        self.elapsed_var.set("")
        self._stop_timer()
        self._set_running(False)
        # Set status AFTER _set_running so it's not overridden
        self.status_var.set("Session cleared.")

        # Jump back to Setup tab
        try:
            self.notebook.select(self.setup_tab)
        except Exception:  # noqa: BLE001
            pass

    def _show_hotkey_help(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Keyboard Shortcuts")
        win.geometry("520x420")
        win.transient(self.root)
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Keyboard Shortcuts", style="Header.TLabel").pack(anchor="w", pady=(0, 8))
        groups = [
            ("Pipeline", [
                ("Ctrl+R / F5", "Run pipeline"),
                ("Esc", "Cancel run"),
                ("Ctrl+B", "Pick images folder"),
                ("Ctrl+O", "Open config"),
            ]),
            ("View", [
                ("Ctrl+T", "Toggle dark/light theme"),
                ("Ctrl+Q", "Quit application"),
                ("F1", "This help"),
            ]),
            ("Labeler", [
                ("N / P", "Next / previous image"),
                ("W / Y / E", "Rect / Polygon / Edit mode"),
                ("Del", "Delete selected shape"),
                ("Ctrl+S", "Save annotations JSON"),
                ("Ctrl+Z", "Undo"),
                ("Right-click / Double-click", "Finish polygon"),
                ("Mouse wheel", "Zoom preview"),
                ("Ctrl + drag", "Pan canvas"),
            ]),
        ]
        body = scrolledtext.ScrolledText(frm, wrap="word", font=("Consolas", 10))
        body.pack(fill="both", expand=True)
        for title, pairs in groups:
            body.insert("end", f"\n── {title} ──\n", "h")
            for key, desc in pairs:
                body.insert("end", f"  {key:<28} {desc}\n")
        body.tag_configure("h", font=("Segoe UI", 10, "bold"))
        body.configure(state="disabled")
        close_btn = ttk.Button(frm, text="Close", command=win.destroy)
        close_btn.pack(anchor="e", pady=(8, 0))
        win.bind("<Escape>", lambda _e: win.destroy())
        win.bind("<Return>", lambda _e: win.destroy())
        close_btn.focus_set()

    # ---------- dashboard ----------
    def _build_dashboard_tab(self) -> None:
        frm = ttk.Frame(self.dashboard_tab, padding=12)
        frm.pack(fill="both", expand=True)

        header = ttk.Frame(frm)
        header.pack(fill="x")
        ttk.Label(header, text="Dashboard", style="Header.TLabel").pack(side="left")
        ttk.Button(header, text="🔄 Refresh", command=self._refresh_dashboard).pack(side="right")
        ttk.Button(header, text="📂 Open output", command=self._open_output).pack(side="right", padx=(0, 6))

        # Stat cards row
        cards = ttk.Frame(frm)
        cards.pack(fill="x", pady=(12, 0))
        self.dash_card_vars: Dict[str, tk.StringVar] = {}

        def _card(parent, label: str, key: str) -> None:
            c = ttk.LabelFrame(parent, text=label, padding=10)
            c.pack(side="left", fill="x", expand=True, padx=4)
            v = tk.StringVar(value="—")
            self.dash_card_vars[key] = v
            ttk.Label(c, textvariable=v, font=("Segoe UI", 22, "bold")).pack(anchor="center")

        _card(cards, "Total images", "total")
        _card(cards, "Accepted", "accepted")
        _card(cards, "Human review", "human_review")
        _card(cards, "Accept rate", "accept_rate")
        _card(cards, "Avg confidence", "avg_conf")
        _card(cards, "Total retries", "retries")

        # Class distribution chart (ASCII bar)
        chart_frm = ttk.LabelFrame(frm, text="Class distribution", padding=8)
        chart_frm.pack(fill="both", expand=True, pady=(12, 0))
        self.dash_chart = scrolledtext.ScrolledText(chart_frm, wrap="none", font=("Consolas", 10), height=14)
        self.dash_chart.pack(fill="both", expand=True)

        # Top issues panel
        issues_frm = ttk.LabelFrame(frm, text="Top QA issues", padding=8)
        issues_frm.pack(fill="x", pady=(8, 0))
        self.dash_issues = scrolledtext.ScrolledText(issues_frm, wrap="word", font=("Consolas", 10), height=6)
        self.dash_issues.pack(fill="x")

    def _refresh_dashboard(self) -> None:
        # Reset
        for k, v in self.dash_card_vars.items():
            v.set("—")
        self.dash_chart.configure(state="normal")
        self.dash_chart.delete("1.0", "end")
        self.dash_issues.configure(state="normal")
        self.dash_issues.delete("1.0", "end")

        out = self.last_output_path or (
            Path(self.output_path.get().strip()) if self.output_path.get().strip() else None
        )
        if not out or not out.exists():
            self.dash_chart.insert("end", "(No run output found. Run pipeline first or pick an output folder.)\n")
            self._lock_dashboard()
            return
        qa_file = out / "qa_report.json"
        if not qa_file.exists():
            self.dash_chart.insert("end", f"(qa_report.json not found in {out})\n")
            self._lock_dashboard()
            return
        try:
            data = json.loads(qa_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.dash_chart.insert("end", f"Could not parse qa_report.json: {exc}\n")
            self._lock_dashboard()
            return

        total = int(data.get("total_images", 0))
        status_counts = data.get("status_counts", {})
        accepted = int(status_counts.get("ACCEPTED", 0))
        hr = int(status_counts.get("HUMAN_REVIEW", 0))
        retries = int(data.get("retries_total", 0))
        accept_rate = float(data.get("accept_rate", 0.0))
        avg_conf = float(data.get("avg_mask_confidence", 0.0))

        self.dash_card_vars["total"].set(str(total))
        self.dash_card_vars["accepted"].set(str(accepted))
        self.dash_card_vars["human_review"].set(str(hr))
        self.dash_card_vars["accept_rate"].set(f"{int(accept_rate * 100)}%")
        self.dash_card_vars["avg_conf"].set(f"{avg_conf:.2f}")
        self.dash_card_vars["retries"].set(str(retries))

        # Class distribution bar chart (ASCII)
        per_class = data.get("per_class_mask_counts", {})
        max_count = max(per_class.values(), default=0) or 1
        bar_w = 40
        for cls, cnt in sorted(per_class.items(), key=lambda x: -x[1]):
            n = int((cnt / max_count) * bar_w)
            bar = "█" * n + "·" * (bar_w - n)
            self.dash_chart.insert("end", f"{cls:<14} {bar} {cnt}\n")
        if not per_class:
            self.dash_chart.insert("end", "(no class detections recorded)\n")

        # Top issues
        issues = data.get("top_issues", []) or []
        if not issues:
            self.dash_issues.insert("end", "(no issues — all clean)\n")
        else:
            for item in issues[:10]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    text, count = item
                    self.dash_issues.insert("end", f"  ×{count}  {text}\n")
                else:
                    self.dash_issues.insert("end", f"  {item}\n")

        self._lock_dashboard()

    def _lock_dashboard(self) -> None:
        self.dash_chart.configure(state="disabled")
        self.dash_issues.configure(state="disabled")

    def _on_tab_changed(self, _e=None) -> None:
        try:
            current = self.notebook.select()
            if current == str(self.dashboard_tab):
                self._refresh_dashboard()
        except Exception:  # noqa: BLE001
            pass

    # ---------- GPU widget ----------
    def _refresh_gpu_widget(self) -> None:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                used_gb = (total - free) / (1024 ** 3)
                total_gb = total / (1024 ** 3)
                self.gpu_var.set(f"GPU {used_gb:.1f}/{total_gb:.1f} GB")
            else:
                self.gpu_var.set("CPU only")
        except Exception:  # noqa: BLE001
            self.gpu_var.set("")
        self.root.after(2000, self._refresh_gpu_widget)

    # ---------- layout ----------
    def _build_ui(self) -> None:
        # Global toolbar above everything
        toolbar = ttk.Frame(self.root, padding=(8, 4))
        toolbar.pack(fill="x")
        self.tb_run_btn = ttk.Button(toolbar, text="\u25b6 Run", style="Accent.TButton", command=self._run_pipeline)
        self.tb_run_btn.pack(side="left", padx=2)
        self.tb_cancel_btn = ttk.Button(toolbar, text="\u25a0 Cancel", command=self._cancel_pipeline, state="disabled")
        self.tb_cancel_btn.pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="\ud83d\udcc1 Dataset", command=self._pick_dataset).pack(side="left", padx=2)
        ttk.Button(toolbar, text="\ud83d\udce4 Output", command=self._open_output).pack(side="left", padx=2)
        ttk.Button(toolbar, text="\ud83c\udff7 YOLO", command=self._open_yolo).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="\ud83c\udf13 Theme", command=self._toggle_theme).pack(side="left", padx=2)
        ttk.Button(toolbar, text="\ud83e\uddf9 Clear", command=self._clear_session).pack(side="left", padx=2)
        ttk.Button(toolbar, text="\u2754 Help (F1)", command=self._show_hotkey_help).pack(side="right", padx=2)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.dashboard_tab = ttk.Frame(notebook)
        self.setup_tab = ttk.Frame(notebook)
        self.log_tab = ttk.Frame(notebook)
        self.results_tab = ttk.Frame(notebook)
        self.video_tab = ttk.Frame(notebook)
        self.labeler_tab = ttk.Frame(notebook)
        self.fewshot_tab = ttk.Frame(notebook)
        notebook.add(self.dashboard_tab, text="Dashboard")
        notebook.add(self.setup_tab, text="Setup")
        notebook.add(self.log_tab, text="Run Log")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.fewshot_tab, text="\ud83c\udfaf Few-Shot")
        notebook.add(self.video_tab, text="Video \u2192 Frames")
        notebook.add(self.labeler_tab, text="Labeler")
        self.notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_dashboard_tab()
        self._build_setup_tab()
        self._build_log_tab()
        self._build_results_tab()
        self._build_fewshot_tab()
        self._build_video_tab()
        self._build_labeler_tab()

        # Status bar
        status = ttk.Frame(self.root)
        status.pack(fill="x", padx=8, pady=(0, 6))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        self.gpu_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.gpu_var, style="Muted.TLabel").pack(side="right", padx=(0, 8))
        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.elapsed_var, style="Muted.TLabel").pack(side="right", padx=(0, 8))
        self.progress = ttk.Progressbar(status, mode="determinate", length=260)
        self.progress.pack(side="right")

        # Apply theme to newly built widgets
        self._apply_theme()
        # Start periodic GPU stat refresh
        self._refresh_gpu_widget()

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
        attach_tooltip(
            sam3_entry,
            "Text prompt for SAM3. Use | for fallback variants tried until masks found\n"
            "e.g. paint mark|dark spot|stain — commas separate per-class prompts",
        )

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
        self.export_accepted_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text="Accepted only", variable=self.export_accepted_only_var).pack(side="left", padx=(2, 6))
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
        if hasattr(self, "tb_run_btn"):
            self.tb_run_btn.configure(state=("disabled" if running else "normal"))
        if hasattr(self, "tb_cancel_btn"):
            self.tb_cancel_btn.configure(state=("normal" if running else "disabled"))
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
        # Auto-refresh dashboard with latest stats
        try:
            self._refresh_dashboard()
        except Exception:  # noqa: BLE001
            pass
        self.notebook.select(self.results_tab)

    # ---------- labels viewer ----------
    def _generate_yolo_lines_for_bundle(self, bundle: AnnotationBundle) -> str:
        if not self._classes:
            classes = sorted({mask.class_id for mask in bundle.masks})
        else:
            classes = self._classes
        class_to_id = {name: i for i, name in enumerate(classes)}

        segmentation = False
        if self.last_config:
            segmentation = self.last_config.yolo_segmentation
        else:
            segmentation = (self.last_output_path / "yolo_seg_labels").exists() if self.last_output_path else False

        lines = []
        width = max(bundle.image.width, 1)
        height = max(bundle.image.height, 1)
        for mask in bundle.masks:
            if mask.class_id not in class_to_id:
                continue
            cid = class_to_id[mask.class_id]

            if segmentation and mask.polygon and len(mask.polygon) >= 3:
                coords = []
                for (px, py) in mask.polygon:
                    nx = min(1.0, max(0.0, px / width))
                    ny = min(1.0, max(0.0, py / height))
                    coords.append(f"{nx:.6f}")
                    coords.append(f"{ny:.6f}")
                lines.append(f"{cid} " + " ".join(coords))
                continue

            x, y, w, h = mask.bbox
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            x_center = min(1.0, max(0.0, (x + (w / 2.0)) / width))
            y_center = min(1.0, max(0.0, (y + (h / 2.0)) / height))
            norm_w = min(1.0, max(0.0, w / width))
            norm_h = min(1.0, max(0.0, h / height))
            lines.append(
                f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )
        return "\n".join(lines)

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

        if self.last_bundles:
            for bundle in self.last_bundles:
                stem = bundle.image.path.stem
                if labels_dir:
                    lf = labels_dir / f"{stem}.txt"
                    if lf.exists():
                        self._label_files[stem] = lf
                        continue
                self._label_files[stem] = None
        else:
            if labels_dir is not None:
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
                if lf is not None:
                    lines = [l for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
                    count = len(lines)
                else:
                    bundle = next((b for b in self.last_bundles if b.image.path.stem == stem), None)
                    count = len(bundle.masks) if bundle else 0
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
        if lf is not None:
            content = lf.read_text(encoding="utf-8")
        else:
            bundle = next((b for b in self.last_bundles if b.image.path.stem == stem), None)
            if bundle:
                content = self._generate_yolo_lines_for_bundle(bundle)
            else:
                content = ""

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
            if lf is not None:
                lines = lf.read_text(encoding="utf-8").splitlines()
            else:
                bundle = next((b for b in self.last_bundles if b.image.path.stem == stem), None)
                if bundle:
                    lines = self._generate_yolo_lines_for_bundle(bundle).splitlines()
                else:
                    lines = []
            for idx, line in enumerate(lines):
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
        self._load_labels_list()
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
        self._load_labels_list()
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

        accepted_only = self.export_accepted_only_var.get()
        if accepted_only:
            bundles = [b for b in bundles if b.status == "ACCEPTED"]
            if not bundles:
                messagebox.showwarning("No accepted annotations", "No ACCEPTED annotations found. Uncheck 'Accepted only' to export all.")
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
            status_note = " (ACCEPTED only)" if accepted_only else ""
            self.status_var.set(f"Exported {len(annotated_bundles)} annotated image(s) to YOLO format{status_note}.")
            messagebox.showinfo(
                "Export complete",
                f"Successfully exported {len(annotated_bundles)} annotated image(s)"
                f"{status_note} to:\n{labels_dir}"
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    # ---------- shortcuts ----------
    def _resolve_output_dir(self) -> Optional[Path]:
        """Pick the best-known output dir: last run > current field > resolved field."""
        if self.last_output_path and self.last_output_path.exists():
            return self.last_output_path
        raw = self.output_path.get().strip()
        if raw:
            p = Path(raw)
            if p.exists():
                return p
            p2 = resolve_path(p)
            if p2.exists():
                return p2
        return None

    def _open_output(self) -> None:
        out = self._resolve_output_dir()
        if out is None:
            messagebox.showinfo(
                "No output yet",
                "No output folder available. Run the pipeline or set an output folder.",
            )
            return
        open_in_explorer(out)

    def _open_yolo(self) -> None:
        out = self._resolve_output_dir()
        if out is None:
            messagebox.showinfo("No output yet", "No YOLO labels yet — run the pipeline first.")
            return
        for cand in ("yolo_seg_labels", "yolo_labels"):
            d = out / cand
            if d.exists():
                open_in_explorer(d)
                return
        open_in_explorer(out)

    # ---------- Few-Shot tab ----------
    def _build_fewshot_tab(self) -> None:
        frm = ttk.Frame(self.fewshot_tab, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="🎯 Few-Shot SAM3 Annotator", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(
            frm,
            text="Annotate a few images manually in Labeler → scan them here → SAM3 propagates masks to all remaining images",
            foreground="#888888",
        ).pack(anchor="w", pady=(2, 10))

        paned = ttk.PanedWindow(frm, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # ── Left: reference images ────────────────────────────────────────────
        left = ttk.LabelFrame(paned, text="Reference Images (manually annotated)", padding=6)
        paned.add(left, weight=1)

        ref_ctrl = ttk.Frame(left)
        ref_ctrl.pack(fill="x", pady=(0, 4))
        ttk.Button(ref_ctrl, text="🔍 Scan from output folder", command=self._fewshot_scan_refs).pack(side="left")
        ttk.Button(ref_ctrl, text="✕ Clear", command=self._fewshot_clear_refs).pack(side="left", padx=6)
        self._fs_ref_count_var = tk.StringVar(value="0 references loaded")
        ttk.Label(ref_ctrl, textvariable=self._fs_ref_count_var, foreground="#888888").pack(side="left", padx=8)

        ref_list_frm = ttk.Frame(left)
        ref_list_frm.pack(fill="both", expand=True)
        ref_scroll = ttk.Scrollbar(ref_list_frm, orient="vertical")
        self._fs_ref_listbox = tk.Listbox(ref_list_frm, font=("Consolas", 9), yscrollcommand=ref_scroll.set, selectmode="extended")
        ref_scroll.config(command=self._fs_ref_listbox.yview)
        ref_scroll.pack(side="right", fill="y")
        self._fs_ref_listbox.pack(fill="both", expand=True)

        # ── Right: targets + settings ─────────────────────────────────────────
        right = ttk.Frame(paned, padding=6)
        paned.add(right, weight=1)

        ttk.Label(right, text="Target Images (unannotated)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self._fs_target_count_var = tk.StringVar(value="0 targets found")
        ttk.Label(right, textvariable=self._fs_target_count_var, foreground="#555555").pack(anchor="w", pady=(2, 8))

        tgt_list_frm = ttk.Frame(right)
        tgt_list_frm.pack(fill="both", expand=True)
        tgt_scroll = ttk.Scrollbar(tgt_list_frm, orient="vertical")
        self._fs_target_listbox = tk.Listbox(tgt_list_frm, font=("Consolas", 9), yscrollcommand=tgt_scroll.set)
        tgt_scroll.config(command=self._fs_target_listbox.yview)
        tgt_scroll.pack(side="right", fill="y")
        self._fs_target_listbox.pack(fill="both", expand=True)

        # Settings
        opts = ttk.LabelFrame(right, text="Settings", padding=6)
        opts.pack(fill="x", pady=(10, 0))
        ttk.Label(opts, text="Class name:").grid(row=0, column=0, sticky="w")
        self._fs_class_var = tk.StringVar(value="object")
        ttk.Entry(opts, textvariable=self._fs_class_var, width=20).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(opts, text="Confidence threshold:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._fs_conf_var = tk.StringVar(value="0.3")
        ttk.Entry(opts, textvariable=self._fs_conf_var, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(4, 0))

        # Run button + progress
        run_frm = ttk.Frame(frm)
        run_frm.pack(fill="x", pady=(10, 0))
        self._fs_run_btn = ttk.Button(run_frm, text="▶ Run Few-Shot SAM3", style="Accent.TButton", command=self._fewshot_run)
        self._fs_run_btn.pack(side="left")
        self._fs_cancel_var = tk.BooleanVar(value=False)
        self._fs_cancel_btn = ttk.Button(run_frm, text="⏹ Cancel", command=lambda: self._fs_cancel_var.set(True), state="disabled")
        self._fs_cancel_btn.pack(side="left", padx=6)
        self._fs_status_var = tk.StringVar(value="")
        ttk.Label(run_frm, textvariable=self._fs_status_var, foreground="#555555").pack(side="left", padx=10)

        self._fs_progress_var = tk.DoubleVar(value=0.0)
        self._fs_progress = ttk.Progressbar(frm, variable=self._fs_progress_var, maximum=100)
        self._fs_progress.pack(fill="x", pady=(6, 0))

        # Internal state
        self._fs_refs: List[Tuple[Path, List[Tuple[int, int, int, int]]]] = []  # (img_path, [bboxes])
        self._fs_targets: List[Path] = []

    def _fewshot_scan_refs(self) -> None:
        """Scan output folder for annotated images → load as references."""
        out_dir = self._resolve_output_dir()
        dataset_dir = Path(self.dataset_path.get().strip() or "")
        if not out_dir or not out_dir.exists():
            messagebox.showwarning("No output folder", "Set the output folder in Setup tab first.")
            return

        labels_dir = None
        for cand in ("yolo_seg_labels", "yolo_labels", "labels"):
            d = out_dir / cand
            if d.exists():
                labels_dir = d
                break
        if labels_dir is None:
            messagebox.showwarning("No labels", f"No label folder found in:\n{out_dir}\nRun pipeline or annotate images first.")
            return

        refs = []
        for lf in sorted(labels_dir.glob("*.txt")):
            if lf.name == "classes.txt":
                continue
            stem = lf.stem
            lines = [l.strip() for l in lf.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                continue
            # Find image
            img_path = None
            if dataset_dir.exists():
                for ext in IMAGE_EXTS:
                    candidates = list(dataset_dir.rglob(f"{stem}{ext}"))
                    if candidates:
                        img_path = candidates[0]
                        break
            if img_path is None:
                continue
            # Parse YOLO bboxes (normalized cx cy w h or polygon coords nx1 ny1 nx2 ny2... → pixel x1y1x2y2)
            try:
                from PIL import Image as _PILImg
                with _PILImg.open(img_path) as im:
                    W, H = im.size
            except Exception:
                continue
            bboxes = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    if len(parts) >= 7 and len(parts) % 2 == 1:
                        # Polygon format: class x1 y1 x2 y2 ...
                        xs = [float(parts[i]) for i in range(1, len(parts), 2)]
                        ys = [float(parts[i]) for i in range(2, len(parts), 2)]
                        x1 = int(min(xs) * W)
                        y1 = int(min(ys) * H)
                        x2 = int(max(xs) * W)
                        y2 = int(max(ys) * H)
                    else:
                        # Bbox format: class cx cy w h
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = int((cx - bw / 2) * W)
                        y1 = int((cy - bh / 2) * H)
                        x2 = int((cx + bw / 2) * W)
                        y2 = int((cy + bh / 2) * H)
                    bboxes.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
                except Exception:
                    continue
            if bboxes:
                refs.append((img_path, bboxes))

        if not refs:
            messagebox.showwarning("No references found", f"No annotated images found in labels folder ({labels_dir.name}).\nAnnotate some images in Labeler or run pipeline first.")
            return

        self._fs_refs = refs
        self._fs_ref_listbox.delete(0, "end")
        for p, bboxes in refs:
            self._fs_ref_listbox.insert("end", f"[{len(bboxes)} box{'es' if len(bboxes)!=1 else ''}]  {p.name}")
        self._fs_ref_count_var.set(f"{len(refs)} reference(s) loaded")

        # Scan targets: images NOT in refs
        ref_stems = {p.stem for p, _ in refs}
        all_images = []
        if dataset_dir.exists():
            for ext in IMAGE_EXTS:
                all_images.extend(dataset_dir.rglob(f"*{ext}"))
        targets = sorted(set(p for p in all_images if p.stem not in ref_stems))
        self._fs_targets = targets
        self._fs_target_listbox.delete(0, "end")
        for p in targets:
            self._fs_target_listbox.insert("end", p.name)
        self._fs_target_count_var.set(f"{len(targets)} target image(s) to annotate")

    def _fewshot_clear_refs(self) -> None:
        self._fs_refs = []
        self._fs_targets = []
        self._fs_ref_listbox.delete(0, "end")
        self._fs_target_listbox.delete(0, "end")
        self._fs_ref_count_var.set("0 references loaded")
        self._fs_target_count_var.set("0 targets found")
        self._fs_status_var.set("")
        self._fs_progress_var.set(0.0)

    def _fewshot_run(self) -> None:
        if not self._fs_refs:
            messagebox.showwarning("No references", "Scan references first.")
            return
        if not self._fs_targets:
            messagebox.showwarning("No targets", "No unannotated target images found.")
            return

        out_dir = self._resolve_output_dir()
        dataset_dir = Path(self.dataset_path.get().strip() or "")
        if not out_dir:
            messagebox.showwarning("No output folder", "Set output folder in Setup tab.")
            return

        class_name = self._fs_class_var.get().strip() or "object"
        try:
            conf = float(self._fs_conf_var.get().strip())
        except ValueError:
            conf = 0.3

        params = {
            "backend": "hf_local",
            "local_dir": "./models/sam3",
            "hf_score_threshold": conf,
            "allow_mock_fallback": True,
        }

        refs = list(self._fs_refs)
        targets = list(self._fs_targets)
        total = len(targets)

        self._fs_run_btn.configure(state="disabled")
        self._fs_cancel_btn.configure(state="normal")
        self._fs_cancel_var.set(False)
        self._fs_progress_var.set(0.0)
        self._fs_status_var.set(f"Starting few-shot run on {total} images…")

        import threading as _threading

        def _worker():
            from src.tools.sam3 import sam3_segment_fewshot
            from src.tools.yolo.exporter import export_yolo
            from src.core.models import AnnotationBundle, ImageRecord, MaskRecord, new_mask_id, ProjectConfig
            import time

            # Determine labels folder name: yolo_seg_labels if segmentation, else yolo_labels
            labels_folder_name = "yolo_labels"
            if self.last_config and self.last_config.yolo_segmentation:
                labels_folder_name = "yolo_seg_labels"
            elif (out_dir / "yolo_seg_labels").exists():
                labels_folder_name = "yolo_seg_labels"

            labels_dir = out_dir / labels_folder_name
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Process in batches of 8 to avoid OOM
            BATCH = 8
            all_results = {}
            for batch_start in range(0, total, BATCH):
                if self._fs_cancel_var.get():
                    break
                batch = targets[batch_start: batch_start + BATCH]
                self.root.after(0, lambda s=batch_start: self._fs_status_var.set(
                    f"Processing {s+1}–{min(s+BATCH, total)}/{total}…"
                ))
                try:
                    batch_res = sam3_segment_fewshot(
                        ref_data=refs,
                        target_paths=batch,
                        class_name=class_name,
                        model_name="facebook/sam3",
                        params=params,
                    )
                    all_results.update(batch_res)
                except Exception as exc:
                    self.root.after(0, lambda e=str(exc): messagebox.showerror("Few-shot error", e))
                    break
                pct = min(100.0, (batch_start + len(batch)) / total * 100)
                self.root.after(0, lambda p=pct: self._fs_progress_var.set(p))

            # Write YOLO label files
            written = 0
            for p in targets:
                masks = all_results.get(str(p), [])
                if not masks:
                    continue
                label_path = labels_dir / f"{p.stem}.txt"
                lines = []
                try:
                    from PIL import Image as _PI
                    with _PI.open(p) as im:
                        W, H = im.size
                except Exception:
                    continue
                for m in masks:
                    if m.polygon and len(m.polygon) >= 3:
                        coords = " ".join(f"{x/W:.6f} {y/H:.6f}" for x, y in m.polygon)
                        lines.append(f"0 {coords}")
                    elif m.bbox:
                        x1, y1, bw, bh = m.bbox
                        cx = (x1 + bw / 2) / W
                        cy = (y1 + bh / 2) / H
                        lines.append(f"0 {cx:.6f} {cy:.6f} {bw/W:.6f} {bh/H:.6f}")
                if lines:
                    label_path.write_text("\n".join(lines), encoding="utf-8")
                    written += 1

            # Ensure classes.txt exists in the output folder
            classes_file = out_dir / "classes.txt"
            if not classes_file.exists():
                classes_file.write_text(class_name, encoding="utf-8")

            # Update in-memory bundles so results list updates immediately
            if self.last_bundles:
                for p in targets:
                    masks_raw = all_results.get(str(p), [])
                    if not masks_raw:
                        continue
                    # Find or create bundle
                    bundle = next((b for b in self.last_bundles if b.image.path == p), None)
                    if not bundle:
                        try:
                            from PIL import Image as _PI
                            with _PI.open(p) as im:
                                w_img, h_img = im.size
                        except Exception:
                            w_img, h_img = 1024, 1024
                        im_rec = ImageRecord(id=f"img_{p.stem}", path=p, width=w_img, height=h_img)
                        bundle = AnnotationBundle(image=im_rec, status="ACCEPTED")
                        self.last_bundles.append(bundle)
                    
                    bundle.status = "ACCEPTED"
                    self._bundle_status[p.stem] = "ACCEPTED"
                    
                    # Convert raw masks to MaskRecords
                    converted_masks = []
                    for m in masks_raw:
                        converted_masks.append(MaskRecord(
                            mask_id=new_mask_id(),
                            image_id=bundle.image.id,
                            class_id=class_name,
                            polygon=m.polygon,
                            bbox=m.bbox,
                            area=m.area,
                            confidence=m.confidence,
                            source="sam3",
                            version=1
                        ))
                    bundle.masks = converted_masks

            cancelled = self._fs_cancel_var.get()
            def _done():
                self._fs_run_btn.configure(state="normal")
                self._fs_cancel_btn.configure(state="disabled")
                self._fs_progress_var.set(100.0 if not cancelled else self._fs_progress_var.get())
                msg = f"Cancelled. " if cancelled else ""
                self._fs_status_var.set(f"{msg}Done. {written}/{total} images annotated → {labels_dir}")
                if written > 0:
                    self._load_labels_list()  # Refresh results tab lists!
                    messagebox.showinfo("Few-Shot Complete", f"Annotated {written} of {total} target images.\nLabels saved to:\n{labels_dir}")
            self.root.after(0, _done)

        _threading.Thread(target=_worker, daemon=True).start()

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
