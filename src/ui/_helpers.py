"""Shared helpers for GUI modules (review_app + labeler)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PALETTE = [
    "#ff4d4d", "#4dff88", "#4d88ff", "#ffd24d",
    "#ff4dff", "#4dffff", "#ff8c4d", "#a04dff",
    "#4dd4ff", "#ff4d99",
]


def project_root() -> Path:
    """Resolve project root from this module's location (src/ui/_helpers.py)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p.resolve()
    if p.exists():
        return p.resolve()

    candidate = project_root() / p
    if candidate.exists():
        return candidate.resolve()

    return p.resolve()


def color_for(class_name: str) -> str:
    # Python's builtin hash() is randomized per process (PYTHONHASHSEED), so a
    # class's color would shift on every restart. Use a stable hash instead —
    # the same class must render the same color across sessions.
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    return PALETTE[int.from_bytes(digest[:4], "big") % len(PALETTE)]


def load_json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_json(path: Path, data) -> None:
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def open_in_explorer(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:  # noqa: BLE001
        pass


def attach_tooltip(widget: tk.Widget, text: str) -> None:
    tip = {"win": None}

    def _show(_e=None):
        if tip["win"] is not None:
            return
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        win = tk.Toplevel(widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            win, text=text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("Segoe UI", 9), padx=6, pady=3,
        )
        lbl.pack()
        tip["win"] = win

    def _hide(_e=None):
        w = tip["win"]
        if w is not None:
            w.destroy()
            tip["win"] = None

    widget.bind("<Enter>", _show)
    widget.bind("<Leave>", _hide)
