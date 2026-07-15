"""Facade refine editor: drag the 4 facade corners / override the grid, re-apply.

A Tk Toplevel over the cutout image.  Apply re-runs only B3->B4->B5 through
`pipeline.rerun_from_facade` (seconds — no TRELLIS, no SAT) on a worker thread, and
every Apply is captured into the dataset sample as an (auto prediction, human
correction) pair — the refine UI doubles as a labeling tool.
"""

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from building import pipeline

_HANDLE_R = 6
_MAX_W, _MAX_H = 640, 480


class FacadeEditor(tk.Toplevel):
    def __init__(self, parent: tk.Tk, build_state: dict, on_applied):
        super().__init__(parent)
        self.title("Facade tools")
        self.configure(bg="#161b22")
        self.resizable(False, False)
        self.state_dict = build_state
        self.on_applied = on_applied
        self._busy = False

        src = build_state.get("cutout_path") or build_state.get("photo_path")
        img = Image.open(src).convert("RGBA")
        bg = Image.new("RGBA", img.size, (30, 34, 42, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
        self.img_w, self.img_h = img.size
        self.scale = min(_MAX_W / self.img_w, _MAX_H / self.img_h, 1.0)
        disp = img.resize((max(1, int(self.img_w * self.scale)),
                           max(1, int(self.img_h * self.scale))), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(disp)

        self.canvas = tk.Canvas(self, width=disp.width, height=disp.height,
                                bg="#0d1117", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # Initial quad: the analyzed facade 0, else the image bounds.
        bundle = build_state.get("facade_bundle")
        if bundle and bundle["ir"]["facades"]:
            f0 = bundle["ir"]["facades"][0]
            quad = [list(p) for p in f0["quad_px"]]
            self._init_rows = f0["grid"]["n_rows"]
            self._init_cols = f0["grid"]["n_cols"]
        else:
            m = 0.05
            quad = [[self.img_w * m, self.img_h * m],
                    [self.img_w * (1 - m), self.img_h * m],
                    [self.img_w * (1 - m), self.img_h * (1 - m)],
                    [self.img_w * m, self.img_h * (1 - m)]]
            self._init_rows, self._init_cols = 8, 8
        self.quad = quad                                  # image px, TL,TR,BR,BL
        self._drag_idx = None

        self._poly = self.canvas.create_polygon(
            *self._canvas_pts(), outline="#58a6ff", fill="", width=2)
        self._grid_lines = []
        self._handles = []
        for i in range(4):
            h = self.canvas.create_oval(0, 0, 0, 0, fill="#58a6ff",
                                        outline="white", width=1)
            self._handles.append(h)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag_idx", None))

        ctrl = tk.Frame(self, bg="#161b22")
        ctrl.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(ctrl, text="Floors:", bg="#161b22", fg="#8b949e",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.rows_var = tk.IntVar(value=self._init_rows)
        ttk.Spinbox(ctrl, from_=1, to=128, width=5,
                    textvariable=self.rows_var).pack(side=tk.LEFT, padx=(4, 12))
        tk.Label(ctrl, text="Bays:", bg="#161b22", fg="#8b949e",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.cols_var = tk.IntVar(value=self._init_cols)
        ttk.Spinbox(ctrl, from_=1, to=128, width=5,
                    textvariable=self.cols_var).pack(side=tk.LEFT, padx=(4, 12))

        self.apply_btn = tk.Button(ctrl, text="Apply", command=self._on_apply,
                                   bg="#238636", fg="white", relief=tk.FLAT,
                                   padx=14, pady=4, cursor="hand2")
        self.apply_btn.pack(side=tk.RIGHT, padx=(6, 0))
        self.reset_btn = tk.Button(ctrl, text="Reset to auto", command=self._on_reset,
                                   bg="#21262d", fg="#c9d1d9", relief=tk.FLAT,
                                   padx=10, pady=4, cursor="hand2")
        self.reset_btn.pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Drag corners to the facade, then Apply.")
        tk.Label(self, textvariable=self.status_var, bg="#161b22", fg="#8b949e",
                 anchor=tk.W, font=("Segoe UI", 8)).pack(fill=tk.X, padx=8, pady=(0, 6))

        self._redraw()

    # ── canvas helpers ────────────────────────────────────────────────────────

    def _canvas_pts(self):
        out = []
        for x, y in self.quad:
            out += [x * self.scale, y * self.scale]
        return out

    def _redraw(self):
        self.canvas.coords(self._poly, *self._canvas_pts())
        for i, h in enumerate(self._handles):
            cx, cy = self.quad[i][0] * self.scale, self.quad[i][1] * self.scale
            self.canvas.coords(h, cx - _HANDLE_R, cy - _HANDLE_R,
                               cx + _HANDLE_R, cy + _HANDLE_R)
        for line in self._grid_lines:
            self.canvas.delete(line)
        self._grid_lines = []
        tl, tr, br, bl = [np.asarray(p, dtype=float) for p in self.quad]
        rows = max(1, int(self.rows_var.get() or 1))
        cols = max(1, int(self.cols_var.get() or 1))
        for k in range(1, rows):
            t = k / rows
            a = (tl + (bl - tl) * t) * self.scale
            b = (tr + (br - tr) * t) * self.scale
            self._grid_lines.append(self.canvas.create_line(
                *a, *b, fill="#58a6ff", stipple="gray50"))
        for k in range(1, cols):
            t = k / cols
            a = (tl + (tr - tl) * t) * self.scale
            b = (bl + (br - bl) * t) * self.scale
            self._grid_lines.append(self.canvas.create_line(
                *a, *b, fill="#58a6ff", stipple="gray50"))

    def _on_press(self, event):
        best, best_d = None, 1e9
        for i, (x, y) in enumerate(self.quad):
            d = ((x * self.scale - event.x) ** 2
                 + (y * self.scale - event.y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = i, d
        self._drag_idx = best if best_d <= 3 * _HANDLE_R else None

    def _on_drag(self, event):
        if self._drag_idx is None or self._busy:
            return
        x = min(max(event.x / self.scale, 0), self.img_w - 1)
        y = min(max(event.y / self.scale, 0), self.img_h - 1)
        self.quad[self._drag_idx] = [x, y]
        self._redraw()

    # ── apply / reset ────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool, msg: str):
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.apply_btn.config(state=state)
        self.reset_btn.config(state=state)
        self.status_var.set(msg)

    def _run(self, quad_override, grid_override, done_msg):
        def work():
            try:
                voxel_data = pipeline.rerun_from_facade(
                    self.state_dict, quad_override=quad_override,
                    grid_override=grid_override)
                self.after(0, lambda: self._on_done(voxel_data, done_msg))
            except Exception as exc:
                msg = str(exc)[:120]
                self.after(0, lambda: self._set_busy(False, "Error: " + msg))
        threading.Thread(target=work, daemon=True).start()

    def _on_apply(self):
        if self._busy:
            return
        self._redraw()
        grid_override = None
        rows, cols = int(self.rows_var.get()), int(self.cols_var.get())
        if (rows, cols) != (self._init_rows, self._init_cols):
            grid_override = {"n_rows": rows, "n_cols": cols}
        self._set_busy(True, "Re-applying facade...")
        self._run([list(p) for p in self.quad], grid_override, "Facade re-applied.")

    def _on_reset(self):
        if self._busy:
            return
        self._set_busy(True, "Restoring automatic facade...")
        self._run(None, None, "Automatic facade restored.")

    def _on_done(self, voxel_data, msg):
        # Refresh the quad/grid display from the recomputed facade.
        bundle = self.state_dict.get("facade_bundle")
        if bundle and bundle["ir"]["facades"]:
            f0 = bundle["ir"]["facades"][0]
            self.quad = [list(p) for p in f0["quad_px"]]
            self._init_rows = f0["grid"]["n_rows"]
            self._init_cols = f0["grid"]["n_cols"]
            self.rows_var.set(self._init_rows)
            self.cols_var.set(self._init_cols)
        self._redraw()
        self._set_busy(False, msg)
        if self.on_applied:
            self.on_applied(voxel_data)
