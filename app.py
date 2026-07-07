#!/usr/bin/env python3
"""
Real2Minecraft v2 — Photo to Minecraft voxels
Image -> TRELLIS (image-to-3D) -> normalize -> voxelize -> textured viewer / .schem

This module is the Tk UI + pipeline orchestration only; the pipeline itself lives in:
  trellis_client  - cutout + TRELLIS API calls
  mesh_normalize  - GLB -> upright, centered mesh
  voxelize        - SAT voxelization + the color pipeline (voxel_sampling,
                    color_repair, color_quantize) + palette matching
  voxel_view      - pyvista viewer (+ matplotlib fallback)
  schematic       - .schem export
"""

import sys
import time
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from config import VOXEL_MIN_BLOCKS, VOXEL_MAX_BLOCKS
from trellis_client import TRELLIS_API_URL, _generate_mesh
from mesh_normalize import normalize_mesh
from voxelize import voxelize_mesh
from schematic import export_schematic
from voxel_view import show_voxel_diagnostic


# ── Main Application ──────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real2Minecraft v2")
        self.root.configure(bg="#0d1117")
        self.root.geometry("500x480")
        self.root.minsize(420, 400)

        self.image_path: Path | None  = None
        self.norm_path: Path | None   = None
        self.norm_info: dict | None   = None
        self.voxel_data: dict | None  = None
        self.is_generating = False
        self._preview_photo = None  # keep ref to prevent GC
        self._timer_start: float | None = None  # stopwatch start (time.time())

        # Voxel settings (exposed in the UI).
        self.res_var = tk.IntVar(value=48)      # target longest-axis size in blocks
        self.colormode_var = tk.StringVar(value="Enhanced (fix shadows & clumps)")

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = tk.Frame(self.root, bg="#161b22")
        ctrl.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        tk.Label(
            ctrl, text="Real2Minecraft v2",
            bg="#161b22", fg="#58a6ff",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(18, 2))
        tk.Label(
            ctrl, text="Image  ->  3D Mesh",
            bg="#161b22", fg="#8b949e",
            font=("Segoe UI", 9),
        ).pack(pady=(0, 14))

        # preview canvas
        self.preview_canvas = tk.Canvas(
            ctrl, width=260, height=260,
            bg="#0d1117", highlightthickness=1,
            highlightbackground="#30363d",
        )
        self.preview_canvas.pack(padx=16, pady=4)
        self.preview_canvas.create_text(
            130, 130,
            text="No image selected",
            fill="#4a5568", font=("Segoe UI", 9),
        )

        btn_frame = tk.Frame(ctrl, bg="#161b22")
        btn_frame.pack(fill=tk.X, padx=16, pady=6)

        self._btn_style = dict(
            relief=tk.FLAT, bd=0, padx=10, pady=6,
            font=("Segoe UI", 9),
            cursor="hand2",
            activeforeground="white",
        )

        tk.Button(
            btn_frame, text="Choose Image",
            command=self._choose_image,
            bg="#21262d", fg="#c9d1d9",
            activebackground="#30363d",
            **self._btn_style,
        ).pack(fill=tk.X, pady=3)

        self.build_btn = tk.Button(
            btn_frame, text="🧱  Voxelize the Build",
            command=self._on_build,
            bg="#238636", fg="white",
            activebackground="#2ea043",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, bd=0, padx=10, pady=8,
            cursor="hand2", activeforeground="white",
            state=tk.DISABLED,
        )
        self.build_btn.pack(fill=tk.X, pady=3)

        # ── voxel settings ───────────────────────────────────────────────────
        settings = tk.Frame(btn_frame, bg="#161b22")
        settings.pack(fill=tk.X, pady=(4, 2))

        self.res_label = tk.Label(
            settings, bg="#161b22", fg="#8b949e", font=("Segoe UI", 8),
            text="Target size: {} blocks".format(self.res_var.get()),
        )
        self.res_label.pack(anchor="w")
        res_scale = ttk.Scale(
            settings, from_=VOXEL_MIN_BLOCKS, to=VOXEL_MAX_BLOCKS, orient=tk.HORIZONTAL,
            variable=self.res_var, command=self._on_res_change,
        )
        res_scale.pack(fill=tk.X)

        # Color mode: Enhanced runs the full repair pipeline (shadow/clump fixes);
        # Raw maps the model's colors as-is (best when the source has no baked
        # shadows and corrections would do more harm than good).
        mode_row = tk.Frame(settings, bg="#161b22")
        mode_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            mode_row, text="Color mode:", bg="#161b22", fg="#8b949e",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)
        ttk.Combobox(
            mode_row, textvariable=self.colormode_var, state="readonly", width=26,
            values=["Enhanced (fix shadows & clumps)", "Raw (model colors as-is)"],
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.export_btn = tk.Button(
            btn_frame, text="💾  Export .schem",
            command=self._on_export_schem,
            bg="#3d1f79", fg="white",
            activebackground="#5a2ea6",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, bd=0, padx=10, pady=8,
            cursor="hand2", activeforeground="white",
            state=tk.DISABLED,
        )
        self.export_btn.pack(fill=tk.X, pady=3)

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # Stopwatch — shows how long the system has been working.
        self.timer_var = tk.StringVar(value="")
        tk.Label(
            btn_frame, textvariable=self.timer_var,
            bg="#161b22", fg="#8b949e",
            font=("Consolas", 9),
        ).pack(fill=tk.X, pady=(2, 0))

        # env info label
        tk.Label(
            ctrl,
            text="Python: {}\n{}".format(Path(sys.executable).name, sys.executable),
            bg="#161b22", fg="#30363d",
            font=("Consolas", 7),
            wraplength=270,
            justify=tk.LEFT,
        ).pack(side=tk.BOTTOM, pady=4, padx=8, anchor="w")

        # ── status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready -- select an image to begin")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#161b22", fg="#8b949e",
            anchor=tk.W, padx=10, pady=3,
            font=("Segoe UI", 8),
        ).pack(side=tk.BOTTOM, fill=tk.X)

    # ── event handlers ────────────────────────────────────────────────────────

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose input image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.image_path = Path(path)
        self._show_preview(path)
        # New picture — clear prior results and the stopwatch.
        self.norm_path = None
        self.norm_info = None
        self.voxel_data = None
        self._reset_timer()
        self.build_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.DISABLED)
        self._set_status("Image: {}".format(self.image_path.name))

    def _show_preview(self, path: str):
        img = Image.open(path).convert("RGBA")
        img.thumbnail((260, 260), Image.LANCZOS)
        # Checkerboard for transparency
        cw = 10
        bg = Image.new("RGBA", img.size, (20, 20, 20, 255))
        for y in range(0, img.height, cw):
            for x in range(0, img.width, cw):
                if (x // cw + y // cw) % 2 == 0:
                    for dy in range(min(cw, img.height - y)):
                        for dx in range(min(cw, img.width - x)):
                            bg.putpixel((x + dx, y + dy), (35, 35, 35, 255))
        composite = Image.alpha_composite(bg, img)
        self._preview_photo = ImageTk.PhotoImage(composite)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(130, 130, anchor=tk.CENTER, image=self._preview_photo)

    def _on_res_change(self, _value):
        # ttk.Scale passes a float string; snap to int and refresh the label.
        self.res_var.set(int(float(_value)))
        self.res_label.config(text="Target size: {} blocks".format(self.res_var.get()))

    def _on_build(self):
        """Run the whole pipeline: model -> normalize -> voxelize -> view."""
        if self.is_generating or not self.image_path:
            return
        if not self.image_path.exists():
            messagebox.showerror("File not found", str(self.image_path))
            return
        # Snapshot settings on the main thread so the worker doesn't touch tk vars.
        self._build_target_blocks = int(self.res_var.get())
        self._build_color_repair = self.colormode_var.get().startswith("Enhanced")
        self.is_generating = True
        self.build_btn.config(state=tk.DISABLED, text="Working...")
        self.export_btn.config(state=tk.DISABLED)
        self.progress.start(12)
        self._start_timer()
        self._set_status("Launching TRELLIS...")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    # ── Build pipeline (TRELLIS -> normalize -> voxelize) ──────────────────────

    def _run_pipeline(self):
        try:
            self._set_status("Generating mesh via TRELLIS ({})...".format(TRELLIS_API_URL))
            obj_path = _generate_mesh(self.image_path,
                                      status_cb=self._set_status,
                                      preview_cb=self._show_cutout_preview)
            if obj_path is None or not obj_path.exists():
                raise FileNotFoundError("TRELLIS produced no mesh under outputs/.")

            self._set_status("Normalizing mesh...")
            norm_path, norm_info = normalize_mesh(obj_path)
            self.norm_path = norm_path
            self.norm_info = norm_info

            # Prefer the archived cutout as the colour reference — its alpha is the
            # exact object mask, so photo-guided repairs don't see background colours.
            ref_image = (next(obj_path.parent.glob(obj_path.stem + "_cutout.png"), None)
                         or self.image_path)

            self._set_status("Voxelizing — this may take a few seconds...")
            voxel_data = voxelize_mesh(
                norm_path, norm_info,
                target_blocks=self._build_target_blocks,
                color_repair=self._build_color_repair,
                image_path=ref_image,
            )
            self.root.after(0, self._on_build_done, voxel_data)

        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    # ── post-build callbacks (must run on main thread) ────────────────────────

    def _on_build_done(self, voxel_data: dict):
        self.voxel_data = voxel_data
        self._reset_controls()
        self._stop_timer()
        self.export_btn.config(state=tk.NORMAL)
        W, H, L = voxel_data["dims"]
        note = "" if voxel_data["watertight"] else " (mesh not watertight — may have holes)"
        self._set_status(
            "Voxelized: {:,} blocks  |  {}x{}x{} (XYZ){}".format(
                voxel_data["block_count"], W, H, L, note
            )
        )
        # Prefer the textured pyvista viewer; fall back to the matplotlib preview if
        # pyvista/VTK isn't available or errors out.
        try:
            from voxel_view import show_voxel_pyvista
            show_voxel_pyvista(voxel_data, mesh_info=self.norm_info)
        except Exception as exc:
            self._set_status("Textured viewer unavailable ({}); using basic preview".format(
                str(exc)[:60]))
            show_voxel_diagnostic(voxel_data)

    def _on_error(self, msg: str):
        self._reset_controls()
        self._stop_timer()
        self._set_status("Error: {}".format(msg[:100]))
        messagebox.showerror("Build failed", msg)

    def _on_export_schem(self):
        if not self.voxel_data:
            return
        path = filedialog.asksaveasfilename(
            title="Save Schematic",
            defaultextension=".schem",
            initialfile="model.schem",
            filetypes=[("Sponge Schematic", "*.schem"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            export_schematic(self.voxel_data, Path(path))
            self._set_status("Exported: {}".format(Path(path).name))
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            self._set_status("Export failed: {}".format(str(exc)[:80]))

    def _reset_controls(self):
        self.is_generating = False
        self.progress.stop()
        self.build_btn.config(state=tk.NORMAL, text="🧱  Voxelize the Build")

    # ── stopwatch ─────────────────────────────────────────────────────────────

    def _start_timer(self):
        self._timer_start = time.time()
        self._tick()

    def _tick(self):
        # Re-schedule itself once per second while a build is running.
        if self._timer_start is None:
            return
        elapsed = int(time.time() - self._timer_start)
        self.timer_var.set("Working for {:02d}:{:02d}".format(elapsed // 60, elapsed % 60))
        self.root.after(1000, self._tick)

    def _stop_timer(self):
        if self._timer_start is None:
            return
        elapsed = int(time.time() - self._timer_start)
        self._timer_start = None  # stops the _tick loop
        self.timer_var.set("Done in {:02d}:{:02d}".format(elapsed // 60, elapsed % 60))

    def _reset_timer(self):
        self._timer_start = None  # stops any running _tick loop
        self.timer_var.set("")

    def _show_cutout_preview(self, path):
        # Called from the worker thread: swap the preview to the background-removed
        # cutout so the user sees exactly what TRELLIS will reconstruct.
        self.root.after(0, lambda: self._show_preview(str(path)))

    def _set_status(self, msg: str):
        # Thread-safe status update
        self.root.after(0, lambda: self.status_var.set(msg))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
