#!/usr/bin/env python3
"""
Real2Minecraft v2 — Photo to 3D Mesh
Image in → InstantMesh → interactive 3D viewer
"""

import os
import sys
import threading
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk


def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation matrix that rotates unit vector a onto unit vector b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float32)
    if c < -1.0 + 1e-8:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return (np.eye(3) + kmat + kmat @ kmat * (1.0 - c) / (s * s)).astype(np.float32)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
INSTANTMESH_DIR = PROJECT_ROOT / "InstantMesh"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Large config gives the best quality; use instant-mesh-base for faster runs.
IM_CONFIG = INSTANTMESH_DIR / "configs" / "instant-mesh-large.yaml"

# ── Resolve the Python interpreter to use for InstantMesh ─────────────────────
def _find_venv_python() -> str:
    """Return the Python executable that has InstantMesh's dependencies."""
    candidates = [
        PROJECT_ROOT / "r2m_v2" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Fall back to the same interpreter running this app.
    return sys.executable


VENV_PYTHON = _find_venv_python()


# ── 3-D Mesh Viewer ───────────────────────────────────────────────────────────
class MeshViewer(tk.Frame):
    """Matplotlib-based interactive 3-D mesh viewer embedded in Tkinter."""

    MAX_FACES = 8_000  # decimate above this for smooth interaction

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#0d1117", **kwargs)

        self.fig = Figure(figsize=(5.5, 5.5), dpi=96, facecolor="#0d1117")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0d1117")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar below the 3-D canvas
        btn_bar = tk.Frame(self, bg="#161b22")
        btn_bar.pack(side=tk.BOTTOM, fill=tk.X)

        _btn_cfg = dict(
            bg="#21262d", fg="#8b949e",
            relief=tk.FLAT, bd=0, padx=8, pady=4,
            font=("Segoe UI", 8), cursor="hand2",
            activebackground="#30363d", activeforeground="white",
            state=tk.DISABLED,
        )
        self._rot_btns: list[tk.Button] = []
        for label, axis, deg in [
            ("Flip ↕", 0, 180),
            ("Turn ←", 2, 90), ("Turn →", 2, -90),
            ("Tilt ↑", 0, 90),  ("Tilt ↓", 0, -90),
        ]:
            b = tk.Button(btn_bar, text=label,
                          command=lambda a=axis, d=deg: self._rotate(a, d),
                          **_btn_cfg)
            b.pack(side=tk.LEFT, padx=2, pady=3)
            self._rot_btns.append(b)

        reset_b = tk.Button(btn_bar, text="Reset orient.",
                            command=self._reset_rot, **_btn_cfg)
        reset_b.pack(side=tk.RIGHT, padx=6, pady=3)
        self._rot_btns.append(reset_b)

        # Stored mesh state (set by load_mesh, used by _do_render)
        self._stored_verts: np.ndarray | None = None
        self._stored_faces: np.ndarray | None = None
        self._stored_vc: np.ndarray | None = None
        self._manual_rot: np.ndarray = np.eye(3, dtype=np.float32)

        # Drag rotation state
        self._drag_xy: tuple[int, int] | None = None
        self._drag_base_elev: float = 30.0
        self._drag_base_azim: float = -60.0
        self._view_elev: float = 30.0
        self._view_azim: float = -60.0
        self.ax.view_init(elev=self._view_elev, azim=self._view_azim)
        self.canvas.mpl_connect("button_press_event",   self._on_drag_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_drag_move)
        self.canvas.mpl_connect("button_release_event", self._on_drag_release)

        self._placeholder()

    # ── public API ────────────────────────────────────────────────────────────

    def show_placeholder(self):
        self._stored_verts = None
        for b in self._rot_btns:
            b.config(state=tk.DISABLED)
        self._placeholder()

    def load_mesh(self, obj_path: str):
        # force="mesh" merges all geometry and keeps vertex colours intact
        mesh = trimesh.load(obj_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if not geoms:
                raise ValueError("OBJ contains no geometry")
            mesh = trimesh.util.concatenate(geoms)

        # Read vertex colours (ColorVisuals has .vertex_colors directly)
        vc_raw = None
        try:
            raw = mesh.visual.vertex_colors
            if raw is not None and len(raw) == len(mesh.vertices):
                vc_raw = np.asarray(raw, dtype=np.uint8)[:, :3]
        except Exception:
            pass

        # Rebuild mesh with colours attached so they survive decimation
        mesh = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=mesh.faces,
            vertex_colors=vc_raw,
            process=False,
        )

        if len(mesh.faces) > self.MAX_FACES:
            saved_vc = None
            try:
                saved_vc = mesh.visual.vertex_colors
            except Exception:
                pass
            orig_verts = np.asarray(mesh.vertices, dtype=np.float32)
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=self.MAX_FACES)
                # QEM drops colors; restore via nearest-original-vertex lookup
                if saved_vc is not None:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(orig_verts)
                    _, idx = tree.query(np.asarray(mesh.vertices, dtype=np.float32), k=1)
                    mesh = trimesh.Trimesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        vertex_colors=np.asarray(saved_vc, dtype=np.uint8)[idx],
                        process=False,
                    )
            except Exception:
                step = max(1, len(mesh.faces) // self.MAX_FACES)
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces[::step],
                    vertex_colors=saved_vc,
                    process=False,
                )

        # Re-read colours after potential decimation
        vc = None
        try:
            raw = mesh.visual.vertex_colors
            if raw is not None and len(raw) == len(mesh.vertices):
                vc = np.asarray(raw, dtype=np.float32)[:, :3] / 255.0
        except Exception:
            pass

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        # Auto-orient: SVD ground-plane fit → rotate so ground is at bottom (Z-up)
        verts = self._auto_orient(verts)

        self._manual_rot = np.eye(3, dtype=np.float32)
        self._stored_verts = verts
        self._stored_faces = faces
        self._stored_vc = vc
        for b in self._rot_btns:
            b.config(state=tk.NORMAL)
        self._do_render()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _auto_orient(verts: np.ndarray) -> np.ndarray:
        """Rotate mesh so the flattest face is the floor (Z-up).

        Tries all 6 cardinal directions as "up" candidates, picks the one
        whose bottom-25% cluster is most planar, then ensures the centroid
        is in the upper half so the model isn't upside-down.
        """
        zup = np.array([0.0, 0.0, 1.0])
        candidates = [
            np.array([ 1., 0., 0.]), np.array([-1., 0., 0.]),
            np.array([ 0., 1., 0.]), np.array([ 0.,-1., 0.]),
            np.array([ 0., 0., 1.]), np.array([ 0., 0.,-1.]),
        ]
        best_rot = np.eye(3, dtype=np.float32)
        best_score = np.inf
        for up in candidates:
            proj = verts @ up
            p_range = float(proj.max() - proj.min())
            if p_range < 1e-6:
                continue
            bottom = verts[proj <= proj.min() + 0.25 * p_range]
            if len(bottom) < 4:
                continue
            c = bottom.mean(axis=0)
            _, sv, _ = np.linalg.svd(bottom - c, full_matrices=False)
            score = float(sv[-1])  # smallest singular value = most planar
            if score < best_score:
                best_score = score
                best_rot = _rotation_from_to(up, zup)
        v2 = (verts @ best_rot.T).astype(np.float32)
        # Flip upside-down: centroid should be above the Z midpoint
        z2 = v2[:, 2]
        if float(v2[:, 2].mean()) < (float(z2.min()) + float(z2.max())) / 2.0:
            flip = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)
            v2 = (v2 @ flip.T).astype(np.float32)
        return v2

    def _rotate(self, axis: int, deg: float):
        """Accumulate a 90° rotation around the chosen axis and re-render."""
        if self._stored_verts is None:
            return
        angle = np.radians(deg)
        c, s = float(np.cos(angle)), float(np.sin(angle))
        if axis == 0:
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        elif axis == 1:
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        self._manual_rot = R @ self._manual_rot
        self._do_render()

    def _reset_rot(self):
        """Undo all manual rotation, keeping only the auto-orient result."""
        self._manual_rot = np.eye(3, dtype=np.float32)
        self._view_elev = 30.0
        self._view_azim = -60.0
        self._do_render()

    # ── drag-to-rotate handlers ───────────────────────────────────────────────

    def _on_drag_press(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        self._drag_xy = (event.x, event.y)
        self._drag_base_elev = self._view_elev
        self._drag_base_azim = self._view_azim

    def _on_drag_move(self, event):
        if self._drag_xy is None or event.button != 1:
            return
        dx = event.x - self._drag_xy[0]
        dy = event.y - self._drag_xy[1]
        self._view_elev = float(np.clip(self._drag_base_elev - dy * 0.4, -90, 90))
        self._view_azim = self._drag_base_azim - dx * 0.4
        self.ax.view_init(elev=self._view_elev, azim=self._view_azim)
        self.canvas.draw_idle()

    def _on_drag_release(self, event):
        if self._drag_xy is None or event.button != 1:
            return
        self._drag_xy = None
        # _view_elev / _view_azim already hold the final camera position from
        # _on_drag_move — nothing to bake, no jerk.

    def _do_render(self):
        if self._stored_verts is None:
            return
        verts = (self._stored_verts @ self._manual_rot.T).astype(np.float32)
        self._render(verts, self._stored_faces, self._stored_vc)
        # ax.cla() inside _render resets the view angles; restore them.
        self.ax.view_init(elev=self._view_elev, azim=self._view_azim)
        self.canvas.draw_idle()

    def _placeholder(self):
        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.text2D(
            0.5, 0.5,
            "No mesh loaded\n\nGenerate a 3-D model\nto view it here",
            ha="center", va="center",
            color="#4a5568", fontsize=11,
            transform=self.ax.transAxes,
        )
        self.canvas.draw_idle()

    def _render(self, verts: np.ndarray, faces: np.ndarray, vc: np.ndarray | None = None):
        tris = verts[faces]  # (F, 3, 3)

        # Phong diffuse — always computed so we can use it for shading
        e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
        normals = np.cross(e1, e2)
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        normals /= lens

        light = np.array([0.4, 0.6, 1.0], dtype=np.float32)
        light /= np.linalg.norm(light)
        diffuse = np.clip(normals @ light, 0.15, 1.0)  # (F,) — min 0.15 so dark faces stay visible

        if vc is not None and len(vc) == len(verts):
            # Average vertex colours per face, then brighten and apply diffuse lighting
            face_rgb = vc[faces].mean(axis=1)               # (F, 3)
            # Normalise: stretch the colour range so the brightest face = 1.0
            max_val = face_rgb.max()
            if max_val > 0.01:
                face_rgb = face_rgb / max_val               # normalise to full range
            face_rgb = np.clip(face_rgb * diffuse[:, None] * 1.2, 0, 1)
            face_colors = np.column_stack([face_rgb, np.ones(len(faces))])
        else:
            # Phong-only fallback (cool-blue tint)
            r = diffuse * 0.35
            g = diffuse * 0.55
            b = diffuse * 0.80
            face_colors = np.stack([r, g, b, np.ones_like(diffuse)], axis=1)

        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.set_facecolor("#0d1117")

        poly = Poly3DCollection(tris, zsort="average")
        poly.set_facecolor(face_colors)
        poly.set_edgecolor("none")
        self.ax.add_collection3d(poly)

        # Centre and scale axes
        centre = verts.mean(axis=0)
        half = max(np.abs(verts - centre).max(), 1e-6)
        self.ax.set_xlim(centre[0] - half, centre[0] + half)
        self.ax.set_ylim(centre[1] - half, centre[1] + half)
        self.ax.set_zlim(centre[2] - half, centre[2] + half)

        self.ax.set_title(
            "Drag to rotate  •  Use buttons below to fix orientation",
            color="#8b949e", fontsize=8, pad=4,
        )
        self.canvas.draw_idle()


# ── Main Application ──────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real2Minecraft v2")
        self.root.configure(bg="#0d1117")
        self.root.geometry("1150x680")
        self.root.minsize(900, 560)

        self.image_path: Path | None = None
        self.is_generating = False
        self._preview_photo = None  # keep ref to prevent GC

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top-level split ──────────────────────────────────────────────────
        paned = tk.PanedWindow(
            self.root, orient=tk.HORIZONTAL,
            bg="#0d1117", sashwidth=5, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        # ── left control panel ───────────────────────────────────────────────
        left = tk.Frame(paned, bg="#161b22", width=300)
        paned.add(left, minsize=260)

        tk.Label(
            left, text="Real2Minecraft v2",
            bg="#161b22", fg="#58a6ff",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(18, 2))
        tk.Label(
            left, text="Image  →  3D Mesh",
            bg="#161b22", fg="#8b949e",
            font=("Segoe UI", 9),
        ).pack(pady=(0, 14))

        # preview canvas
        self.preview_canvas = tk.Canvas(
            left, width=260, height=260,
            bg="#0d1117", highlightthickness=1,
            highlightbackground="#30363d",
        )
        self.preview_canvas.pack(padx=16, pady=4)
        self.preview_canvas.create_text(
            130, 130,
            text="No image selected",
            fill="#4a5568", font=("Segoe UI", 9),
        )

        btn_frame = tk.Frame(left, bg="#161b22")
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

        self.generate_btn = tk.Button(
            btn_frame, text="▶  Generate 3D Model",
            command=self._on_generate,
            bg="#238636", fg="white",
            activebackground="#2ea043",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, bd=0, padx=10, pady=8,
            cursor="hand2", activeforeground="white",
            state=tk.DISABLED,
        )
        self.generate_btn.pack(fill=tk.X, pady=3)

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(4, 0))

        # env info label
        tk.Label(
            left,
            text=f"Python: {Path(VENV_PYTHON).name}\n{VENV_PYTHON}",
            bg="#161b22", fg="#30363d",
            font=("Consolas", 7),
            wraplength=270,
            justify=tk.LEFT,
        ).pack(side=tk.BOTTOM, pady=4, padx=8, anchor="w")

        # ── right 3-D viewer ─────────────────────────────────────────────────
        right = tk.Frame(paned, bg="#0d1117")
        paned.add(right, minsize=500)

        self.viewer = MeshViewer(right)
        self.viewer.pack(fill=tk.BOTH, expand=True)

        # ── status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready — select an image to begin")
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
        self.generate_btn.config(state=tk.NORMAL)
        self._set_status(f"Image: {self.image_path.name}")

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

    def _on_generate(self):
        if self.is_generating or not self.image_path:
            return
        if not self.image_path.exists():
            messagebox.showerror("File not found", str(self.image_path))
            return
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED, text="Generating…")
        self.progress.start(12)
        self._set_status("Launching InstantMesh…")
        threading.Thread(target=self._run_instantmesh, daemon=True).start()

    # ── InstantMesh pipeline ──────────────────────────────────────────────────

    def _run_instantmesh(self):
        try:
            # Detect pre-existing alpha — skip rembg if image already has transparency
            img = Image.open(self.image_path)
            skip_rembg = img.mode == "RGBA" and np.array(img)[:, :, 3].min() < 255

            cmd = [
                VENV_PYTHON,
                str(INSTANTMESH_DIR / "run.py"),
                str(IM_CONFIG),
                str(self.image_path),
                "--output_path", str(OUTPUTS_DIR),
                # --export_texmap requires nvdiffrast compiled from source (MSVC + CUDA toolkit).
                # Omit it here so we get a vertex-coloured OBJ that works without nvdiffrast.
            ]
            if skip_rembg:
                cmd.append("--no_rembg")

            env = os.environ.copy()
            # Make sure InstantMesh's internal imports resolve correctly
            env["PYTHONPATH"] = str(INSTANTMESH_DIR)

            self._set_status("Running InstantMesh — loading models (first run downloads weights)…")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(INSTANTMESH_DIR),
            )

            log_lines: list[str] = []
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                log_lines.append(line)
                # Route status messages to the status bar
                ll = line.lower()
                if "loading diffusion" in ll:
                    self._set_status("Loading diffusion model…")
                elif "loading custom" in ll or "unet" in ll:
                    self._set_status("Loading custom UNet…")
                elif "loading reconstruction" in ll:
                    self._set_status("Loading reconstruction model…")
                elif "rembg" in ll or "background" in ll:
                    self._set_status("Removing background…")
                elif "imagin" in ll or "multiview" in ll or "novel view" in ll:
                    self._set_status("Generating multi-view images…")
                elif "reconstruct" in ll or "flexicubes" in ll:
                    self._set_status("Reconstructing 3-D mesh…")
                elif "saving" in ll or "export" in ll:
                    self._set_status("Saving mesh…")

            proc.wait()

            if proc.returncode != 0:
                tail = "\n".join(log_lines[-30:])
                raise RuntimeError(
                    f"InstantMesh exited with code {proc.returncode}.\n\n"
                    f"Last output:\n{tail}"
                )

            # Find the newest .obj in outputs/
            obj_files = sorted(
                OUTPUTS_DIR.rglob("*.obj"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not obj_files:
                raise FileNotFoundError(
                    "InstantMesh finished but no .obj was found under outputs/.\n"
                    "Check that --export_texmap is supported by your build."
                )

            obj_path = obj_files[0]
            self._set_status(f"Mesh ready — loading viewer…  ({obj_path.name})")
            self.root.after(0, self._on_mesh_ready, str(obj_path))

        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    # ── post-generation callbacks (must run on main thread) ───────────────────

    def _on_mesh_ready(self, obj_path: str):
        self._reset_controls()
        self._set_status("Loading mesh into 3-D viewer…")
        try:
            self.viewer.load_mesh(obj_path)
            self._set_status(
                f"Done  ✓  Drag to rotate • Scroll to zoom   [{obj_path}]"
            )
        except Exception as exc:
            self._set_status(f"Mesh generated but viewer error: {exc}")
            messagebox.showwarning(
                "Viewer error",
                f"The mesh was generated at:\n{obj_path}\n\nBut the viewer raised:\n{exc}",
            )

    def _on_error(self, msg: str):
        self._reset_controls()
        self._set_status(f"Error: {msg[:100]}")
        messagebox.showerror("Generation failed", msg)

    def _reset_controls(self):
        self.is_generating = False
        self.progress.stop()
        self.generate_btn.config(state=tk.NORMAL, text="▶  Generate 3D Model")

    def _set_status(self, msg: str):
        # Thread-safe status update
        self.root.after(0, lambda: self.status_var.set(msg))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
