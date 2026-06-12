#!/usr/bin/env python3
"""
Real2Minecraft v2 — Photo to 3D Mesh
Image in -> InstantMesh -> mesh normalization -> diagnostic view
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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import trimesh

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
INSTANTMESH_DIR = PROJECT_ROOT / "InstantMesh"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Large config gives the best quality; use instant-mesh-base for faster runs.
IM_CONFIG = INSTANTMESH_DIR / "configs" / "instant-mesh-large.yaml"

# ── YOLO semantic lookup tables ────────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.6

VERTICAL_CLASSES = {
    "person", "bottle", "fire hydrant", "traffic light", "parking meter",
    "clock", "potted plant",
}

HORIZONTAL_CLASSES = {
    "car", "truck", "bus", "boat", "airplane", "bicycle", "motorcycle",
    "bench", "couch", "dining table", "bed", "train",
    "horse", "cow", "sheep", "dog", "cat", "elephant",
}

REAL_WORLD_HEIGHTS = {
    "person": 1.8,  "car": 1.5,   "truck": 3.5,  "bus": 3.2,
    "bicycle": 1.0, "motorcycle": 1.1, "airplane": 5.0, "boat": 2.5,
    "bottle": 0.25, "fire hydrant": 0.6, "bench": 0.9,
    "traffic light": 3.0, "potted plant": 0.5, "parking meter": 1.2,
    "horse": 1.6,   "cow": 1.4,   "elephant": 3.3, "dog": 0.6, "cat": 0.3,
    "couch": 0.9,   "dining table": 0.75, "bed": 0.6, "train": 3.8,
    "skateboard": 0.15, "surfboard": 0.3, "sheep": 1.0, "clock": 0.3,
}

_yolo_model = None  # loaded once per session on first use

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


# ── YOLO inference ────────────────────────────────────────────────────────────

def _run_yolo(image_path: Path) -> tuple:
    """Run YOLOv8n on image_path. Returns (label, confidence, bbox_xyxy) or ("", 0.0, None).

    Picks the highest-confidence detection. Falls back silently if ultralytics
    is not installed or inference fails. Downloads yolov8n.pt on first call.
    """
    global _yolo_model
    try:
        from ultralytics import YOLO as _YOLO
    except ImportError:
        return "", 0.0, None
    try:
        if _yolo_model is None:
            _yolo_model = _YOLO("yolov8n.pt")
        results = _yolo_model(str(image_path), verbose=False)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return "", 0.0, None
        boxes = results[0].boxes
        best  = int(boxes.conf.argmax())
        label = results[0].names[int(boxes.cls[best])]
        conf  = float(boxes.conf[best])
        bbox  = boxes.xyxy[best].tolist()
        return label, conf, bbox
    except Exception:
        return "", 0.0, None


# ── Orientation helpers ───────────────────────────────────────────────────────

def _rotation_to_y_axis(v: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix that maps unit vector v to +Y = [0,1,0]."""
    v = v / np.linalg.norm(v)
    up = np.array([0.0, 1.0, 0.0])
    if np.allclose(v, up):
        return np.eye(3)
    if np.allclose(v, -up):
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(v, up)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(v, up), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _describe_axis(v: np.ndarray) -> str:
    """Name of the closest cardinal axis to unit vector v (e.g. '+Z', '-X')."""
    v = v / np.linalg.norm(v)
    candidates = [('+X', [1,0,0]), ('-X', [-1,0,0]),
                  ('+Y', [0,1,0]), ('-Y', [0,-1,0]),
                  ('+Z', [0,0,1]), ('-Z', [0,0,-1])]
    return max(candidates, key=lambda kv: np.dot(v, kv[1]))[0]


def _extract_image_up_signal(image_path: Path) -> tuple:
    """Analyze the input image for a sky-up / ground-down visual gradient.

    Compares mean luminance of the top third vs bottom third of the image.
    For typical real-world photos the sky (top) is brighter than the ground
    (bottom), giving a positive confidence score.

    Returns (tier, confidence):
      tier:       'strong' | 'weak' | 'none'
      confidence: float in [-1, +1]
                  positive  -> top of image is brighter (sky-like, up is up)
                  negative  -> bottom is brighter (unusual / flipped shot)
                  near zero -> ambiguous, no useful signal

    Uses only PIL/numpy; no opencv required.
    """
    try:
        with Image.open(image_path) as img:
            # Composite transparent images onto neutral gray so transparent
            # areas don't falsely drag luminance toward black.
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
                img = Image.alpha_composite(bg, img)
            gray = img.convert("L").resize((64, 64), Image.LANCZOS)

        arr = np.array(gray, dtype=float)
        h = arr.shape[0]
        top_mean = arr[: h // 3].mean()
        bot_mean = arr[2 * h // 3 :].mean()
        total = top_mean + bot_mean
        if total < 1.0:
            return "none", 0.0

        # Normalized difference in [-1, +1]
        confidence = (top_mean - bot_mean) / total

        if abs(confidence) >= 0.08:
            return "strong", float(confidence)
        if abs(confidence) >= 0.03:
            return "weak", float(confidence)
        return "none", float(confidence)
    except Exception:
        return "none", 0.0


def _auto_orient(verts: np.ndarray, image_path: Path = None) -> tuple:
    """Rotate verts so the natural 'up' axis aligns with +Y.

    Scores each of the six +/-principal-component candidates as:
      score = planarity / xz_b^2

    where planarity = smallest singular value of the bottom-20% cluster
    (lower = flatter) and xz_b = XZ bounding-box area of that cluster
    (larger = wider ground contact).  Minimise the score.

    Three-tier image-hint supplement (does not replace the geometry loop):
      strong signal -> image may override geometry among candidates within 3x
                       of the best geometry score
      weak signal   -> image breaks near-ties (scores within 20% of best)
      no signal     -> pure geometry scoring, identical to original behaviour

    The image hint assumes InstantMesh outputs meshes with the input image's
    vertical axis corresponding to the raw mesh's -Z axis.  Internally
    InstantMesh uses Z+ as world up (camera_util.py: up_world=[0,0,1]), but
    save_obj in mesh_util.py negates Z on export, so the saved OBJ has -Z as
    the object's up direction.
    A positive confidence (top-of-image brighter) is taken as evidence that
    the raw -Z direction is the correct "up".

    Returns (rotated_verts, raw_up_axis_name, orient_tier_label).
    """
    img_tier, img_conf = (
        _extract_image_up_signal(image_path) if image_path is not None else ("none", 0.0)
    )

    centered = verts - verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    best_score   = np.inf
    best_R       = np.eye(3)
    best_pc      = Vt[0]
    second_score = np.inf
    second_R     = np.eye(3)
    second_pc    = Vt[0]
    all_cands    = []

    for pc in Vt:
        for candidate in (pc, -pc):
            R = _rotation_to_y_axis(candidate)
            rotated = verts @ R.T

            bot_thresh = np.percentile(rotated[:, 1], 20)
            bottom = rotated[rotated[:, 1] <= bot_thresh]

            if len(bottom) < 3:
                continue

            c = bottom - bottom.mean(axis=0)
            _, s, _ = np.linalg.svd(c, full_matrices=False)
            planarity = float(s[-1])

            xz_b = ((bottom[:, 0].max() - bottom[:, 0].min()) *
                    (bottom[:, 2].max() - bottom[:, 2].min()))

            # Reward orientations where the mesh is taller than it is wide.
            # For a tower lying on its side h_after << max(w,d), so bonus = 1
            # (no effect). For a tower standing upright h_after >> max(w,d),
            # bonus is large and the score shrinks, making upright strongly
            # preferred. Flat/wide objects also get bonus = 1 (h < max(w,d)).
            h_after = rotated[:, 1].max() - rotated[:, 1].min()
            w_after = rotated[:, 0].max() - rotated[:, 0].min()
            d_after = rotated[:, 2].max() - rotated[:, 2].min()
            aspect_bonus = max(1.0, h_after / (max(w_after, d_after) + 1e-9)) ** 2

            score = planarity / ((xz_b ** 2 + 1e-9) * aspect_bonus)

            all_cands.append((score, R, candidate))

            if score < best_score:
                second_score, second_R, second_pc = best_score, best_R, best_pc
                best_score, best_R, best_pc = score, R, candidate
            elif score < second_score:
                second_score, second_R, second_pc = score, R, candidate

    # ── Apply image hint ───────────────────────────────────────────────────────
    # sign: +1 -> top of image brighter -> raw -Z should be "up" (save_obj negates Z)
    #        -1 -> bottom of image brighter -> raw +Z should be "up" (unusual)
    orient_tier = "geometry"

    if img_tier == "strong":
        # Image gives a confident sky-up signal.  Consider ALL six geometry
        # candidates so the correct orientation is never excluded by a score
        # window (geometry can be 100-500x wrong for thin upright objects).
        sign = 1.0 if img_conf > 0 else -1.0
        winner = max(
            all_cands,
            key=lambda x: float(np.dot(x[2], [0.0, 0.0, -1.0])) * sign,
        )
        best_R, best_pc = winner[1], winner[2]
        orient_tier = "image-direct (conf={:+.2f})".format(img_conf)

    elif img_tier == "weak" and second_score <= best_score * 1.2:
        # Geometry is nearly tied; image breaks the deadlock.
        sign = 1.0 if img_conf > 0 else -1.0
        cands = [
            (best_score, best_R, best_pc),
            (second_score, second_R, second_pc),
        ]
        winner = max(
            cands,
            key=lambda x: float(np.dot(x[2], [0.0, 0.0, -1.0])) * sign,
        )
        best_R, best_pc = winner[1], winner[2]
        orient_tier = "image-tiebreak (conf={:+.2f})".format(img_conf)

    return verts @ best_R.T, _describe_axis(best_pc), orient_tier


# ── Mesh normalization ────────────────────────────────────────────────────────
def normalize_mesh(obj_path: Path, image_path: Path = None, yolo_info: dict = None) -> tuple:
    """Normalize a raw InstantMesh OBJ to Minecraft block coordinates.

    Steps (applied in order):
      0. Orient upright (YOLO direct rotation OR PCA + image-hint fallback)
      1. Ground placement -- min Y -> Y=0
      2. Center on XZ bounding box
      3. Scale: YOLO real-world height OR default 1 raw unit = 1/0.9 blocks

    yolo_info, if provided, should contain keys: label, conf, orient_type,
    real_height_m.  When conf >= YOLO_CONF_THRESHOLD and orient_type is set,
    PCA is skipped and the known InstantMesh -Z-up rotation is applied directly.

    Returns (norm_path, info_dict). The raw file is not modified.
    """
    mesh = trimesh.load(str(obj_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            raise ValueError("OBJ contains no geometry")
        mesh = trimesh.util.concatenate(geoms)

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # 0. Orient upright
    yolo_fired = (
        yolo_info is not None
        and yolo_info.get("conf", 0.0) >= YOLO_CONF_THRESHOLD
        and yolo_info.get("orient_type") is not None
    )

    if yolo_fired:
        # InstantMesh always outputs with -Z as up (save_obj negates Z).
        # YOLO tells us what the object is, so we can apply this rotation directly.
        R_yolo = _rotation_to_y_axis(np.array([0.0, 0.0, -1.0]))
        verts = verts @ R_yolo.T
        raw_up = "-Z"
        orient_tier = "yolo-{} ({}, conf={:.2f})".format(
            yolo_info["orient_type"], yolo_info["label"], yolo_info["conf"]
        )
    else:
        verts, raw_up, orient_tier = _auto_orient(verts, image_path)

    # 1. Ground placement: translate so the lowest vertex sits at Y=0.
    # Using min ensures no geometry ends up underground.
    y_floor = float(verts[:, 1].min())
    verts[:, 1] -= y_floor

    # 2. Center on XZ bounding box
    verts[:, 0] -= (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    verts[:, 2] -= (verts[:, 2].min() + verts[:, 2].max()) / 2.0

    # 3. Scale to Minecraft blocks (1 block = 0.9 m)
    yolo_h = yolo_info.get("real_height_m") if yolo_fired else None
    if yolo_h is not None:
        raw_h = float(verts[:, 1].max() - verts[:, 1].min())
        scale = yolo_h / (raw_h * 0.9) if raw_h > 1e-6 else 1.0 / 0.9
        scale_source = "yolo"
    else:
        scale = 1.0 / 0.9
        scale_source = "default"
    verts *= scale

    norm_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    norm_path = obj_path.parent / "{}_normalized.obj".format(obj_path.stem)
    norm_mesh.export(str(norm_path))

    W = float(verts[:, 0].max() - verts[:, 0].min())
    D = float(verts[:, 2].max() - verts[:, 2].min())
    H = float(verts[:, 1].max() - verts[:, 1].min())

    return norm_path, {
        "name": obj_path.stem,
        "face_count": len(faces),
        "y_floor_raw": y_floor,
        "W": W, "D": D, "H": H,
        "raw_up": raw_up,
        "orient_tier": orient_tier,
        "yolo_label":    yolo_info.get("label")         if yolo_info else None,
        "yolo_conf":     yolo_info.get("conf")          if yolo_info else None,
        "yolo_height_m": yolo_info.get("real_height_m") if yolo_info else None,
        "scale_source":  scale_source,
    }


# ── Diagnostic window ─────────────────────────────────────────────────────────
def show_diagnostic(norm_path: Path, info: dict):
    """Open a standalone matplotlib window to verify the normalized mesh.

    Shows a 3D scatter of vertices on a 1-block XZ grid, with a Steve
    reference line at Y=2 for scale. Blocking (plt.show()).
    """
    mesh = trimesh.load(str(norm_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms)

    verts = np.asarray(mesh.vertices, dtype=np.float32)

    # Subsample for performance
    if len(verts) > 5000:
        idx = np.random.default_rng(0).choice(len(verts), 5000, replace=False)
        pts = verts[idx]
    else:
        pts = verts

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Axis mapping: mesh X -> plt X, mesh Z -> plt Y, mesh Y (height) -> plt Z
    # This keeps height visually upward in the default matplotlib 3D view.
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               s=0.8, c=pts[:, 1], cmap="viridis", alpha=0.6)

    # XZ grid at mesh Y=0 (plt Z=0), lines every 1 block
    x_min = float(verts[:, 0].min())
    x_max = float(verts[:, 0].max())
    z_min = float(verts[:, 2].min())
    z_max = float(verts[:, 2].max())
    pad = 0.5
    W_blk = x_max - x_min
    D_blk = z_max - z_min
    H_blk = float(verts[:, 1].max()) - float(verts[:, 1].min())
    # Equal-cube display: all axes span the same range so object shape reads as
    # how much of the cube it fills (car = flat slab, tower = tall column).
    # Always reach at least Steve's height so the reference line stays visible.
    view_size = max(W_blk, D_blk, H_blk, 2.2) * 1.08
    cx = (x_min + x_max) / 2.0
    cz = (z_min + z_max) / 2.0
    max_dim = max(W_blk, D_blk, H_blk)
    grid_step = 10.0 if max_dim > 50 else 5.0 if max_dim > 20 else 2.0 if max_dim > 5 else 1.0
    gx = np.arange(np.floor(x_min - pad), np.ceil(x_max + pad) + grid_step, grid_step)
    gz = np.arange(np.floor(z_min - pad), np.ceil(z_max + pad) + grid_step, grid_step)
    for xv in gx:
        ax.plot([xv, xv], [gz[0], gz[-1]], [0, 0],
                color="#888888", lw=0.5, alpha=0.4)
    for zv in gz:
        ax.plot([gx[0], gx[-1]], [zv, zv], [0, 0],
                color="#888888", lw=0.5, alpha=0.4)

    # Steve reference line at mesh Y=2 (plt Z=2)
    ax.plot([x_min, x_max], [0, 0], [2, 2],
            color="red", lw=1.5, label="Steve = 2 blocks")

    # +Y (UP) arrow at origin -- plotted in plot-Z direction since mesh Y -> plt Z
    h_up = float(verts[:, 1].max()) * 0.25
    h_up = max(h_up, 0.5)
    ax.quiver(0, 0, 0, 0, 0, h_up,
              color="cyan", arrow_length_ratio=0.3, linewidth=1.5, label="+Y (UP)")

    # Bounding box wireframe
    bx0, bx1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    bz0, bz1 = float(verts[:, 2].min()), float(verts[:, 2].max())
    by0, by1 = float(verts[:, 1].min()), float(verts[:, 1].max())
    # corners in plot coords: (plot_x, plot_y, plot_z) = (mesh_x, mesh_z, mesh_y)
    bbox_corners = np.array([
        [bx0, bz0, by0], [bx1, bz0, by0], [bx0, bz1, by0], [bx1, bz1, by0],
        [bx0, bz0, by1], [bx1, bz0, by1], [bx0, bz1, by1], [bx1, bz1, by1],
    ])
    bbox_edges = [(0,1),(2,3),(4,5),(6,7),
                  (0,2),(1,3),(4,6),(5,7),
                  (0,4),(1,5),(2,6),(3,7)]
    for i, j in bbox_edges:
        ax.plot([bbox_corners[i,0], bbox_corners[j,0]],
                [bbox_corners[i,1], bbox_corners[j,1]],
                [bbox_corners[i,2], bbox_corners[j,2]],
                color="orange", lw=0.8, alpha=0.8)

    ax.set_xlim(cx - view_size / 2, cx + view_size / 2)
    ax.set_ylim(cz - view_size / 2, cz + view_size / 2)
    ax.set_zlim(0, view_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (height in blocks)")
    raw_up = info.get("raw_up", "?")
    orient_tier = info.get("orient_tier", "geometry")
    yolo_label = info.get("yolo_label")
    yolo_line = ""
    if yolo_label:
        scale_str = (
            "{:.2f}m real-world".format(info["yolo_height_m"])
            if info.get("yolo_height_m") else "default scale"
        )
        yolo_line = "\nYOLO: {} (conf={:.2f})  |  scale: {} ({})".format(
            yolo_label, info["yolo_conf"], info.get("scale_source", "default"), scale_str
        )
    ax.set_title(
        "{name}  |  {fc:,} faces\n"
        "Bounds: {W:.1f} x {D:.1f} x {H:.1f} blocks  "
        "|  Y-floor: {yf:.4f} m\n"
        "raw up: {raw_up} -> +Y  |  orient: {tier}"
        "{yolo_line}".format(
            name=info["name"],
            fc=info["face_count"],
            W=info["W"], D=info["D"], H=info["H"],
            yf=info["y_floor_raw"],
            raw_up=raw_up,
            tier=orient_tier,
            yolo_line=yolo_line,
        )
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── Main Application ──────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real2Minecraft v2")
        self.root.configure(bg="#0d1117")
        self.root.geometry("500x480")
        self.root.minsize(420, 400)

        self.image_path: Path | None = None
        self.is_generating = False
        self._preview_photo = None  # keep ref to prevent GC

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
            ctrl,
            text="Python: {}\n{}".format(Path(VENV_PYTHON).name, VENV_PYTHON),
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
        self.generate_btn.config(state=tk.NORMAL)
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

    def _on_generate(self):
        if self.is_generating or not self.image_path:
            return
        if not self.image_path.exists():
            messagebox.showerror("File not found", str(self.image_path))
            return
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED, text="Generating...")
        self.progress.start(12)
        self._set_status("Launching InstantMesh...")
        threading.Thread(target=self._run_instantmesh, daemon=True).start()

    # ── InstantMesh pipeline ──────────────────────────────────────────────────

    def _run_instantmesh(self):
        try:
            # Extract image up-signal from the original photo before InstantMesh
            # consumes / background-removes it, so we always analyse the source image.
            img_tier, img_conf = _extract_image_up_signal(self.image_path)
            hint_msg = img_tier + (" (conf={:+.2f})".format(img_conf) if img_tier != "none" else "")
            self._set_status("Image hint: {}".format(hint_msg))

            # YOLO object detection on the original photo
            self._set_status("Running YOLO object detection...")
            yolo_label, yolo_conf, _ = _run_yolo(self.image_path)

            orient_type   = None
            real_height_m = None
            if yolo_label:
                if yolo_label in VERTICAL_CLASSES:
                    orient_type = "vertical"
                elif yolo_label in HORIZONTAL_CLASSES:
                    orient_type = "horizontal"
                real_height_m = REAL_WORLD_HEIGHTS.get(yolo_label)
                height_str = "{:.2f}m".format(real_height_m) if real_height_m else "unknown height"
                yolo_status = "YOLO: {} (conf={:.2f}) → {}, {}".format(
                    yolo_label, yolo_conf,
                    orient_type or "unrecognized class",
                    height_str,
                )
            else:
                yolo_status = "YOLO: no detection"
            self._set_status(yolo_status)

            yolo_info = {
                "label":         yolo_label,
                "conf":          yolo_conf,
                "orient_type":   orient_type,
                "real_height_m": real_height_m,
            }

            # Detect pre-existing alpha -- skip rembg if image already has transparency
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

            self._set_status("Running InstantMesh -- loading models (first run downloads weights)...")

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
                    self._set_status("Loading diffusion model...")
                elif "loading custom" in ll or "unet" in ll:
                    self._set_status("Loading custom UNet...")
                elif "loading reconstruction" in ll:
                    self._set_status("Loading reconstruction model...")
                elif "rembg" in ll or "background" in ll:
                    self._set_status("Removing background...")
                elif "imagin" in ll or "multiview" in ll or "novel view" in ll:
                    self._set_status("Generating multi-view images...")
                elif "reconstruct" in ll or "flexicubes" in ll:
                    self._set_status("Reconstructing 3-D mesh...")
                elif "saving" in ll or "export" in ll:
                    self._set_status("Saving mesh...")

            proc.wait()

            if proc.returncode != 0:
                tail = "\n".join(log_lines[-30:])
                raise RuntimeError(
                    "InstantMesh exited with code {}.\n\nLast output:\n{}".format(
                        proc.returncode, tail
                    )
                )

            # Find the newest raw .obj in outputs/ (exclude _normalized.obj from previous runs)
            obj_files = sorted(
                (p for p in OUTPUTS_DIR.rglob("*.obj")
                 if not p.name.endswith("_normalized.obj")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not obj_files:
                raise FileNotFoundError(
                    "InstantMesh finished but no .obj was found under outputs/.\n"
                    "Check that --export_texmap is supported by your build."
                )

            obj_path = obj_files[0]
            self._set_status("Normalizing mesh...")
            norm_path, norm_info = normalize_mesh(obj_path, image_path=self.image_path, yolo_info=yolo_info)
            self.root.after(0, self._on_mesh_ready, str(obj_path), str(norm_path), norm_info)

        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    # ── post-generation callbacks (must run on main thread) ───────────────────

    def _on_mesh_ready(self, raw_path: str, norm_path: str, info: dict):
        self._reset_controls()
        name, W, D, H = info["name"], info["W"], info["D"], info["H"]
        self._set_status(
            "Normalization complete -- {}_normalized.obj  |  "
            "{:.1f} x {:.1f} x {:.1f} blocks".format(name, W, D, H)
        )
        show_diagnostic(Path(norm_path), info)

    def _on_error(self, msg: str):
        self._reset_controls()
        self._set_status("Error: {}".format(msg[:100]))
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
