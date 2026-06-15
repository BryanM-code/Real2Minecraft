#!/usr/bin/env python3
"""
Real2Minecraft v2 — Photo to 3D Mesh
Image in -> InstantMesh -> mesh normalization -> diagnostic view
"""

import os
import sys
import time
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

OLLAMA_URL = "http://localhost:11434"

# Voxel grid resolution: fit the longest axis to this many voxels for a
# detailed, MagicaVoxel-like result (decoupled from real-world metres).
VOXEL_TARGET_RES = 64

# Minecraft scale: one block represents one cubic metre of the real object.
# Used only for the LLaVA height -> block-count status display.
METERS_PER_BLOCK = 1.0

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


# ── LLaVA via Ollama ──────────────────────────────────────────────────────────

def _run_llava(image_path: Path) -> dict | None:
    """Query LLaVA via Ollama for object label, orientation, and real-world height.

    Returns dict with keys: object (str), orientation ('vertical'|'horizontal'|None),
    height_metres (float|None).  Returns None on any failure so callers fall back
    to geometry-only mode.
    """
    import base64
    import json
    try:
        import requests
    except ImportError:
        return None
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception:
        return None

    prompt = (
        "What is the main object in this image? "
        "Reply ONLY with a JSON object (no markdown, no extra text) with these fields:\n"
        '"object": short label (e.g. "castle tower", "car", "oak tree")\n'
        '"orientation": "vertical" if the object stands upright and is taller than wide, '
        'or "horizontal" if it is wider or longer than it is tall\n'
        '"height_metres": your best estimate of the real-world height of this object '
        "in metres as a plain number (e.g. 1.8 for a person, 15 for a building)"
    )

    try:
        resp = requests.post(
            OLLAMA_URL + "/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        text = raw.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        data = json.loads(text[start:end])
        orient = str(data.get("orientation", "")).strip().lower()
        result = {
            "object":        str(data.get("object", "")).strip(),
            "orientation":   orient if orient in ("vertical", "horizontal") else None,
            "height_metres": None,
        }
        h = data.get("height_metres")
        if h is not None:
            result["height_metres"] = float(h)
        return result
    except Exception:
        return None


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



def _auto_orient(verts: np.ndarray, llava_hint: dict = None) -> tuple:
    """Rotate verts so the natural 'up' axis aligns with +Y.

    Scores each of the six +/-principal-component candidates as:
      score = planarity / xz_b^2

    where planarity = smallest singular value of the bottom-20% cluster
    (lower = flatter) and xz_b = XZ bounding-box area of that cluster
    (larger = wider ground contact).  Minimise the score.

    LLaVA hint supplement (does not replace the geometry loop):
      'vertical' or 'horizontal' -> selects the candidate most aligned with -Z,
        because InstantMesh save_obj negates Z so the OBJ's up is always -Z.
      None / absent              -> pure geometry scoring.

    Returns (rotated_verts, raw_up_axis_name, orient_tier_label).
    """
    # Both "vertical" and "horizontal" map to sign=+1 (select the -Z-up candidate).
    # InstantMesh save_obj negates Z, so -Z is always the object's up in the OBJ.
    if llava_hint and llava_hint.get("orientation") in ("vertical", "horizontal"):
        img_tier, img_conf = "strong", 1.0
    else:
        img_tier, img_conf = "none", 0.0

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
        orient_tier = "llava-{} ({})".format(
            llava_hint.get("orientation", "?") if llava_hint else "?",
            llava_hint.get("object", "?")      if llava_hint else "?",
        )

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
        orient_tier = "geometry-tiebreak"

    return verts @ best_R.T, _describe_axis(best_pc), orient_tier


# ── Mesh normalization ────────────────────────────────────────────────────────
def normalize_mesh(obj_path: Path, llava_info: dict = None) -> tuple:
    """Normalize a raw InstantMesh OBJ to Minecraft block coordinates.

    Steps (applied in order):
      0. Orient upright via PCA (LLaVA hint guides candidate selection)
      1. Ground placement -- min Y -> Y=0
      2. Center on XZ bounding box
      3. Scale: LLaVA real-world height (1 block = 1 m) OR default 1 raw unit = 1 block

    llava_info, if provided, should contain keys: object, orientation,
    height_metres.

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

    # Preserve per-vertex colors (raw InstantMesh OBJs carry `v x y z r g b`) so the
    # voxelizer can color blocks by the real object.  Captured before any processing
    # so it stays aligned 1:1 with `verts`.
    try:
        vcolors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()
        if len(vcolors) != len(verts):
            vcolors = None
    except Exception:
        vcolors = None

    # 0. Orient upright — PCA always runs; LLaVA hint guides candidate selection.
    verts, raw_up, orient_tier = _auto_orient(verts, llava_hint=llava_info)

    # 1. Ground placement: translate so the lowest vertex sits at Y=0.
    # Using min ensures no geometry ends up underground.
    y_floor = float(verts[:, 1].min())
    verts[:, 1] -= y_floor

    # 2. Center on XZ bounding box
    verts[:, 0] -= (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    verts[:, 2] -= (verts[:, 2].min() + verts[:, 2].max()) / 2.0

    # 3. Scale to Minecraft blocks (1 block = METERS_PER_BLOCK metres)
    llava_h = llava_info.get("height_metres") if llava_info else None
    if llava_h is not None:
        raw_h = float(verts[:, 1].max() - verts[:, 1].min())
        # Make the normalized height equal the real-world metres, then 1 block = 1 m.
        scale = (llava_h / METERS_PER_BLOCK) / raw_h if raw_h > 1e-6 else 1.0
        scale_source = "llava"
    else:
        scale = 1.0
        scale_source = "default"
    verts *= scale

    norm_mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_colors=vcolors, process=False
    )
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
        "llava_object":      llava_info.get("object")        if llava_info else None,
        "llava_orientation": llava_info.get("orientation")   if llava_info else None,
        "llava_height_m":    llava_info.get("height_metres") if llava_info else None,
        "scale_source":      scale_source,
    }


# ── Voxelization ─────────────────────────────────────────────────────────────
def _sample_voxel_colors(pts, verts, faces, vcolors):
    """RGB color per voxel center, sampled from the colored mesh surface.

    For each voxel center, find the closest point on the mesh surface and
    barycentric-blend the three vertex colors of that triangle.  This is more
    accurate than nearest-vertex (which can grab a color from the wrong side of
    a thin feature).  Falls back to nearest-vertex, then to None.

    `verts`/`faces`/`vcolors` describe the colored mesh already scaled into the
    voxel frame so it lines up with `pts`.  Returns a list of (r,g,b) ints in the
    same order as `pts`, or None.
    """
    if vcolors is None:
        return None
    try:
        cmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        closest, _, tri_id = trimesh.proximity.closest_point(cmesh, pts)
        tris = cmesh.triangles[tri_id]                                   # (K,3,3)
        bary = trimesh.triangles.points_to_barycentric(tris, closest)    # (K,3)
        tri_cols = vcolors[faces[tri_id]].astype(np.float64)             # (K,3,3)
        blended = (bary[:, :, None] * tri_cols).sum(axis=1)              # (K,3)
        blended = np.clip(np.rint(blended), 0, 255).astype(int)
        return [tuple(int(c) for c in rgb) for rgb in blended]
    except Exception:
        pass
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(verts).query(pts)
        return [tuple(int(c) for c in rgb) for rgb in vcolors[idx]]
    except Exception:
        return None


def voxelize_mesh(norm_path: Path, norm_info: dict) -> dict:
    """Load normalized OBJ, repair, scale to target resolution, voxelize.

    The longest axis is fit to VOXEL_TARGET_RES voxels (decoupled from metres)
    for a detailed, MagicaVoxel-like result.

    Returns dict:
      blocks       - list of (x, y, z) int tuples, all >= 0
      colors       - list of (r,g,b) ints parallel to blocks, or None
      block_type   - 'stone' (hard-coded for now)
      dims         - (W, H, L) bounding box in blocks
      block_count  - number of occupied blocks
      scale_factor - float applied before voxelizing
      watertight   - bool; False = warning, voxelization still proceeds
    """
    mesh = trimesh.load(str(norm_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms)

    # Snapshot the colored geometry BEFORE repair — fill_holes can add uncolored
    # vertices, which would break the vertex<->color alignment used for sampling.
    orig_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    orig_faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    try:
        orig_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3].copy()
        if len(orig_colors) != len(orig_verts):
            orig_colors = None
    except Exception:
        orig_colors = None

    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    # Fit the longest axis to VOXEL_TARGET_RES voxels for a detailed result.
    W, D, H = norm_info["W"], norm_info["D"], norm_info["H"]
    longest = max(W, D, H)
    scale_factor = VOXEL_TARGET_RES / longest if longest > 1e-6 else 1.0
    mesh.apply_scale(scale_factor)

    voxels = mesh.voxelized(pitch=1.0)
    voxels = voxels.fill()

    pts = voxels.points
    bx = np.floor(pts[:, 0]).astype(int)
    by = np.floor(pts[:, 1]).astype(int)
    bz = np.floor(pts[:, 2]).astype(int)

    bx -= bx.min()
    bz -= bz.min()
    if by.min() < 0:
        by -= by.min()

    blocks = list(zip(bx.tolist(), by.tolist(), bz.tolist()))
    W_v = int(bx.max() - bx.min() + 1)
    H_v = int(by.max() - by.min() + 1)
    L_v = int(bz.max() - bz.min() + 1)

    # Color each block from the colored mesh surface (scaled to the voxel frame).
    colors = _sample_voxel_colors(
        pts, orig_verts * scale_factor, orig_faces, orig_colors
    )

    return {
        "blocks":       blocks,
        "colors":       colors,
        "block_type":   "stone",
        "dims":         (W_v, H_v, L_v),
        "block_count":  len(blocks),
        "scale_factor": scale_factor,
        "watertight":   bool(mesh.is_watertight),
    }


# ax.voxels draws every cube face (~1.5 ms/cube), so it's only snappy for small
# models.  Above this many surface cubes, render fast square-marker points instead
# (visually near-identical at high resolution, but instant).
VOXEL_RENDER_LIMIT = 3500


def _draw_steve(ax):
    """Draw a 1(X) x 1(Z) x 2(height) wireframe box as a Steve scale reference.

    Placed just off the model's -X side.  Plot axes: (plt_x, plt_y, plt_z) =
    (block_x, block_z, block_height).
    """
    x0, x1 = -2.0, -1.0
    z0, z1 = 0.0, 1.0
    y0, y1 = 0.0, 2.0
    corners = np.array([
        [x0, z0, y0], [x1, z0, y0], [x0, z1, y0], [x1, z1, y0],
        [x0, z0, y1], [x1, z0, y1], [x0, z1, y1], [x1, z1, y1],
    ])
    edges = [(0,1),(2,3),(4,5),(6,7),
             (0,2),(1,3),(4,6),(5,7),
             (0,4),(1,5),(2,6),(3,7)]
    for n, (i, j) in enumerate(edges):
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color="red", lw=1.5,
                label="Steve = 2x1 blocks" if n == 0 else None)


def show_voxel_diagnostic(voxel_data: dict):
    """Render voxels as Minecraft-style colored cubes. Blocking (plt.show())."""
    blocks = voxel_data.get("blocks", [])
    if not blocks:
        return
    colors = voxel_data.get("colors")
    xs, ys, zs = zip(*blocks)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    W, H, L = voxel_data["dims"]

    # Per-block RGBA (0-1).  Fall back to a height gradient when no colors exist.
    if colors:
        rgba = np.ones((len(blocks), 4), dtype=float)
        rgba[:, :3] = np.asarray(colors, dtype=float) / 255.0
    else:
        norm = ys / max(ys.max(), 1)
        rgba = plt.cm.viridis(norm)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot axes: block X -> plt X, block Z -> plt Y, block height (Y) -> plt Z (up).
    # Occupancy + facecolor grids, indexed [x, z, height] to match the plot axes.
    occ = np.zeros((W, L, H), dtype=bool)
    facecolors = np.zeros((W, L, H, 4), dtype=float)
    occ[xs, zs, ys] = True
    facecolors[xs, zs, ys] = rgba

    # Only the surface (exposed) voxels are visible — a fully interior voxel is
    # hidden behind its neighbours.  Rendering just the shell is identical to the
    # eye but vastly cheaper than drawing every solid-fill voxel.
    pad = np.pad(occ, 1, constant_values=False)
    surrounded = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] &
        pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] &
        pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    shell = occ & ~surrounded

    if int(shell.sum()) <= VOXEL_RENDER_LIMIT:
        ax.voxels(shell, facecolors=facecolors, edgecolor="black", linewidth=0.15)
        render_note = ""
    else:
        # High-res model — draw the surface voxels as square markers (instant).
        sx, sz, sy = np.nonzero(shell)
        ax.scatter(sx + 0.5, sz + 0.5, sy + 0.5,
                   s=12, marker="s", c=facecolors[sx, sz, sy], depthshade=False)
        render_note = "  [points: >{:,} surface blocks]".format(VOXEL_RENDER_LIMIT)

    _draw_steve(ax)

    # Equal unit length on every axis so cubes read as cubes; -3 leaves room for Steve.
    ax.set_xlim(-3, W)
    ax.set_ylim(0, L)
    ax.set_zlim(0, H)
    ax.set_box_aspect((W + 3, L, H))

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (height in blocks)")

    scale_note = (
        "  [voxel scale: {:.2f}x]".format(voxel_data["scale_factor"])
        if voxel_data["scale_factor"] != 1.0 else ""
    )
    watertight_note = "" if voxel_data["watertight"] else "  ⚠ mesh not watertight"

    ax.set_title(
        "Voxel result  |  {:,} blocks  |  {}x{}x{} (XYZ){}{}{}".format(
            voxel_data["block_count"], W, H, L,
            scale_note, watertight_note, render_note
        )
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def export_schematic(voxel_data: dict, out_path: Path):
    """Write voxel_data to a Sponge Schematic v2 (.schem) gzipped NBT file."""
    try:
        import nbtlib
        from nbtlib import Compound, Int, Short, ByteArray, IntArray
    except ImportError:
        raise RuntimeError(
            "nbtlib is required for .schem export.\n"
            "Run:  r2m_v2\\Scripts\\pip install nbtlib"
        )

    W, H, L = voxel_data["dims"]
    blocks   = voxel_data["blocks"]

    # Build 3D occupancy in YZX order (Sponge Schematic v2 convention)
    # 0 = minecraft:air  |  1 = minecraft:stone
    grid = np.zeros((H, L, W), dtype=np.uint8)
    for x, y, z in blocks:
        if 0 <= y < H and 0 <= z < L and 0 <= x < W:
            grid[y, z, x] = 1

    def _varint(v: int) -> bytes:
        buf = bytearray()
        while True:
            if v & ~0x7F == 0:
                buf.append(v)
                break
            buf.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(buf)

    block_data = bytearray()
    for val in grid.flatten():
        block_data.extend(_varint(int(val)))

    schematic = Compound({
        "Version":     Int(2),
        "DataVersion": Int(3465),       # Minecraft 1.20.1
        "Width":       Short(W),
        "Height":      Short(H),
        "Length":      Short(L),
        "PaletteMax":  Int(2),
        "Palette": Compound({
            "minecraft:air":   Int(0),
            "minecraft:stone": Int(1),
        }),
        "BlockData": ByteArray(block_data),
        "Offset":    IntArray([0, 0, 0]),
    })

    nbtlib.File({"": schematic}).save(str(out_path), gzipped=True)


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

    def _on_build(self):
        """Run the whole pipeline: InstantMesh -> normalize -> voxelize -> view."""
        if self.is_generating or not self.image_path:
            return
        if not self.image_path.exists():
            messagebox.showerror("File not found", str(self.image_path))
            return
        self.is_generating = True
        self.build_btn.config(state=tk.DISABLED, text="Working...")
        self.export_btn.config(state=tk.DISABLED)
        self.progress.start(12)
        self._start_timer()
        self._set_status("Launching InstantMesh...")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    # ── Build pipeline (InstantMesh -> normalize -> voxelize) ──────────────────

    def _run_pipeline(self):
        try:
            # Query LLaVA (via Ollama) for object label, orientation hint, and scale.
            self._set_status("Querying LLaVA via Ollama...")
            llava_info = _run_llava(self.image_path)
            if llava_info and llava_info.get("object"):
                h = llava_info.get("height_metres")
                size_str = (
                    "~{:.0f}m → {:.0f} blocks".format(h, h / METERS_PER_BLOCK)
                    if h else "unknown size"
                )
                self._set_status("LLaVA: {} | {} | {}".format(
                    llava_info["object"],
                    llava_info.get("orientation") or "?",
                    size_str,
                ))
            else:
                llava_info = None
                self._set_status("LLaVA unavailable — using geometry fallback")

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
            norm_path, norm_info = normalize_mesh(obj_path, llava_info=llava_info)
            self.norm_path = norm_path
            self.norm_info = norm_info

            self._set_status("Voxelizing — this may take a few seconds...")
            voxel_data = voxelize_mesh(norm_path, norm_info)
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

    def _set_status(self, msg: str):
        # Thread-safe status update
        self.root.after(0, lambda: self.status_var.set(msg))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
