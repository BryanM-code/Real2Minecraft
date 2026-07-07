#!/usr/bin/env python3
"""
Real2Minecraft v2 — Photo to Minecraft voxels
Image -> TRELLIS (image-to-3D) -> normalize -> voxelize -> textured viewer / .schem
"""

import os
import sys
import time
import threading
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
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Resolution slider bounds: the longest axis is scaled to the chosen target (in blocks).
# Higher targets preserve architectural detail (crenellations, windows) — only pays off
# once the source mesh actually carries that detail (see TRELLIS_MESH_SIMPLIFY).
VOXEL_MIN_BLOCKS = 16     # slider minimum
VOXEL_MAX_BLOCKS = 128    # slider maximum

# ── TRELLIS image-to-3D backend ───────────────────────────────────────────────
# TRELLIS runs in its own env (C:\TRELLIS) and is driven over its local FastAPI server.
TRELLIS_API_URL = os.environ.get("R2M_TRELLIS_URL", "http://127.0.0.1:7960")

# TRELLIS's simplify is "ratio of faces to REMOVE".  Lower keeps more detail, but its
# postprocessing (xatlas UV-unwrap + 100-view texture bake) blows up super-linearly with
# face count: 0.95 -> ~30s, 0.1 -> minutes.  0.9 keeps ~2x the detail of the 0.95 default
# while staying fast.  Lower it for finer builds at the cost of generation time.
TRELLIS_MESH_SIMPLIFY = 0.9

# Baked texture resolution.  The voxelizer samples this texture directly per voxel, so
# raising it (e.g. 2048) buys sharper block colors at the cost of a slower TRELLIS bake.
TRELLIS_TEXTURE_SIZE = 1024

# Diffusion steps for both TRELLIS stages (sparse structure + slat).  The server default
# is 12 (max 50); 24 buys noticeably cleaner geometry/texture latents for a few extra
# seconds per run.
TRELLIS_SAMPLING_STEPS = 24

# Background-removal model for the client-side cutout.  'isnet-general-use' handles
# whole scenes (house + yard + sky) better than the u2net the TRELLIS server would run
# internally; _prepare_cutout falls back to u2net if this model can't be loaded.
R2M_REMBG_MODEL = os.environ.get("R2M_REMBG_MODEL", "isnet-general-use")

_REMBG_SESSION = None


def _prepare_cutout(image_path: Path, out_path: Path) -> Path | None:
    """Cut the object out of the photo (background removal) BEFORE TRELLIS sees it.

    Left alone, the TRELLIS server runs its own u2net rembg invisibly; on outdoor
    scenes that cutout often keeps lawn/sky fringe, which the model then reconstructs
    as part of the object (a lawn baked onto a house's underside, sky-teal walls).
    Doing it client-side (a) uses a stronger segmentation model, (b) lets the UI SHOW
    the cutout so a bad segmentation is visible instead of silently poisoning the
    build, and (c) gives the colour pipeline an exact object mask.  The saved RGBA
    png's alpha makes the server's preprocess skip its own rembg and trust ours
    (bundle code: trellis_image_to_3d.py preprocess_image).

    Returns out_path, or None when rembg is unavailable or the mask looks broken
    (callers then send the original photo, restoring the old behaviour).
    """
    global _REMBG_SESSION
    try:
        import rembg
        if _REMBG_SESSION is None:
            try:
                _REMBG_SESSION = rembg.new_session(
                    R2M_REMBG_MODEL, providers=["CPUExecutionProvider"])
            except Exception:
                _REMBG_SESSION = rembg.new_session(
                    "u2net", providers=["CPUExecutionProvider"])
        img = Image.open(image_path).convert("RGB")
        if max(img.size) > 1024:            # match the server's own pre-clamp
            scale = 1024 / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)),
                             Image.LANCZOS)
        cut = rembg.remove(img, session=_REMBG_SESSION)
        alpha = np.asarray(cut)[:, :, 3]
        # A usable object mask keeps a sane fraction of the frame; outside that the
        # segmentation failed (kept everything / nothing) and the raw photo is safer.
        frac = float((alpha > 128).mean())
        if not 0.05 <= frac <= 0.97:
            return None
        cut.save(out_path)
        return out_path
    except Exception:
        return None


def _run_trellis_api(image_path: Path, seed: int, out_path: Path,
                     mesh_simplify: float = TRELLIS_MESH_SIMPLIFY,
                     texture_size: int = TRELLIS_TEXTURE_SIZE) -> Path:
    """Generate a colored GLB via the TRELLIS portable bundle's FastAPI server.

    Server contract (IgorAherne/trellis-stable-projectorz, default port 7960):
      POST /generate_no_preview  (form `image_base64` + seed/sampling steps/
                                   mesh_simplify_ratio/texture_size/output_format)
                                 -> JSON {model_url}
      GET  <model_url>           -> binary GLB (model/gltf-binary)

    The image is sent as base64 rather than a multipart file on purpose: the bundle's
    request validator calls len() on the upload, which crashes on a single UploadFile
    ("object of type 'UploadFile' has no len()").  The base64 field skips that path.

    Writes the GLB to exactly `out_path` (caller picks a unique per-run name so no build
    can ever pick up a previous run's file) and validates it loads as a textured mesh
    before returning — a corrupt or stale download fails loudly here, not downstream.
    Raises RuntimeError with guidance when the server isn't reachable.
    """
    import base64
    import requests
    base = TRELLIS_API_URL.rstrip("/")
    try:
        requests.get(base + "/ping", timeout=5).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "TRELLIS API not reachable at {} ({}).\n"
            "Launch the portable bundle's API server (run.bat — full precision fits in "
            "16 GB VRAM and beats run-fp16.bat on quality; port 7960) before building."
            .format(base, exc)
        )

    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()
    resp = requests.post(
        base + "/generate_no_preview",
        data={
            "image_base64": img_b64,
            "seed": str(seed),
            "ss_sampling_steps": str(TRELLIS_SAMPLING_STEPS),
            "slat_sampling_steps": str(TRELLIS_SAMPLING_STEPS),
            "mesh_simplify_ratio": str(mesh_simplify),
            "texture_size": str(texture_size),
            "output_format": "glb",
        },
        timeout=1800,   # first call also loads the diffusion + reconstruction models
    )
    resp.raise_for_status()
    model_url = resp.json().get("model_url") or "/download/model"
    glb = requests.get(base + model_url, timeout=180)
    glb.raise_for_status()
    if len(glb.content) < 1024:
        raise RuntimeError("TRELLIS returned a suspiciously small model "
                           "({} bytes) — generation likely failed.".format(len(glb.content)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(glb.content)

    # Fail loudly on a corrupt/textureless download instead of proceeding.
    try:
        mesh = trimesh.load(str(out_path), force="mesh")
        uv, tex = _extract_texture(mesh)
        if len(mesh.faces) == 0:
            raise ValueError("no faces")
        if tex is None:
            raise ValueError("no baked texture")
    except Exception as exc:
        raise RuntimeError(
            "TRELLIS output {} failed validation ({}) — the server may be serving a "
            "stale/broken model; restart it and retry.".format(out_path.name, exc))
    return out_path


def _extract_texture(mesh):
    """(uv, image) from a textured trimesh — (N,2) float64 UVs aligned with the mesh's
    vertices and an (H,W,3) uint8 texture with glTF baseColorFactor applied — or
    (None, None) when the mesh carries no texture."""
    try:
        visual = mesh.visual
        if getattr(visual, "uv", None) is None:
            return None, None
        uv = np.asarray(visual.uv, dtype=np.float64)
        img = (getattr(visual.material, "baseColorTexture", None)
               or getattr(visual.material, "image", None))
        if img is None or len(uv) != len(mesh.vertices):
            return None, None
        arr = np.asarray(img.convert("RGB"), dtype=np.float64)
        # glTF multiplies the texture by baseColorFactor when present.
        factor = getattr(visual.material, "baseColorFactor", None)
        if factor is not None:
            f = np.asarray(factor, dtype=np.float64).ravel()[:3]
            if f.max() > 1.0:
                f = f / 255.0
            arr = arr * f
        return uv.copy(), np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return None, None


def _generate_mesh(image_path: Path, status_cb=None, preview_cb=None) -> Path:
    """Generate one TRELLIS GLB from the photo (cutout-conditioned, random seed).

    The object is first CUT OUT of the photo client-side (see _prepare_cutout) and the
    cutout — not the raw photo — conditions TRELLIS; `preview_cb(path)` is invoked with
    the cutout so the UI can show what will actually be reconstructed.

    Every run writes to a unique `<stem>_<runid>.glb` (nothing can reuse a previous
    run's files) and archives the input photo + cutout next to it (dev_rebuild.py looks
    them up as `<glb stem>_input.<ext>` / `<glb stem>_cutout.png`).
    """
    import random
    import shutil

    out_dir = OUTPUTS_DIR / "trellis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runid = time.strftime("%Y%m%d-%H%M%S")
    stem = "{}_{}".format(image_path.stem, runid)

    # Archive the input photo: the voxelizer needs it as the colour ground truth, and
    # offline rebuilds (dev_rebuild.py) look it up as <glb stem>_input.<ext>.
    try:
        shutil.copyfile(image_path,
                        out_dir / (stem + "_input" + image_path.suffix.lower()))
    except Exception:
        pass

    # Client-side background removal; the cutout drives generation AND the colour
    # reference (its alpha is the exact object mask).  None -> raw photo (old path).
    if status_cb:
        status_cb("Cutting the object out of the photo...")
    cutout = _prepare_cutout(image_path, out_dir / (stem + "_cutout.png"))
    source = cutout or image_path
    if cutout is not None and preview_cb:
        preview_cb(cutout)
    if cutout is None and status_cb:
        status_cb("Background removal unavailable — sending the raw photo")

    seed = random.randrange(1, 2**31 - 1)
    if status_cb:
        status_cb("TRELLIS generating (seed {})...".format(seed))
    return _run_trellis_api(source, seed=seed, out_path=out_dir / (stem + ".glb"))


# ── Mesh normalization ────────────────────────────────────────────────────────
def normalize_mesh(obj_path: Path) -> tuple:
    """Normalize a raw TRELLIS GLB to Minecraft block coordinates.

    TRELLIS already exports an upright Y-up mesh, so no reorientation is needed.
    Steps: ground placement (min Y -> Y=0), XZ centering.  Absolute size doesn't
    matter here — voxelize_mesh scales the longest axis to the Target-size slider.

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

    # Preserve the mesh's color so the voxelizer can color blocks by the real object.
    # TRELLIS GLBs carry a full texture image + per-vertex UVs — keep BOTH: sampling the
    # texture directly per voxel preserves its full resolution, whereas baking to
    # per-vertex colors (to_color = one nearest texel per vertex) aliases badly on
    # simplified meshes.  Baked vertex colors stay as the fallback for texture-less
    # meshes (InstantMesh OBJs carry `v x y z r g b`).  Captured before any processing
    # so everything stays aligned 1:1 with `verts`.
    tex_uv, tex_image = _extract_texture(mesh)
    vcolors = None
    try:
        visual = mesh.visual
        if getattr(visual, "uv", None) is not None:
            visual = visual.to_color()      # texture -> per-vertex colors (fallback path)
        vcolors = np.asarray(visual.vertex_colors, dtype=np.uint8).copy()
        if len(vcolors) != len(verts):
            vcolors = None
    except Exception:
        vcolors = None

    # 1. Ground placement: translate so the lowest vertex sits at Y=0 (TRELLIS is
    # already upright Y-up, so no reorientation needed).
    y_floor = float(verts[:, 1].min())
    verts[:, 1] -= y_floor

    # 2. Center on XZ bounding box.
    verts[:, 0] -= (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    verts[:, 2] -= (verts[:, 2].min() + verts[:, 2].max()) / 2.0

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
        "raw_up": "+Y (native)",
        "orient_tier": "trellis-native",
        # Normalized geometry, always kept (drives the mesh preview); the texture pair
        # is None when the source carried no texture — the voxelizer then uses the
        # baked vertex colors and the preview falls back to a plain surface.
        "tex_verts": verts.copy(),
        "tex_faces": faces.astype(np.int64),
        "tex_uv":    tex_uv,
        "tex_image": tex_image,
    }


# ── Voxelization ─────────────────────────────────────────────────────────────
def _bilinear_sample(image, uv):
    """Bilinear-sample an (H,W,3) uint8 image at (N,2) glTF UVs.

    glTF convention: u wraps, v has its origin at the bottom-left (so image row 0 is
    v=1).  Returns (N,3) float64 colors.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float64)
    u = np.mod(uv[:, 0], 1.0) * (w - 1)
    v = (1.0 - np.mod(uv[:, 1], 1.0)) * (h - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def _sample_voxel_colors_texture(occ, origin, grid_shape, verts, faces, uv, image,
                                 vert_thresh: float = 0.35,
                                 conflict_frac: float = 0.20,
                                 conflict_min: int = 2,
                                 mode_radius: float = 40.0):
    """RGB per voxel: mode-seeking mean of the texture over the voxel's surface patch.

    Surface splatting with supersampling: every triangle is covered with barycentric
    sample points at ~1/3-voxel spacing, each sample looks up the texture at its
    interpolated UV (bilinear).  The voxel's color is the mean of ONLY the samples
    within `mode_radius` (RGB) of its median-luminance sample — i.e. the majority
    material's samples.  A plain mean would average a bimodal voxel (black window
    frame + cream wall, dark ridge trim + brown shingles) into a mid-grey that
    exists nowhere in the texture and reads as a different material downstream; a
    plain median keeps raw texture noise that inflates cluster spreads.  The
    mode-seeking mean does both jobs: boundary voxels snap to the majority material
    (windows/doors/trim survive) while within-material noise still averages out.

    Side-aware: samples are bucketed by face orientation (UP / DOWN / SIDE at
    |n_y| > `vert_thresh`).  A voxel that collects meaningful samples from BOTH the
    top and bottom of a thin slab (a roof ~1 voxel thick: bright shingles above,
    baked-shadow underside below) keeps only the brighter vertical side — the lit
    albedo — instead of averaging them into a muddy grey that then reads as a
    different material.  Minecraft re-lights the build, so the baked-shadow side is
    lighting, not material.  Walls (all SIDE) and thick surfaces are unaffected.

    Voxels no surface sample lands in (flood-filled interiors) fall back to a
    closest-point texture lookup.  `verts` are in the scaled voxel frame (axis 1 =
    up); a point p lies in voxel floor(p) - origin.  Returns [(r,g,b)] aligned with
    `occ`, or None if sampling fails (caller then uses the vertex-color path).
    """
    try:
        occ = np.asarray(occ)
        tri_v = verts[faces]                                        # (F,3,3)
        tri_uv = uv[faces]                                          # (F,3,2)

        # Face up-components for the orientation buckets.  tex_faces never went
        # through fix_normals, so guard against globally inverted winding via the
        # signed volume (sum dot(v0, e1 x e2) = 6V; negative -> flipped).
        fn = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
        nrm = np.linalg.norm(fn, axis=1)
        ny = np.divide(fn[:, 1], nrm, out=np.zeros_like(nrm), where=nrm > 1e-12)
        if float(np.einsum("ij,ij->i", fn, tri_v[:, 0]).sum()) < 0.0:
            ny = -ny
        tri_bucket = np.where(ny > vert_thresh, 0,                  # 0=UP
                              np.where(ny < -vert_thresh, 1, 2))    # 1=DOWN 2=SIDE

        edges = np.linalg.norm(tri_v - np.roll(tri_v, 1, axis=1), axis=2)
        n_div = np.clip(np.ceil(edges.max(axis=1) * 3).astype(np.int64), 1, 96)

        ncells = int(np.prod(grid_shape))
        sums = np.zeros((3 * ncells, 3), dtype=np.float32)          # per bucket
        counts = np.zeros(3 * ncells, dtype=np.int32)
        hi = np.asarray(grid_shape, dtype=np.int64) - 1

        # Row lookup: linear cell -> occ row (-1 = unoccupied); the batch loop only
        # stores samples landing in occupied cells (the only ones ever read back).
        lin_occ = np.ravel_multi_index((occ[:, 0], occ[:, 1], occ[:, 2]), grid_shape)
        voxrow = np.full(ncells, -1, dtype=np.int32)
        voxrow[lin_occ] = np.arange(len(occ), dtype=np.int32)
        parts_row, parts_sb, parts_col = [], [], []

        # Batch triangles by subdivision count so each group splats in one shot.
        for n in np.unique(n_div):
            sel = n_div == n
            gi, gj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = (gi + gj) <= n
            b0 = gi[keep] / n
            b1 = gj[keep] / n
            bary = np.stack([b0, b1, 1.0 - b0 - b1], axis=1)        # (M,3)
            pos = np.einsum("mk,tkd->tmd", bary, tri_v[sel]).reshape(-1, 3)
            uvs = np.einsum("mk,tkd->tmd", bary, tri_uv[sel]).reshape(-1, 2)
            cols = _bilinear_sample(image, uvs)
            vidx = np.clip(np.floor(pos).astype(np.int64) - origin, 0, hi)
            lin = np.ravel_multi_index((vidx[:, 0], vidx[:, 1], vidx[:, 2]), grid_shape)
            sb = np.repeat(tri_bucket[sel], len(b0))    # sample bucket (tri-major,
            np.add.at(sums, lin + sb * ncells, cols)    # matching the reshapes above)
            np.add.at(counts, lin + sb * ncells, 1)
            row = voxrow[lin]
            m = row >= 0
            parts_row.append(row[m])
            parts_sb.append(sb[m].astype(np.uint8))
            parts_col.append(np.clip(np.rint(cols[m]), 0, 255).astype(np.uint8))

        s_up, s_dn = (sums[lin_occ + b * ncells].astype(np.float64) for b in (0, 1))
        c_up, c_dn, c_sd = (counts[lin_occ + b * ncells].astype(np.float64)
                            for b in (0, 1, 2))
        total = c_up + c_dn + c_sd

        wl = np.array([0.299, 0.587, 0.114])
        lum_up = np.divide(s_up @ wl, c_up, out=np.zeros_like(c_up), where=c_up > 0)
        lum_dn = np.divide(s_dn @ wl, c_dn, out=np.zeros_like(c_dn), where=c_dn > 0)
        # Top/bottom conflict -> keep the brighter vertical side (ties keep UP).
        conflict = np.minimum(c_up, c_dn) >= np.maximum(conflict_min,
                                                        conflict_frac * total)
        drop_up = conflict & (lum_up < lum_dn)
        drop_dn = conflict & ~drop_up
        if os.environ.get("R2M_DEBUG"):
            print("[sample] {} voxels, {} top/bottom conflicts "
                  "(dropped underside {}, top {})".format(
                      len(occ), int(conflict.sum()),
                      int(drop_dn.sum()), int(drop_up.sum())))

        # Mode-seeking mean per voxel: find the median-by-luminance sample (the
        # majority material — a real texture color, never a blend), then average
        # only the samples within mode_radius of it.  Boundary voxels snap to the
        # majority material; within-material noise still averages out.
        out = np.zeros((len(occ), 3), dtype=np.float64)
        have = np.zeros(len(occ), dtype=bool)
        if parts_row:
            s_row = np.concatenate(parts_row)
            s_sb = np.concatenate(parts_sb)
            s_col = np.concatenate(parts_col)
            kept = ~((drop_up[s_row] & (s_sb == 0)) |
                     (drop_dn[s_row] & (s_sb == 1)))
            s_row, s_col = s_row[kept], s_col[kept]
            if len(s_row):
                s_lum = s_col.astype(np.float32) @ wl.astype(np.float32)
                order = np.lexsort((s_lum, s_row))   # stable: same input, same pick
                r_sorted = s_row[order]
                starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
                seg_len = np.diff(np.r_[starts, len(r_sorted)])
                med = order[starts + (seg_len - 1) // 2]   # lower median: real sample
                rows = r_sorted[starts]
                med_by_row = np.zeros((len(occ), 3), dtype=np.float32)
                med_by_row[rows] = s_col[med].astype(np.float32)
                near = (np.linalg.norm(s_col.astype(np.float32)
                                       - med_by_row[s_row], axis=1) < mode_radius)
                acc = np.zeros((len(occ), 3), dtype=np.float64)
                cnt = np.zeros(len(occ), dtype=np.int64)
                np.add.at(acc, s_row[near], s_col[near].astype(np.float64))
                np.add.at(cnt, s_row[near], 1)
                have = cnt > 0                 # every sampled voxel keeps >= its median
                out[have] = acc[have] / cnt[have, None]

        missing = ~have
        if missing.any():
            cmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            centers = occ[missing] + 0.5 + origin
            closest, _, tri_id = trimesh.proximity.closest_point(cmesh, centers)
            bary = trimesh.triangles.points_to_barycentric(
                cmesh.triangles[tri_id], closest)
            bary = np.nan_to_num(bary, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
            uv_i = (bary[:, :, None] * uv[faces[tri_id]]).sum(axis=1)
            out[missing] = _bilinear_sample(image, uv_i)

        out = np.clip(np.rint(np.nan_to_num(out, nan=0.0)), 0, 255).astype(np.int16)
        return [tuple(int(c) for c in rgb) for rgb in out]
    except Exception:
        return None


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
        # Degenerate triangles give NaN/inf barycentric coords; sanitize before the int
        # cast, otherwise NaN->int yields a garbage huge value (breaks downstream int casts).
        blended = np.nan_to_num(blended, nan=0.0, posinf=255.0, neginf=0.0)
        blended = np.clip(np.rint(blended), 0, 255).astype(np.int16)
        return [tuple(int(c) for c in rgb) for rgb in blended]
    except Exception:
        pass
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(verts).query(pts)
        return [tuple(int(c) for c in rgb) for rgb in vcolors[idx]]
    except Exception:
        return None


def _photo_reference_lab(image_path, size: int = 96, border: int = 3,
                         bg_de: float = 10.0, min_bin_frac: float = 0.02,
                         max_colors: int = 4096):
    """LAB colour population of the input photo's object (foreground) pixels.

    The photo is the ground truth for which colours belong on the build (CLAUDE.md:
    the photo is the *reference*).  When the image carries a real alpha channel (a
    _prepare_cutout output), that alpha IS the object mask.  Otherwise background is
    estimated colour-wise from the image border ring (sky, lawn, backdrop); pixels far
    from every border colour — plus the centre patch, where the object sits — count as
    foreground.

    Only the DOMINANT colour families are kept (ab-plane bins holding >= `min_bin_frac`
    of the foreground): rare pixels — shadow slivers, anti-aliased silhouette blends —
    would otherwise legitimise hallucinations of nearly their colour (a house's teal
    bake finding support in a few cool-shadow pixels).  Returns (K,3) float64 LAB
    (unweighted), or None on any failure.
    """
    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
        from block_palette import _srgb_to_lab

        rgba = Image.open(image_path).convert("RGBA").resize((size, size))
        arr = np.asarray(rgba, dtype=np.float64)
        lab = _srgb_to_lab(arr[..., :3]).reshape(-1, 3)
        alpha = arr[..., 3]
        if (alpha < 250).any():
            # A real cutout (our _prepare_cutout output): the alpha IS the object mask
            # — exact, no heuristics.  Erode to drop anti-aliased silhouette blends.
            fg2d = ndimage.binary_erosion(alpha > 128, iterations=2)
            if int(fg2d.sum()) < 50:
                return None
        else:
            # Plain photo: estimate background colour-wise from the border ring, keep
            # pixels far from it plus the centre patch where the object sits.
            interior = np.zeros((size, size), dtype=bool)
            interior[border:-border, border:-border] = True
            d, _ = cKDTree(lab[~interior.ravel()]).query(lab)
            fg2d = interior & (d > bg_de).reshape(size, size)
            # Erode: strips the anti-aliased silhouette (object-sky blends read as
            # colours that exist in neither object nor background).
            fg2d = ndimage.binary_erosion(fg2d, iterations=2)
            c0, c1 = int(size * 0.3), int(size * 0.7)
            fg2d[c0:c1, c0:c1] = True
        cols = lab[fg2d.ravel()]
        # Keep only dominant colour families (>= min_bin_frac of foreground per ab bin).
        bins = np.floor(cols[:, 1:] / 8.0).astype(np.int64)
        key = bins[:, 0] * 4096 + bins[:, 1]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        dominant = cols[cnt[inv] >= min_bin_frac * len(cols)]
        if len(dominant) >= 50:
            cols = dominant
        if len(cols) > max_colors:
            keep = np.random.default_rng(0).choice(len(cols), max_colors, replace=False)
            cols = cols[keep]
        return cols
    except Exception:
        return None


# Lightness weight in the photo-support distance: the TRELLIS bake is globally darker
# than the photo, so support leans on hue/saturation with only a mild L penalty.
# The repaint threshold is deliberately CONSERVATIVE: it flags only egregious
# inventions (a lawn baked onto a house's underside, ~dE 48); measured real-but-
# drifted colors reach ~dE 13.6 (hard tower's cream facade), so tightening past ~16
# starts repainting real surfaces.
PHOTO_L_WEIGHT = 0.25


def _repair_unsupported_colors(occ, colors, photo_lab, thresh: float = 16.0,
                               max_frac: float = 0.5):
    """Repaint voxels whose colour does not exist anywhere in the input photo.

    TRELLIS hallucinates colours on surfaces the photo never saw (teal walls on a
    white house, gold specks in green foliage).  Any voxel colour with no close match
    in the photo's foreground population (hue-weighted LAB, see PHOTO_L_WEIGHT) is
    replaced: preferably by its mirror across X/Z when the mirror is supported (keeps
    the build's structure), else by the nearest supported voxel.

    Safety gate: if more than `max_frac` of voxels would change, the photo mask is
    probably wrong (object fills the frame) — leave everything alone.
    """
    if not colors or photo_lab is None or not len(photo_lab):
        return colors
    from scipy.spatial import cKDTree
    from block_palette import _srgb_to_lab

    occ = np.asarray(occ)
    col = np.asarray(colors, dtype=float)
    w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
    d, _ = cKDTree(np.asarray(photo_lab, dtype=np.float64) * w).query(
        _srgb_to_lab(col) * w)
    bad = d > thresh
    if not bad.any() or float(bad.mean()) > max_frac:
        return colors

    out = col.copy()
    good = ~bad
    dims = occ.max(axis=0) + 1
    gidx = np.full(tuple(dims), -1, dtype=np.int64)
    gidx[occ[:, 0], occ[:, 1], occ[:, 2]] = np.arange(len(occ))
    fixed = np.zeros(len(occ), dtype=bool)
    for ax in (0, 2):
        lo_c, hi_c = int(occ[:, ax].min()), int(occ[:, ax].max())
        mcoord = occ.copy()
        mcoord[:, ax] = (lo_c + hi_c) - occ[:, ax]
        mj = gidx[mcoord[:, 0], mcoord[:, 1], mcoord[:, 2]]     # mirror index, or -1
        use = bad & ~fixed & (mj >= 0) & good[np.where(mj >= 0, mj, 0)]
        out[use] = col[mj[use]]
        fixed |= use
    rem = bad & ~fixed
    if rem.any() and good.any():
        _, idx = cKDTree(occ[good]).query(occ[rem])
        out[rem] = col[good][idx]
    return [tuple(int(c) for c in rgb) for rgb in out]


def _snap_suspect_hues(colors, photo_lab, lo: float = 8.0, hi: float = 16.0):
    """Snap the HUE of suspicious colours onto the photo's palette, keeping lightness.

    Colours whose photo-support distance falls in the [lo, hi) band are too far from the
    photo to be trusted but not egregious enough for `_repair_unsupported_colors` to
    repaint (that threshold must stay >= ~16, see PHOTO_L_WEIGHT note).  For those, the
    voxel keeps its own L (shading/detail survives) but takes the LAB ab of its nearest
    photo colour — a wrong-hued clump becomes a shaded patch of a colour that actually
    exists in the photo, while real-but-drifted colours snap to their own colour family
    (a near no-op).  Colours below `lo` are already photo-true and stay untouched.
    """
    if not colors or photo_lab is None or not len(photo_lab):
        return colors
    from scipy.spatial import cKDTree
    from block_palette import _srgb_to_lab, _lab_to_srgb

    col = np.asarray(colors, dtype=np.float64)
    lab = _srgb_to_lab(col)
    w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
    ref = np.asarray(photo_lab, dtype=np.float64)
    d, idx = cKDTree(ref * w).query(lab * w)
    band = (d >= lo) & (d < hi)
    if not band.any():
        return colors
    lab[band, 1:] = ref[idx[band], 1:]
    col[band] = _lab_to_srgb(lab[band])
    return [tuple(int(c) for c in np.rint(rgb)) for rgb in col]


def _fill_unobserved_colors(occ, colors, photo_lab=None, pair_de: float = 20.0,
                            min_pairs: int = 30):
    """Repair TRELLIS's hallucinated far side by mirror-pair disagreement.

    TRELLIS bakes the side facing the input photo faithfully; surfaces the photo never
    saw are diffusion-generated, which shows up as regions that are DARK (grey tower
    back) or WRONG-HUED (teal house walls).  Whole-half statistics miss these (a big
    shared roof dilutes any half-mean), so the repair works pair-by-pair.  Runs on
    TONE-LIFTED colours so photo comparisons are at photo brightness.  Per horizontal
    axis (X then Z):

      1. Pair every voxel with its mirror across the model's centre plane and keep the
         pairs that STRONGLY disagree (LAB dE >= `pair_de`).  Agreeing voxels — shared
         roofs, symmetric bodies — drop out untouched.  Fewer than `min_pairs`
         disagreements means real asymmetry/noise, not a baked artifact: skip.
      2. Decide which side of the disagreeing set to trust, by cascade:
           a. a LARGE median-lightness gap -> the brighter side (a dark grey back vs a
              lit facade; hallucinated fills trend dim).  This outranks the photo test
              because a real-but-drifted bright facade can look photo-unsupported;
           b. PHOTO SUPPORT -> when one side's colours clearly exist in the input
              photo and the other's clearly don't (teal walls on a cream house),
              trust the supported side;
           c. small lightness gap, then colour diversity, else leave the axis alone.
      3. Suspect members of disagreeing pairs take their mirror's colour
         position-for-position, so the hallucinated patch inherits the real material.

    Always runs (not gated on any symmetry guess); colour-only, never changes geometry.
    `occ` is the (N,3) voxel-index array aligned 1:1 with `colors`.
    """
    if not colors or len(colors) < 4:
        return colors
    from block_palette import _srgb_to_lab

    occ = np.asarray(occ)
    out = np.asarray(colors, dtype=float)

    ptree = None
    if photo_lab is not None and len(photo_lab):
        try:
            from scipy.spatial import cKDTree
            w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
            ptree = cKDTree(np.asarray(photo_lab, dtype=np.float64) * w)
        except Exception:
            ptree = None

    dims = occ.max(axis=0) + 1
    gidx = np.full(tuple(dims), -1, dtype=np.int64)
    gidx[occ[:, 0], occ[:, 1], occ[:, 2]] = np.arange(len(occ))

    # One sweep per horizontal axis.  (A second sweep was tried and rejected: after the
    # first repaint the pair statistics describe repaired colours, and the follow-up
    # decisions repaint real surfaces — e.g. half a roof with wall colours.)
    for ax in (0, 2):
        lo_c, hi_c = int(occ[:, ax].min()), int(occ[:, ax].max())
        if hi_c - lo_c < 3:
            continue
        lab = _srgb_to_lab(out)

        mcoord = occ.copy()
        mcoord[:, ax] = (lo_c + hi_c) - occ[:, ax]
        mj = gidx[mcoord[:, 0], mcoord[:, 1], mcoord[:, 2]]     # mirror index, or -1
        center = (lo_c + hi_c) / 2.0
        is_lo = occ[:, ax] < center
        # One index per pair: lo-side voxels that have a hi-side mirror partner.
        paired = is_lo & (mj >= 0) & ~is_lo[np.where(mj >= 0, mj, 0)]
        if not paired.any():
            continue
        pi = np.where(paired)[0]                                # lo member
        pj = mj[pi]                                             # hi member
        de = np.linalg.norm(lab[pi] - lab[pj], axis=1)
        bad = de >= pair_de
        if int(bad.sum()) < min_pairs:
            continue
        bi, bj = pi[bad], pj[bad]

        # 2. Trust cascade (see docstring).
        lum_lo = float(np.median(lab[bi][:, 0]))
        lum_hi = float(np.median(lab[bj][:, 0]))
        sup_lo = sup_hi = None
        if ptree is not None:
            w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
            sup_lo = float(np.median(ptree.query(lab[bi] * w)[0]))
            sup_hi = float(np.median(ptree.query(lab[bj] * w)[0]))

        div_lo = float(lab[bi].std(axis=0).mean())
        div_hi = float(lab[bj].std(axis=0).mean())
        if abs(lum_lo - lum_hi) >= 25.0:
            # An unambiguous lit-vs-dark split.  Only this outranks the photo test: a
            # real-but-hue-drifted bright facade can look photo-unsupported (hard
            # tower, dL~35), while a hallucination is rarely THIS much brighter.
            trust_lo, rule = lum_lo > lum_hi, "bright-strong"
        elif sup_lo is not None and (sup_lo - sup_hi) >= 3.0:
            trust_lo, rule = False, "photo"  # lo's colours aren't in the photo -> fake
        elif sup_lo is not None and (sup_hi - sup_lo) >= 3.0:
            trust_lo, rule = True, "photo"
        elif abs(lum_lo - lum_hi) >= 6.0:
            trust_lo, rule = lum_lo > lum_hi, "bright-weak"
        elif div_lo >= 1.3 * div_hi:
            trust_lo, rule = True, "diversity"
        elif div_hi >= 1.3 * div_lo:
            trust_lo, rule = False, "diversity"
        else:
            continue                       # can't tell which side is real -> hands off

        # 3. Repaint the suspect member of each disagreeing pair from its mirror.
        suspect = bj if trust_lo else bi
        donor = bi if trust_lo else bj
        suspect_med = np.median(lab[suspect], axis=0)
        out[suspect] = out[donor]

        # 4. Unpaired voxels on the suspect side wearing the same (fake) colouring —
        # the hallucinated patch usually extends past where mirror partners exist —
        # follow their nearest trusted/repainted neighbour.  When the photo test made
        # the call, membership is per-voxel photo support (the patch varies more than
        # a fixed radius around its median covers); otherwise colour similarity.
        repainted = np.zeros(len(occ), dtype=bool)
        repainted[suspect] = True
        side = ~is_lo if trust_lo else is_lo
        if rule == "photo" and ptree is not None:
            w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
            cand = side & ~repainted & (ptree.query(lab * w)[0] >= 7.5)
        else:
            cand = side & ~repainted & \
                (np.linalg.norm(lab - suspect_med, axis=1) <= 12.0)
        donors = ~side | repainted
        n_extra = int(cand.sum())
        if n_extra and donors.any():
            try:
                from scipy.spatial import cKDTree
                _, idx = cKDTree(occ[donors]).query(occ[cand])
                out[cand] = out[donors][idx]
            except Exception:
                n_extra = 0
        if os.environ.get("R2M_DEBUG"):
            print("  [fill_unobs] ax={} bad={} extra={} lumL {:.1f}/{:.1f} sup {}/{} "
                  "div {:.1f}/{:.1f} -> {} trust {}".format(
                      ax, len(bi), n_extra, lum_lo, lum_hi,
                      "-" if sup_lo is None else "{:.1f}".format(sup_lo),
                      "-" if sup_hi is None else "{:.1f}".format(sup_hi),
                      div_lo, div_hi, rule, "lo" if trust_lo else "hi"))

    return [tuple(int(c) for c in rgb) for rgb in out]


def _inpaint_dark_voxels(occ, colors, floor_frac: float = 0.35,
                         abs_floor: float = 30.0, max_frac: float = 0.40,
                         min_blob_frac: float = 0.01, min_blob_floor: int = 8):
    """Repair baked-shadow / unobserved dark BLOBS from the nearest well-lit voxel.

    TRELLIS bakes parts the photo never saw (a roof top shot from ground level,
    undersides) near-black, leaving dark blobs that _fill_unobserved_colors's
    front/back mirror can't reach — they aren't a half-space split on X/Z.  Any voxel
    whose luminance falls below max(abs_floor, floor_frac * the model's median
    luminance) takes the colour of the nearest voxel above that floor.

    Only LARGE dark components are repainted (26-connected size >=
    max(min_blob_floor, min_blob_frac * N)): bake-shadow blobs span hundreds of
    voxels, while genuine small dark detail — window frames, doors, trim — forms
    small components that must survive to the palette matcher.

    Safety gate: skips entirely when more than `max_frac` of voxels qualify — then the
    object is genuinely dark (a black car), not shadow-damaged.  Colour-only.
    """
    if not colors or len(colors) < 4:
        return colors
    occ = np.asarray(occ)
    col = np.asarray(colors, dtype=float)
    lum = col @ np.array([0.299, 0.587, 0.114])
    floor = max(abs_floor, floor_frac * float(np.median(lum)))
    dark_raw = lum < floor                 # inpaint SOURCE excludes all of these
    dark = dark_raw
    n_dark_raw = int(dark_raw.sum())
    if dark.any() and float(dark.mean()) <= max_frac:
        # Keep small dark components: genuine detail, not bake shadow.
        try:
            from scipy import ndimage
            min_blob = max(min_blob_floor, int(round(min_blob_frac * len(occ))))
            shape = tuple(int(m) + 1 for m in occ.max(axis=0))
            grid = np.zeros(shape, dtype=bool)
            didx = occ[dark]
            grid[didx[:, 0], didx[:, 1], didx[:, 2]] = True
            lab, n_comp = ndimage.label(
                grid, structure=ndimage.generate_binary_structure(3, 3))
            sizes = np.bincount(lab.ravel())
            keep_small = sizes[lab[didx[:, 0], didx[:, 1], didx[:, 2]]] < min_blob
            d2 = dark.copy()
            d2[np.flatnonzero(dark)[keep_small]] = False
            dark = d2
        except Exception:
            pass
    if os.environ.get("R2M_DEBUG"):
        print("[inpaint] floor={:.1f} dark={}/{} ({:.0%}), {} in large blobs{}".format(
            floor, n_dark_raw, len(lum), n_dark_raw / max(len(lum), 1),
            int(dark.sum()),
            "  SKIPPED (over max_frac)" if n_dark_raw / max(len(lum), 1) > max_frac
            else ""))
    if not dark.any() or n_dark_raw / max(len(lum), 1) > max_frac:
        return colors
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(occ[~dark_raw]).query(occ[dark])
        col[dark] = col[~dark_raw][idx]
    except Exception:
        return colors
    return [tuple(int(c) for c in rgb) for rgb in col]


def _normalize_tone(colors, hi_pct: float = 98.0, target_hi: int = 235,
                    max_gain: float = 2.0):
    """Lift globally-dark TRELLIS textures so colors map to sensible materials.

    TRELLIS bakes low-value textures, so a whole model's colors can sit in a dark band
    that the palette matches to near-black stone (a green tree coming out mostly
    blackstone).  Compute a single brightness gain that brings the model's bright end (the
    `hi_pct` luminance percentile) up to `target_hi`, clamped to [1, max_gain] (a model that
    is already bright is left alone, never darkened).

    Hue-preserving: instead of `col*gain` then clipping each channel at 255 (which turns a
    bright green into yellow once R and G clip), the per-voxel gain is capped so the voxel's
    BRIGHTEST channel lands at exactly 255 — the colour saturates in brightness without ever
    clipping a channel, so hue/saturation are kept (green stays green).
    """
    if not colors:
        return colors
    col = np.asarray(colors, dtype=float)
    lum = col @ np.array([0.299, 0.587, 0.114])
    p_hi = float(np.percentile(lum, hi_pct))
    if p_hi < 1e-3:
        return colors                      # essentially all black -> leave as-is
    gain = min(max_gain, max(1.0, target_hi / p_hi))
    if gain <= 1.0 + 1e-3:
        return colors                      # already bright enough
    chan_max = col.max(axis=1)             # brightest channel per voxel
    safe_gain = np.divide(255.0, chan_max, out=np.full_like(chan_max, gain),
                          where=chan_max > 0)                    # cap so no channel clips
    g = np.minimum(gain, safe_gain)[:, None]
    out = np.clip(col * g, 0, 255)         # clip is now a no-op safety net; hue preserved
    return [tuple(int(c) for c in rgb) for rgb in out]


def _quantize_colors(occ, colors, grid_shape,
                     k_init: int = 24,
                     merge_de: float = 12.0,
                     merge_spread_factor: float = 1.2,
                     region_divisor: int = 2000,
                     min_region_floor: int = 3,
                     absorb_de: float = 25.0,
                     absorb_always_size: int = 1,
                     shadow_chroma_max: float = 6.0,
                     shadow_l_min: float = 28.0,
                     shadow_dl_min: float = 12.0,
                     donor_chroma_min: float = 8.0,
                     shadow_contact_frac: float = 0.25,
                     shadow_max_frac: float = 0.35):
    """Collapse noisy per-voxel colors into a few flat color regions before matching.

    TRELLIS textures carry baked lighting/AO/noise, so a surface a human reads as ONE
    material is hundreds of slightly different shades; matched voxel-by-voxel they
    straddle palette boundaries and come out as random near-miss block patches.

    Stage A — global quantization: k-means in the palette's weighted-LAB match space
    (L x L_WEIGHT, so cluster boundaries align with block boundaries), then close
    centroids are agglomeratively merged.  The merge radius is spread-aware: two
    clusters merge when their distance is under max(merge_de,
    merge_spread_factor * (rms_spread_i + rms_spread_j)) capped at 2*merge_de —
    k-means loves carving ONE noisy material into sub-clusters, and those betray
    themselves by internal spread comparable to their separation, while genuinely
    distinct materials are tight relative to their distance.  Global on purpose:
    disconnected same-material surfaces (front and back wall) land in the same
    cluster and therefore the same block.  Down-weighted L + the merge radius also
    absorb baked shading gradients into the surface's one color.

    Stage A2 — shadow → albedo repaint: a cast shadow baked into the texture (roof
    overhang darkening a wall) forms its own near-neutral dark cluster; Minecraft
    re-lights builds, so that's lighting, not material.  A cluster D qualifies when
    it is near-neutral (C < shadow_chroma_max — genuine materials, even warm beige,
    carry more chroma) and not true-black (L >= shadow_l_min, protecting window
    frames/posts).  Shadows are LOCAL, so each connected COMPONENT of D picks its
    own donor: the brighter chromatic cluster (L_B - L_D >= shadow_dl_min,
    C_B >= donor_chroma_min) holding the largest share of the component's
    brighter-neighbour boundary (>= shadow_contact_frac; contacts with darker
    clusters say nothing about D's material, shadows grade into deeper shadow).
    The component's voxels take the donor's centroid — the LIT albedo — so a wall
    shadow rejoins the wall while a roof-edge shadow rejoins the roof.  If accepted
    repaints sum past shadow_max_frac of the build, everything is skipped (misfire
    guard: a legitimately dark/two-tone model, not a shadow).

    Stage B — spatial coherence: connected components (26-conn) smaller than
    max(min_region_floor, N/region_divisor) are absorbed into their most-adjacent
    neighbouring cluster — unconditionally at size <= `absorb_always_size` (lone
    specks), otherwise only when the two centroids sit within `absorb_de`, so small
    genuine detail (a dark window on a light wall) survives.

    Runs AFTER the photo-guided repairs (their calibration needs the raw per-voxel
    variation) and only simplifies — it never re-judges hallucinations.  Fully
    deterministic (fixed k-means rng, index-ordered tie-breaks): same GLB -> same
    build.  Color-only; returns [(r,g,b)] parallel to `occ`, geometry untouched.
    """
    if not colors or len(colors) < 8:
        return colors
    import warnings
    from scipy import ndimage
    from block_palette import L_WEIGHT, _srgb_to_lab, _lab_to_srgb

    col = np.clip(np.asarray(colors, dtype=np.float64), 0, 255)
    labw = _srgb_to_lab(col)
    labw[:, 0] *= L_WEIGHT

    # ── Stage A: global quantization in weighted LAB ──────────────────────────
    uniq, inv = np.unique(col, axis=0, return_inverse=True)
    if len(uniq) <= k_init:                # tiny/flat builds: uniques ARE the clusters
        cent = _srgb_to_lab(uniq)
        cent[:, 0] *= L_WEIGHT
        labels = inv.reshape(-1).astype(np.int64)
    else:
        from scipy.cluster.vq import kmeans2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # empty-cluster warnings are handled below
            cent, labels = kmeans2(labw, k_init, iter=20, minit="++", rng=0)
        labels = labels.astype(np.int64)
        keep = np.bincount(labels, minlength=len(cent)) > 0
        cent = cent[keep]
        labels = (np.cumsum(keep) - 1)[labels]
    sizes = np.bincount(labels, minlength=len(cent)).astype(np.float64)
    k_start = len(cent)

    # Per-cluster scatter about the centroid, kept as sum-of-squares so merges can
    # combine it exactly (parallel-axis theorem); rms spread = sqrt(ssq/size).
    diffs = labw - cent[labels]
    ssq = np.bincount(labels, weights=np.einsum("ij,ij->i", diffs, diffs),
                      minlength=len(cent))

    # Merge closest pair (relative to its threshold) until none qualifies.
    cap = 2.0 * merge_de
    while len(cent) > 1:
        d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
        spread = np.sqrt(ssq / np.maximum(sizes, 1.0))
        thr = np.clip(merge_spread_factor * (spread[:, None] + spread[None, :]),
                      merge_de, cap)
        ratio = d / thr
        ratio[np.diag_indices(len(cent))] = np.inf
        i, j = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
        if ratio[i, j] >= 1.0:
            break
        if i > j:
            i, j = j, i                    # keep the lower index (determinism)
        merged = (cent[i] * sizes[i] + cent[j] * sizes[j]) / (sizes[i] + sizes[j])
        ssq[i] = (ssq[i] + sizes[i] * float(np.sum((cent[i] - merged) ** 2))
                  + ssq[j] + sizes[j] * float(np.sum((cent[j] - merged) ** 2)))
        cent[i] = merged
        sizes[i] += sizes[j]
        cent = np.delete(cent, j, axis=0)
        sizes = np.delete(sizes, j)
        ssq = np.delete(ssq, j)
        labels = np.where(labels == j, i, labels)
        labels = np.where(labels > j, labels - 1, labels)

    occ_a = np.asarray(occ)
    n_vox = len(occ_a)
    min_region = max(min_region_floor, int(round(n_vox / region_divisor)))
    struct = ndimage.generate_binary_structure(3, 3)   # 26-connectivity
    X, Y, Z = grid_shape

    def _sl(d, n):                         # (source, neighbour) slices for offset d
        return (slice(0, n - d), slice(d, n)) if d >= 0 else (slice(-d, n), slice(0, n + d))

    dbg = bool(os.environ.get("R2M_DEBUG"))

    # ── Stage A2: shadow → albedo repaint (per connected shadow patch) ────────
    K = len(cent)
    if K >= 2:
        Lu = cent[:, 0] / L_WEIGHT                   # unweighted lightness
        Cc = np.linalg.norm(cent[:, 1:], axis=1)     # chroma
        grid = np.full(grid_shape, -1, dtype=np.int64)
        grid[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]] = labels
        shadowy = np.where((Cc < shadow_chroma_max) & (Lu >= shadow_l_min))[0]
        new_lab = labels.copy()
        n_repaint = 0
        for D in shadowy:
            D = int(D)
            donor_ok = (Lu - Lu[D] >= shadow_dl_min) & (Cc >= donor_chroma_min)
            donor_ok[shadowy] = False                # shadow clusters can't donate
            if not donor_ok.any():
                continue
            comp, n_comp = ndimage.label(grid == D, structure=struct)
            if n_comp == 0:
                continue
            # Per component: contact counts with each neighbouring cluster.
            pair_keys = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if (dx, dy, dz) == (0, 0, 0):
                            continue
                        (s0, t0), (s1, t1), (s2, t2) = (_sl(dx, X), _sl(dy, Y),
                                                        _sl(dz, Z))
                        a = comp[s0, s1, s2].ravel()     # component id (1..n_comp)
                        b = grid[t0, t1, t2].ravel()     # neighbour cluster id
                        m = (a > 0) & (b >= 0) & (b != D)
                        if m.any():
                            pair_keys.append((a[m] - 1) * K + b[m])
            if not pair_keys:
                continue
            keys, counts = np.unique(np.concatenate(pair_keys), return_counts=True)
            cids, ncl = keys // K, keys % K
            brighter = Lu[ncl] > Lu[D]               # denominator: brighter contacts
            bright_tot = np.bincount(cids[brighter], weights=counts[brighter],
                                     minlength=n_comp)
            ok = donor_ok[ncl]
            # Best qualifying donor per component (max contact; ties -> low cluster).
            order = np.lexsort((ncl[ok], -counts[ok], cids[ok]))
            c_ok, n_ok, cnt_ok = cids[ok][order], ncl[ok][order], counts[ok][order]
            first = np.ones(len(order), dtype=bool)
            first[1:] = c_ok[1:] != c_ok[:-1]
            tgt_comp, tgt_donor, tgt_cnt = c_ok[first], n_ok[first], cnt_ok[first]
            frac = tgt_cnt / np.maximum(bright_tot[tgt_comp], 1)
            take = frac >= shadow_contact_frac
            if not take.any():
                if dbg:
                    print("[shadow] D#{} L={:.1f} C={:.1f}: no component passed "
                          "contact {:.2f}".format(D, Lu[D], Cc[D],
                                                  shadow_contact_frac))
                continue
            comp_to_donor = np.full(n_comp + 1, -1, dtype=np.int64)
            comp_to_donor[tgt_comp[take] + 1] = tgt_donor[take]
            vox_comp = comp[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]]
            donor_lab = comp_to_donor[vox_comp]
            upd = donor_lab >= 0
            new_lab = np.where(upd, donor_lab, new_lab)
            n_repaint += int(upd.sum())
            if dbg:
                for c, dn, fr in zip(tgt_comp[take], tgt_donor[take], frac[take]):
                    print("[shadow] D#{} L={:.1f} C={:.1f} comp {} ({} vox) -> "
                          "B#{} L={:.1f} C={:.1f} (contact {:.2f})".format(
                              D, Lu[D], Cc[D], int(c) + 1,
                              int((vox_comp == c + 1).sum()), int(dn),
                              Lu[dn], Cc[dn], fr))
        if n_repaint:
            if n_repaint > shadow_max_frac * n_vox:
                if dbg:
                    print("[shadow] skipped: {:.0%} of voxels would repaint "
                          "(> {:.0%} guard) — treating as real dark material".format(
                              n_repaint / max(n_vox, 1), shadow_max_frac))
            else:
                labels = new_lab

    # ── Stage B: absorb small spatial fragments ───────────────────────────────
    absorbed = []
    for _pass in range(2):
        K = len(cent)
        grid = np.full(grid_shape, -1, dtype=np.int64)
        grid[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]] = labels

        # Connected components per cluster, packed into one global component grid.
        comp = np.full(grid_shape, -1, dtype=np.int64)
        comp_cluster, comp_sizes, n_comp = [], [], 0
        for k in range(K):
            lab_k, n_k = ndimage.label(grid == k, structure=struct)
            if n_k == 0:
                continue
            mask = lab_k > 0
            comp[mask] = lab_k[mask] - 1 + n_comp
            comp_sizes.extend(np.bincount(lab_k[mask])[1:].tolist())
            comp_cluster.extend([k] * n_k)
            n_comp += n_k
        comp_sizes = np.asarray(comp_sizes, dtype=np.int64)
        comp_cluster = np.asarray(comp_cluster, dtype=np.int64)
        small = comp_sizes < min_region
        if not small.any():
            break

        # (small component, adjacent other-cluster) contact counts over all 26 offsets.
        pair_keys = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    (s0, t0), (s1, t1), (s2, t2) = _sl(dx, X), _sl(dy, Y), _sl(dz, Z)
                    a = comp[s0, s1, s2].ravel()
                    b = grid[t0, t1, t2].ravel()
                    m = (a >= 0) & (b >= 0)
                    a, b = a[m], b[m]
                    m = small[a] & (comp_cluster[a] != b)
                    if m.any():
                        pair_keys.append(a[m] * K + b[m])
        if not pair_keys:
            break
        keys, counts = np.unique(np.concatenate(pair_keys), return_counts=True)
        comp_ids, neigh = keys // K, keys % K
        # Modal neighbour per component; ties -> lower cluster index (determinism).
        order = np.lexsort((neigh, -counts, comp_ids))
        first = np.ones(len(order), dtype=bool)
        first[1:] = comp_ids[order][1:] != comp_ids[order][:-1]
        tgt_comp, tgt_cluster = comp_ids[order][first], neigh[order][first]

        # Absorb: lone specks always; bigger fragments only across a small color gap.
        d_cent = np.linalg.norm(cent[comp_cluster[tgt_comp]] - cent[tgt_cluster], axis=1)
        allow = (comp_sizes[tgt_comp] <= absorb_always_size) | (d_cent < absorb_de)
        tgt_comp, tgt_cluster = tgt_comp[allow], tgt_cluster[allow]
        if len(tgt_comp) == 0:
            break
        comp_to_new = np.full(n_comp, -1, dtype=np.int64)
        comp_to_new[tgt_comp] = tgt_cluster
        new_lab = comp_to_new[comp[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]]]
        upd = new_lab >= 0
        if not upd.any():
            break
        labels = np.where(upd, new_lab, labels)
        absorbed.append(int(upd.sum()))

    if os.environ.get("R2M_DEBUG"):
        top = ", ".join("{}".format(int(s))
                        for s in sorted(np.bincount(labels, minlength=len(cent)),
                                        reverse=True)[:8])
        print("[quantize] {} voxels: {} clusters -> {} after merge_de={} "
              "(top sizes {}); min_region={}, absorbed/pass={}".format(
                  n_vox, k_start, len(cent), merge_de, top, min_region,
                  absorbed or [0]))

    cent_lab = cent.copy()
    cent_lab[:, 0] /= L_WEIGHT
    cent_rgb = np.clip(np.rint(_lab_to_srgb(cent_lab)), 0, 255).astype(np.int64)
    return [tuple(int(c) for c in cent_rgb[l]) for l in labels]


def _consolidate_block_ids(colors, block_ids, max_de: float = 10.0):
    """Post-match cleanup: merge block ids of clusters that landed on near-identical
    MATERIALS (distance in the weighted-LAB match space).

    Quantization leaves only a handful of distinct colors, but two surviving clusters
    can still match palette blocks that read as the same material (stone vs
    cobblestone, ~4 apart) — a fake material boundary.  Working per unique color
    (cheap: quantization guarantees few), the smaller cluster adopts the bigger
    cluster's id whenever their MATCHED palette colors sit within `max_de`.  Global
    and deterministic — same fix on every surface, unlike a per-voxel majority vote.
    """
    if not colors or not block_ids:
        return block_ids
    from block_palette import BLOCK_PALETTE, _match_space

    col = np.clip(np.asarray(colors, dtype=np.float64), 0, 255)
    uniq, first, inv, counts = np.unique(
        col, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    if len(uniq) <= 1:
        return block_ids
    final = [block_ids[i] for i in first]      # one id per unique color (same-color
                                               # voxels always match the same block)
    pal = {bid: _match_space(np.asarray(BLOCK_PALETTE[bid], dtype=np.float64))
           for bid in set(final) if bid in BLOCK_PALETTE}
    order = np.lexsort((np.arange(len(uniq)), -counts))   # biggest cluster first
    for pos in range(1, len(order)):
        i = int(order[pos])
        for pos2 in range(pos):
            j = int(order[pos2])
            if final[i] == final[j]:
                break
            if final[i] in pal and final[j] in pal and \
                    float(np.linalg.norm(pal[final[i]] - pal[final[j]])) <= max_de:
                final[i] = final[j]
                break
    return [final[u] for u in inv]


def _voxelize_surface_sat(verts, faces):
    """Conservative triangle-box voxelization → gap-free surface occupancy grid.

    `verts` are already scaled so one voxel == one unit (pitch 1).  Every voxel a
    triangle actually intersects is marked, using the Akenine-Möller separating-axis
    test (3 AABB axes + triangle normal + 9 edge cross-products) against a unit box.
    Unlike vertex-rasterization this leaves no holes between triangles.

    Returns (surf, origin): a bool grid and the integer min-corner so a voxel index
    (i,j,k) corresponds to world center (i,j,k) + 0.5 + origin.
    """
    origin = np.floor(verts.min(axis=0)).astype(np.int64)
    upper = np.floor(verts.max(axis=0)).astype(np.int64)
    sx, sy, sz = (upper - origin + 1).tolist()
    surf = np.zeros((sx, sy, sz), dtype=bool)

    tris = (verts - origin)[faces]   # (F,3,3) in grid coords
    lo_axis = np.array([0, 0, 0])
    hi_axis = np.array([sx - 1, sy - 1, sz - 1])
    h = 0.5

    for tri in tris:
        v0, v1, v2 = tri
        lo = np.clip(np.floor(tri.min(axis=0)).astype(np.int64), lo_axis, hi_axis)
        hi = np.clip(np.floor(tri.max(axis=0)).astype(np.int64), lo_axis, hi_axis)

        gx, gy, gz = np.meshgrid(
            np.arange(lo[0], hi[0] + 1),
            np.arange(lo[1], hi[1] + 1),
            np.arange(lo[2], hi[2] + 1),
            indexing="ij",
        )
        idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)   # (M,3)
        C = idx + 0.5                                                  # voxel centers
        u0, u1, u2 = v0 - C, v1 - C, v2 - C                            # box-local verts

        # 1. AABB of the triangle vs the unit box on the 3 coordinate axes.
        ok = np.ones(len(C), dtype=bool)
        for a in range(3):
            tmin = np.minimum(np.minimum(u0[:, a], u1[:, a]), u2[:, a])
            tmax = np.maximum(np.maximum(u0[:, a], u1[:, a]), u2[:, a])
            ok &= ~((tmin > h) | (tmax < -h))
        if not ok.any():
            continue

        # 2. Triangle plane vs box (skip degenerate triangles).
        f0, f1, f2 = v1 - v0, v2 - v1, v0 - v2
        n = np.cross(f0, f1)
        if float(n @ n) < 1e-18:
            continue
        ok &= np.abs(u0 @ n) <= h * np.abs(n).sum()
        if not ok.any():
            continue

        # 3. Nine edge cross-product axes (box axis × triangle edge), reduced form.
        for f in (f0, f1, f2):
            fx, fy, fz = f
            # Each axis (box axis × edge) projects all three vertices; a separating
            # axis exists if the projected triangle clears the box radius either side.
            pa = np.stack([-u0[:, 1]*fz + u0[:, 2]*fy,
                           -u1[:, 1]*fz + u1[:, 2]*fy,
                           -u2[:, 1]*fz + u2[:, 2]*fy], axis=0)
            ra = h * (abs(fz) + abs(fy))
            ok &= ~((pa.min(0) > ra) | (pa.max(0) < -ra))
            pb = np.stack([u0[:, 0]*fz - u0[:, 2]*fx,
                           u1[:, 0]*fz - u1[:, 2]*fx,
                           u2[:, 0]*fz - u2[:, 2]*fx], axis=0)
            rb = h * (abs(fz) + abs(fx))
            ok &= ~((pb.min(0) > rb) | (pb.max(0) < -rb))
            pc = np.stack([-u0[:, 0]*fy + u0[:, 1]*fx,
                           -u1[:, 0]*fy + u1[:, 1]*fx,
                           -u2[:, 0]*fy + u2[:, 1]*fx], axis=0)
            rc = h * (abs(fy) + abs(fx))
            ok &= ~((pc.min(0) > rc) | (pc.max(0) < -rc))
            if not ok.any():
                break

        if ok.any():
            sel = idx[ok]
            surf[sel[:, 0], sel[:, 1], sel[:, 2]] = True

    return surf, origin


def _clean_voxels(solid, close: bool = True, despur: bool = False):
    """Tidy a solid occupancy grid: drop floating debris, fill pinholes.

    InstantMesh's non-watertight meshes leave stray cubes and 1-voxel pits after
    the flood fill (the floating blocks around the castle base).  Cleanup steps:

      1. close   - binary_closing fills 1-voxel pits/cracks (dilate then erode), so
                   surfaces read as solid walls instead of speckled ones.
      2. largest - keep only the largest 6-connected component, deleting every
                   floating island that isn't part of the main body.
      3. despur  - (opt-in) remove voxels with <2 occupied face-neighbours; trims
                   noise specks but can nibble legit 1-voxel spires/flagpoles, so
                   it defaults off.

    Returns a cleaned bool grid the same shape as `solid`.
    """
    from scipy import ndimage

    out = solid.copy()
    # 6-connectivity (face neighbours) — matches the flood-fill connectivity.
    struct = ndimage.generate_binary_structure(3, 1)

    if close:
        out = ndimage.binary_closing(out, structure=struct)

    labels, n = ndimage.label(out, structure=struct)
    if n > 1:
        # Component 0 is background; pick the largest of the real components.
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        out = labels == int(sizes.argmax())

    if despur:
        neighbours = ndimage.convolve(
            out.astype(np.int8), struct.astype(np.int8), mode="constant", cval=0
        ) - out  # subtract self (struct centre is 1)
        out &= neighbours >= 2

    return out


def voxelize_mesh(norm_path: Path, norm_info: dict,
                  target_blocks: int = 48,
                  match_palette: bool = True,
                  image_path: Path | None = None) -> dict:
    """Load normalized OBJ, repair, scale to Minecraft size, voxelize solid.

    Pipeline: conservative triangle-box SAT rasterization (gap-free surface shell)
    -> exterior flood fill (everything the outside can't reach becomes solid, which
    is robust to the non-watertight meshes InstantMesh produces).  The longest axis is
    scaled to exactly `target_blocks` (up OR down) so the user controls how much detail
    survives.  `match_palette` maps voxel colors to real Minecraft blocks (else
    everything is stone).

    Returns dict:
      blocks       - list of (x, y, z) int tuples, all >= 0
      colors       - list of (r,g,b) ints parallel to blocks, or None
      block_ids    - list of minecraft block ids parallel to blocks
      block_type   - 'stone' (legacy field, kept for compatibility)
      dims         - (W, H, L) bounding box in blocks
      block_count  - number of occupied blocks
      scale_factor - float applied before voxelizing
      watertight   - bool; False = warning, voxelization still proceeds
    """
    from scipy import ndimage

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

    # Scale so the longest axis lands at exactly `target_blocks` (up or down) — this is
    # the detail knob: a bigger target means more, finer voxels.
    W, D, H = norm_info["W"], norm_info["D"], norm_info["H"]
    longest = max(W, D, H)
    scale_factor = (target_blocks / longest) if longest > 1e-6 else 1.0
    mesh.apply_scale(scale_factor)

    # 1. Gap-free surface shell via conservative SAT rasterization.
    scaled_verts = np.asarray(mesh.vertices, dtype=np.float64)
    surf, origin = _voxelize_surface_sat(scaled_verts, np.asarray(mesh.faces, dtype=np.int64))

    # 2. Exterior flood fill: pad so the model is surrounded by air, label the
    # empty space (6-connectivity so 1-voxel diagonal gaps don't leak), then mark
    # everything the corner-connected outside region can't reach as solid.
    padded = np.pad(surf, 2)
    labels, _ = ndimage.label(~padded)
    exterior = labels == labels[0, 0, 0]
    solid = (~exterior)[2:-2, 2:-2, 2:-2]

    # 2b. Cleanup: drop floating debris and fill pinholes (keeps the shell fallback
    # below honest if cleanup wipes everything on a degenerate mesh).
    cleaned = _clean_voxels(solid)
    if cleaned.any():
        solid = cleaned

    occ = np.argwhere(solid)              # (N,3) int voxel indices
    if len(occ) == 0:                     # degenerate mesh — fall back to the shell
        occ = np.argwhere(surf)

    # 3. Color each voxel from the mesh surface.  Prefer sampling the TRELLIS texture
    # directly (keeps its full resolution); fall back to baked per-vertex colors for
    # texture-less sources.  Voxel centers map back to the scaled-mesh frame via
    # (index + 0.5 + origin), matching orig_verts*scale.
    centers = occ + 0.5 + origin
    colors = None
    color_source = None
    if norm_info.get("tex_uv") is not None:
        colors = _sample_voxel_colors_texture(
            occ, origin, solid.shape,
            norm_info["tex_verts"] * scale_factor,
            norm_info["tex_faces"],
            norm_info["tex_uv"],
            norm_info["tex_image"],
        )
        if colors is not None:
            color_source = "texture"
    if colors is None:
        colors = _sample_voxel_colors(
            centers, orig_verts * scale_factor, orig_faces, orig_colors
        )
        if colors is not None:
            color_source = "vertex"

    # Block coords/dims from the occupancy, shifted to origin 0.
    bx, by, bz = occ[:, 0].copy(), occ[:, 1].copy(), occ[:, 2].copy()
    bx -= bx.min(); by -= by.min(); bz -= bz.min()
    blocks = list(zip(bx.tolist(), by.tolist(), bz.tolist()))
    W_v = int(bx.max() + 1)
    H_v = int(by.max() + 1)
    L_v = int(bz.max() + 1)

    # 3b. Inpaint near-black voxels (unlit bake) from the nearest well-lit voxel.
    colors = _inpaint_dark_voxels(occ, colors)

    # 3c. Lift globally-dark TRELLIS textures so the model spans a usable tonal range
    # (otherwise dark bakes map almost entirely to near-black stone).  Also a
    # prerequisite for the photo comparisons below: an RGB gain amplifies faint tints
    # (a near-grey bake with a slight cyan cast lifts into plainly teal), and only at
    # photo brightness can voxel colours be judged against the photo honestly.
    colors = _normalize_tone(colors)

    # 3d. Unobserved-side repairs, both photo-guided: mirror-pair disagreement repair
    # (dark/wrong-hued far side inherits the observed side), then repaint any colour
    # that doesn't exist in the photo at all (TRELLIS inventions, e.g. a lawn baked
    # onto the model's underside).
    photo_lab = (_photo_reference_lab(image_path)
                 if colors and image_path is not None else None)
    colors = _fill_unobserved_colors(occ, colors, photo_lab)
    if colors and photo_lab is not None:
        colors = _repair_unsupported_colors(occ, colors, photo_lab)
        colors = _snap_suspect_hues(colors, photo_lab)

    # 3e. Region quantization: collapse the noisy per-voxel colors (baked lighting/AO/
    # texture noise) into a few flat color regions so matching can't turn one wall
    # into random near-miss materials; small spatial fragments are absorbed too.
    colors = _quantize_colors(occ, colors, solid.shape)

    # 4. Map each block's sampled color to the nearest real Minecraft block (LAB),
    # matching every voxel independently against the full palette (no per-object presets).
    # Falls back to all-stone when disabled or when the mesh carried no color.
    if match_palette and colors:
        import block_palette
        block_ids = block_palette.match_blocks(colors)
        # 4b. Material coherence: quantized clusters can still land on near-identical
        # palette blocks (stone vs cobblestone); merge those ids globally so a fake
        # material boundary never appears.
        block_ids = _consolidate_block_ids(colors, block_ids)
    else:
        block_ids = ["minecraft:stone"] * len(blocks)

    return {
        "blocks":       blocks,
        "colors":       colors,
        "block_ids":    block_ids,
        "block_type":   "stone",
        "dims":         (W_v, H_v, L_v),
        "block_count":  len(blocks),
        "scale_factor": scale_factor,
        "watertight":   bool(mesh.is_watertight),
        "color_source": color_source,   # "texture" | "vertex" | None (no color carried)
    }


# ax.voxels draws every cube face (~1.5 ms/cube), so it's only snappy for small
# models.  Above this many surface cubes, render fast square-marker points instead
# (visually near-identical at high resolution, but instant).
VOXEL_RENDER_LIMIT = 5000


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
    block_ids = voxel_data.get("block_ids")
    xs, ys, zs = zip(*blocks)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    W, H, L = voxel_data["dims"]

    # Per-block RGBA (0-1).  Prefer the matched-block palette colors so the preview shows
    # what the .schem will actually be; fall back to raw sampled colors, then to a height
    # gradient.
    palette_rgb = None
    if block_ids:
        try:
            from block_palette import BLOCK_PALETTE
            palette_rgb = np.array(
                [BLOCK_PALETTE.get(b, (128, 128, 128)) for b in block_ids], dtype=float
            )
        except Exception:
            palette_rgb = None

    if palette_rgb is not None:
        rgba = np.ones((len(blocks), 4), dtype=float)
        rgba[:, :3] = palette_rgb / 255.0
    elif colors:
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
        # High-res model — drawing real cubes (ax.voxels) costs ~1.5 ms each, so a
        # 6k-voxel shell takes ~7 s per redraw and rotation crawls.  Instead draw the
        # surface voxels as square markers (instant), but size each marker to the
        # on-screen voxel pitch so they tile into a gap-free surface — and re-fit them
        # on resize, since a fixed point size leaves gaps once the window is maximised.
        sx, sz, sy = np.nonzero(shell)
        sc = ax.scatter(sx + 0.5, sz + 0.5, sy + 0.5,
                        marker="s", c=facecolors[sx, sz, sy], depthshade=False)
        extent = max(W + 3, L, H)

        def _fit_markers(_evt=None):
            try:
                bbox = ax.get_window_extent()
                px_per_block = min(bbox.width, bbox.height) / max(extent, 1)
                side_pts = px_per_block * 72.0 / fig.dpi
                sc.set_sizes([max(9.0, (side_pts * 1.3) ** 2)])  # 1.3x overlap = no gaps
            except Exception:
                pass

        fig.canvas.mpl_connect("resize_event", _fit_markers)
        _fit_markers()
        render_note = "  [points]"

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
    blocks    = voxel_data["blocks"]
    block_ids = voxel_data.get("block_ids") or ["minecraft:stone"] * len(blocks)

    # Palette: air is always id 0; every distinct block type used gets the next id.
    palette: dict[str, int] = {"minecraft:air": 0}
    for bid in block_ids:
        if bid not in palette:
            palette[bid] = len(palette)

    # Build 3D block-id grid in YZX order (Sponge Schematic v2 convention); 0 = air.
    grid = np.zeros((H, L, W), dtype=np.int32)
    for (x, y, z), bid in zip(blocks, block_ids):
        if 0 <= y < H and 0 <= z < L and 0 <= x < W:
            grid[y, z, x] = palette[bid]

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
        "PaletteMax":  Int(len(palette)),
        "Palette":     Compound({bid: Int(idx) for bid, idx in palette.items()}),
        "BlockData":   ByteArray(block_data),
        "Offset":      IntArray([0, 0, 0]),
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

        # Voxel settings (exposed in the UI).
        self.res_var = tk.IntVar(value=48)      # target longest-axis size in blocks
        self.palette_var = tk.BooleanVar(value=True)          # match blocks vs stone-only

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

        tk.Checkbutton(
            settings, text="Match Minecraft blocks (color)",
            variable=self.palette_var,
            bg="#161b22", fg="#c9d1d9", selectcolor="#0d1117",
            activebackground="#161b22", activeforeground="#c9d1d9",
            font=("Segoe UI", 8), anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))

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
        self._build_match_palette = bool(self.palette_var.get())
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
                match_palette=self._build_match_palette,
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
