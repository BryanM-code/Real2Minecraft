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
# seconds per candidate.
TRELLIS_SAMPLING_STEPS = 24

# Candidates per build: each gets a fresh random seed, is scored against the input photo,
# and the best one wins.  Hallucinated colours vary a lot between seeds, so best-of-N is
# the single strongest lever against baked-in artifact clumps.  Override with R2M_BEST_OF
# (e.g. 1 for fast dev iterations).
TRELLIS_BEST_OF = max(1, int(os.environ.get("R2M_BEST_OF", "3")))

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


def _score_candidate(glb_path: Path, photo_lab, n_samples: int = 20000) -> float:
    """Photo-agreement score for a candidate GLB — LOWER is better.

    Area-weighted surface samples -> interpolated UV -> texture colour -> tone lift ->
    weighted-LAB distance to the input photo's colour population.  Three terms:
      1. mean support distance (general colour drift);
      2. a heavy penalty for the fraction of surface that is outright unsupported
         (a hallucinated patch: teal walls, a lawn on the underside);
      3. a COVERAGE penalty for photo colour families the bake failed to reproduce —
         without it a washed-out grey bake wins on terms 1-2 (grey is always "legal")
         over one that actually carries the roof's brown.

    Returns 0.0 when scoring isn't possible (no photo/texture) so a lone candidate
    still passes through.
    """
    try:
        if photo_lab is None or not len(photo_lab):
            return 0.0
        from scipy.spatial import cKDTree
        from block_palette import _srgb_to_lab

        mesh = trimesh.load(str(glb_path), force="mesh")
        uv, tex = _extract_texture(mesh)
        if uv is None:
            return 0.0
        points, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
        bary = trimesh.triangles.points_to_barycentric(
            mesh.triangles[face_idx], points)
        bary = np.nan_to_num(bary, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
        uv_s = (bary[:, :, None] * uv[np.asarray(mesh.faces)[face_idx]]).sum(axis=1)
        cols = np.clip(np.rint(_bilinear_sample(tex, uv_s)), 0, 255)
        cols = _normalize_tone([tuple(int(c) for c in rgb) for rgb in cols])

        w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
        lab = _srgb_to_lab(np.asarray(cols, dtype=np.float64))
        ref = np.asarray(photo_lab, dtype=np.float64)
        d, _ = cKDTree(ref * w).query(lab * w)

        # Coverage: the bake should reproduce each dominant photo colour family in
        # roughly its photo proportion — mere existence is too weak a test (a washed-out
        # grey bake keeps a few brown texels and would pass), so penalize MASS deficits:
        # every bake sample is assigned to its nearest family, and any family holding
        # less of the bake than of the photo contributes its shortfall.
        bins = np.floor(ref[:, 1:] / 8.0).astype(np.int64)
        key = bins[:, 0] * 4096 + bins[:, 1]
        uniq, inv = np.unique(key, return_inverse=True)
        reps = np.array([np.median(ref[inv == i], axis=0) for i in range(len(uniq))])
        p_mass = np.bincount(inv, minlength=len(uniq)) / len(inv)      # photo share
        _, nearest = cKDTree(reps * w).query(lab * w)
        b_mass = np.bincount(nearest, minlength=len(uniq)) / len(lab)  # bake share
        deficit = float(np.maximum(0.0, p_mass - b_mass).sum())        # 0..1

        return (float(d.mean()) + 50.0 * float((d > 16.0).mean())
                + 20.0 * deficit)
    except Exception:
        return 0.0


def _generate_best_candidate(image_path: Path, n: int | None = None,
                             status_cb=None, preview_cb=None) -> Path:
    """Generate `n` TRELLIS candidates with fresh random seeds and keep the one that
    agrees best with the input photo.

    The object is first CUT OUT of the photo client-side (see _prepare_cutout) and the
    cutout — not the raw photo — conditions TRELLIS; `preview_cb(path)` is invoked with
    the cutout so the UI can show what will actually be reconstructed.

    Every run writes to unique `<stem>_<runid>_*` paths (nothing can reuse a previous
    run's files), archives the input photo + cutout next to the winner, and hashes each
    download to detect a stale server (two seeds returning byte-identical models).
    Losing candidates are deleted; the winner is renamed to `<stem>_<runid>.glb` and
    returned.
    """
    import hashlib
    import random
    import shutil

    n = TRELLIS_BEST_OF if n is None else max(1, n)
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

    photo_lab = _photo_reference_lab(source) if n > 1 else None
    candidates = []                       # (score, path, seed)
    hashes: dict[str, int] = {}
    for i in range(n):
        seed = random.randrange(1, 2**31 - 1)
        if status_cb:
            status_cb("TRELLIS candidate {}/{} (seed {})...".format(i + 1, n, seed))
        glb = _run_trellis_api(source, seed=seed,
                               out_path=out_dir / "{}_c{}.glb".format(stem, i + 1))
        digest = hashlib.sha256(glb.read_bytes()).hexdigest()
        if digest in hashes:
            raise RuntimeError(
                "TRELLIS returned byte-identical models for two different seeds "
                "(candidates {} and {}) — the server is serving a stale result. "
                "Restart the TRELLIS API server and retry.".format(hashes[digest], i + 1))
        hashes[digest] = i + 1
        score = _score_candidate(glb, photo_lab) if n > 1 else 0.0
        candidates.append((score, glb, seed))
        if status_cb and n > 1:
            status_cb("Candidate {}/{}: photo-agreement {:.1f} (lower = better)".format(
                i + 1, n, score))

    candidates.sort(key=lambda c: c[0])
    best_score, best_glb, best_seed = candidates[0]
    for _, glb, _ in candidates[1:]:
        try:
            glb.unlink()
        except Exception:
            pass
    final = out_dir / (stem + ".glb")
    best_glb.rename(final)
    if status_cb and n > 1:
        status_cb("Picked candidate seed {} (score {:.1f} of {})".format(
            best_seed, best_score,
            "/".join("{:.1f}".format(s) for s, _, _ in candidates)))
    return final


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
        # Full-resolution texture for direct per-voxel sampling (None when the source
        # carried no texture; the voxelizer then uses the baked vertex colors).
        "tex_verts": verts.copy() if tex_uv is not None else None,
        "tex_faces": faces.astype(np.int64) if tex_uv is not None else None,
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


def _sample_voxel_colors_texture(occ, origin, grid_shape, verts, faces, uv, image):
    """RGB per voxel by area-averaging the mesh TEXTURE over each voxel's surface patch.

    Surface splatting with supersampling: every triangle is covered with barycentric
    sample points at ~1/3-voxel spacing, each sample looks up the texture at its
    interpolated UV (bilinear), and all samples landing in a voxel are averaged.
    Unlike the baked vertex-color path this keeps the texture's full resolution —
    a stray bright texel averages out instead of tinting a whole region, and detail
    smaller than a triangle (window rows, trim) survives down to voxel scale.

    Voxels no surface sample lands in (flood-filled interiors) fall back to a
    closest-point texture lookup.  `verts` are in the scaled voxel frame; a point p
    lies in voxel floor(p) - origin.  Returns [(r,g,b)] aligned with `occ`, or None
    if sampling fails (caller then uses the vertex-color path).
    """
    try:
        occ = np.asarray(occ)
        tri_v = verts[faces]                                        # (F,3,3)
        tri_uv = uv[faces]                                          # (F,3,2)
        edges = np.linalg.norm(tri_v - np.roll(tri_v, 1, axis=1), axis=2)
        n_div = np.clip(np.ceil(edges.max(axis=1) * 3).astype(np.int64), 1, 96)

        ncells = int(np.prod(grid_shape))
        sums = np.zeros((ncells, 3), dtype=np.float64)
        counts = np.zeros(ncells, dtype=np.int64)
        hi = np.asarray(grid_shape, dtype=np.int64) - 1

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
            np.add.at(sums, lin, cols)
            np.add.at(counts, lin, 1)

        lin_occ = np.ravel_multi_index((occ[:, 0], occ[:, 1], occ[:, 2]), grid_shape)
        out = np.zeros((len(occ), 3), dtype=np.float64)
        have = counts[lin_occ] > 0
        out[have] = sums[lin_occ[have]] / counts[lin_occ[have], None]

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

    The photo is the ground truth for which colours belong on the build (the photo
    is the *reference*).  When the image carries a real alpha channel (a
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
                         abs_floor: float = 30.0, max_frac: float = 0.40):
    """Repair baked-shadow / unobserved dark PATCHES from the nearest well-lit voxel.

    TRELLIS bakes parts the photo never saw (a roof top shot from ground level,
    undersides) near-black, leaving dark blobs that _fill_unobserved_colors's
    front/back mirror can't reach — they aren't a half-space split on X/Z.  Any voxel
    whose luminance falls below max(abs_floor, floor_frac * the model's median
    luminance) takes the colour of the nearest voxel above that floor.

    Safety gate: skips entirely when more than `max_frac` of voxels qualify — then the
    object is genuinely dark (a black car), not shadow-damaged.  Colour-only.
    """
    if not colors or len(colors) < 4:
        return colors
    occ = np.asarray(occ)
    col = np.asarray(colors, dtype=float)
    lum = col @ np.array([0.299, 0.587, 0.114])
    floor = max(abs_floor, floor_frac * float(np.median(lum)))
    dark = lum < floor
    if not dark.any() or float(dark.mean()) > max_frac:
        return colors
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(occ[~dark]).query(occ[dark])
        col[dark] = col[~dark][idx]
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


def _denoise_colors(occ, colors, outlier_dist: float = 60.0):
    """Outlier-only spatial denoise: a voxel is replaced by the per-channel median of
    its occupied 26-neighbours ONLY when it sits far from that median (one pass).

    Per-voxel sampling can still leave the odd stray block (a speck the texture
    area-averaging didn't dilute).  Pulling just the outliers to the neighbourhood
    median absorbs those while leaving everything else untouched — unlike an
    unconditional median, genuine 1-voxel detail (a window row on a facade, trim
    lines) survives.  `outlier_dist` is the RGB euclidean distance that counts as
    "far" (~60 keeps window-vs-wall contrast but catches yellow-in-green specks).

    `occ` is the (N,3) integer voxel-index array aligned 1:1 with `colors`.
    """
    if not colors or len(colors) < 3:
        return colors
    col = np.clip(np.asarray(colors, dtype=np.float64), 0, 255).astype(np.int16)
    occ = np.asarray(occ)
    index = {(int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(occ)}
    out = col.copy()
    offs = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)]
    for i, (x, y, z) in enumerate(occ):
        neigh = [col[index[(x + dx, y + dy, z + dz)]]
                 for dx, dy, dz in offs
                 if (x + dx, y + dy, z + dz) in index]
        if len(neigh) >= 3:                # enough support to trust the median
            med = np.median(np.asarray(neigh), axis=0)
            if float(np.linalg.norm(col[i] - med)) > outlier_dist:
                out[i] = med
    return [tuple(int(c) for c in rgb) for rgb in out]


def _smooth_materials(blocks, block_ids, max_de: float = 10.0):
    """Post-match cleanup: a block adopts the majority block-id among its 6 face
    neighbours, but only when that id's palette colour is within `max_de` of its own
    (distance in the weighted-LAB match space).

    With a >100-block palette, two near-identical materials can alternate voxel-to-
    voxel across a flat wall (stone/cobblestone patchwork).  Snapping within-tolerance
    minorities to the local majority makes surfaces read as one material; the ΔE gate
    guarantees a real colour boundary (window vs wall) is never crossed.
    """
    from collections import Counter
    from block_palette import BLOCK_PALETTE, _match_space

    lab = {bid: _match_space(np.asarray(rgb, dtype=np.float64))
           for bid, rgb in BLOCK_PALETTE.items()}
    index = {tuple(b): i for i, b in enumerate(blocks)}
    offs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    out = list(block_ids)
    for i, (x, y, z) in enumerate(blocks):
        me = block_ids[i]
        neigh = [block_ids[index[(x + dx, y + dy, z + dz)]]
                 for dx, dy, dz in offs if (x + dx, y + dy, z + dz) in index]
        if len(neigh) < 3:
            continue
        top, n = Counter(neigh).most_common(1)[0]
        if top == me or n < 3:               # need a clear local majority
            continue
        if me in lab and top in lab and \
                float(np.linalg.norm(lab[me] - lab[top])) <= max_de:
            out[i] = top
    return out


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


def _mirror_brighter_half(solid, cgrid, axis):
    """Overwrite the darker half of `solid`/`cgrid` with the mirror of the brighter half
    across the grid centre on `axis`.  The observed (front/lit) half is brighter, so this
    copies its shape + color onto the unobserved side, fixing one-sided lumps and dark backs.
    """
    n = solid.shape[axis]
    mid = n // 2
    bright = cgrid.max(axis=3)
    lo_sum = float(bright.take(range(0, mid), axis=axis).sum())
    hi_sum = float(bright.take(range(n - mid, n), axis=axis).sum())
    sol_m = np.flip(solid, axis=axis)
    cg_m = np.flip(cgrid, axis=axis)
    idx = np.arange(n)
    sel = (idx >= n - mid) if lo_sum >= hi_sum else (idx < mid)   # the darker half
    shape = [1, 1, 1]; shape[axis] = n
    selm = np.broadcast_to(sel.reshape(shape), solid.shape)
    out_sol = np.where(selm, sol_m, solid)
    out_cg = np.where(selm[..., None], cg_m, cgrid)
    return out_sol, out_cg


def _enforce_symmetry(solid, occ, colors, force: bool = False):
    """Make a build symmetric by mirroring its observed (brighter) half onto the other side.

    The axis to mirror is the one with the largest FRONT/BACK BRIGHTNESS imbalance — that
    is the observed-vs-unobserved split (e.g. a tower's bright front vs dark back), which is
    exactly where TRELLIS leaves lumps/dark colour and where mirroring helps.  (Picking the
    most-self-similar axis instead, as before, often mirrored the wrong plane and left the
    dark-back seam.)  If both horizontal axes are clearly imbalanced it's radial -> mirror
    both.  Mirrors occupancy AND per-voxel colour brighter->darker.  Unless `force`, skips
    when neither axis is meaningfully imbalanced (nothing to repair).

    Returns (new_solid, new_occ, new_colors).
    """
    occ = np.asarray(occ)
    cols = np.clip(np.asarray(colors, dtype=np.float64), 0, 255).astype(np.int16)
    cgrid = np.zeros(solid.shape + (3,), dtype=np.int16)
    cgrid[occ[:, 0], occ[:, 1], occ[:, 2]] = cols
    lum = cols @ np.array([0.299, 0.587, 0.114])

    def imbalance(ax):
        n = solid.shape[ax]; mid = n // 2
        m = occ[:, ax]
        lo, hi = lum[m < mid], lum[m >= n - mid]
        return abs(float(lo.mean()) - float(hi.mean())) if len(lo) and len(hi) else 0.0

    bx, bz = imbalance(0), imbalance(2)
    if max(bx, bz) < 5.0 and not force:
        return solid, occ, colors                 # already balanced -> nothing to repair
    ranked = sorted([(bx, 0), (bz, 2)], reverse=True)
    axes = [ranked[0][1]]
    if ranked[1][0] >= 8.0:                        # both axes imbalanced -> radial
        axes.append(ranked[1][1])

    for ax in axes:
        solid, cgrid = _mirror_brighter_half(solid, cgrid, ax)

    occ2 = np.argwhere(solid)
    colors2 = [tuple(int(v) for v in cgrid[x, y, z]) for x, y, z in occ2]
    return solid, occ2, colors2


def voxelize_mesh(norm_path: Path, norm_info: dict,
                  target_blocks: int = 48,
                  match_palette: bool = True,
                  symmetry_mode: str = "off",
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

    # 3a. Symmetry repair (user opt-in): mirror the cleaner (observed) half across the
    # detected vertical plane — removes one-sided lumps/floating blocks and backfills
    # the unobserved side with the front's real shape + color.
    if colors and symmetry_mode == "force":
        solid, occ, colors = _enforce_symmetry(solid, occ, colors, force=True)

    # Block coords/dims from the (possibly symmetrized) occupancy, shifted to origin 0.
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

    # 3e. Spatial denoise: median-filter colors over each voxel's neighbourhood so lone
    # off-colour specks (e.g. yellow in green foliage) are absorbed before matching.
    colors = _denoise_colors(occ, colors)

    # 4. Map each block's sampled color to the nearest real Minecraft block (LAB),
    # matching every voxel independently against the full palette (no per-object presets).
    # Falls back to all-stone when disabled or when the mesh carried no color.
    if match_palette and colors:
        import block_palette
        block_ids = block_palette.match_blocks(colors)
        # 4b. Material coherence: merge near-identical materials that alternate across
        # a flat surface into the local majority (never crosses a real colour boundary).
        block_ids = _smooth_materials(blocks, block_ids)
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
        self.symmetry_var = tk.StringVar(value="Off")         # Off | Force symmetric

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

        # Symmetry: Force = always mirror the observed half; Off = never.
        sym_row = tk.Frame(settings, bg="#161b22")
        sym_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            sym_row, text="Symmetry:", bg="#161b22", fg="#8b949e", font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)
        ttk.Combobox(
            sym_row, textvariable=self.symmetry_var, state="readonly", width=14,
            values=["Force symmetric", "Off"],
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
        self._build_match_palette = bool(self.palette_var.get())
        self._build_symmetry_mode = {
            "Force symmetric": "force", "Off": "off",
        }.get(self.symmetry_var.get(), "off")
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
            obj_path = _generate_best_candidate(self.image_path,
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
                symmetry_mode=self._build_symmetry_mode,
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
            show_voxel_pyvista(voxel_data)
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
