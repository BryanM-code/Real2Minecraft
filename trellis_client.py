"""TRELLIS image-to-3D client: cutout preparation + FastAPI server calls.

TRELLIS runs in its own env (C:\\TRELLIS) and is driven over its local FastAPI
server; this module owns everything up to a validated GLB on disk.
"""

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from config import OUTPUTS_DIR
from mesh_normalize import _extract_texture

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
