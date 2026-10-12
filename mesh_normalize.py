"""Mesh normalization: reorient/center a raw TRELLIS GLB for voxelization.

Also home to `_extract_texture`, the shared texture-extraction helper used both to
validate TRELLIS downloads and to carry the full texture into the voxelizer.
"""

from pathlib import Path

import numpy as np
import trimesh


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
    # meshes (e.g. OBJs that carry `v x y z r g b`).  Captured before any processing
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
