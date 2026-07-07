#!/usr/bin/env python3
"""Offline rebuild harness: re-voxelize the cached TRELLIS GLBs and screenshot each result.

Skips TRELLIS/the UI entirely — just normalize_mesh + voxelize_mesh + an off-screen
pyvista render, so palette/color-pipeline changes can be compared before/after without
regenerating any meshes.

Usage:
    r2m_app\\Scripts\\python.exe dev_rebuild.py <tag> [glb_stem ...]

Writes outputs/dev_renders/<tag>/<name>.png (+ a per-model block-id histogram to stdout).
`glb_stem` filters to specific cached models (e.g. "Tree_for_M2R"); default is all.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from config import OUTPUTS_DIR
from mesh_normalize import normalize_mesh
from voxelize import voxelize_mesh

TARGET_BLOCKS = 48          # match the UI default so renders are comparable to the reports


def render_screenshot(voxel_data: dict, out_png: Path):
    """Off-screen pyvista render of a voxel build (same look as the interactive viewer)."""
    import pyvista as pv
    from block_textures import build_atlas
    from voxel_view import _build_polydata

    blocks = [tuple(b) for b in voxel_data["blocks"]]
    block_ids = voxel_data.get("block_ids") or ["minecraft:stone"] * len(blocks)
    atlas, uv_map = build_atlas(block_ids)
    pts, faces, tcoords = _build_polydata(blocks, block_ids, uv_map)
    if len(pts) == 0:
        return

    mesh = pv.PolyData(pts, faces)
    mesh.active_texture_coordinates = tcoords
    mesh.compute_normals(cell_normals=True, point_normals=True,
                         auto_orient_normals=False, inplace=True)
    tex = pv.Texture(atlas)
    try:
        tex.interpolate = False
    except Exception:
        pass

    # Front and back isometric views, so unobserved-side artifacts are visible too.
    for suffix, azimuth in (("", 30), ("_back", 210)):
        pl = pv.Plotter(off_screen=True, window_size=(1024, 800))
        pl.set_background("white")
        pl.add_mesh(mesh, texture=tex, show_edges=True, edge_color="black", line_width=1,
                    lighting=True, ambient=0.45, diffuse=0.65, specular=0.0, culling=False)
        W, H, L = voxel_data["dims"]
        pl.add_text("{:,} blocks   {}x{}x{} (XYZ)".format(
            voxel_data["block_count"], W, H, L), font_size=11, color="black")
        pl.view_isometric()
        pl.camera.azimuth = azimuth
        pl.screenshot(str(out_png.with_name(out_png.stem + suffix + ".png")))
        pl.close()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "dev"
    only = set(sys.argv[2:])
    glb_dir = OUTPUTS_DIR / "trellis"
    out_dir = OUTPUTS_DIR / "dev_renders" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    glbs = sorted(glb_dir.glob("*.glb"))
    if only:
        glbs = [g for g in glbs if g.stem in only]
    if not glbs:
        print("No cached GLBs found under", glb_dir)
        return 1

    for glb in glbs:
        t0 = time.time()
        # The archived cutout (exact object mask) beats the raw input photo as the
        # colour reference; fall back to the input for pre-cutout archives.
        photo = (next(glb.parent.glob(glb.stem + "_cutout.png"), None)
                 or next(glb.parent.glob(glb.stem + "_input.*"), None))
        norm_path, norm_info = normalize_mesh(glb)
        voxel_data = voxelize_mesh(norm_path, norm_info,
                                   target_blocks=TARGET_BLOCKS,
                                   image_path=photo)
        dt = time.time() - t0
        png = out_dir / (glb.stem + ".png")
        render_screenshot(voxel_data, png)

        W, H, L = voxel_data["dims"]
        print("\n== {}  ({:.1f}s)  {:,} blocks  {}x{}x{}  [colors: {}, photo: {}]".format(
            glb.stem, dt, voxel_data["block_count"], W, H, L,
            voxel_data.get("color_source"), photo.name if photo else "none"))
        hist = Counter(voxel_data["block_ids"]).most_common(8)
        total = len(voxel_data["block_ids"])
        for bid, n in hist:
            print("   {:5.1f}%  {}".format(100.0 * n / total, bid))
        print("   ->", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
