#!/usr/bin/env python3
"""Interactive textured voxel viewer (pyvista/VTK) — the Minecraft-looking preview.

Renders the build as real cubes with per-block textures (from block_textures) instead of
matplotlib's flat point markers: only the exposed faces of each block are emitted, each
face is UV-mapped into a single texture atlas, edges are drawn so it reads as a crisp
blocky build, and the atlas uses nearest-neighbour sampling for the pixelated look.

`show_voxel_pyvista(voxel_data)` blocks until the window closes (like plt.show()).
Falls back to the matplotlib renderer if pyvista isn't available.
"""

import numpy as np

# Cube faces with OUTWARD-pointing winding (CCW seen from outside) so VTK never treats an
# exposed face as a back-face and drops it.  world maps block (X, Y=height, Z) -> world
# (X, Z, Y); a block occupies world cell [X,X+1] x [Z,Z+1] x [Y,Y+1].  Each entry is
# (block-space neighbour delta, 4 world corner offsets (a,b,c) added to (X, Z, Y)).
_FACES = [
    ((1, 0, 0),  [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),   # +X  (normal +worldX)
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),   # -X
    ((0, 0, 1),  [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),   # +Z depth -> +worldY
    ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),   # -Z depth
    ((0, 1, 0),  [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),   # +Y height (top) -> +worldZ
    ((0, -1, 0), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),   # -Y (bottom)
]
_FACE_UV = [(0, 0), (1, 0), (1, 1), (0, 1)]   # corner -> (u-pick, v-pick) within tile rect


def _build_polydata(blocks, block_ids, uv_map):
    """Return (points, quad_faces, tcoords) for the exposed faces of all blocks."""
    bset = set(blocks)
    pts, faces, uvs = [], [], []
    n = 0
    for (x, y, z), bid in zip(blocks, block_ids):
        rect = uv_map.get(bid)
        if rect is None:
            continue
        u0, v0, u1, v1 = rect
        for (dx, dy, dz), corners in _FACES:
            if (x + dx, y + dy, z + dz) in bset:
                continue                                  # neighbour present -> hidden
            for (a, b, c) in corners:
                # corner offset (a,b,c) on world cell at (X, Z, Y)
                pts.append((x + a, z + b, y + c))
            for (pu, pv) in _FACE_UV:
                uvs.append((u1 if pu else u0, v1 if pv else v0))
            faces.append((4, n, n + 1, n + 2, n + 3))
            n += 4
    return (np.asarray(pts, dtype=float),
            np.asarray(faces, dtype=np.int64).ravel(),
            np.asarray(uvs, dtype=float))


def show_voxel_pyvista(voxel_data: dict):
    """Render the voxel build as textured cubes in an interactive pyvista window."""
    import pyvista as pv
    from block_textures import build_atlas

    blocks = voxel_data.get("blocks", [])
    if not blocks:
        return
    block_ids = voxel_data.get("block_ids") or ["minecraft:stone"] * len(blocks)
    blocks = [tuple(b) for b in blocks]

    atlas, uv_map = build_atlas(block_ids)
    pts, faces, tcoords = _build_polydata(blocks, block_ids, uv_map)
    if len(pts) == 0:
        return

    mesh = pv.PolyData(pts, faces)
    mesh.active_texture_coordinates = tcoords
    # Per-face flat normals (faces don't share vertices) for correct lighting.
    mesh.compute_normals(cell_normals=True, point_normals=True,
                         auto_orient_normals=False, inplace=True)

    tex = pv.Texture(atlas)
    try:
        tex.interpolate = False        # nearest-neighbour -> crisp pixels
    except Exception:
        try:
            tex.SetInterpolate(False)
        except Exception:
            pass

    W, H, L = voxel_data["dims"]
    pl = pv.Plotter(title="Real2Minecraft — voxel build")
    pl.set_background("white")
    actor = pl.add_mesh(
        mesh, texture=tex,
        show_edges=True, edge_color="black", line_width=1,
        lighting=True, ambient=0.45, diffuse=0.65, specular=0.0,
        culling=False,                 # never drop a face by orientation
    )
    # Belt-and-suspenders: render both sides so nothing is ever see-through, even if a
    # stray face is wound the wrong way.
    try:
        actor.prop.backface_culling = False
        actor.prop.frontface_culling = False
        actor.prop.show_edges = True
    except Exception:
        pass

    # Steve scale reference: a 1x1x2 (W x D x H) red wireframe box just off the -X side.
    steve = pv.Box(bounds=(-2.0, -1.0, 0.0, 1.0, 0.0, 2.0))
    pl.add_mesh(steve, style="wireframe", color="red", line_width=2)

    note = "" if voxel_data["watertight"] else "  (mesh not watertight)"
    pl.add_text(
        "{:,} blocks   {}x{}x{} (XYZ){}".format(
            voxel_data["block_count"], W, H, L, note),
        font_size=11, color="black",
    )
    pl.add_axes()
    pl.view_isometric()
    pl.camera.azimuth = 30

    _constrain_to_upper_hemisphere(pl)
    pl.show()


def _constrain_to_upper_hemisphere(pl):
    """Stop the camera dropping below the model's horizon (no viewing the underside).

    World up is +Z.  On every interaction, if the camera has gone below the focal
    point's horizontal plane (elevation < 0 -> looking up at the bottom), snap it back
    onto the horizon, preserving azimuth and distance so orbiting/pitching stays smooth.
    Pitching down to a top view is still allowed.
    """
    def _clamp(*_):
        cam = pl.camera
        px, py, pz = cam.position
        fx, fy, fz = cam.focal_point
        if pz >= fz:
            return                       # at or above horizon -> fine
        hx, hy = px - fx, py - fy
        r = (hx * hx + hy * hy) ** 0.5
        if r < 1e-6:
            return                       # straight overhead -> nothing to clamp
        d = (hx * hx + hy * hy + (pz - fz) ** 2) ** 0.5   # keep current distance
        cam.position = (fx + hx / r * d, fy + hy / r * d, fz)
        cam.up = (0.0, 0.0, 1.0)
        pl.render()

    try:
        pl.iren.add_observer("InteractionEvent", _clamp)
        pl.iren.add_observer("EndInteractionEvent", _clamp)
    except Exception:
        pass                              # non-interactive/offscreen -> no constraint needed
