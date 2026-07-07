#!/usr/bin/env python3
"""Interactive textured voxel viewer (pyvista/VTK) — the Minecraft-looking preview.

Renders the build as real cubes with per-block textures (from block_textures) instead of
matplotlib's flat point markers: only the exposed faces of each block are emitted, each
face is UV-mapped into a single texture atlas, edges are drawn so it reads as a crisp
blocky build, and the atlas uses nearest-neighbour sampling for the pixelated look.

`show_voxel_pyvista(voxel_data)` blocks until the window closes (like plt.show()).
When `mesh_info` (the dict from normalize_mesh) is passed, the window splits into two
linked views: the normalized TRELLIS mesh on the left, the voxel build on the right —
so hallucinated texture clumps can be traced back to the source mesh.

`show_voxel_diagnostic(voxel_data)` is the matplotlib fallback used when pyvista/VTK
is unavailable or errors out; both matplotlib and pyvista are imported lazily so this
module stays cheap to import headlessly (dev_rebuild).
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


def _add_source_mesh(pl, mesh_info, scale):
    """Add the normalized TRELLIS mesh to the active subplot in voxel-world coords.

    Same axis mapping as _build_polydata (mesh (X, Y-up, Z) -> world (X, Z, Y)) and
    scaled by the voxel scale_factor, with the min corner at the origin — so it sits
    in the voxel build's frame and linked cameras show matching orientation/size.
    Textured when UV + image are present, plain light-gray surface otherwise.
    """
    import pyvista as pv

    v = np.asarray(mesh_info["tex_verts"], dtype=float) * scale
    v = v[:, [0, 2, 1]]                    # world Z is height, like the voxel points
    v -= v.min(axis=0)                     # min corner -> block (0,0,0)
    f = np.asarray(mesh_info["tex_faces"], dtype=np.int64)
    faces = np.column_stack([np.full(len(f), 3, dtype=np.int64), f]).ravel()
    mesh = pv.PolyData(v, faces)

    uv, img = mesh_info.get("tex_uv"), mesh_info.get("tex_image")
    if uv is not None and img is not None:
        mesh.active_texture_coordinates = np.asarray(uv, dtype=float)
        pl.add_mesh(mesh, texture=pv.Texture(np.ascontiguousarray(img)),
                    lighting=True, ambient=0.45, diffuse=0.65, specular=0.0,
                    culling=False)         # axis swap flips winding; never cull
    else:
        pl.add_mesh(mesh, color="lightgray", lighting=True,
                    ambient=0.45, diffuse=0.65, specular=0.0, culling=False)


def show_voxel_pyvista(voxel_data: dict, mesh_info: dict | None = None):
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
    have_mesh = (mesh_info is not None
                 and mesh_info.get("tex_verts") is not None
                 and mesh_info.get("tex_faces") is not None
                 and len(mesh_info["tex_faces"]) > 0)
    if have_mesh:
        pl = pv.Plotter(shape=(1, 2), window_size=(1500, 750),
                        title="Real2Minecraft — TRELLIS mesh vs voxel build")
        pl.subplot(0, 0)
        pl.set_background("white")
        _add_source_mesh(pl, mesh_info, float(voxel_data.get("scale_factor") or 1.0))
        pl.add_text("TRELLIS mesh (normalized)", font_size=11, color="black")
        pl.subplot(0, 1)
    else:
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
    if have_mesh:
        pl.link_views()
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


# ── Matplotlib fallback viewer ────────────────────────────────────────────────
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
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

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
