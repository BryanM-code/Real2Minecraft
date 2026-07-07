"""Mesh → solid voxel grid → colored Minecraft blocks.

`voxelize_mesh` is the pipeline orchestrator: SAT surface rasterization, exterior
flood fill, cleanup, per-voxel color sampling (voxel_sampling), the color pipeline
(color_repair + color_quantize, gated by `color_repair`), and palette matching.
"""

from pathlib import Path

import numpy as np
import trimesh

from voxel_sampling import _sample_voxel_colors_texture, _sample_voxel_colors
from color_repair import (_photo_reference_lab, _inpaint_dark_voxels,
                          _normalize_tone, _fill_unobserved_colors,
                          _repair_unsupported_colors, _snap_suspect_hues)
from color_quantize import _quantize_colors, _consolidate_block_ids


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
                  color_repair: bool = True,
                  image_path: Path | None = None) -> dict:
    """Load normalized OBJ, repair, scale to Minecraft size, voxelize solid.

    Pipeline: conservative triangle-box SAT rasterization (gap-free surface shell)
    -> exterior flood fill (everything the outside can't reach becomes solid, which
    is robust to the non-watertight meshes InstantMesh produces).  The longest axis is
    scaled to exactly `target_blocks` (up OR down) so the user controls how much detail
    survives.

    `color_repair` selects the color mode: True (Enhanced) runs the full pipeline —
    dark-blob inpaint, photo-guided hallucination repairs, region quantization with
    shadow→albedo repaint — so clumps and baked shadows are fixed; False (Raw) keeps
    only the brightness lift and maps the model's colors as-is (for sources with no
    baked shadows, where corrections can do more harm than good).  Palette matching
    to real Minecraft blocks is ALWAYS on; the all-stone fallback only covers meshes
    that carry no color at all.

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
    if color_repair:
        colors = _inpaint_dark_voxels(occ, colors)

    # 3c. Lift globally-dark TRELLIS textures so the model spans a usable tonal range
    # (otherwise dark bakes map almost entirely to near-black stone).  Runs in BOTH
    # color modes.  Also a prerequisite for the photo comparisons below: an RGB gain
    # amplifies faint tints (a near-grey bake with a slight cyan cast lifts into
    # plainly teal), and only at photo brightness can voxel colours be judged against
    # the photo honestly.
    colors = _normalize_tone(colors)

    if color_repair:
        # 3d. Unobserved-side repairs, both photo-guided: mirror-pair disagreement
        # repair (dark/wrong-hued far side inherits the observed side), then repaint
        # any colour that doesn't exist in the photo at all (TRELLIS inventions, e.g.
        # a lawn baked onto the model's underside).
        photo_lab = (_photo_reference_lab(image_path)
                     if colors and image_path is not None else None)
        colors = _fill_unobserved_colors(occ, colors, photo_lab)
        if colors and photo_lab is not None:
            colors = _repair_unsupported_colors(occ, colors, photo_lab)
            colors = _snap_suspect_hues(colors, photo_lab)

        # 3e. Region quantization: collapse the noisy per-voxel colors (baked
        # lighting/AO/texture noise) into a few flat color regions so matching can't
        # turn one wall into random near-miss materials; small spatial fragments are
        # absorbed too.
        colors = _quantize_colors(occ, colors, solid.shape)

    # 4. Map each block's sampled color to the nearest real Minecraft block (LAB),
    # matching every voxel independently against the full palette (no per-object
    # presets).  Always on; all-stone only when the mesh carried no color at all.
    if colors:
        import block_palette
        block_ids = block_palette.match_blocks(colors)
        if color_repair:
            # 4b. Material coherence: quantized clusters can still land on
            # near-identical palette blocks (stone vs cobblestone); merge those ids
            # globally so a fake material boundary never appears.  (Enhanced only:
            # on raw colors the per-unique-color pass would be O(uniques²).)
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
