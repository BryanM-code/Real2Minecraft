"""Mesh → solid voxel grid → colored Minecraft blocks (the app's sculpture mode).

`voxelize_mesh` is the pipeline orchestrator: SAT surface rasterization, exterior
flood fill, cleanup, per-voxel color sampling (voxel_sampling), the color pipeline
(color_repair + color_quantize, gated by `color_repair`), palette matching, and
partial-block shape classification (block_shapes).

The geometry and block steps come from the shared `mcvoxel` package; the TRELLIS-specific,
photo-guided colour repairs (`color_repair`) belong to this app and stay here with it.
"""

import os
from pathlib import Path

import numpy as np
import trimesh

from mcvoxel import block_shapes
from mcvoxel.voxel_sampling import _sample_voxel_colors_texture, _sample_voxel_colors
from color_repair import (_photo_reference_lab, _inpaint_dark_voxels,
                          _normalize_tone, _lift_shadows, _fill_unobserved_colors,
                          _repair_unsupported_colors, _snap_suspect_hues)
from mcvoxel.color_quantize import quantize_colors, consolidate_block_ids
from mcvoxel.raster import clean_solid, fine_solid_aligned, rasterize_surface, solid_from_surface


def voxelize_mesh(norm_path: Path, norm_info: dict,
                  target_blocks: int = 48,
                  color_repair: bool = True,
                  partial_shapes: bool = True,
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

    `partial_shapes` (default True): after matching, classify each voxel's sub-cell
    geometry into full cube / slab / stairs (block_shapes) and encode the choice as a
    block-state string on the id.  Materials without a Minecraft variant swap to the
    closest one that has (block_shapes.SWAP_MAX_DE); shapes never add or remove voxels.

    Returns dict:
      blocks       - list of (x, y, z) int tuples, all >= 0
      colors       - list of (r,g,b) ints parallel to blocks, or None
      block_ids    - list of minecraft block ids parallel to blocks (may carry [state])
      shapes       - list of block_shapes codes parallel to blocks, or None
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
    faces_arr = np.asarray(mesh.faces, dtype=np.int64)
    surf, origin = rasterize_surface(scaled_verts, faces_arr)

    # 2. Exterior flood fill: pad so the model is surrounded by air, label the
    # empty space (6-connectivity so 1-voxel diagonal gaps don't leak), then mark
    # everything the corner-connected outside region can't reach as solid.
    solid = solid_from_surface(surf)

    # 2b. Cleanup: drop floating debris and fill pinholes (keeps the shell fallback
    # below honest if cleanup wipes everything on a degenerate mesh).
    cleaned = clean_solid(solid)
    if cleaned.any():
        solid = cleaned

    occ = np.argwhere(solid)              # (N,3) int voxel indices
    degenerate = len(occ) == 0
    if degenerate:                        # degenerate mesh — fall back to the shell
        occ = np.argwhere(surf)

    # 2c. Partial-block shapes: voxelize AGAIN at a finer pitch so each coarse voxel has
    # sub-cell occupancy (8 octants), then classify full cube / slab / stairs.  Skipped
    # on the degenerate shell path (no solid interior to classify against).  A memory
    # gate drops the fine scale (4 -> 2 -> off) so large targets never blow up.
    shapes = None
    if partial_shapes and not degenerate:
        scale = int(os.environ.get("R2M_PARTIAL_SCALE", "4"))
        extent = scaled_verts.max(axis=0) - scaled_verts.min(axis=0)

        def _fine_cells(s):
            return float(np.prod(np.floor(extent * s) + 3))

        while scale > 1 and _fine_cells(scale) > 40e6:
            scale //= 2
        if scale >= 2 and _fine_cells(scale) <= 40e6:
            fine_solid, fine_origin = fine_solid_aligned(
                scaled_verts, faces_arr, origin, solid.shape, scale)
            shapes = block_shapes.classify_shapes(
                occ, solid, fine_solid, fine_origin, origin, scale)
            if shapes is None and os.environ.get("R2M_DEBUG"):
                print("[shapes] fine-flood leak guard tripped -> all full cubes")

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
        colors = quantize_colors(occ, colors, solid.shape)

    # 3f. Shadow lift: brighten dark regions (TRELLIS bakes dark) so they match mid-tone
    # blocks instead of near-black.  Runs LAST, after any repairs/quantization, so it only
    # affects block choice — not the calibrated photo comparisons.  Both color modes.
    colors = _lift_shadows(colors)

    # 4. Map each block's sampled color to the nearest real Minecraft block (LAB),
    # matching every voxel independently against the full palette (no per-object
    # presets).  Always on; all-stone only when the mesh carried no color at all.
    if colors:
        from mcvoxel import block_palette
        block_ids = block_palette.match_blocks(colors)
        if color_repair:
            # 4b. Material coherence: quantized clusters can still land on
            # near-identical palette blocks (stone vs cobblestone); merge those ids
            # globally so a fake material boundary never appears.  (Enhanced only:
            # on raw colors the per-unique-color pass would be O(uniques²).)
            block_ids = consolidate_block_ids(colors, block_ids)
    else:
        block_ids = ["minecraft:stone"] * len(blocks)

    # 5. Partial-block shapes: rewrite ids into [state] strings (slab/stairs) and swap
    # materials that lack a variant.  Runs AFTER consolidate_block_ids so state strings
    # are never merged; covers Enhanced, Raw, and the no-color branch alike.
    if shapes is not None:
        block_ids, shapes = block_shapes.apply_shapes(block_ids, colors, shapes)

    return {
        "blocks":       blocks,
        "colors":       colors,
        "block_ids":    block_ids,
        "shapes":       (shapes.tolist() if shapes is not None else None),
        "block_type":   "stone",
        "dims":         (W_v, H_v, L_v),
        "block_count":  len(blocks),
        "scale_factor": scale_factor,
        "watertight":   bool(mesh.is_watertight),
        "color_source": color_source,   # "texture" | "vertex" | None (no color carried)
    }
