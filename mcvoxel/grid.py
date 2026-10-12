"""Scene -> block-scale voxel model, at any size, in a caller-chosen grid frame.

This is the array-level twin of `voxelize.voxelize_mesh` for the offline dataset pipeline.
It uses the same pieces -- the conservative SAT surface, the exterior flood fill,
`clean_solid`, the octant shape classifier, the mode-seeking side-aware colour sampler --
and differs in four ways that a training target needs:

* **Explicit scale and frame.** `blocks_per_unit` is chosen by the caller (the file's own
  units are meaningless for asset-library models), and `origin`/`shape` pin the grid, so a
  clean model and its degraded copy land voxel-for-voxel in the same frame.
* **Any size.** Slab/stair shapes come from a 4x-finer grid computed chunk by chunk in a
  band around the surface, so a 300-block tower keeps its stairs. `voxelize_mesh` builds
  that grid globally and has to fall back to 2x (where stairs never fire) or to full cubes
  once it passes 40M fine cells.
* **The visible shell.** Minecraft buildings are hollow and a photo never shows the inside,
  so the model is supervised on the shell -- solid voxels touching outside air -- and the
  interior is reported separately so it can be ignored.
* **Materials survive.** Each shell voxel carries the share of its surface samples that
  came from glass and door materials (`scene.KIND_*`), not just a colour.

Arrays are (X, Y, Z) with Y up, like the rest of mcvoxel; Sponge `.schem` order is
(Y, Z, X) and the conversion happens where schematics are written.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import block_shapes
from .raster import clean_solid, rasterize_surface, solid_from_surface
from .voxel_sampling import sample_voxel_surface

__all__ = ["VoxelModel", "voxelize_scene", "pad_model", "shell_of", "occluded_interior",
           "shape_shell", "remove_ground_plate", "remove_site", "classify_shapes_banded", "resolve_slivers",
           "keep_bridges"]


@dataclass
class VoxelModel:
    solid: np.ndarray                  # (X,Y,Z) bool, after cleanup
    surface: np.ndarray                # (X,Y,Z) bool, conservative surface (before fill)
    shell: np.ndarray                  # (X,Y,Z) bool, solid voxels touching outside air
    origin: np.ndarray                 # (3,) int64 grid min corner in the scaled frame
    blocks_per_unit: float
    shapes: np.ndarray | None = None   # (X,Y,Z) int8 block_shapes code, -1 where not solid
    rgb: np.ndarray | None = None      # (X,Y,Z,3) uint8, meaningful on shell voxels
    kind_frac: np.ndarray | None = None  # (X,Y,Z,n_kinds) float16 on shell voxels
    info: dict = field(default_factory=dict)
    post_mask: np.ndarray | None = None  # fine-width, anchored vertical runs; not a material guess
    post_family: np.ndarray | None = None  # 0 unknown, 1 wood, 2 masonry, 3 incompatible

    @property
    def interior(self):
        return self.solid & ~self.shell


def pad_model(vm: VoxelModel, pad) -> VoxelModel:
    """`vm` embedded in a margin of air: pad = ((x0, x1), (y0, y1), (z0, z1)) voxels.

    Every per-voxel array is padded with empty values and the origin moves with it, so the
    padded model still maps voxel index + origin to the same point of the scaled mesh.
    """
    from dataclasses import replace

    pad = tuple((int(a), int(b)) for a, b in pad)

    def grow(a, fill):
        if a is None:
            return None
        extra = ((0, 0),) * (a.ndim - 3)
        return np.pad(a, pad + extra, constant_values=fill)

    solid = grow(vm.solid, False)
    info = dict(vm.info)
    info["padded"] = [list(p) for p in pad]
    return replace(vm, solid=solid, surface=grow(vm.surface, False), shell=grow(vm.shell, False),
                   origin=np.asarray(vm.origin) - np.array([p[0] for p in pad]),
                   shapes=grow(vm.shapes, -1), rgb=grow(vm.rgb, 0), kind_frac=grow(vm.kind_frac, 0),
                   post_mask=grow(vm.post_mask, False), post_family=grow(vm.post_family, 0),
                   info=info)


def shell_of(solid, ground: bool = True):
    """Solid voxels with at least one 6-neighbour that is air.

    Outside the grid counts as air -- except below it when `ground` is set: a building
    stands on the ground, so its underside is neither visible nor supervised.
    """
    pad = np.pad(solid, 1)
    if ground:
        pad[:, 0, :] = True
    inner = (pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] &
             pad[1:-1, :-2, 1:-1] & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2])
    return solid & ~inner


def shape_shell(solid, shapes):
    """Include backing blocks visible through the empty part of a slab or stair.

    Cube-only shell culling drops roof risers behind bottom slabs. Conservatively retain
    the six solid neighbours of partial cells; it is a bounded one-cell surface band.
    """
    from scipy import ndimage
    partial = solid & (shapes > block_shapes.SHAPE_FULL)
    return shell_of(solid) | (solid & ndimage.binary_dilation(partial))


def occluded_interior(occupied):
    """Empty voxels a building encloses even when its mesh is not watertight.

    Architectural models routinely leave the bottom open and model windows as holes, so the
    exterior flood leaks in and the building comes out as a paper-thin shell. Here an empty
    voxel counts as inside when it is covered from ABOVE and from at least four of the
    other five axis directions (+-X, +-Z, below). Rooms behind glassless windows and
    floorless ground storeys pass; spaces open above do not. This axis-ray heuristic can
    also mistake porches and colonnades for rooms when other columns block the rays,
    including at high resolution. It is not a semantic interior classifier.

    Isolated compact cross-sections and one-cell tangent spans do not count as
    horizontal wall blockers. This prevents a colonnade from enclosing the axis rays
    while retaining broad walls. Attached/ambiguous columns remain a limitation.
    """
    occ = np.asarray(occupied, dtype=bool)
    from scipy import ndimage
    horizontal = occ.copy()
    # Thick columns also block individual rays. Identify compact, isolated cross-
    # sections at each storey instead of letting a row of columns impersonate walls.
    # A room's connected wall perimeter, long partition, or filled mass survives.
    for y in range(occ.shape[1]):
        layer = occ[:,y,:]
        pts = np.argwhere(layer)
        if len(pts) < 2:
            continue
        span = np.ptp(pts,axis=0)+1
        limit = max(2.,float(span.min())*.12)
        labels,n = ndimage.label(layer)
        if n <= 1:
            continue
        for label,box in enumerate(ndimage.find_objects(labels),1):
            if box is None:
                continue
            widths = [s.stop-s.start for s in box]
            if max(widths) <= limit and max(widths) <= 2*min(widths):
                horizontal[:,y,:][labels==label] = False

    def seen(axis, from_high):
        # A post is not a room boundary. Only a three-cell tangential wall run can
        # occlude a horizontal ray; vertical rays still see roofs and floors.
        blockers = occ
        if axis != 1:
            from scipy import ndimage
            structure = np.ones((1, 1, 3) if axis == 0 else (3, 1, 1), bool)
            blockers = ndimage.binary_opening(horizontal, structure=structure)
        a = np.flip(blockers, axis) if from_high else blockers
        cum = np.maximum.accumulate(a, axis=axis)
        # strictly beyond: shift by one so a voxel does not block itself
        shifted = np.zeros_like(cum)
        idx = [slice(None)] * 3
        src = [slice(None)] * 3
        idx[axis] = slice(1, None)
        src[axis] = slice(0, -1)
        shifted[tuple(idx)] = cum[tuple(src)]
        return np.flip(shifted, axis) if from_high else shifted

    above = seen(1, True)
    others = (seen(0, False).astype(np.uint8) + seen(0, True) + seen(2, False) + seen(2, True)
              + seen(1, False))
    return ~occ & above & (others >= 4)


def remove_ground_plate(solid, max_thickness=3, min_ratio: float = 1.6):
    """Strip the slab a building stands on (lawn, plinth, map tile).

    The plate is the bottom run of layers ending where the footprint drops most sharply --
    a layer at least `min_ratio` x wider than the one above it -- no thicker than
    `max_thickness` layers (an int, or a float fraction of the building's height). Three
    layers by default, so a podium that is part of the building survives; callers that
    know the object stands on a modelled plot pass a larger bound. Returns (solid, info).
    """
    ys = np.flatnonzero(solid.any(axis=(0, 2)))
    if len(ys) < 4:
        return solid, {"removed": False}
    y0, y1 = int(ys[0]), int(ys[-1])
    height = y1 - y0 + 1
    limit = int(max_thickness * height) if isinstance(max_thickness, float) else int(max_thickness)
    limit = max(1, min(limit, height - 3))
    area = solid.sum(axis=(0, 2)).astype(np.float64)
    best, best_t = 0.0, 0
    for t in range(1, limit + 1):
        above = area[y0 + t]
        if above <= 0:
            continue
        plate_top = area[y0 + t - 1]
        plate_max = area[y0:y0 + t].max()
        ratio = min(plate_top, plate_max) / above
        if ratio >= min_ratio and ratio > best:
            best, best_t = ratio, t
    if not best_t:
        return solid, {"removed": False}
    out = solid.copy()
    out[:, y0:y0 + best_t, :] = False
    return out, {"removed": True, "thickness": best_t, "area_ratio": round(float(best), 2)}


def remove_site(solid, site_height: int, margin: int = 2, core_width: int = 3):
    """Strip the site around a building: lawns, paths, pools, garden walls, fences, cars.

    The building's footprint is every column that rises at least `site_height` voxels above
    the lowest solid layer AND belongs to a mass at least `core_width` columns wide in plan
    (a 2D opening drops thinner tall features -- garden and boundary walls, fences, gate and
    lamp posts), grown by `margin` voxels (eaves, entrance steps, a porch, the first metres
    of a wall that meets the building). Every solid voxel in a column outside that footprint
    is site and is removed. A building with no mass that wide (a slender tower, a lone wall)
    is left untouched: a chimney or sign above a low roof is not enough evidence to crop
    away the building below it. One with no column that tall is also left as it is.
    Returns (solid, info).
    """
    from scipy import ndimage

    solid = np.asarray(solid, dtype=bool)
    ys = np.flatnonzero(solid.any(axis=(0, 2)))
    if len(ys) == 0:
        return solid, {"removed": 0}
    y0 = int(ys[0])
    Y = solid.shape[1]
    has = solid.any(axis=1)
    top = np.where(has, Y - 1 - np.argmax(solid[:, ::-1, :], axis=1), -1)
    tall = top >= y0 + int(site_height)
    if not tall.any():
        return solid, {"removed": 0, "no_tall_columns": True}
    core = tall
    if core_width > 1:
        core = ndimage.binary_opening(tall, structure=np.ones((core_width, core_width), bool))
        if not core.any():
            return solid, {"removed": 0, "no_broad_tall_core": True}
    square = np.ones((3, 3), bool)
    footprint = ndimage.binary_dilation(core, structure=square, iterations=int(margin)) if margin > 0 else core
    site = solid & ~footprint[:, None, :]
    return solid & ~site, {"removed": int(site.sum()), "footprint_columns": int(footprint.sum()),
                           "thin_tall_columns_dropped": int((tall & ~footprint).sum())}


def _local_exterior(free, air_faces, scale):
    """Fine cells connected to an air-facing side WITHOUT leaving their own coarse voxel.

    `free` is the fine free space of a box; `air_faces` maps each of the six directions to
    a coarse bool grid "this voxel's neighbour on that side is outside air". A fine cell is
    exterior only if it reaches, through free cells of the SAME coarse voxel, a face of that
    voxel that looks at air. Letting the flood run from voxel to voxel instead (a global or
    band flood) leaks under single-plane roofs from any open gable or eave and turns the
    whole roof into thin slabs.
    """
    n = [d * scale for d in air_faces["-x"].shape]
    idx = [np.arange(k) % scale for k in n]
    first = [(i == 0) for i in idx]                 # fine layer at each block's low face
    last = [(i == scale - 1) for i in idx]          # ... and high face

    def up(m):
        return m.repeat(scale, 0).repeat(scale, 1).repeat(scale, 2)

    def along(axis, mask1d):
        shape = [1, 1, 1]
        shape[axis] = -1
        return mask1d.reshape(shape)

    seeds = np.zeros(free.shape, dtype=bool)
    for axis, (neg, pos) in enumerate((("-x", "+x"), ("-y", "+y"), ("-z", "+z"))):
        seeds |= along(axis, first[axis]) & up(air_faces[neg])
        seeds |= along(axis, last[axis]) & up(air_faces[pos])
    ext = seeds & free

    # Moves that stay inside one coarse voxel: into a cell that is not a block's first
    # layer (moving +) or not its last layer (moving -).
    stay_pos = [along(a, ~first[a]) for a in range(3)]
    stay_neg = [along(a, ~last[a]) for a in range(3)]
    for _ in range(3 * (scale - 1) + 1):
        grow = ext.copy()
        for a in range(3):
            fwd = np.zeros_like(ext)
            sl_dst = [slice(None)] * 3
            sl_src = [slice(None)] * 3
            sl_dst[a], sl_src[a] = slice(1, None), slice(0, -1)
            fwd[tuple(sl_dst)] = ext[tuple(sl_src)]
            grow |= fwd & stay_pos[a]
            bwd = np.zeros_like(ext)
            sl_dst[a], sl_src[a] = slice(0, -1), slice(1, None)
            bwd[tuple(sl_dst)] = ext[tuple(sl_src)]
            grow |= bwd & stay_neg[a]
        grow &= free
        if (grow == ext).all():
            break
        ext = grow
    return ext


def classify_shapes_banded(solid, shell, verts, faces, origin, scale: int = 4,
                           chunk: int = 32, margin: int = 0):
    """Slab/stair shape per shell voxel from a 4x-finer grid built chunk by chunk.

    For each chunk of the coarse grid that holds shell voxels, the fine surface is
    rasterised over the chunk, and each shell voxel's fine cells are split into outside
    (reachable from an air-facing side within the voxel) and inside (everything else). The
    octant classifier then reads that fine occupancy. Everything is local to a voxel, so
    the chunking is exact and memory stays bounded for any building size.

    Returns (shapes int8 grid with -1 off-solid, fine solid fraction, empty-mask, layers,
    info). `layers` (X, Y, Z, 3) uint8 is, per axis, a bit mask of which of the voxel's
    `scale` fine layers along that axis hold any solid (bit k = layer k from the low face).
    Its population count says how thick the content is along the axis -- a wall seen edge-on
    fills few layers along its normal and all of them along the other two -- and its lowest
    and highest bits say which face the content lies against; `resolve_slivers` reads both.
    """
    if scale > 8:
        raise ValueError("layer masks hold at most 8 fine layers")
    X, Y, Z = solid.shape
    shapes = np.full(solid.shape, -1, dtype=np.int8)
    shapes[solid] = block_shapes.SHAPE_FULL
    frac = np.zeros(solid.shape, dtype=np.float16)       # fine solid fraction, shell voxels
    frac[solid] = 1.0
    empty_mask = np.zeros(solid.shape, dtype=bool)       # no octant reached OCTANT_MIN
    layers = np.zeros(solid.shape + (3,), dtype=np.uint8)
    layers[solid] = (1 << scale) - 1
    bits = (1 << np.arange(scale)).astype(np.uint8)
    if not shell.any():
        return shapes, frac, empty_mask, layers, {"chunks": 0}

    v = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = v[faces] - np.asarray(origin, dtype=np.float64)       # coarse grid coords
    tmin = np.floor(tri.min(axis=1)).astype(np.int64)
    tmax = np.floor(tri.max(axis=1)).astype(np.int64)
    air = np.pad(~solid, 1, constant_values=True)               # outside the grid is air...
    air[:, 0, :] = False                                         # ...except the ground below
    n_chunks = 0
    t0 = time.time()
    for cx in range(0, X, chunk):
        for cy in range(0, Y, chunk):
            for cz in range(0, Z, chunk):
                core = (slice(cx, min(cx + chunk, X)), slice(cy, min(cy + chunk, Y)),
                        slice(cz, min(cz + chunk, Z)))
                core_shell = shell[core]
                if not core_shell.any():
                    continue
                n_chunks += 1
                b0 = np.array([s.start for s in core])
                b1 = np.array([s.stop for s in core])
                sel = np.all((tmax >= b0 - margin) & (tmin < b1 + margin), axis=1)
                fine_surf, _ = rasterize_surface(v * scale, faces[sel],
                                                 origin=(np.asarray(origin) + b0) * scale,
                                                 shape=(b1 - b0) * scale)
                # Air status of each core voxel's six neighbours (from the padded grid).
                p = [slice(int(a) + 1, int(b) + 1) for a, b in zip(b0, b1)]

                def shifted(axis, d):
                    q = list(p)
                    q[axis] = slice(p[axis].start + d, p[axis].stop + d)
                    return air[tuple(q)]

                air_faces = {"-x": shifted(0, -1), "+x": shifted(0, 1), "-y": shifted(1, -1),
                             "+y": shifted(1, 1), "-z": shifted(2, -1), "+z": shifted(2, 1)}
                ext = _local_exterior(~fine_surf, air_faces, scale)
                fine = ~ext
                counts = block_shapes.octant_counts(fine, scale)
                code = block_shapes.shapes_from_counts(counts, scale).astype(np.int8)
                fr = (counts.sum(axis=-1) / float(scale ** 3)).astype(np.float16)
                em = counts.max(axis=-1) < block_shapes.OCTANT_MIN[scale]
                bx, by, bz = (int(n) for n in (b1 - b0))
                f6 = fine.reshape(bx, scale, by, scale, bz, scale)
                lay = np.stack([(f6.any(axis=(3, 5)) * bits[None, :, None, None]).sum(axis=1),
                                (f6.any(axis=(1, 5)) * bits[None, None, :, None]).sum(axis=2),
                                (f6.any(axis=(1, 3)) * bits[None, None, None, :]).sum(axis=3)],
                               axis=-1).astype(np.uint8)
                for dst, src in ((shapes, code), (frac, fr), (empty_mask, em), (layers, lay)):
                    view = dst[core]
                    view[core_shell] = src[core_shell]
                    dst[core] = view
    return shapes, frac, empty_mask, layers, {"chunks": n_chunks, "scale": scale,
                                              "seconds": round(time.time() - t0, 2)}


def _layer_tables(scale):
    """(count, lowest, highest) lookup tables over layer bit masks 0 .. 2**scale - 1."""
    n = 1 << scale
    count = np.array([bin(m).count("1") for m in range(n)], dtype=np.int16)
    lowest = np.array([(m & -m).bit_length() - 1 if m else scale for m in range(n)], dtype=np.int16)
    highest = np.array([m.bit_length() - 1 if m else -1 for m in range(n)], dtype=np.int16)
    return count, lowest, highest


def _neighbour(a, axis, d, fill):
    """Value of `a` at the neighbour one step `d` along `axis` (`fill` outside the grid)."""
    out = np.full(a.shape, fill, dtype=a.dtype)
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if d > 0:
        src[axis], dst[axis] = slice(1, None), slice(0, -1)
    else:
        src[axis], dst[axis] = slice(0, -1), slice(1, None)
    out[tuple(dst)] = a[tuple(src)]
    return out


def resolve_slivers(solid, frac, empty_mask, layers, scale: int = 4, max_frac: float = 0.3,
                    anchor_frac: float = 0.5):
    """Shell voxels that only repeat a surface a neighbouring voxel already represents.

    The conservative surface marks every voxel a face touches, so a wall that runs close to
    a voxel boundary -- or is thick enough to straddle one -- occupies two layers of voxels,
    one of them nearly empty; so does the edge of a sloped or curved surface. Removing the
    nearly empty one rounds the surface to the nearest voxel boundary. A voxel is such a
    repeat only when its content lies AGAINST the face it shares with the neighbour that
    represents the surface -- within half its fine layers of that face (`layers` masks):

    * A **sheet** voxel (less than half full, filling at most half its fine layers along one
      axis and nearly all of them along the other two -- a wall, slab or roof plane seen
      edge-on) is a repeat if its content lies against a face along its normal and the
      neighbour across that face is solid and holds more (a thicker fill, or the same
      sheet's larger share; ties go to the lower index, so one of a split pair survives).
    * Any other nearly empty voxel (the corner or edge of a surface, a sloped plane clipped
      at a boundary) is a repeat if its content lies against a face whose neighbour is
      substantially solid.

    A thin element that stands clear of its neighbours -- a parapet on a roof, a porch post
    between floor and roof, a balcony slab, a fin, a mullion in the middle of a voxel -- lies
    against no such face and is kept. (Pilot v2 used "touches a substantial neighbour on any
    side": it deleted parapets and the outer course of roofs, 14-38% of every shell, and
    whole porch posts.)
    """
    solid = np.asarray(solid, dtype=bool)
    f = np.where(solid, frac.astype(np.float32), -1.0)
    count_t, lowest_t, highest_t = _layer_tables(scale)
    count = [count_t[layers[..., a]] for a in range(3)]
    half, near_full = scale // 2, scale - 1
    against_low = [highest_t[layers[..., a]] < half for a in range(3)]            # content in the low half
    against_high = [lowest_t[layers[..., a]] >= scale - half for a in range(3)]   # ... in the high half
    low = solid & (f < 0.5)
    sheets = [low & (count[a] <= half) & (count[(a + 1) % 3] >= near_full) & (count[(a + 2) % 3] >= near_full)
              for a in range(3)]
    any_sheet = sheets[0] | sheets[1] | sheets[2]

    duplicate = np.zeros(solid.shape, dtype=bool)
    for a, sheet in enumerate(sheets):
        if not sheet.any():
            continue
        lower, upper = _neighbour(f, a, -1, -2.0), _neighbour(f, a, +1, -2.0)
        duplicate |= sheet & against_low[a] & (lower >= 0) & (lower >= f)
        duplicate |= sheet & against_high[a] & (upper >= 0) & (upper > f)

    sub = solid & ((f >= anchor_frac) | ~shell_of(solid))
    held = np.zeros(solid.shape, dtype=bool)
    for a in range(3):
        held |= against_low[a] & _neighbour(sub, a, -1, False)
        held |= against_high[a] & _neighbour(sub, a, +1, False)
    graze = solid & ~any_sheet & empty_mask & (f < max_frac) & held
    return duplicate | graze


def keep_bridges(solid, removed):
    """`removed` minus the voxels whose removal would cut `solid` into pieces.

    Removing surface repeats can take out the only face-connected link of an eave course,
    a roof edge or a post to the rest of the building, leaving it touching only at an edge
    or a corner -- in Minecraft, blocks that visibly float. Grow links back from detached
    pieces, but only through removed components touching two distinct kept components.
    Stop as soon as they reconnect: restoring an entire candidate component can thicken a
    whole roof to repair one eave. Unrelated repeats on a detached wing stay removed, and
    pieces already separate in the original surface are never joined.
    Returns the reduced `removed`.
    """
    from scipy import ndimage

    struct = ndimage.generate_binary_structure(3, 1)
    removed = np.asarray(removed, dtype=bool) & solid
    while removed.any():
        labels, n = ndimage.label(solid & ~removed, structure=struct)
        if n <= 1:
            break
        candidates, count = ndimage.label(removed, structure=struct)
        lowest = np.full(count + 1, n + 1, dtype=np.int32)
        highest = np.zeros(count + 1, dtype=np.int32)
        for axis in range(3):
            for step in (-1, 1):
                neighbour = _neighbour(labels, axis, step, 0)
                touch = removed & (neighbour > 0)
                np.minimum.at(lowest, candidates[touch], neighbour[touch])
                np.maximum.at(highest, candidates[touch], neighbour[touch])
        eligible = (highest > lowest)[candidates] & removed
        if not eligible.any():
            break
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        detached = (labels > 0) & (labels != int(sizes.argmax()))
        bridge = eligible & ndimage.binary_dilation(detached, structure=struct)
        if not bridge.any():
            break
        removed &= ~bridge
    return removed


def voxelize_scene(sm, blocks_per_unit: float, offset=(0.0, 0.0, 0.0), origin=None, shape=None,
                   fine_scale: int = 4, keep_largest: bool = True, drop_ground: bool = True,
                   with_shapes: bool = True, with_colors: bool = True, clean: bool = True,
                   crop: bool | None = None, max_plate=3, color_fallback: str = "closest_point",
                   trace: dict | None = None, site_height: int | None = None, site_margin: int = 2,
                   color_mode: str = "exposed", architecture: bool = False):
    """Voxelise a `scene.SceneMesh` at `blocks_per_unit` after shifting by `offset` blocks.

    verts_blocks = verts * blocks_per_unit + offset. With `origin`/`shape` the grid frame
    is fixed (use the target's frame to voxelise a degraded input); otherwise it is the
    model's own bounding box.

    `site_height` (voxels), when given, removes the low site around the building after the
    ground plate (`remove_site`). `color_mode` "exposed" colours each shell voxel from what
    is seen through its air-facing faces (`surface_color.exposed_colors`); "volume" pools
    every surface sample inside the voxel (`voxel_sampling.sample_voxel_surface`).

    `trace`, when a dict is passed, receives a copy of the grid after every stage (surface,
    flood, enclosed, after_ground, after_site, after_clean in the uncropped frame; crop_lo;
    slivers in the cropped frame) so each stage's effect can be inspected --
    `synth.voxel_debug` draws them. Tracing never changes the result.
    """
    from .scene import KIND_NAMES, scene_color_fns

    info = {}
    t0 = time.time()
    if crop is None:
        # Decided before `origin` is overwritten with the computed frame below: pilot v1-v3
        # tested it afterwards, so no grid was ever cropped to the building and every target
        # kept the frame of the whole model, site and all.
        crop = origin is None and shape is None        # a caller-fixed frame is kept as given
    v = sm.verts * float(blocks_per_unit) + np.asarray(offset, dtype=np.float64)
    surf, origin = rasterize_surface(v, sm.faces, origin=origin, shape=shape)
    solid = solid_from_surface(surf)
    if trace is not None:
        trace.update(surface=surf.copy(), flood=solid.copy(), frame_origin=np.asarray(origin).copy())
    info["surface_voxels"] = int(surf.sum())
    info["flood_voxels"] = int(solid.sum())
    if architecture:
        from .architecture import enclosed_cross_sections
        enclosed = enclosed_cross_sections(solid)
        info['enclosure_policy'] = 'closed_horizontal_cross_sections'
    else:
        enclosed = occluded_interior(solid)
    solid = solid | enclosed
    if trace is not None:
        trace["enclosed"] = enclosed.copy()
    info["occlusion_filled"] = int(enclosed.sum())
    info["filled_voxels"] = int(solid.sum())
    info["fill_gain"] = round(info["filled_voxels"] / max(1, info["surface_voxels"]), 3)

    if drop_ground:
        solid, info["ground"] = remove_ground_plate(solid, max_thickness=max_plate)
    if trace is not None:
        trace["after_ground"] = solid.copy()
    if site_height:
        solid, info["site"] = remove_site(solid, site_height, site_margin)
    if trace is not None:
        trace["after_site"] = solid.copy()
    before = int(solid.sum())
    if clean or keep_largest:
        cleaned = clean_solid(solid, close=clean, keep_largest=keep_largest and not architecture)
        if cleaned.any():
            solid = cleaned
    if trace is not None:
        trace["after_clean"] = solid.copy()
    info["kept_fraction"] = round(int(solid.sum()) / max(1, before), 3)
    # Crop the grid to the building: the lowest solid layer becomes ground level, and a
    # degraded copy voxelised later with this origin/shape lands in the same frame.
    if crop and solid.any():
        nz = [np.flatnonzero(solid.any(axis=tuple(a for a in range(3) if a != ax))) for ax in range(3)]
        lo = np.array([int(n[0]) for n in nz])
        hi = np.array([int(n[-1]) + 1 for n in nz])
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        solid, surf = solid[box], surf[box]
        origin = np.asarray(origin) + lo
        info["cropped_from"] = [int(v) for v in lo]
        if trace is not None:
            trace["crop_lo"], trace["crop_hi"] = lo.copy(), hi.copy()
    shell = shell_of(solid)
    info["shell_voxels"] = int(shell.sum())
    info["t_geometry"] = round(time.time() - t0, 2)

    vm = VoxelModel(solid=solid, surface=surf & solid, shell=shell, origin=np.asarray(origin),
                    blocks_per_unit=float(blocks_per_unit), info=info)

    if with_shapes and solid.any():
        shapes, frac, empty_mask, layers, info["shapes"] = classify_shapes_banded(
            solid, shell, v, sm.faces, origin, fine_scale)
        from .details import structural_posts
        posts, families, duplicates, detail_info = structural_posts(solid, frac, layers, fine_scale, sm, v, origin)
        vm.post_mask, vm.post_family = posts, families
        info["details"] = detail_info
        slivers = resolve_slivers(solid, frac, empty_mask, layers, fine_scale) | duplicates
        slivers &= ~posts
        if slivers.any() and keep_largest:
            slivers = keep_bridges(solid, slivers)
        if trace is not None:
            trace["slivers"] = slivers.copy()
        info["slivers_removed"] = int(slivers.sum())
        if slivers.any():
            solid = solid & ~slivers
            shapes[slivers] = -1
            shell = shell_of(solid)          # voxels behind a sliver are now outside-facing
            vm.solid, vm.shell, vm.surface = solid, shell, surf & solid
            info["shell_voxels"] = int(shell.sum())
        if architecture:
            from .architecture import coherent_wall_shapes
            shapes, info['architecture_shapes'] = coherent_wall_shapes(solid, shapes, v, sm.faces, origin)
        vm.shapes = shapes
        vm.shell = shell = shape_shell(solid, shapes)
        info["shell_voxels"] = int(shell.sum())

    if with_colors and shell.any():
        t1 = time.time()
        occ = np.argwhere(shell)
        group_colors, point_colors = scene_color_fns(sm)
        if color_mode == "exposed":
            from .surface_color import exposed_colors
            seen = exposed_colors(solid, shell, origin, v, sm.faces, point_colors,
                                  face_labels=sm.face_kind, n_labels=len(KIND_NAMES))
            res = {"rgb": seen["rgb"], "sampled": seen["seen"], "label_frac": seen["label_frac"]}
            info["seen_shell"] = int(seen["seen"].sum())
            color_fallback = "nearest"                   # an unseen voxel takes its nearest seen one
        else:
            res = sample_voxel_surface(occ, origin, solid.shape, v, sm.faces, group_colors,
                                       point_colors if color_fallback == "closest_point" else None,
                                       face_labels=sm.face_kind, n_labels=len(KIND_NAMES))
        if color_fallback == "nearest" and (~res["sampled"]).any() and res["sampled"].any():
            from scipy.spatial import cKDTree
            have = res["sampled"]
            _, j = cKDTree(occ[have]).query(occ[~have])
            res["rgb"][~have] = res["rgb"][have][j]
            res["label_frac"][~have] = res["label_frac"][have][j]
        rgb = np.zeros(solid.shape + (3,), dtype=np.uint8)
        rgb[occ[:, 0], occ[:, 1], occ[:, 2]] = np.clip(res["rgb"], 0, 255).astype(np.uint8)
        kf = np.zeros(solid.shape + (len(KIND_NAMES),), dtype=np.float16)
        kf[occ[:, 0], occ[:, 1], occ[:, 2]] = res["label_frac"].astype(np.float16)
        vm.rgb, vm.kind_frac = rgb, kf
        info["unsampled_shell"] = int((~res["sampled"]).sum())
        info["t_colors"] = round(time.time() - t1, 2)
    info["t_total"] = round(time.time() - t0, 2)
    return vm
