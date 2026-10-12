"""Surface rasterisation and solid fill, vectorised.

`rasterize_surface` is the same conservative triangle-box test as
`voxelize._voxelize_surface_sat` (Akenine-Moller separating axes: 3 box axes, the triangle
normal, 9 edge cross products) and marks exactly the same voxels -- `tests/test_raster.py`
checks that on random and real meshes. The difference is speed. The original loops over
triangles in Python, which is fine for a 40k-face TRELLIS mesh at 48 blocks and hopeless for
a 200k-face architectural model at 150 blocks, or for the 4x-finer grid the slab/stair
classifier needs. Here triangles are grouped by the shape of their candidate box, and each
group is tested in one numpy pass.

It also takes an explicit `origin` and `shape`, so two meshes (a clean model and its
degraded copy) can be rasterised into the SAME grid frame -- the prerequisite for using one
as the training target of the other.

`solid_from_surface` and `fine_solid_aligned` are the flood-fill and fine-grid embedding
steps that `voxelize.voxelize_mesh` and `building.massing` used to inline, and `clean_solid`
is the debris / pinhole cleanup that follows the fill.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rasterize_surface", "solid_from_surface", "fine_solid_aligned", "grid_frame",
           "clean_solid"]

_H = 0.5
_CHUNK = 1 << 22          # candidate cells tested per numpy pass (~4M x a few arrays)


def grid_frame(verts):
    """(origin, shape) the original rasteriser would use for `verts` at pitch 1."""
    origin = np.floor(verts.min(axis=0)).astype(np.int64)
    upper = np.floor(verts.max(axis=0)).astype(np.int64)
    return origin, (upper - origin + 1)


def rasterize_surface(verts, faces, origin=None, shape=None, pairs: list | None = None):
    """Conservative surface voxelisation -> (surf bool grid, origin int64[3]).

    `verts` are already scaled so one voxel == one unit. Voxel (i,j,k) spans
    [origin + (i,j,k), origin + (i,j,k) + 1). With `origin`/`shape` omitted the frame is
    the mesh's own bounding box, exactly as `_voxelize_surface_sat` chose it.

    `pairs`, when a list is passed, receives (flat voxel index, triangle index) arrays for
    every voxel/triangle intersection -- which triangles lie in which voxel, the index
    `surface_color` casts its rays against.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if origin is None or shape is None:
        o, s = grid_frame(verts)
        origin = o if origin is None else np.asarray(origin, dtype=np.int64)
        shape = s if shape is None else np.asarray(shape, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.int64)
    shape = np.asarray(shape, dtype=np.int64)
    surf = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    if len(faces) == 0:
        return surf, origin

    tris = (verts - origin)[faces]                     # (F,3,3) grid coordinates
    hi_axis = shape - 1
    lo = np.clip(np.floor(tris.min(axis=1)).astype(np.int64), 0, hi_axis)
    hi = np.clip(np.floor(tris.max(axis=1)).astype(np.int64), 0, hi_axis)

    # Degenerate triangles never mark anything (the original `continue`s on them).
    e0 = tris[:, 1] - tris[:, 0]
    e1 = tris[:, 2] - tris[:, 1]
    e2 = tris[:, 0] - tris[:, 2]
    normal = np.cross(e0, e1)
    ok_tri = np.einsum("ij,ij->i", normal, normal) >= 1e-18

    dims = hi - lo + 1                                  # candidate box per triangle
    keys = dims[:, 0] * 1_000_003 + dims[:, 1] * 1_009 + dims[:, 2]
    keys = np.where(ok_tri, keys, -1)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], len(order)]

    for s, e in zip(starts, ends):
        if sorted_keys[s] < 0:
            continue
        group = order[s:e]
        d = dims[group[0]]
        offs = np.stack(np.meshgrid(np.arange(d[0]), np.arange(d[1]), np.arange(d[2]),
                                    indexing="ij"), axis=-1).reshape(-1, 3)
        m = len(offs)
        step = max(1, _CHUNK // m)
        for c in range(0, len(group), step):
            idx = group[c:c + step]
            cells = lo[idx][:, None, :] + offs[None, :, :]          # (T,M,3)
            hit = _sat(tris[idx], normal[idx], e0[idx], e1[idx], e2[idx], cells)
            if hit.any():
                sel = cells[hit]
                surf[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                if pairs is not None:
                    flat = (sel[:, 0] * shape[1] + sel[:, 1]) * shape[2] + sel[:, 2]
                    pairs.append((flat, np.broadcast_to(idx[:, None], hit.shape)[hit]))
    return surf, origin


def _sat(tri, n, f0, f1, f2, cells):
    """Separating-axis test of T triangles against their (T,M) candidate unit boxes."""
    c = cells + 0.5                                               # (T,M,3) box centres
    u0 = tri[:, None, 0, :] - c
    u1 = tri[:, None, 1, :] - c
    u2 = tri[:, None, 2, :] - c

    ok = np.ones(c.shape[:2], dtype=bool)
    for a in range(3):                                            # box face axes
        tmin = np.minimum(np.minimum(u0[..., a], u1[..., a]), u2[..., a])
        tmax = np.maximum(np.maximum(u0[..., a], u1[..., a]), u2[..., a])
        ok &= ~((tmin > _H) | (tmax < -_H))

    ok &= np.abs(np.einsum("tmk,tk->tm", u0, n)) <= _H * np.abs(n).sum(axis=1)[:, None]

    for f in (f0, f1, f2):                                        # 9 edge axes
        fx, fy, fz = (f[:, i][:, None] for i in range(3))
        pa = np.stack([-u0[..., 1] * fz + u0[..., 2] * fy,
                       -u1[..., 1] * fz + u1[..., 2] * fy,
                       -u2[..., 1] * fz + u2[..., 2] * fy], axis=0)
        ra = _H * (np.abs(fz) + np.abs(fy))
        ok &= ~((pa.min(0) > ra) | (pa.max(0) < -ra))
        pb = np.stack([u0[..., 0] * fz - u0[..., 2] * fx,
                       u1[..., 0] * fz - u1[..., 2] * fx,
                       u2[..., 0] * fz - u2[..., 2] * fx], axis=0)
        rb = _H * (np.abs(fz) + np.abs(fx))
        ok &= ~((pb.min(0) > rb) | (pb.max(0) < -rb))
        pc = np.stack([-u0[..., 0] * fy + u0[..., 1] * fx,
                       -u1[..., 0] * fy + u1[..., 1] * fx,
                       -u2[..., 0] * fy + u2[..., 1] * fx], axis=0)
        rc = _H * (np.abs(fy) + np.abs(fx))
        ok &= ~((pc.min(0) > rc) | (pc.max(0) < -rc))
    return ok


def solid_from_surface(surf, pad: int = 2):
    """Everything the outside cannot reach through empty space (6-connected) is solid.

    Robust to non-watertight meshes in the way that matters here: a hole smaller than a
    voxel does not leak, because the conservative surface already closes it.
    """
    from scipy import ndimage

    padded = np.pad(surf, pad)
    labels, _ = ndimage.label(~padded)
    exterior = labels == labels[0, 0, 0]
    sl = tuple(slice(pad, -pad) for _ in range(3))
    return (~exterior)[sl]


def clean_solid(solid, close: bool = True, keep_largest: bool = True, despur: bool = False):
    """Tidy a solid occupancy grid: fill pinholes, drop floating debris.

      1. close        - binary closing (dilate then erode, 6-connected) bridges 1-voxel
                        slots and cracks between solid parts, so a wall split by a crack
                        reads as one wall. (A pit open to the outside is not filled: some
                        placement of the 6-connected element reaches it from outside
                        without touching solid.)
      2. keep_largest - keep only the largest 6-connected component, deleting every
                        floating island that is not part of the main body.
      3. despur       - (opt-in) remove voxels with < 2 occupied face-neighbours; trims noise
                        specks but can nibble legitimate 1-voxel spires, so it defaults off.

    The result depends only on the shape, never on where the grid happens to end: cleaning a
    grid and cleaning the same grid inside a wider margin of air give the same voxels.

    Returns a new bool grid the same shape as `solid`.
    """
    from scipy import ndimage

    out = np.array(solid, dtype=bool)
    struct = ndimage.generate_binary_structure(3, 1)       # 6-connectivity, like the flood
    if close:
        # Closing is a dilation followed by an erosion. scipy evaluates both inside the array
        # only, treating everything outside it as empty, so the dilation cannot spill past the
        # boundary and the erosion then strips every solid voxel ON the boundary (a solid 6^3
        # cube used to come back as 4^3, and meshes voxelised in their own bounding box lost
        # their outer layer on all six sides). Embedding the grid in a margin of empty space as
        # wide as the structuring element's reach lets the dilation spill outward and the
        # erosion take it back, which is exactly the closing of the shape in unbounded space;
        # nothing the closing adds can lie in that margin, so cropping it off loses nothing.
        pad = 1                                            # reach of one 6-connected step
        closed = ndimage.binary_closing(np.pad(out, pad), structure=struct)
        out = closed[pad:-pad, pad:-pad, pad:-pad]
    if keep_largest:
        labels, n = ndimage.label(out, structure=struct)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0                                   # label 0 is the background
            out = labels == int(sizes.argmax())
    if despur:
        neighbours = ndimage.convolve(out.astype(np.int8), struct.astype(np.int8),
                                      mode="constant", cval=0) - out
        out &= neighbours >= 2
    return out


def fine_solid_aligned(scaled_verts, faces, origin, coarse_shape, scale):
    """Solid occupancy at `scale`x finer pitch, embedded in the coarse grid's frame.

    Every coarse voxel gets exactly scale^3 fine cells; without the embedding the +X/+Z
    boundary voxels are truncated and misclassify (see the partial-blocks notes).
    Returns (fine_solid, fine_origin).
    """
    fine_origin = np.asarray(origin, dtype=np.int64) * scale
    shape = np.asarray(coarse_shape, dtype=np.int64) * scale
    fine_surf, _ = rasterize_surface(np.asarray(scaled_verts, dtype=np.float64) * scale, faces,
                                     origin=fine_origin, shape=shape)
    return solid_from_surface(fine_surf), fine_origin
