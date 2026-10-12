"""Colour each block the way it looks from outside: rays through its air-facing faces.

The volume sampler (`voxel_sampling.sample_voxel_surface`) pools every surface point that
falls inside a voxel -- front and back of a wall, a balcony's white floor and its red front,
a roof's thatch and the beams laid over it, interior walls behind a window -- and a voxel
takes the majority. Measured on the pilot, that is where most "wrong material" targets came
from: painted windows vanished into the wall, balcony fronts took the floor's colour.

A block's appearance is what you see through its exposed faces. For every shell voxel and
every face that looks at air, `exposed_colors` casts a small grid of rays inward, starting
half a block outside the face (so a surface just outside the voxel -- where a peeled sliver
used to be -- is still seen), and keeps the first surface each ray meets within the voxel.
Hits through SIDE faces win over hits through the top or bottom: a facade is what the street
sees, a roof or floor only matters when nothing else is exposed. The voxel's colour is the
per-channel median of its hits; its material kinds (glass, door) are the share of hits on
faces of that kind, so a window counts as glass exactly when it is glass from outside.

Rays run along the grid axes, so each ray/triangle test is a 2D point-in-triangle test plus
one interpolated depth, and each ray only meets the triangles that the rasteriser found in
its voxel and the voxel in front of it.
"""
from __future__ import annotations

import numpy as np

from .raster import rasterize_surface

__all__ = ["exposed_colors"]

_DIRS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))      # (axis, sign): +x -x +y -y +z -z
_CHUNK = 1 << 21


def _ragged(starts, counts):
    """Concatenated ranges [starts[i], starts[i] + counts[i])."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = np.repeat(starts - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    return np.arange(total, dtype=np.int64) + offs


def exposed_colors(solid, shell, origin, verts, faces, point_colors, face_labels=None, n_labels: int = 0,
                   rays: int = 3, outside: float = 0.5, depth: float = 1.0, ground: bool = True):
    """Colours of the shell voxels as seen from outside.

    `verts` are in the scaled (block) frame; `origin` is the grid's min corner in that frame.
    `point_colors(tri_ids, bary)` gives (N, 3) colours 0..255 (see `scene.scene_color_fns`).
    Returns dict(occ (N,3) shell voxel indices, rgb (N,3) float, seen (N,) bool -- False where
    no ray met a surface, label_frac (N, n_labels), hits (N,) int).
    """
    solid = np.asarray(solid, dtype=bool)
    X, Y, Z = solid.shape
    occ = np.argwhere(shell)
    n = len(occ)
    out = {"occ": occ, "rgb": np.zeros((n, 3)), "seen": np.zeros(n, dtype=bool),
           "label_frac": np.zeros((n, max(n_labels, 1))), "hits": np.zeros(n, dtype=np.int64)}
    if n == 0 or len(faces) == 0:
        return out

    v = np.asarray(verts, dtype=np.float64) - np.asarray(origin, dtype=np.float64)   # grid coords
    f = np.asarray(faces, dtype=np.int64)
    pairs = []
    rasterize_surface(v, f, origin=np.zeros(3, dtype=np.int64), shape=solid.shape, pairs=pairs)
    if not pairs:
        return out
    cells = np.concatenate([p[0] for p in pairs])
    tris = np.concatenate([p[1] for p in pairs])
    order = np.argsort(cells, kind="stable")
    cells, tris = cells[order], tris[order]
    ucells, starts, counts = np.unique(cells, return_index=True, return_counts=True)

    def tris_of(flat):
        """(start, count) of each flat cell's triangle list; count 0 where it has none."""
        pos = np.searchsorted(ucells, flat)
        pos_c = np.minimum(pos, len(ucells) - 1)
        found = (pos < len(ucells)) & (ucells[pos_c] == flat)
        return np.where(found, starts[pos_c], 0), np.where(found, counts[pos_c], 0)

    air = np.pad(~solid, 1, constant_values=True)
    if ground:
        air[:, 0, :] = False                              # below the grid is ground, not air
    T = v[f]                                              # (F, 3, 3)
    sub = (np.arange(rays) + 0.5) / rays
    uu, ww = np.meshgrid(sub, sub, indexing="ij")
    uu, ww = uu.ravel(), ww.ravel()                       # (R,)
    R = len(uu)

    # Per voxel, per direction: hit colours accumulated as lists of (voxel row, rgb, label).
    side_rows, side_rgb, side_lab = [], [], []
    cap_rows, cap_rgb, cap_lab = [], [], []
    labels = None if face_labels is None else np.asarray(face_labels)
    flat_occ = (occ[:, 0] * Y + occ[:, 1]) * Z + occ[:, 2]
    for axis, sign in _DIRS:
        nb = occ.copy()
        nb[:, axis] += sign
        exposed = air[nb[:, 0] + 1, nb[:, 1] + 1, nb[:, 2] + 1]
        rows = np.flatnonzero(exposed)
        if len(rows) == 0:
            continue
        # Candidate triangles: those in the voxel and in the air voxel in front of it.
        s1, c1 = tris_of(flat_occ[rows])
        inside = np.all((nb[rows] >= 0) & (nb[rows] < np.array(solid.shape)), axis=1)
        nflat = (nb[rows, 0] * Y + nb[rows, 1]) * Z + nb[rows, 2]
        s2, c2 = tris_of(np.where(inside, nflat, -1))
        c2 = np.where(inside, c2, 0)
        face_of = np.concatenate([np.repeat(np.arange(len(rows)), c1), np.repeat(np.arange(len(rows)), c2)])
        cand = tris[np.concatenate([_ragged(s1, c1), _ragged(s2, c2)])]
        if len(cand) == 0:
            continue
        b_ax, c_ax = [a for a in range(3) if a != axis]
        plane = occ[rows, axis] + (1 if sign > 0 else 0) + sign * outside       # ray start along axis
        best_t = np.full((len(rows), R), np.inf)
        best_tri = np.full((len(rows), R), -1, dtype=np.int64)
        best_bary = np.zeros((len(rows), R, 3))
        step = max(1, _CHUNK // R)
        for c0 in range(0, len(cand), step):
            fo = face_of[c0:c0 + step]                    # (P,)
            tt = cand[c0:c0 + step]
            tri = T[tt]                                   # (P, 3, 3)
            # A triangle edge-on to the ray axis cannot be hit; drop it before dividing by
            # its (zero) projected area.
            proj = (tri[:, 1, b_ax] - tri[:, 0, b_ax]) * (tri[:, 2, c_ax] - tri[:, 0, c_ax]) - \
                   (tri[:, 2, b_ax] - tri[:, 0, b_ax]) * (tri[:, 1, c_ax] - tri[:, 0, c_ax])
            facing = np.abs(proj) > 1e-12
            if not facing.all():
                fo, tt, tri, proj = fo[facing], tt[facing], tri[facing], proj[facing]
            if len(fo) == 0:
                continue
            pb = occ[rows[fo], b_ax][:, None] + uu[None, :]          # (P, R) ray coords
            pc = occ[rows[fo], c_ax][:, None] + ww[None, :]
            x0, y0 = tri[:, 0, b_ax][:, None], tri[:, 0, c_ax][:, None]
            x1, y1 = tri[:, 1, b_ax][:, None], tri[:, 1, c_ax][:, None]
            x2, y2 = tri[:, 2, b_ax][:, None], tri[:, 2, c_ax][:, None]
            area = proj[:, None]
            w0 = ((x1 - pb) * (y2 - pc) - (x2 - pb) * (y1 - pc)) / area
            w1 = ((x2 - pb) * (y0 - pc) - (x0 - pb) * (y2 - pc)) / area
            w2 = 1.0 - w0 - w1
            ok = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            h = w0 * tri[:, 0, axis][:, None] + w1 * tri[:, 1, axis][:, None] + w2 * tri[:, 2, axis][:, None]
            t = (plane[fo][:, None] - h) * sign           # distance travelled inward
            ok &= (t >= 0) & (t <= outside + depth)
            t = np.where(ok, t, np.inf)
            # Keep the nearest hit per (face, ray) across this chunk and earlier ones.
            pi, ri = np.nonzero(np.isfinite(t))
            if len(pi) == 0:
                continue
            tv = t[pi, ri]
            fr = fo[pi]
            o = np.lexsort((tv, ri, fr))
            fr, ri, tv, pi = fr[o], ri[o], tv[o], pi[o]
            first = np.r_[True, (fr[1:] != fr[:-1]) | (ri[1:] != ri[:-1])]
            fr, ri, tv, pi = fr[first], ri[first], tv[first], pi[first]
            better = tv < best_t[fr, ri]
            fr, ri, tv, pi = fr[better], ri[better], tv[better], pi[better]
            best_t[fr, ri] = tv
            best_tri[fr, ri] = tt[pi]
            best_bary[fr, ri] = np.stack([w0[pi, ri], w1[pi, ri], w2[pi, ri]], axis=1)
        hit = best_tri >= 0
        if not hit.any():
            continue
        hf, hr = np.nonzero(hit)
        tri_ids = best_tri[hf, hr]
        rgb = np.asarray(point_colors(tri_ids, best_bary[hf, hr]), dtype=np.float64)
        lab = labels[tri_ids] if labels is not None else np.zeros(len(tri_ids), dtype=np.int64)
        target = (side_rows, side_rgb, side_lab) if axis != 1 else (cap_rows, cap_rgb, cap_lab)
        target[0].append(rows[hf])
        target[1].append(rgb)
        target[2].append(lab)

    def gather(rows_l, rgb_l, lab_l):
        if not rows_l:
            return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64)
        return np.concatenate(rows_l), np.concatenate(rgb_l), np.concatenate(lab_l)

    s_rows, s_rgb, s_lab = gather(side_rows, side_rgb, side_lab)
    k_rows, k_rgb, k_lab = gather(cap_rows, cap_rgb, cap_lab)
    has_side = np.zeros(n, dtype=bool)
    has_side[s_rows] = True
    keep_cap = ~has_side[k_rows]                          # top/bottom only where no side is seen
    rows_all = np.concatenate([s_rows, k_rows[keep_cap]])
    rgb_all = np.concatenate([s_rgb, k_rgb[keep_cap]])
    lab_all = np.concatenate([s_lab, k_lab[keep_cap]])
    if len(rows_all) == 0:
        return out
    order = np.argsort(rows_all, kind="stable")
    rows_all, rgb_all, lab_all = rows_all[order], rgb_all[order], lab_all[order]
    uniq, first, cnt = np.unique(rows_all, return_index=True, return_counts=True)
    # Per-voxel median per channel: split the sorted hits at voxel boundaries.
    med = np.zeros((len(uniq), 3))
    for ch in range(3):
        vals = rgb_all[:, ch]
        # sort each voxel's values: order by (row, value)
        o = np.lexsort((vals, rows_all))
        sv = vals[o]
        lo_i = first + (cnt - 1) // 2
        hi_i = first + cnt // 2
        med[:, ch] = 0.5 * (sv[lo_i] + sv[hi_i])
    out["rgb"][uniq] = med
    out["seen"][uniq] = True
    out["hits"][uniq] = cnt
    if n_labels:
        lf = np.zeros((len(uniq), n_labels))
        np.add.at(lf, (np.searchsorted(uniq, rows_all), np.clip(lab_all, 0, n_labels - 1)), 1.0)
        out["label_frac"][uniq] = lf / cnt[:, None]
    return out
