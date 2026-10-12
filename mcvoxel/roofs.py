"""Deterministic Minecraft roof construction from connected source-mesh planes.

Upward mesh hits supply pitch, direction, height and source material. Connected planes
choose one lattice: 1:1 and 1:2 stairs with full backing, or 2:1 and 3:1 slab/full
courses. A bounded rational fit handles long noncanonical roofs. Direction changes
form explicit ridge/hip/valley seams; narrow flat details retain geometric thickness.
Only mesh-supported caps and a shallow connecting band are rebuilt. Source openings,
glass, eaves and unsampled vertical parapets remain intact. Materials are coherent
within a plane and source material boundary. Neighbour-dependent states are derived
later, after the full grid is assembled, by blockstates.finish_states.

Grids are (X, Y, Z), Y up, +X east, +Z south.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import block_shapes as bs
from .scene import KIND_GLASS

__all__ =["RoofParams", "RoofPlan", "top_surface", "plan_roofs", "apply_roofs", "builder_roofs",
           "harmonize_roof_materials", "SLOPE_CLASSES"]

SLOPE_CLASSES = ("none", "flat", "low", "pitched", "steep")
_NONE, _FLAT, _LOW, _PITCHED, _STEEP = range(5)
_STAIR_OF = {"east": bs.SHAPE_STAIR_E, "west": bs.SHAPE_STAIR_W,
             "north": bs.SHAPE_STAIR_N, "south": bs.SHAPE_STAIR_S}
_FACINGS = ("east", "west", "north", "south")
_CHUNK = 1 << 21


@dataclass
class RoofParams:
    coherent: bool = False          # opt-in architecture-v1 surface evidence; legacy remains replayable
    sub: int = 4                    # sub-columns per block edge for the top-surface raster
    flat_max: float = 0.25          # rise per block below which a cap is flat
    low_max: float = 0.75           # ... low pitch
    pitched_max: float = 1.5        # ... pitched; above is steep
    min_samples: int = 4            # sub-columns (of sub*sub) that must hit the mesh
    glass_max: float = 0.5          # caps with more glass than this are left geometric
    max_shift: int = 1              # legacy API field; region fit now bounds displacement
    # The whole-block step of a slab roof: "full" (bottom slab, full block, ...) or "slab_top"
    # (bottom slab, top slab, ... -- a half-block-thick roof). Builders use both: low-pitch
    # caps on the 66 real builds are 29% full / 17% top slabs; on the 21 of them outside the
    # evaluation set, 11% / 38%.
    low_upper: str = "full"


@dataclass
class RoofPlan:
    """Per-column decisions, (X, Z) arrays; -1 / 0 where a column is not rewritten."""
    cls: np.ndarray                 # int8 SLOPE_CLASSES index
    facing: np.ndarray              # int8 index into ("east","west","north","south"), -1 none
    level: np.ndarray               # int32 new cap y, -1 where unchanged
    shape: np.ndarray               # int8 block_shapes code of the new cap, -1 where unchanged
    region: np.ndarray              # int32 region id (one phase each), 0 = none
    group: np.ndarray               # int32 plane/material component id, 0 = none
    info: dict = field(default_factory=dict)
    support_low: np.ndarray | None = None  # shallow roof backing retained in exported shell
    layers: list = field(default_factory=list)  # additional exposed lower roof plans (architecture-v1)


def top_surface(verts, faces, shape_xz, sub: int = 4, face_labels=None, ceiling=None):
    """Highest mesh surface above each sub-column, and the slope of the face it lies on.

    `verts` are in grid coordinates (voxel (i, j, k) spans [i, i+1) x [j, j+1) x [k, k+1)).
    Sub-column (a, c) is centred at ((a + 0.5) / sub, (c + 0.5) / sub) in X and Z.
    Returns (height (X*sub, Z*sub) float32, -inf where nothing is above; grad (X*sub,
    Z*sub, 2) float32 = (dy/dx, dy/dz) of that face). Vertical faces project to lines and
    never win a sub-column, so the gradient is always a roof or floor gradient.
    """
    X, Z = int(shape_xz[0]), int(shape_xz[1])
    nx, nz = X * sub, Z * sub
    height = np.full(nx * nz, -np.inf, dtype=np.float64)
    gx_out = np.zeros(nx * nz, dtype=np.float32)
    gz_out = np.zeros(nx * nz, dtype=np.float32)
    label_out = np.full(nx * nz, -1, dtype=np.int32)

    def result():
        gradient = np.stack([gx_out, gz_out], axis=-1).reshape(nx, nz, 2)
        pair = (height.reshape(nx, nz).astype(np.float32), gradient)
        return pair if face_labels is None else pair + (label_out.reshape(nx, nz),)
    tri = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]      # (F,3,3)
    if len(tri) == 0:
        return result()
    p0, p1, p2 = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(p1 - p0, p2 - p0)
    ok = np.abs(n[:, 1]) > 1e-9 * (np.abs(n).sum(axis=1) + 1e-30)
    tri, n = tri[ok], n[ok]
    labels = None if face_labels is None else np.asarray(face_labels)[ok]
    if len(tri) == 0:
        return result()
    gx_t = -n[:, 0] / n[:, 1]
    gz_t = -n[:, 2] / n[:, 1]
    xs, zs = tri[:, :, 0] * sub - 0.5, tri[:, :, 2] * sub - 0.5          # sub-column index space
    lo_a = np.clip(np.ceil(xs.min(axis=1)).astype(np.int64), 0, nx)
    hi_a = np.clip(np.floor(xs.max(axis=1)).astype(np.int64), -1, nx - 1)
    lo_c = np.clip(np.ceil(zs.min(axis=1)).astype(np.int64), 0, nz)
    hi_c = np.clip(np.floor(zs.max(axis=1)).astype(np.int64), -1, nz - 1)
    da, dc = hi_a - lo_a + 1, hi_c - lo_c + 1
    has = (da > 0) & (dc > 0)
    keys = np.where(has, da * 100_003 + dc, -1)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        if sk[s] < 0:
            continue
        group = order[s:e]
        a_n, c_n = int(da[group[0]]), int(dc[group[0]])
        oa, oc = np.meshgrid(np.arange(a_n), np.arange(c_n), indexing="ij")
        oa, oc = oa.ravel(), oc.ravel()
        step = max(1, _CHUNK // len(oa))
        for c0 in range(0, len(group), step):
            idx = group[c0:c0 + step]
            A = lo_a[idx][:, None] + oa[None, :]                     # (T,M) sub-column indices
            C = lo_c[idx][:, None] + oc[None, :]
            px, pz = (A + 0.5) / sub, (C + 0.5) / sub                # world X, Z of the samples
            t = tri[idx]
            x0, z0 = t[:, 0, 0][:, None], t[:, 0, 2][:, None]
            x1, z1 = t[:, 1, 0][:, None], t[:, 1, 2][:, None]
            x2, z2 = t[:, 2, 0][:, None], t[:, 2, 2][:, None]
            area = (x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)
            w0 = ((x1 - px) * (z2 - pz) - (x2 - px) * (z1 - pz)) / area
            w1 = ((x2 - px) * (z0 - pz) - (x0 - px) * (z2 - pz)) / area
            w2 = 1.0 - w0 - w1
            eps = -1e-9
            inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps) & (A < nx) & (C < nz)
            if not inside.any():
                continue
            y = w0 * t[:, 0, 1][:, None] + w1 * t[:, 1, 1][:, None] + w2 * t[:, 2, 1][:, None]
            tt, mm = np.nonzero(inside)
            flat = A[tt, mm] * nz + C[tt, mm]
            yv = y[tt, mm]
            if ceiling is not None:
                keep = yv < np.asarray(ceiling).ravel()[flat]
                flat, yv, tt = flat[keep], yv[keep], tt[keep]
                if not len(flat):
                    continue
            o = np.lexsort((yv, flat))                               # by sample, then height
            flat, yv, tsel = flat[o], yv[o], idx[tt[o]]
            last = np.r_[flat[1:] != flat[:-1], True]                # highest hit per sample
            flat, yv, tsel = flat[last], yv[last], tsel[last]
            better = yv > height[flat]
            height[flat[better]] = yv[better]
            gx_out[flat[better]] = gx_t[tsel[better]]
            gz_out[flat[better]] = gz_t[tsel[better]]
            if labels is not None:
                label_out[flat[better]] = labels[tsel[better]]
    return result()


def plan_roofs(solid, kind_frac, height, grad, params: RoofParams | None = None,
               surface_material=None) -> RoofPlan:
    """Fit connected mesh-backed planes, then lay one canonical lattice per plane.

    Missing raster caps do not veto a mesh-covered roof column. Genuine openings have
    no roof samples and are never closed. Direction ties at hips/ridges get full caps,
    rather than an arbitrary cardinal stair. No whole-building morphology is used.
    """
    from scipy import ndimage
    from collections import deque, Counter

    p = params or RoofParams()
    X, Y, Z = solid.shape
    sub = p.sub
    has = solid.any(axis=1)
    top = np.where(has, Y - 1 - np.argmax(solid[:, ::-1, :], axis=1), -1)
    H = height.reshape(X, sub, Z, sub).transpose(0, 2, 1, 3).reshape(X, Z, -1)
    G = grad.reshape(X, sub, Z, sub, 2).transpose(0, 2, 1, 3, 4).reshape(X, Z, -1, 2)
    finite = np.isfinite(H)
    n = finite.sum(axis=2)
    # Estimate the surface at the BLOCK centre, including partially covered eave cells.
    # The old mean of clipped samples changed the phase at every edge and ridge.
    dx, dz = np.meshgrid((np.arange(sub) + .5) / sub - .5,
                         (np.arange(sub) + .5) / sub - .5, indexing="ij")
    centre = H - G[..., 0] * dx.ravel() - G[..., 1] * dz.ravel()
    h = np.sum(np.where(finite, centre, 0), axis=2) / np.maximum(n, 1)
    g = np.sum(np.where(finite[..., None], G, 0), axis=2) / np.maximum(n, 1)[..., None]
    slope_samples = np.linalg.norm(G, axis=-1)
    # A narrow parapet and its lower flat roof can share one coarse column.
    # Averaging those heights erases the parapet and merges it with the broad roof.
    # Keep the highest flat surface; narrow connected strips remain geometric below.
    all_flat = ((slope_samples < p.flat_max) | ~finite).all(axis=2) & (n > 0)
    h[all_flat] = np.max(np.where(finite, H, -np.inf), axis=2)[all_flat]
    # Mixed opposing slopes cancel at a ridge. Preserve that seam explicitly.
    sample_x = np.abs(G[...,0]) >= np.abs(G[...,1])
    sample_facing = np.where(sample_x,np.where(G[...,0]>0,0,1),np.where(G[...,1]>0,3,2))
    directional = np.stack([((sample_facing==f)&finite&(slope_samples>=p.flat_max)).sum(axis=2)
                            for f in range(4)],axis=-1)
    # Different magnitudes on the SAME slope are bevels, not a ridge.
    seam = np.sort(directional,axis=-1)[...,-2] >= max(2,p.min_samples)
    h[seam] = np.max(np.where(finite, H, -np.inf), axis=2)[seam]
    slope = np.linalg.norm(g, axis=-1)
    usable = (n >= p.min_samples) & (h >= 1) & (h < Y) & (slope <= 3.0)
    # A vertical parapet thinner than the ray spacing may have no upward hit.
    # The lower roof behind it is not permission to cut down that existing wall.
    highest_hit = np.max(np.where(finite, H, -np.inf), axis=2)
    usable &= ~has | (top <= np.ceil(highest_hit) + 1)
    # Respect site/component removal. Extend only into nearby mesh-backed roof gaps,
    # with continuity in height; never regrow an unrelated removed site object.
    anchored = usable & has & (np.abs(h - (top + .5)) <= 2.5)
    for _ in range(2):
        grow = anchored.copy()
        for axis in (0, 1):
            for step in (-1, 1):
                from .grid import _neighbour
                near = _neighbour(anchored, axis, step, False)
                nh = _neighbour(h, axis, step, -100.)
                grow |= usable & near & (np.abs(nh - h) <= 2.5)
        anchored = grow
    usable &= anchored
    glass = np.zeros((X, Z), bool)
    if kind_frac is not None and kind_frac.shape[-1] > KIND_GLASS:
        xs, zs = np.nonzero(has)
        glass[xs, zs] = kind_frac[xs, top[xs, zs], zs, KIND_GLASS] > p.glass_max
    usable &= ~glass
    below = np.zeros((X, Z), bool)
    xs, zs = np.nonzero(has & (top > 0))
    below[xs, zs] = solid[xs, top[xs, zs] - 1, zs]
    cls = np.zeros((X, Z), np.int8)
    cls[usable & (slope < p.flat_max) & below] = _FLAT
    cls[usable & (slope >= p.flat_max) & (slope < p.low_max)] = _LOW
    cls[usable & (slope >= p.low_max) & (slope < p.pitched_max)] = _PITCHED
    cls[usable & (slope >= p.pitched_max)] = _STEEP
    cls[usable & seam] = _FLAT
    ax_x = np.abs(g[..., 0]) >= np.abs(g[..., 1])
    facing = np.where(ax_x, np.where(g[..., 0] > 0, 0, 1), np.where(g[..., 1] > 0, 3, 2)).astype(np.int8)
    facing[cls <= _FLAT] = -1
    pitches = np.array([1/3, .5, 1., 2.])
    patterns = ("3:1", "2:1", "1:1", "1:2")
    pitch_id = np.argmin(np.abs(np.arctan(slope[..., None]) - np.arctan(pitches)), axis=2)
    region = np.zeros((X, Z), np.int32)
    level = np.full((X, Z), -1, np.int32)
    shape = np.full((X, Z), -1, np.int8)
    records = []
    # Connected components with a HEIGHT-continuity check: two parallel roofs above
    # one another are different planes even when their projections touch.
    for x0, z0 in np.argwhere(cls > 0):
        if region[x0, z0]:
            continue
        r = len(records) + 1
        c, f, pi = int(cls[x0, z0]), int(facing[x0, z0]), int(pitch_id[x0, z0])
        todo, cells = deque([(int(x0), int(z0))]), []
        region[x0, z0] = r
        while todo:
            x, z = todo.popleft()
            cells.append((x, z))
            for dx1, dz1 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = x + dx1, z + dz1
                if not (0 <= a < X and 0 <= b < Z) or region[a, b]:
                    continue
                if p.coherent and c >= _LOW and cls[a,b] >= _LOW:
                    # Smooth source-supported slopes can straddle a local pitch bin
                    # because of tiles/bevels. They still need one shared course phase.
                    if facing[a,b] != f:
                        continue
                elif cls[a, b] != c or facing[a, b] != f or (c >= _LOW and pitch_id[a, b] != pi):
                    continue
                predicted = .5 * ((g[x, z, 0] + g[a, b, 0]) * dx1 +
                                  (g[x, z, 1] + g[a, b, 1]) * dz1)
                if c == _FLAT:
                    predicted = 0.
                if abs(h[a, b] - h[x, z] - predicted) > (.15 if c == _FLAT else .6):
                    continue
                region[a, b] = r
                todo.append((a, b))
        xs, zs = np.array(cells).T
        hs = h[xs, zs]
        source_pitch = float(np.median(slope[xs, zs]))
        if p.coherent and c >= _LOW:
            # A region, not whichever corner happened to be visited first, chooses pitch.
            pi = int(np.argmin(np.abs(np.arctan(source_pitch)-np.arctan(pitches))))
            c = _LOW if source_pitch < p.low_max else (_PITCHED if source_pitch < p.pitched_max else _STEEP)
            cls[xs,zs] = c
        if c == _FLAT and not seam[xs,zs].any() and not ndimage.binary_erosion(region==r,structure=np.ones((3,3))).any():
            # Thin flat strips are parapets, ledges and ornament. Their geometric
            # slabs/full caps already express their thickness; don't flatten them
            # into the height of an adjacent broad roof.
            cls[xs,zs]=_NONE;facing[xs,zs]=-1;region[xs,zs]=-1
            records.append({"region":r,"columns":len(cells),"direction":None,"source_pitch":source_pitch,
                            "selected_pitch":0.,"pattern":"preserve-flat-detail","planned":False})
            continue
        if c >= _LOW:
            pitch = float(pitches[pi])
            u = (xs if f in (0, 1) else zs) * (1 if f in (0, 3) else -1)
            u = u - u.min()  # up-slope distance, invariant under rotation/mirroring
            intercept = float(np.median(hs - pitch * u))
            fit = intercept + pitch * u
            # Extremely noncanonical/long planes retain a source-following lattice.
            # This is a region decision, never a per-column rejection creating holes.
            fallback = bool(np.max(np.abs(fit - hs)) > 1.5)
            if fallback:
                from fractions import Fraction
                rational = Fraction(source_pitch).limit_denominator(8)
                pitch = float(rational)
                intercept = float(np.median(hs - pitch * u))
            if pi >= 2:
                b = np.floor(intercept - .75 + .5 + pitch * u + 1e-6).astype(int)
                s = np.full(len(xs), _STAIR_OF[_FACINGS[f]], np.int8)
            else:
                # 2:1: slab/full. 3:1: repeated slab/full half-steps over 3 columns.
                q0 = int(np.floor(2 * intercept + .5 + 1e-6))
                q = q0 + np.floor(2 * pitch * u + 1e-6).astype(int)
                b = (q - 1) // 2
                upper = bs.SHAPE_SLAB_TOP if p.low_upper == "slab_top" else bs.SHAPE_FULL
                s = np.where(q % 2, bs.SHAPE_SLAB_BOTTOM, upper).astype(np.int8)
            pattern = patterns[pi] if not fallback else f"{rational.denominator}:{rational.numerator} bounded-fit"
        else:
            pitch, intercept, pattern = 0., float(np.median(hs)), "seam" if seam[xs, zs].any() else "flat"
            b = np.full(len(xs), int(np.floor(intercept + .5 + 1e-6)) - 1)
            s = np.full(len(xs), bs.SHAPE_FULL, np.int8)
        b = np.clip(b, 0, Y - 1)
        level[xs, zs], shape[xs, zs] = b, s
        records.append({"region": r, "columns": len(cells), "direction": _FACINGS[f] if f >= 0 else None,
                        "source_pitch": round(source_pitch, 6), "selected_pitch": pitch,
                        "angle_degrees": round(float(np.degrees(np.arctan(pitch))), 2),
                        "pattern": pattern, "phase_height": round(intercept, 6),
                        "max_surface_error": round(float(np.max(np.abs(b + np.where(s == bs.SHAPE_SLAB_BOTTOM, .5,
                                                     np.where(np.isin(s,list(bs._FACING)), .75, 1.)) - hs))), 4)})
    sloped = cls >= _LOW
    pad = np.pad(sloped, 1)
    ridge = (cls == _FLAT) & ((pad[2:, 1:-1] & pad[:-2, 1:-1]) | (pad[1:-1, 2:] & pad[1:-1, :-2]))
    roof = sloped | ridge | (usable & seam)
    # Material groups follow source material boundaries if the mesh provides them.
    group = np.zeros((X, Z), np.int32)
    labels = np.zeros((X, Z), int) if surface_material is None else surface_material
    ng = 0
    for r in np.unique(region[roof]):
        for mat in np.unique(labels[roof & (region==r)]):
            lab, k = ndimage.label(roof & (region==r) & (labels == mat), structure=np.ones((3, 3), bool))
            group[lab > 0] = lab[lab > 0] + ng
            ng += k
    region[region<0]=0
    counts = {SLOPE_CLASSES[i]: int((cls == i).sum()) for i in range(1, 5)}
    info = {"columns": int(has.sum()), "planned": int((level >= 0).sum()), "by_class": counts,
            "regions": len(records), "roof_groups": ng, "glass_columns_skipped": int(glass.sum()),
            "shift_rejected": 0, "plane_decisions": records,
            "pattern_distribution": dict(Counter(row["pattern"] for row in records if row.get("planned",True))),
            "mesh_backed_missing_caps": int(((level >= 0) & (top < np.floor(h) - 1)).sum())}
    return RoofPlan(cls, facing, level, shape, region, group, info)


def apply_roofs(vm, plan: RoofPlan):
    """Copy of voxel model `vm` with every planned column's cap rewritten.

    A shallow band under raised caps is filled (colour/kind copied from the old top),
    cells above a lowered cap cleared, and a partial shape left directly under a new cap is
    made full so no gap opens beneath it. The shell is recomputed.
    """
    from .grid import shape_shell, shell_of

    solid = vm.solid.copy()
    shapes = vm.shapes.copy()
    rgb = None if vm.rgb is None else vm.rgb.copy()
    kind = None if vm.kind_frac is None else vm.kind_frac.copy()
    X, Y, Z = solid.shape
    top = np.where(solid.any(axis=1), Y - 1 - np.argmax(solid[:, ::-1, :], axis=1), -1)
    added = removed = supports = 0
    support_mask = np.zeros_like(solid)
    plan.support_low = plan.level.copy()
    repairs = {r: {"cap_added": 0, "cap_removed": 0, "support_added": 0} for r in np.unique(plan.region) if r > 0}
    for x, z in zip(*np.nonzero(plan.level >= 0)):
        b, t = int(plan.level[x, z]), int(top[x, z])
        count = repairs[int(plan.region[x,z])]
        if b > t:
            # A missing cap over a porch must not become a floor-to-roof column.
            lo = max(t + 1, b - 2)
            solid[x, lo:b + 1, z] = True
            shapes[x, lo:b + 1, z] = bs.SHAPE_FULL
            if rgb is not None:
                rgb[x, lo:b + 1, z] = rgb[x, max(t, 0), z]
            if kind is not None:
                kind[x, lo:b + 1, z] = kind[x, max(t, 0), z]
            added += b - lo + 1
            count["cap_added"] += b - lo + 1
        elif b < t:
            solid[x, b + 1:t + 1, z] = False
            shapes[x, b + 1:t + 1, z] = -1
            removed += t - b
            count["cap_removed"] += t - b
            if not vm.solid[x, b, z]:                     # new cap in what was air
                solid[x, b, z] = True
                added += 1
                count["cap_added"] += 1
                if rgb is not None:
                    rgb[x, b, z] = rgb[x, t, z]
                if kind is not None:
                    kind[x, b, z] = kind[x, t, z]
        shapes[x, b, z] = plan.shape[x, z]
        if b > 0 and solid[x, b - 1, z] and shapes[x, b - 1, z] not in (bs.SHAPE_FULL, -1):
            shapes[x, b - 1, z] = bs.SHAPE_FULL
    # Connect adjacent courses through a shallow roof band. A 1:2 stair needs two
    # riser blocks; shallow slab courses need one. This never extends to porch floors.
    for x, z in zip(*np.nonzero(plan.group > 0)):
        b = int(plan.level[x, z])
        low = b
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, c = x + dx, z + dz
            if 0 <= a < X and 0 <= c < Z and plan.group[a, c] > 0:
                nb = int(plan.level[a, c])
                if 0 <= b - nb <= 2:
                    low = min(low, nb)
        # Retain backing beneath same-level partial caps too, where already present.
        for y in range(max(0, low), b):
            support_mask[x,y,z] = True
            supports += int(not solid[x, y, z])
            repairs[int(plan.region[x,z])]["support_added"] += int(not solid[x,y,z])
            solid[x, y, z] = True
            shapes[x, y, z] = bs.SHAPE_FULL
            if rgb is not None:
                rgb[x, y, z] = rgb[x, b, z]
            if kind is not None:
                kind[x, y, z] = kind[x, b, z]
        plan.support_low[x,z] = low
    shell = shape_shell(solid, shapes) | support_mask
    exposed = _inherit_surface(vm, shell, rgb, kind)
    info = dict(vm.info)
    plan.info["plane_decisions"] = [{**row, "repairs": repairs.get(row["region"], {})}
                                    for row in plan.info.get("plane_decisions", [])]
    info["roofs"] = {**plan.info, "voxels_added": int(added + supports), "voxels_removed": int(removed),
                     "support_voxels_added": int(supports), "newly_exposed": int(exposed),
                     "partial_shape_backing": int((shell & ~shell_of(solid)).sum())}
    posts = None if vm.post_mask is None else vm.post_mask.copy()
    if posts is not None:
        posts[support_mask] = False
        xx,zz=np.nonzero(plan.level>=0)
        posts[xx,plan.level[xx,zz],zz]=False
    return replace(vm, solid=solid, shell=shell, surface=vm.surface & solid, shapes=shapes,
                   rgb=rgb, kind_frac=kind, info=info, post_mask=posts)


def _inherit_surface(vm, shell, rgb, kind) -> int:
    """Colour newly exposed shell voxels from the nearest voxel of the ORIGINAL shell.

    Colours and material kinds are sampled only on the geometric model's shell. Moving a cap
    exposes voxels that were interior before -- the cap itself when a column is lowered, the
    riser of every step beside it -- and those carry no colour (black) until they inherit
    one. Returns the number of voxels coloured.
    """
    from scipy import ndimage

    new = shell & ~vm.shell
    if rgb is None or not new.any() or not vm.shell.any():
        return 0
    pts = np.argwhere(new)
    lo = np.maximum(pts.min(axis=0) - 2, 0)
    hi = np.minimum(pts.max(axis=0) + 3, np.array(shell.shape))
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    ref = vm.shell[box]
    if not ref.any():                                 # no original surface nearby: use all of it
        box, lo, ref = tuple(slice(None) for _ in range(3)), np.zeros(3, int), vm.shell
    _, idx = ndimage.distance_transform_edt(~ref, return_indices=True)
    rel = pts - lo
    src = tuple(idx[a][rel[:, 0], rel[:, 1], rel[:, 2]] + lo[a] for a in range(3))
    rgb[pts[:, 0], pts[:, 1], pts[:, 2]] = vm.rgb[src]
    if kind is not None and vm.kind_frac is not None:
        kind[pts[:, 0], pts[:, 1], pts[:, 2]] = vm.kind_frac[src]
    return len(pts)


def builder_roofs(vm, verts, faces, params: RoofParams | None = None, scene=None,
                  ceiling=None, _include_lower=True):
    """(builder-roof copy of `vm`, plan). `verts` in the grid frame of `vm` (verts - origin)."""
    p = params or RoofParams()
    X, Y, Z = vm.solid.shape
    if scene is None:
        height, grad = top_surface(verts, faces, (X, Z), p.sub, ceiling=ceiling)
        materials = None
    else:
        height, grad, face = top_surface(verts, faces, (X, Z), p.sub, np.arange(len(faces)), ceiling=ceiling)
        mats = np.where(face >= 0, scene.face_mat[np.maximum(face, 0)], -1)
        materials = np.full((X, Z), -1, np.int32)
        for x, z in np.argwhere(np.isfinite(height).reshape(X, p.sub, Z, p.sub).any(axis=(1, 3))):
            ids = mats[x*p.sub:(x+1)*p.sub, z*p.sub:(z+1)*p.sub].ravel()
            ids = ids[ids >= 0]
            materials[x, z] = np.bincount(ids).argmax()
    raw_height = height
    evidence = None
    if p.coherent:
        from .architectural_roofs import surface_evidence, constrain_plan
        height, grad, evidence = surface_evidence(height, grad, (X, Z), p.sub)
    plan = plan_roofs(vm.solid, vm.kind_frac, height, grad, p, materials)
    if evidence is not None:
        plan = constrain_plan(plan, evidence, materials)
    out = apply_roofs(vm, plan)
    if scene is not None and out.rgb is not None:
        # Roof colour comes from upward mesh hits, not a wall/fascia visible at an eave.
        from .scene import scene_color_fns
        _, colours = scene_color_fns(scene)
        aa, cc = np.nonzero(face >= 0)
        fi = face[aa, cc]
        tri = np.asarray(verts)[np.asarray(faces)[fi]]
        points = np.c_[(aa + .5) / p.sub, raw_height[aa, cc], (cc + .5) / p.sub]
        import trimesh
        bary = trimesh.triangles.points_to_barycentric(tri, points)
        rgb = colours(fi, bary)
        sums = np.zeros((X, Z, 3), float)
        counts = np.zeros((X, Z), int)
        np.add.at(sums, (aa // p.sub, cc // p.sub), rgb)
        np.add.at(counts, (aa // p.sub, cc // p.sub), 1)
        for x, z in np.argwhere(plan.group > 0):
            b = int(plan.level[x, z])
            out.rgb[x, b, z] = np.clip(sums[x, z] / max(counts[x, z], 1), 0, 255)
    if p.coherent and _include_lower:
        from .architectural_roofs import lower_roofs
        out, lower = lower_roofs(out, verts, faces, p, scene)
        plan.layers = lower
        out.info['roofs']['lower_layers'] = [r.info for r in lower]
    return out, plan


def harmonize_roof_materials(target, plan: RoofPlan, palette=None, rgb=None):
    """Give every roof group's caps ONE stair-capable material, in place on a BlockTarget.

    With `rgb` (the voxel model's (X, Y, Z, 3) colours) the group's material is the
    stair-capable block nearest the MEDIAN colour of its caps: a speckled grey shingle roof
    whose quantised caps are split between white and grey blocks becomes grey, where the
    most frequent block (white, by a few caps) turned the whole roof white. Without `rgb` the
    dominant base material of the caps wins (ties: alphabetical), swapped for the nearest
    stair-capable block by colour if it has no stairs. Glass caps are left alone. `palette`
    (optional) limits the choice to those ids. Returns an info dict.
    """
    from collections import Counter

    from .block_palette import AUX_PALETTE, BLOCK_PALETTE, match_blocks_restricted
    from .vocab import encode_state

    X, Y, Z = target.shape.shape
    index = {(int(x), int(y), int(z)): i for i, (x, y, z) in enumerate(target.occ)}
    changed, groups = 0, 0
    for g in np.unique(plan.group[plan.group > 0]):
        cols = np.argwhere(plan.group == g)
        cells = [(int(x), int(plan.level[x, z]), int(z)) for x, z in cols]
        cells = [c for c in cells if c in index]
        bases = Counter()
        for c in cells:
            base = bs.base_block(target.states[index[c]])
            if base not in AUX_PALETTE:
                bases[base] += 1
        if not bases:
            continue
        pool = bs._STAIRS_CAPABLE
        if palette is not None:
            pool = [b for b in pool if b in palette] or pool
        if rgb is not None:
            caps = [c for c in cells if bs.base_block(target.states[index[c]]) not in AUX_PALETTE]
            med = np.median(np.array([rgb[c] for c in caps], dtype=np.float64), axis=0)
            top = match_blocks_restricted([tuple(med)], pool)[0][0]
        else:
            top = sorted(bases.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            if top not in bs.BLOCK_VARIANTS or bs.BLOCK_VARIANTS[top][1] is None:
                colour = BLOCK_PALETTE.get(top)
                if colour is None:
                    continue
                top = match_blocks_restricted([colour], pool)[0][0]
        groups += 1
        slab, stairs = bs.BLOCK_VARIANTS[top]
        caps = set(cells)
        if plan.support_low is not None:
            for x,z in cols:
                cells.extend((int(x),y,int(z)) for y in range(int(plan.support_low[x,z]),int(plan.level[x,z]))
                             if (int(x),y,int(z)) in index)
        for c in cells:
            i = index[c]
            state = target.states[i]
            if bs.base_block(state) in AUX_PALETTE:
                continue
            code = int(plan.shape[c[0], c[2]]) if c in caps else bs.SHAPE_FULL
            if code == bs.SHAPE_FULL:
                new = top
            elif code in (bs.SHAPE_SLAB_BOTTOM, bs.SHAPE_SLAB_TOP):
                new = "{}[type={}]".format(slab, "bottom" if code == bs.SHAPE_SLAB_BOTTOM else "top")
            else:
                new = "{}[facing={},half=bottom]".format(stairs, bs._FACING[code])
            if new != state:
                target.states[i] = new
                sh, mat = encode_state(new)
                target.shape[c] = sh
                target.material[c] = mat
                changed += 1
    lower_stats = [harmonize_roof_materials(target, layer, palette, rgb) for layer in plan.layers]
    return {"roof_groups": groups, "caps_restated": changed, **({'lower_layers': lower_stats} if lower_stats else {})}
