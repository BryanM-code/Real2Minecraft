"""B2 — massing regularization: blobby voxel solid -> crisp stacked rectilinear prisms.

Two steps:
  solid_from_mesh  - the yaw-aligned mesh -> raw solid occupancy (reusing the sculpture
                     pipeline's SAT rasterizer + exterior flood fill + cleanup), cropped
                     to its occupied bounding box so every downstream grid shares one
                     index frame.
  build_massing    - group Y slices into levels (footprint IoU), cover each level's
                     majority footprint with a greedy set of inscribed rectangles,
                     inflate rect edges back out over the raw footprint (squares off
                     eroded corners), and snap near-identical rect edges across levels
                     (kills 1-block jitter between stacked prisms).

The rect list IS the massing IR — exactly the representation a future learned model
should predict.  Everything is deterministic (fixed scan orders, first-argmax ties).
"""

import numpy as np
import trimesh
from scipy import ndimage

from voxelize import _voxelize_surface_sat, _clean_voxels
from building.errors import BuildingModeError


def solid_from_mesh(verts: np.ndarray, faces: np.ndarray, target_blocks: int):
    """Yaw-aligned mesh -> (raw_solid bool (W,H,L), scale_factor, origin, watertight).

    `origin` is the integer world min-corner of the CROPPED grid: voxel (i,j,k) has its
    center at (i,j,k) + 0.5 + origin in the scaled-mesh frame — the mapping the texture
    sampler needs.
    """
    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=False)
    # Same repair the sculpture pipeline applies before SAT (voxelize.py): without
    # fill_holes the exterior flood can leak into the interior of an open mesh.
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    longest = float(extents.max())
    scale_factor = (target_blocks / longest) if longest > 1e-6 else 1.0
    mesh.apply_scale(scale_factor)

    scaled_verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces_arr = np.asarray(mesh.faces, dtype=np.int64)
    surf, sat_origin = _voxelize_surface_sat(scaled_verts, faces_arr)

    padded = np.pad(surf, 2)
    labels, _ = ndimage.label(~padded)
    exterior = labels == labels[0, 0, 0]
    solid = (~exterior)[2:-2, 2:-2, 2:-2]

    cleaned = _clean_voxels(solid)
    if cleaned.any():
        solid = cleaned
    if not solid.any():
        raise BuildingModeError("massing", "voxelization produced no solid volume")

    occ = np.argwhere(solid)
    lo = occ.min(axis=0)
    hi = occ.max(axis=0)
    raw_solid = solid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    origin = np.asarray(sat_origin, dtype=np.int64) + lo
    return raw_solid, float(scale_factor), origin, bool(mesh.is_watertight)


# ── rectangle cover ───────────────────────────────────────────────────────────

def _largest_rectangle(mask: np.ndarray):
    """Largest all-true axis-aligned rectangle in a 2D bool mask.

    Classic histogram-stack method, O(W*L).  Returns (area, (x0, z0, x1, z1))
    half-open, or (0, None).  Ties resolve to the first found in scan order
    (row-major, stack pops) — deterministic.
    """
    W, L = mask.shape
    heights = np.zeros(L, dtype=np.int64)
    best_area, best = 0, None
    for x in range(W):
        heights = np.where(mask[x], heights + 1, 0)
        stack = []                                   # (start_col, height)
        for z in range(L + 1):
            h = heights[z] if z < L else 0
            start = z
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (z - s)
                if area > best_area:
                    best_area = area
                    best = (x - sh + 1, s, x + 1, z)
                start = s
            if not stack or h > 0:
                stack.append((start, h))
    return best_area, best


def _cover_rects(mask: np.ndarray, params: dict) -> list:
    """Greedy inscribed-rectangle cover of a footprint mask -> [(x0,z0,x1,z1), ...]."""
    mask_area = int(mask.sum())
    if mask_area == 0:
        return []
    min_area = max(params["massing_cover_min_area"],
                   params["massing_cover_min_area_frac"] * mask_area)
    covered = np.zeros_like(mask)
    rects = []
    while len(rects) < params["massing_cover_max_rects"]:
        area, rect = _largest_rectangle(mask & ~covered)
        if rect is None or area < min_area:
            break
        rects.append(rect)
        x0, z0, x1, z1 = rect
        covered[x0:x1, z0:z1] = True
        if (covered & mask).sum() / mask_area >= params["massing_cover_target"]:
            break
    # Anything still uncovered would become a HOLE in the shell wall; cover leftovers
    # with per-component bounding boxes (slight overfill beats an open slit).
    remaining = mask & ~covered
    if remaining.any():
        labels, n = ndimage.label(remaining)
        for i in range(1, n + 1):
            occ = np.argwhere(labels == i)
            lo, hi = occ.min(axis=0), occ.max(axis=0)
            rects.append((int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1))
    if not rects:                                    # tiny footprint: keep its bbox
        occ = np.argwhere(mask)
        lo, hi = occ.min(axis=0), occ.max(axis=0)
        rects = [(int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1)]
    return rects


def _inflate_rects(rects: list, raw_fp: np.ndarray, params: dict) -> list:
    """Grow rect edges outward over the RAW footprint to square off eroded corners."""
    W, L = raw_fp.shape
    min_fill = params["massing_inflate_min_fill"]
    out = []
    for x0, z0, x1, z1 in rects:
        for _ in range(params["massing_inflate_max"]):
            if x0 > 0 and raw_fp[x0 - 1, z0:z1].mean() >= min_fill:
                x0 -= 1
            if x1 < W and raw_fp[x1, z0:z1].mean() >= min_fill:
                x1 += 1
            if z0 > 0 and raw_fp[x0:x1, z0 - 1].mean() >= min_fill:
                z0 -= 1
            if z1 < L and raw_fp[x0:x1, z1].mean() >= min_fill:
                z1 += 1
        out.append((x0, z0, x1, z1))
    return out


def _snap_edges(levels: list, params: dict) -> list:
    """Cluster rect edge coordinates across ALL levels within snap_gap; snap each to the
    cluster median.  Stacked prisms whose edges differ by 1 block line up afterwards."""
    gap = params["massing_snap_gap"]

    def snap_map(values):
        vals = sorted(set(values))
        mapping = {}
        cluster = []
        for v in vals + [None]:
            if v is not None and (not cluster or v - cluster[-1] <= gap):
                cluster.append(v)
            else:
                target = int(round(float(np.median(cluster))))
                for c in cluster:
                    mapping[c] = target
                cluster = [v] if v is not None else []
        return mapping

    xs, zs = [], []
    for lv in levels:
        for x0, z0, x1, z1 in lv["rects"]:
            xs += [x0, x1]
            zs += [z0, z1]
    mx, mz = snap_map(xs), snap_map(zs)

    for lv in levels:
        snapped = []
        for x0, z0, x1, z1 in lv["rects"]:
            nx0, nx1 = mx[x0], mx[x1]
            nz0, nz1 = mz[z0], mz[z1]
            if nx1 <= nx0 or nz1 <= nz0:             # snap collapsed it: keep original
                snapped.append((x0, z0, x1, z1))
            else:
                snapped.append((nx0, nz0, nx1, nz1))
        lv["rects"] = snapped
    return levels


# ── slice grouping + assembly ────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def _fill_small_holes(fp: np.ndarray, max_size: int = 8) -> np.ndarray:
    """Fill enclosed footprint holes up to max_size cells (voxelization noise that
    would otherwise become air shafts through the build).  Real courtyards are
    bigger and survive."""
    inv = ~fp
    labels, n = ndimage.label(inv)
    if n == 0:
        return fp
    border = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    sizes = np.bincount(labels.ravel())
    out = fp.copy()
    for i in range(1, n + 1):
        if i not in border and sizes[i] <= max_size:
            out[labels == i] = True
    return out


def build_massing(raw_solid: np.ndarray, params: dict):
    """raw_solid -> (massing_ir, reg_solid).  See module docstring."""
    W, H, L = raw_solid.shape

    # 1. Group consecutive Y slices by footprint IoU against the group majority.
    groups = []                                       # {"y0","y1","stack"}
    for y in range(H):
        fp = raw_solid[:, y, :]
        if groups:
            g = groups[-1]
            n = g["y1"] - g["y0"]
            majority = g["stack"] * 2 >= n
            if _iou(fp, majority) >= params["massing_group_iou"]:
                g["stack"] = g["stack"] + fp
                g["y1"] = y + 1
                continue
        groups.append({"y0": y, "y1": y + 1, "stack": fp.astype(np.int32)})

    # 1b. Absorb 1-slice groups (except the topmost) into the more-similar neighbour.
    def majority_of(g):
        n = g["y1"] - g["y0"]
        return g["stack"] * 2 >= n

    merged = True
    while merged and len(groups) > 1:
        merged = False
        for i, g in enumerate(groups):
            if g["y1"] - g["y0"] > 1 or i == len(groups) - 1:
                continue
            fp = majority_of(g)
            prev_iou = _iou(fp, majority_of(groups[i - 1])) if i > 0 else -1.0
            next_iou = _iou(fp, majority_of(groups[i + 1]))
            j = i - 1 if prev_iou >= next_iou else i + 1
            tgt = groups[j]
            tgt["stack"] = tgt["stack"] + g["stack"]
            tgt["y0"] = min(tgt["y0"], g["y0"])
            tgt["y1"] = max(tgt["y1"], g["y1"])
            del groups[i]
            merged = True
            break

    # 2. Roof taper: THIN top groups that keep shrinking inside the group below.
    # (A tall setback tower shrinks too — the height cap keeps it body.)
    roles = ["body"] * len(groups)
    for i in range(len(groups) - 1, 0, -1):
        below = majority_of(groups[i - 1])
        here = majority_of(groups[i])
        thin = (groups[i]["y1"] - groups[i]["y0"]) <= params["massing_roof_max_step_h"]
        shrinking = here.sum() < params["massing_roof_area_ratio"] * below.sum()
        inside = not (here & ~ndimage.binary_dilation(below)).any()
        if thin and shrinking and inside:
            roles[i] = "roof_taper"
        else:
            break
    taper_ys = [groups[i] for i, r in enumerate(roles) if r == "roof_taper"]
    taper_h = sum(g["y1"] - g["y0"] for g in taper_ys)
    roof = {"type": "taper" if taper_h >= 2 else "flat",
            "y_start": int(taper_ys[0]["y0"]) if taper_ys else int(H)}

    # Re-split tall taper groups into <=2-slice steps: a hip/pyramid roof grouped into
    # one ring loses its slope; fine steps read as a proper stepped Minecraft roof.
    split_groups, split_roles = [], []
    for g, role in zip(groups, roles):
        h_g = g["y1"] - g["y0"]
        if role == "roof_taper" and h_g > 2:
            for y0 in range(g["y0"], g["y1"], 2):
                y1 = min(y0 + 2, g["y1"])
                stack = raw_solid[:, y0:y1, :].sum(axis=1).astype(np.int32)
                split_groups.append({"y0": y0, "y1": y1, "stack": stack})
                split_roles.append(role)
        else:
            split_groups.append(g)
            split_roles.append(role)
    groups, roles = split_groups, split_roles

    # 3. Rectangle cover + inflate per level, then snap edges across levels.
    levels = []
    for g, role in zip(groups, roles):
        fp = _fill_small_holes(majority_of(g))
        rects = _cover_rects(fp, params)
        rects = _inflate_rects(rects, fp, params)
        levels.append({"y0": int(g["y0"]), "y1": int(g["y1"]), "role": role,
                       "rects": [[int(v) for v in r] for r in rects],
                       "_raw_fp": fp})
    levels = _snap_edges(levels, params)

    # Snapping often makes adjacent levels' footprints identical — merge those so the
    # massing IR reads as a few clean prisms instead of stacked near-duplicates.
    compact = []
    for lv in levels:
        if (compact and compact[-1]["y1"] == lv["y0"]
                and sorted(map(tuple, compact[-1]["rects"]))
                == sorted(map(tuple, lv["rects"]))):
            compact[-1]["y1"] = lv["y1"]
            compact[-1]["_raw_fp"] = compact[-1]["_raw_fp"] | lv["_raw_fp"]
            if lv["role"] == "body":
                compact[-1]["role"] = "body"
        else:
            compact.append(lv)
    levels = compact

    # 4. Rebuild the regularized solid + per-level fidelity.
    reg_solid = np.zeros_like(raw_solid)
    ious, weights = [], []
    for lv in levels:
        fp = np.zeros((W, L), dtype=bool)
        for x0, z0, x1, z1 in lv["rects"]:
            fp[max(0, x0):min(W, x1), max(0, z0):min(L, z1)] = True
        reg_solid[:, lv["y0"]:lv["y1"], :] = fp[:, None, :]
        lv["iou_raw"] = _iou(fp, lv.pop("_raw_fp"))
        lv["area"] = int(fp.sum())
        ious.append(lv["iou_raw"])
        weights.append(lv["y1"] - lv["y0"])

    mean_iou = float(np.average(ious, weights=weights))
    if mean_iou < params["massing_min_mean_iou"]:
        raise BuildingModeError(
            "massing",
            "mesh does not read as a building (footprint fidelity {:.2f} < {:.2f}) — "
            "try Sculpture mode".format(mean_iou, params["massing_min_mean_iou"]))

    massing_ir = {
        "grid_dims": [int(W), int(H), int(L)],
        "levels": levels,
        "roof": roof,
        "mean_iou": mean_iou,
    }
    return massing_ir, reg_solid
