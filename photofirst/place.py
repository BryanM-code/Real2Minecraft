"""Placing a multi-photo reconstruction in the block grid without the human build (real
photos): which way is up, how the walls meet the grid, how big a block is, where the ground
is. (The evaluation and the training pairs place it on the target, `front_mv.refine_to_target`;
a real photo has none.)

    up      from the cameras: a hand-held camera is level (no roll), so every camera's right
            axis is horizontal and the up is perpendicular to all of them: 1.5 degrees off
            (median, 2.4 at worst) on the 25 evaluation builds' four photos. Levelling it on
            the reconstruction's own surfaces (`level_up`: walls perpendicular, roofs and
            ground along) is worse there (p90 7.3, worst 22: the surfaces are noisy and
            slopes pull it), so it is used only for a single photo, whose own up is otherwise
            just the camera's
    walls   the walls' horizontal directions modulo 90 degrees (their circular mean) turned onto
            the grid's x and z, by the smallest turn (so a compass-hinted north stays north:
            -z, as in Minecraft); walls along the grid are straight, not stepped
    scale   MapAnything's points are in metres: `metres_per_block` (1, Minecraft's usual
            scale), or the build's height in blocks (`height_blocks`, the app's choice)
    shift   the build centred, `margin` blocks of air round it (the seen air the completion
            model reads) and `below` under its lowest points: the photos rarely show a build's
            base (a ground plate, a foundation, what the cutout drops), the human builds have
            one, and the model can only add it with room below (and it reads absolute height);
            at most `max_blocks` a side

Returns the similarity `front_mv.to_grid` takes (grid = s R world + t), the grid's dims and
what was found.
"""
from __future__ import annotations

import numpy as np

from .front import map_normals


def fused_points(views: list, masks=None, every: int = 2) -> tuple:
    """(points (N, 3), unit normals (N, 3) facing their camera) of every photo's building
    pixels, in the reconstruction's world frame."""
    pts, nrm = [], []
    for k, v in enumerate(views):
        m = v["mask"] if masks is None else (np.asarray(masks[k], bool) & v["mask"])
        p = np.asarray(v["pts"], np.float64)
        n_map, ok = map_normals(p, m)
        sel = np.zeros(m.shape, bool)
        sel[1:-1, 1:-1] = ok
        sub = np.zeros(m.shape, bool)
        sub[::every, ::every] = True
        sel &= sub
        n = np.zeros_like(p)
        n[1:-1, 1:-1] = n_map
        cam = np.asarray(v["cam2world"])[:3, 3]
        P, N = p[sel], n[sel]
        N = np.where(((N * (cam - P)).sum(1) < 0)[:, None], -N, N)
        pts.append(P)
        nrm.append(N)
    return np.concatenate(pts), np.concatenate(nrm)


def up_from_cameras(views: list) -> np.ndarray:
    """The up direction the cameras imply: perpendicular to every camera's right axis (level
    cameras), on the side the cameras' own up axes point to; one photo: its own up."""
    rights = np.stack([np.asarray(v["cam2world"])[:3, 0] for v in views])
    cam_up = -np.mean([np.asarray(v["cam2world"])[:3, 1] for v in views], axis=0)
    if len(views) == 1:
        up = cam_up
    else:
        up = np.linalg.svd(rights)[2][-1]
        if up @ cam_up < 0:
            up = -up
    return up / np.linalg.norm(up)


def _basis(up: np.ndarray) -> tuple:
    e1 = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def level_up(normals: np.ndarray, up0: np.ndarray, max_tilt: float = 15.0) -> np.ndarray:
    """The up within `max_tilt` degrees of `up0` that most normals are perpendicular to
    (walls) or along (roofs, ground): `front.frame_from_normals`' smooth score, searched on a
    1-degree grid, then 0.2 degrees round the best."""
    n = normals[np.isfinite(normals).all(1)]
    n = n[:: max(1, len(n) // 30000)]
    up0 = np.asarray(up0, float) / np.linalg.norm(up0)
    e1, e2 = _basis(up0)

    def score(u):
        d = np.abs(n @ u)
        return np.mean(np.exp(-0.5 * (d / 0.05) ** 2)) + np.mean(np.exp(-0.5 * ((1.0 - d) / 0.05) ** 2))

    def search(centre, span, step):
        best = None
        for a in np.radians(np.arange(-span, span + 1e-9, step)):
            for b in np.radians(np.arange(-span, span + 1e-9, step)):
                u = centre[2] + np.tan(a) * centre[0] + np.tan(b) * centre[1]
                u /= np.linalg.norm(u)
                s = score(u)
                if best is None or s > best[0]:
                    best = (s, u)
        return best[1]
    u = search((e1, e2, up0), max_tilt, 1.0)
    f1, f2 = _basis(u)
    return search((f1, f2, u), 1.0, 0.2)


def wall_axes(normals: np.ndarray, up: np.ndarray, reference=(1.0, 0.0, 0.0)) -> tuple:
    """(x axis, z axis): the walls' horizontal directions modulo 90 degrees (circular mean of
    4 theta over the near-horizontal normals), the quarter turn nearest `reference`'s
    horizontal part; and the share of normals that are walls."""
    walls = normals[np.abs(normals @ up) < 0.2]
    e1, e2 = _basis(up)
    h = walls - (walls @ up)[:, None] * up
    h /= np.maximum(np.linalg.norm(h, axis=1), 1e-9)[:, None]
    theta = np.arctan2(h @ e2, h @ e1)
    t0 = np.arctan2(np.sin(4 * theta).sum(), np.cos(4 * theta).sum()) / 4.0
    ref = np.asarray(reference, float)
    ref = ref - (ref @ up) * up
    ref = ref / max(np.linalg.norm(ref), 1e-9)
    cands = [np.cos(t0 + k * np.pi / 2) * e1 + np.sin(t0 + k * np.pi / 2) * e2 for k in range(4)]
    x_axis = max(cands, key=lambda c: float(c @ ref))
    return x_axis, np.cross(x_axis, up), float(len(walls) / max(len(normals), 1))


def ground_from_cameras(views: list, up: np.ndarray, eye_units: float, lowest: float) -> tuple:
    """(the ground's height along `up`, how it was found): hand-held photos are taken at eye
    height, so the ground lies `eye_units` below the cameras (the lowest of them: a photo from
    a window or a slope across the street is higher, hardly ever lower). It never lies above
    the lowest visible point (`lowest`): a photographer below the building's base, or a raised
    spot for every photo, then gives way to the points."""
    h = float(min(np.asarray(v["cam2world"])[:3, 3] @ up for v in views)) - float(eye_units)
    return (h, "cameras") if h <= lowest else (lowest, "lowest_point")


BACK_DEPTH = 1.0      # the unseen back: a footprint at least this deep, in widths, away from the cameras


def extend_back(lo, hi, cams) -> list:
    """Make room for the back the photos do not show: along each horizontal grid axis where
    every camera stands on one side of the visible box, the box is extended away from them
    until it is `BACK_DEPTH` times its wider side deep (in place). Returns the axes extended."""
    w = float(max(hi[0] - lo[0], hi[2] - lo[2]))
    out = []
    for a in (0, 2):
        if hi[a] - lo[a] >= BACK_DEPTH * w:
            continue
        side = np.sign(cams[:, a] - (lo[a] + hi[a]) / 2.0)
        if (side < 0).all():
            hi[a] = lo[a] + BACK_DEPTH * w
        elif (side > 0).all():
            lo[a] = hi[a] - BACK_DEPTH * w
        else:
            continue
        out.append("xyz"[a])
    return out


def free_placement(views: list, masks=None, metres_per_block: float = 1.0, height_blocks=None,
                   margin: int = 8, max_blocks: int = 192, max_tilt: float = 15.0, level=None,
                   below: int = 2, eye_units=None, back: bool = True) -> tuple:
    """(similarity {"s", "R", "t"}, dims (X, Y, Z), info) for `front_mv.to_grid`, from the
    reconstruction alone (module docstring). `level`: level the up on the surfaces (default:
    only for a single photo). `eye_units`: the photographer's eye height in the
    reconstruction's units; given, the ground is put `below` blocks above the grid's floor
    (`ground_from_cameras`), else the lowest visible point is: on street photo sets that is
    the better estimate (the true base lands at its median, 2.0; within a block for 56% of 80
    sets, against 42% from the lowest camera: MapAnything's camera heights are noisy, and
    another photo usually shows a base one photo hides). `margin` 8: a margin of 4 kept a
    quarter of sets under 76% of the build inside the grid (8: 91%). `back`: room for the back
    photos from one side do not show (`extend_back`)."""
    P, N = fused_points(views, masks)
    up_cam = up_from_cameras(views)
    if level is None:
        level = len(views) == 1
    up = level_up(N, up_cam, max_tilt) if level else up_cam
    x_axis, z_axis, wall_share = wall_axes(N, up)
    R = np.stack([x_axis, up, z_axis])                         # rows: grid axes in world coordinates
    q = P @ R.T
    lo, hi = np.percentile(q, 0.2, axis=0), np.percentile(q, 99.8, axis=0)   # as front.to_blocks
    ground_how = "lowest_point"
    if eye_units is not None:
        lo[1], ground_how = ground_from_cameras(views, up, eye_units, float(lo[1]))
    visible_height = float(hi[1] - lo[1])
    extended = extend_back(lo, hi, np.stack([np.asarray(v["cam2world"])[:3, 3] for v in views]) @ R.T) if back else []
    if height_blocks is not None:
        s = float(height_blocks) / max(float(hi[1] - lo[1]), 1e-9)
    else:
        s = 1.0 / float(metres_per_block)
    span = (hi - lo) * s
    capped = bool(span.max() + 2 * margin > max_blocks)
    if capped:                                                 # keep it within the model's reach
        s *= (max_blocks - 2 * margin) / span.max()
        span = (hi - lo) * s
    dims = np.ceil(span + np.array([2 * margin, margin + 1 + below, 2 * margin])).astype(int)
    # the low corner of the points' box (its floor: the ground) at (margin, below, margin)
    t = -s * lo + np.array([margin, float(below), margin])
    tilt = float(np.degrees(np.arccos(np.clip(up @ up_cam, -1, 1))))
    info = {"up_world": up.tolist(), "up_from_cameras": up_cam.tolist(), "levelling_deg": tilt,
            "wall_share": wall_share, "blocks_per_unit": s, "height_units": visible_height,
            "dims": dims.tolist(), "capped_to_max_blocks": capped, "ground": ground_how,
            "back_extended": extended}
    return {"s": s, "R": R, "t": t}, tuple(int(d) for d in dims), info
