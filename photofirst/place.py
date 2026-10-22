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


def free_placement(views: list, masks=None, metres_per_block: float = 1.0, height_blocks=None,
                   margin: int = 4, max_blocks: int = 192, max_tilt: float = 15.0, level=None,
                   below: int = 2) -> tuple:
    """(similarity {"s", "R", "t"}, dims (X, Y, Z), info) for `front_mv.to_grid`, from the
    reconstruction alone (module docstring). `level`: level the up on the surfaces (default:
    only for a single photo)."""
    P, N = fused_points(views, masks)
    up_cam = up_from_cameras(views)
    if level is None:
        level = len(views) == 1
    up = level_up(N, up_cam, max_tilt) if level else up_cam
    x_axis, z_axis, wall_share = wall_axes(N, up)
    R = np.stack([x_axis, up, z_axis])                         # rows: grid axes in world coordinates
    q = P @ R.T
    lo, hi = np.percentile(q, 0.2, axis=0), np.percentile(q, 99.8, axis=0)   # as front.to_blocks
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
    # the low corner of the points' box at (margin, below, margin)
    t = -s * lo + np.array([margin, float(below), margin])
    tilt = float(np.degrees(np.arccos(np.clip(up @ up_cam, -1, 1))))
    info = {"up_world": up.tolist(), "up_from_cameras": up_cam.tolist(), "levelling_deg": tilt,
            "wall_share": wall_share, "blocks_per_unit": s, "height_units": float(hi[1] - lo[1]),
            "dims": dims.tolist(), "capped_to_max_blocks": capped}
    return {"s": s, "R": R, "t": t}, tuple(int(d) for d in dims), info
