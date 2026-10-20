"""The photo-first front end: a photo -> the blocks it shows (seen solid, seen air, colours).

MoGe-2 (`Ruicheng/moge-2-vitl-normal`, Microsoft's monocular geometry model) gives a 3D point
per pixel in the camera frame; the building mask (the photo's alpha, or the app's cutout)
keeps the building's pixels. Then:

1. **Frame**: camera axes -> the build's axes (x, y up, z). Known for the evaluation renders
   (their camera, `visibility.view_axes`); `frame_from_normals` estimates it for a real photo
   from the point map's normals: up from the walls and flat surfaces (the camera's pitch,
   no roll), the yaw from the walls' directions (walls meet at right angles).
2. **Stretch** (orthographic renders only): MoGe reads depth from perspective, and on the
   orthographic evaluation renders flattens it (relief 0.76 of the truth, median over the 25
   builds). `estimate_stretch` restores some of it: the depth scale at which the point map's
   normals are most axis-aligned. Perspective photos need none (it lowers the scores there).
3. **Scale**: to the build's height in blocks (the user's choice in the app; the target's
   height in evaluation).
4. **Blocks**: points pushed half a block into the surface (along the predicted normal)
   mark seen solid where a block holds `min_points`; the blocks between a point and the
   camera are seen air; a block's colour is the mean of its pixels.

The result is in the front end's own frame (origin at the points' low corner, a quarter-turn
ambiguity left when the frame is estimated); `place` puts it into another frame.
"""
from __future__ import annotations

import numpy as np

MODEL = "Ruicheng/moge-2-vitl-normal"
MODELS = {2: MODEL, 3: "Ruicheng/moge-3-vitl"}     # MoGe-3 (2026-08) needs FlexGEMM on Triton
STRETCHES = np.round(np.arange(0.8, 2.61, 0.05), 3)
AXIS_DEG = 10.0


def load_moge(device="cuda", version: int = 2):
    if version == 3:
        from moge.model.v3 import MoGeModel
    else:
        from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(MODELS[version]).to(device).eval()


def infer(model, rgb: np.ndarray, fov_x=None, device="cuda") -> dict:
    """MoGe on an (H, W, 3) uint8 image -> numpy points (camera: x right, y down, z forward),
    depth, mask, intrinsics and, when the model gives them, normals."""
    import torch
    x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(x, fov_x=fov_x)
    out = {k: v.float().cpu().numpy() for k, v in out.items() if hasattr(v, "float")}
    if "normal" not in out:                              # facing the camera, for the edges only
        p = out["points"]
        out["normal"] = -p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-9)
    return out


def known_frame(right, down, near) -> np.ndarray:
    """Camera -> build rotation for a render whose camera axes are known (grid = R @ camera)."""
    return np.stack([np.asarray(right), np.asarray(down), -np.asarray(near)], axis=1)


def map_normals(points: np.ndarray, valid: np.ndarray) -> tuple:
    """Normals of a point map from its pixel neighbours: (normals (H-2, W-2, 3), valid)."""
    du = points[1:-1, 2:] - points[1:-1, :-2]
    dv = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    ok = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & (norm > 0)
    return n / np.maximum(norm, 1e-12)[..., None], ok


def axis_share(normals: np.ndarray, deg: float = AXIS_DEG) -> float:
    """Share of unit normals within `deg` of a coordinate axis."""
    if not len(normals):
        return 0.0
    return float(np.mean(np.abs(normals).max(-1) > np.cos(np.radians(deg))))


def stretch_points(points: np.ndarray, valid: np.ndarray, s: float) -> np.ndarray:
    """Depth relief scaled by `s` about the median depth, each point moved along its own pixel
    ray (so it still projects to its pixel)."""
    z = points[..., 2]
    ok = valid & np.isfinite(z) & (np.abs(z) > 1e-9)        # MoGe leaves pixels outside its mask infinite
    z0 = float(np.median(z[ok]))
    ratio = np.ones_like(z)
    ratio[ok] = (z0 + s * (z[ok] - z0)) / z[ok]
    return points * ratio[..., None]


def estimate_stretch(points: np.ndarray, valid: np.ndarray, R: np.ndarray, stretches=STRETCHES) -> tuple:
    """(stretch, share): the depth relief scale at which the point map's normals, turned into
    the build frame by `R`, are most often within AXIS_DEG of an axis."""
    best = (1.0, -1.0)
    for s in stretches:
        n, ok = map_normals(stretch_points(points, valid, s), valid)
        share = axis_share(n[ok][::7] @ R.T)
        if share > best[1]:
            best = (float(s), share)
    return best


def frame_from_normals(normals: np.ndarray, image_up=(0.0, -1.0, 0.0), roll: bool = False) -> np.ndarray:
    """Camera -> build rotation from surface normals (camera frame, unit) alone: the up
    direction is the one most normals are perpendicular to (walls) or along (roofs, ground),
    searched over the camera's pitch (and, with `roll`, its roll; photos are rarely rolled);
    the walls' directions then fix the yaw (modulo 90 degrees). The point map's own normals
    (`map_normals`) work better than MoGe's predicted ones. Two passes: the share of normals
    within hard thresholds finds the right region robustly but has a plateau a few degrees
    wide; a smooth score then centres it within 3 degrees. On the 25 perspective renders the
    up direction is off by 1.5 degrees (median; 3.9 at p90): hard alone 2.0 (5.0), smooth
    alone 1.5 (8.1), MoGe's predicted normals 4.0 (13.1)."""
    n = normals[np.isfinite(normals).all(-1)]
    n = n[:: max(1, len(n) // 20000)]
    a0 = np.asarray(image_up, float)

    def best_up(pitches, rolls, smooth):
        best = None
        for pitch in np.radians(pitches):
            for r in np.radians(rolls):
                # tilt the image's up about the camera x axis (pitch), then about z (roll)
                cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(r), np.sin(r)
                u = np.array([a0[0], a0[1] * cp - a0[2] * sp, a0[1] * sp + a0[2] * cp])
                u = np.array([u[0] * cr - u[1] * sr, u[0] * sr + u[1] * cr, u[2]])
                d = np.abs(n @ u)
                if smooth:
                    score = np.mean(np.exp(-0.5 * (d / 0.05) ** 2)) + np.mean(np.exp(-0.5 * ((1.0 - d) / 0.05) ** 2))
                else:
                    score = np.mean(d < 0.1) + np.mean(d > 0.97)
                if best is None or score > best[0]:
                    best = (score, u, np.degrees(pitch), np.degrees(r))
        return best
    _, _, p0, r0 = best_up(np.arange(-20, 60.01, 0.5), np.arange(-12, 12.01, 1.0) if roll else [0.0], smooth=False)
    _, up, _, _ = best_up(np.arange(p0 - 3, p0 + 3.01, 0.25),
                          np.arange(r0 - 2, r0 + 2.01, 0.5) if roll else [0.0], smooth=True)
    up = up / np.linalg.norm(up)
    walls = n[np.abs(n @ up) < 0.2]
    h = walls - (walls @ up)[:, None] * up
    h /= np.maximum(np.linalg.norm(h, axis=1), 1e-9)[:, None]
    e1 = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    theta = np.arctan2(h @ e2, h @ e1)
    t0 = np.arctan2(np.sin(4 * theta).sum(), np.cos(4 * theta).sum()) / 4.0
    x_axis = np.cos(t0) * e1 + np.sin(t0) * e2
    z_axis = np.cross(x_axis, up)
    return np.stack([x_axis, up, z_axis], axis=0)       # rows: build axes in camera coords -> grid = R @ cam


def snap_planes(p: np.ndarray, normals: np.ndarray, min_points: int = 40, reach: float = 1.0,
                facing: float = 0.9) -> tuple:
    """Walls, floors and flat roofs are planes on block faces. Points (block units) whose
    normal is within ~25 degrees of an axis vote for planes along it: the frame first gets the
    sub-block shift that puts those planes on whole blocks (the alignment later shifts by
    whole blocks only), then every such point within `reach` of a plane with `min_points`
    votes is moved onto it and its normal set to the axis. Depth noise across a facade (a
    block either way) goes; slopes and curves are left as they are.
    -> (points, normals, the frame shift applied to every point)."""
    p, normals = p.copy(), normals.copy()
    shift = np.zeros(3)
    for a in range(3):
        sel = np.abs(normals[:, a]) > facing
        if sel.sum() < min_points:
            continue
        ang = 2 * np.pi * (p[sel, a] % 1.0)
        phase = (np.arctan2(np.sin(ang).sum(), np.cos(ang).sum()) / (2 * np.pi)) % 1.0
        p[:, a] -= phase                                     # planes now near whole numbers
        shift[a] = -phase
        k = np.round(p[sel, a]).astype(np.int64)
        lo = int(k.min())
        votes = np.bincount(k - lo)
        planes = np.nonzero(votes >= min_points)[0] + lo
        if not len(planes):
            continue
        x = p[sel, a]
        j = np.clip(np.searchsorted(planes, x), 1, len(planes) - 1) if len(planes) > 1 else np.zeros(len(x), np.int64)
        cand = np.stack([planes[np.maximum(j - 1, 0)], planes[np.minimum(j, len(planes) - 1)]], 1)
        nearest = cand[np.arange(len(x)), np.argmin(np.abs(cand - x[:, None]), 1)]
        close = np.abs(nearest - x) <= reach
        idx = np.nonzero(sel)[0][close]
        p[idx, a] = nearest[close]
        n_axis = np.zeros((len(idx), 3))
        n_axis[np.arange(len(idx)), a] = np.sign(normals[idx, a])
        normals[idx] = n_axis
    return p, normals, shift


def _march(start: np.ndarray, d: np.ndarray, t: np.ndarray, dims, out: np.ndarray, chunk: int = 4096):
    """Mark every block of `out` that the rays start + t * d pass through."""
    for s in range(0, len(start), chunk):
        q = np.floor(start[s:s + chunk, None, :] + t[None, :, None] * d[s:s + chunk, None, :]).astype(np.int64)
        inside = ((q >= 0) & (q < dims)).all(-1)
        a = q[inside]
        out[a[:, 0], a[:, 1], a[:, 2]] = True


def to_blocks(points: np.ndarray, normals: np.ndarray, colours: np.ndarray, camera: np.ndarray,
              height_blocks: float, min_points: int = 2, air_step: float = 0.5, air_every: int = 3,
              background=None, margin: int = 6, max_cells: int = 40_000_000, snap: bool = False) -> dict:
    """Points in the build frame (N, 3; any unit), their outward normals and colours, the
    camera centre in the same frame -> {"solid", "air", "rgb"} grids in blocks, scaled so the
    points span `height_blocks` vertically, and the transform {"scale", "origin"}.

    Seen air is the blocks between each point and the camera, and every block along
    `background` (unit directions from the camera through the photo's pixels that show no
    building): the empty space around the building, which the completion model reads as
    such. The grid keeps `margin` blocks around the points (above and to the sides) for it.
    Outlying points (a photo with its background) cannot blow the grid up past `max_cells`."""
    lo, hi = np.percentile(points, 0.2, axis=0), np.percentile(points, 99.8, axis=0)
    scale = float(height_blocks) / max(float(hi[1] - lo[1]), 1e-9)
    h_lo, h_hi = np.percentile(points, 1.0, axis=0), np.percentile(points, 99.0, axis=0)
    lo[[0, 2]], hi[[0, 2]] = h_lo[[0, 2]], h_hi[[0, 2]]
    span = (hi - lo) * scale
    while np.prod(span + 2 * margin + 2) > max_cells:            # a background photo: keep the middle
        mid = (lo + hi) / 2.0
        lo[[0, 2]] = mid[[0, 2]] - (mid[[0, 2]] - lo[[0, 2]]) * 0.8
        hi[[0, 2]] = mid[[0, 2]] + (hi[[0, 2]] - mid[[0, 2]]) * 0.8
        span = (hi - lo) * scale
    pad = np.array([margin, 1, margin], float)
    origin = lo - pad / scale
    p = (points - origin) * scale
    cam = (camera - origin) * scale
    dims = np.ceil(span + pad + np.array([margin, margin, margin])).astype(np.int64) + 1
    if snap:
        p, normals, shift = snap_planes(p, normals)
        cam = cam + shift                                    # the camera moves with the frame
    inward = p - 0.5 * normals
    idx = np.floor(inward).astype(np.int64)
    keep = ((idx >= 0) & (idx < dims)).all(1)
    idx, col = idx[keep], colours[keep].astype(np.float64)
    flat = np.ravel_multi_index(idx.T, dims)
    count = np.bincount(flat, minlength=int(np.prod(dims)))
    solid = (count >= min_points).reshape(dims)
    rgb = np.zeros((int(np.prod(dims)), 3))
    for c in range(3):
        rgb[:, c] = np.bincount(flat, weights=col[:, c], minlength=int(np.prod(dims)))
    rgb = (rgb / np.maximum(count, 1)[:, None]).reshape(tuple(dims) + (3,))
    air = np.zeros(tuple(dims), bool)
    start = p[::air_every]
    d = cam[None, :] - start if cam.ndim == 1 else cam[::air_every] - start
    d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    _march(start, d, np.arange(0.75, float(np.linalg.norm(dims)) + 1.0, air_step), dims, air)
    if background is not None and len(background):
        corners = np.array([[x, y, z] for x in (0, dims[0]) for y in (0, dims[1]) for z in (0, dims[2])], float)
        dist = np.linalg.norm(corners - cam, axis=1)
        t0 = max(float(dist.min()) - float(np.linalg.norm(dims)), 0.0)
        bg = np.asarray(background, float)
        _march(np.broadcast_to(cam, bg.shape), bg, np.arange(t0, float(dist.max()) + 1.0, air_step), dims, air)
    air &= ~solid
    rgb = np.where(solid[..., None], np.clip(rgb, 0, 255), 0).astype(np.uint8)
    return {"solid": solid, "air": air, "rgb": rgb, "scale": scale, "origin": origin.tolist()}


def background_rays(intrinsics: np.ndarray, where: np.ndarray, every: int = 4) -> np.ndarray:
    """Unit directions (camera frame) through every `every`-th pixel where `where` is True,
    from MoGe's normalised intrinsics (focal lengths and centre in image widths / heights)."""
    h, w = where.shape
    ii, jj = np.nonzero(where[::every, ::every])
    u, v = (jj * every + 0.5) / w, (ii * every + 0.5) / h
    K = np.asarray(intrinsics, float)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def place(grid: np.ndarray, turns: int, shift, shape, fill=0) -> np.ndarray:
    """`grid` (X, Y, Z[, C]) turned by `turns` quarter turns about +Y (as
    `mcvoxel.blockstates.rotate_y`) and shifted into an array of `shape` (X, Y, Z)."""
    from training.score import _place
    g = np.rot90(grid, turns, axes=(0, 2)) if turns % 4 else grid
    if g.ndim == 3:
        return _place(np.ascontiguousarray(g), shift, tuple(shape), fill)
    return np.stack([_place(np.ascontiguousarray(g[..., c]), shift, tuple(shape), fill)
                     for c in range(g.shape[-1])], axis=-1)


def photo_to_blocks(model, rgba: np.ndarray, height_blocks: float, R=None, fov_x=None,
                    stretch: bool = False, min_points: int = 2, snap: bool = False) -> dict:
    """The whole front end on an RGBA photo (alpha = the building mask): MoGe, frame (`R`
    given, else estimated), stretch, scale, blocks."""
    alpha = rgba[..., 3] > 127
    rgb = np.where(alpha[..., None], rgba[..., :3], 255).astype(np.uint8)
    return blocks_from_moge(infer(model, rgb, fov_x=fov_x), rgba, height_blocks, R, stretch, min_points, snap)


def blocks_from_moge(out: dict, rgba: np.ndarray, height_blocks: float, R=None, stretch: bool = False,
                     min_points: int = 2, snap: bool = False) -> dict:
    """Everything after MoGe (`infer`'s output for this photo): frame, stretch, scale, blocks."""
    alpha = rgba[..., 3] > 127
    valid = out["mask"].astype(bool) & alpha & np.isfinite(out["points"]).all(-1)
    pts = np.where(valid[..., None], out["points"], 0.0)       # MoGe leaves pixels outside its mask infinite
    if R is None:
        n_map, ok = map_normals(pts, valid)
        R = frame_from_normals(n_map[ok])
    s, share = estimate_stretch(pts, valid, R) if stretch else (1.0, None)
    pts = stretch_points(pts, valid, s)
    n_map, ok = map_normals(pts, valid)                      # normals of the stretched surface
    n_cam = np.zeros_like(pts)
    n_cam[1:-1, 1:-1] = np.where(ok[..., None], n_map, 0.0)
    fallback = np.linalg.norm(n_cam, axis=-1) < 0.5          # edges: MoGe's own normal
    n_cam[fallback] = out["normal"][fallback]
    P = pts[valid] @ R.T
    N = n_cam[valid] @ R.T
    cam = np.zeros(3) @ R.T                                  # the camera centre is the camera frame's origin
    towards = cam[None, :] - P
    N = np.where(((N * towards).sum(1) < 0)[:, None], -N, N)  # outward: towards the camera
    blocks = to_blocks(P, N, rgba[..., :3][valid], cam, height_blocks, min_points,
                       background=background_rays(out["intrinsics"], ~alpha) @ R.T, snap=snap)
    blocks.update({"stretch": s, "axis_share": share, "R": R.tolist(),
                   "fov_deg": float(2 * np.degrees(np.arctan(0.5 / out["intrinsics"][0, 0])))})
    return blocks
