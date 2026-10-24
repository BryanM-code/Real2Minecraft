"""The multi-photo front end: 2-4 photos of a build -> the blocks they show, in one frame.

A multi-view geometry model reconstructs all photos together: their cameras (pose and
intrinsics) and a 3D point per pixel, in one world frame. Supported: MapAnything
(`facebook/map-anything-apache`, Apache 2.0) here; Depth Anything 3 runs in its own
environment (numpy < 2, xformers) and writes the same per-view arrays (`save_views`).

`to_grid` maps that world into a block grid with a similarity (scale, rotation, shift):
`similarity_from_cameras` fits it from the reconstructed cameras to known ones (the renders':
exact, for training pairs and for scoring reconstruction). Then, in the grid:

  seen solid  each pixel's point pushed half a block into the surface (along the point
              map's normal), where a block holds `min_points` points
  seen air    the blocks between each point and its camera, and along the rays through the
              pixels that show no building (the empty space around it)
  colour      per block and photo the mean colour of its pixels; of the photos that saw the
              block, the brightest one (shadows are what differ between photos most)

The result has the completion model's input layout (seen solid, seen air, R, G, B).
"""
from __future__ import annotations

import os

import numpy as np

from .front import map_normals

MAPANYTHING = "facebook/map-anything-apache"
FFT_WORKERS = 8               # threads per FFT: all cores each starve the other processes running alongside


def load_mapanything(device="cuda"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from mapanything.models import MapAnything
    return MapAnything.from_pretrained(MAPANYTHING).to(device).eval()


def run_mapanything(model, image_paths: list, intrinsics=None, poses=None) -> list:
    """Per photo: {"pts" (H, W, 3) world, "ray" (H, W, 3) camera-frame unit rays, "cam2world"
    (4, 4) OpenCV, "mask" (H, W) bool, "rgb" (H, W, 3) uint8} at the model's resolution.
    Optional hints: `intrinsics` (per photo 3x3, pixels of the original image) and `poses`
    (per photo 4x4 cam2world, OpenCV, any scale): MapAnything conditions on them."""
    import torch
    from mapanything.utils.image import load_images, preprocess_inputs
    if intrinsics is None and poses is None:
        views = load_images([str(p) for p in image_paths])
    else:
        from PIL import Image
        views = []
        for k, p in enumerate(image_paths):
            v = {"img": torch.tensor(np.array(Image.open(p).convert("RGB")), dtype=torch.float32)}
            if intrinsics is not None:
                v["intrinsics"] = torch.tensor(np.asarray(intrinsics[k]), dtype=torch.float32)
            if poses is not None:
                v["camera_poses"] = torch.tensor(np.asarray(poses[k]), dtype=torch.float32)
                v["is_metric_scale"] = torch.tensor([False])
            views.append(v)
        views = preprocess_inputs(views)
    with torch.no_grad():
        pred = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                           apply_mask=True, mask_edges=True)
    out = []
    for p in pred:
        img = p["img_no_norm"][0].float().cpu().numpy()
        out.append({"pts": p["pts3d"][0].float().cpu().numpy(),
                    "ray": p["ray_directions"][0].float().cpu().numpy(),
                    "cam2world": p["camera_poses"][0].float().cpu().numpy(),
                    "mask": p["mask"][0, ..., 0].float().cpu().numpy() > 0.5,
                    "rgb": np.clip(img * (255.0 if img.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)})
    return out


VIEW_KEYS = ("pts", "ray", "cam2world", "mask", "rgb")


def save_views(path, views: list) -> None:
    """Per-photo arrays (`run_mapanything`'s layout) to one .npz, for a model that runs in
    another environment (`front_da3`)."""
    np.savez_compressed(path, n=len(views), **{f"{k}_{i}": v[k] for i, v in enumerate(views) for k in VIEW_KEYS})


def load_views(path) -> list:
    z = np.load(path)
    return [{k: z[f"{k}_{i}"] for k in VIEW_KEYS} for i in range(int(z["n"]))]


def depth_edges(depth: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """Pixels whose 3x3 neighbourhood spans more than `rtol` of their depth: the mixed
    foreground/background pixels on silhouettes (MapAnything's `depth_edge`)."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(depth, 1, mode="edge")
    win = sliding_window_view(pad, (3, 3))
    return (win.max((-1, -2)) - win.min((-1, -2))) > rtol * np.maximum(depth, 1e-9)


def rough_poses(yaws_deg, pitch_deg=10.0, radius: float = 3.0) -> list:
    """Rough cam2world poses (OpenCV) from where each photo was taken: its compass direction
    round the build (`yaws_deg`, as `visibility.view_axes`) and a typical pitch, on a unit-free
    circle. What a user can say ("front-left corner, then back-left...") and enough to break
    the multi-view model's ambiguities between look-alike sides."""
    from .visibility import view_axes
    out = []
    for i, yaw in enumerate(np.atleast_1d(yaws_deg)):
        p = pitch_deg[i] if np.ndim(pitch_deg) else pitch_deg
        r, d, n = view_axes(float(yaw), float(p))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = np.stack([r, d, -n], 1), radius * np.asarray(n)
        out.append(T)
    return out


SUBSET_SIZES = (4, 4, 3, 3, 2, 2)
_PATTERNS = {4: ((0, 2, 4, 6), (0, 2, 5, 7), (0, 1, 4, 6)), 3: ((0, 3, 5), (0, 2, 5), (0, 3, 6)),
             2: ((0, 4), (0, 3), (0, 2))}       # in eighths of a turn


def view_subsets(n_views: int, rng, sizes=SUBSET_SIZES) -> list:
    """Photo sets for training pairs from a build's `n_views` photos in yaw order (about evenly
    round it: `render.train_views`): per size, the photos about a pattern's eighths of a turn
    apart (opposite pairs, near-corners, three round one side) from a random start, as people
    take 2-4 photos walking round a building. Distinct sets, sorted indices."""
    out = []
    for n in sizes:
        for _ in range(20):
            pat = _PATTERNS[n][int(rng.integers(len(_PATTERNS[n])))]
            r = int(rng.integers(n_views))
            sub = tuple(sorted({(r + int(round(p * n_views / 8.0))) % n_views for p in pat}))
            if len(sub) == n and sub not in out:
                out.append(sub)
                break
    return out


HINTS = ("none", "intrinsics", "rough", "coarse", "side", "compass_sim", "true")


def render_hints(cams: dict, keys: list, mode: str, rng) -> tuple:
    """(intrinsics, poses) hints for renders `keys` (`render`'s cameras.json): "none"; the true
    "intrinsics"; plus "true" poses; plus "rough" poses (yaw within 10 degrees, pitch within 5);
    "coarse": focal within 10%, yaw within 25 degrees and pitch taken as 0 (eye level), about
    what a user's "which side is each photo from" and the photo's EXIF focal give; "side": the
    same yaw and focal, pitch within 3 degrees (estimated from the photo itself: a depth
    model's walls give it to about 2, `front.frame_from_normals`); "compass_sim": a phone's
    compass (yaw within 12 degrees) and the tilt measured from the photo, its error drawn as
    measured on the renders (`real.tilt`: normal, 2.7 degrees sd: median 1.8), the focal known."""
    if mode == "none":
        return None, None
    f = [cams[k]["focal_px"] * (rng.uniform(0.9, 1.1) if mode in ("coarse", "side") else 1.0) for k in keys]
    K = [np.array([[fk, 0.0, cams[k]["size"] / 2], [0.0, fk, cams[k]["size"] / 2], [0.0, 0.0, 1.0]])
         for fk, k in zip(f, keys)]
    if mode == "intrinsics":
        return K, None
    if mode == "true":
        poses = []
        for k in keys:
            T = np.eye(4)
            T[:3, :3] = np.stack([np.array(cams[k]["right"]), np.array(cams[k]["down"]), -np.array(cams[k]["near"])], 1)
            T[:3, 3] = cams[k]["centre"]
            poses.append(T)
        return K, poses
    if mode == "compass_sim":
        yaws = [cams[k]["yaw"] + rng.uniform(-12.0, 12.0) for k in keys]
        pitches = [cams[k]["pitch"] + rng.normal(0.0, 2.7) for k in keys]
        return K, rough_poses(yaws, pitches)
    yaw_err, pitch_err = {"rough": (10.0, 5.0), "coarse": (25.0, None), "side": (25.0, 3.0)}[mode]
    yaws = [cams[k]["yaw"] + rng.uniform(-yaw_err, yaw_err) for k in keys]
    pitches = [0.0 if pitch_err is None else cams[k]["pitch"] + rng.uniform(-pitch_err, pitch_err) for k in keys]
    return K, rough_poses(yaws, pitches)


def street_subsets(cams: dict, rng) -> list:
    """Photo sets from street-level renders (`render.street_views`: five views across the
    street front, three round the back), as people photograph a building that has neighbours
    behind it: three and two photos from the front arc, four of its five, and one set of four
    round it (both ends of the front, its middle, one from the back). Keys, sorted."""
    front = sorted((k for k, c in cams.items() if c.get("arc") == "front"), key=int)
    back = sorted((k for k, c in cams.items() if c.get("arc") == "back"), key=int)
    pick = lambda pool, n: sorted((pool[i] for i in rng.choice(len(pool), n, replace=False)), key=int)
    sets = [pick(front, 3), pick(front, 2), pick(front, 4)]
    if len(front) == 5 and back:
        sets.append(sorted([front[0], front[2], front[4], back[int(rng.integers(len(back)))]], key=int))
    return sets


STREET_EVAL = {"street3": ("1", "2", "3"), "street2": ("1", "3"), "around4": ("0", "2", "4", "6")}


def pair_jobs(bid: str, cams: dict, hints: str, seed: int = 0) -> list:
    """A training build's photo sets and their camera hints: [(j, keys, intrinsics, poses)]
    for its `render.render_build` or `render.render_street_build` cameras (`cams`, keyed by
    view index; street renders get `street_subsets`). The same for every front end (seeded
    by the build), so their pairs differ only in the reconstruction."""
    rng = np.random.default_rng([seed, *bid.encode()])
    order = sorted(cams, key=int)
    if any("arc" in c for c in cams.values()):
        subsets = street_subsets(cams, rng)
    else:
        subsets = [[order[i] for i in sub] for sub in view_subsets(len(order), rng)]
    return [(j, keys, *render_hints(cams, keys, hints, rng)) for j, keys in enumerate(subsets)]


def true_rotations(cams: dict, keys: list) -> list:
    return [np.stack([np.array(cams[k]["right"]), np.array(cams[k]["down"]), -np.array(cams[k]["near"])], 1)
            for k in keys]


def similarity_from_cameras(cam2world: list, centres, rotations=None) -> dict:
    """The similarity (s, R, t: grid = s R world + t) taking the reconstructed camera centres
    onto known ones (Umeyama); with `rotations` (known cam2world 3x3), R is the mean of the
    per-camera rotations instead (better conditioned for a few cameras on a circle)."""
    src = np.stack([np.asarray(c)[:3, 3] for c in cam2world])
    dst = np.asarray(centres, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    if rotations is not None:
        M = sum(np.asarray(Rk) @ np.asarray(c)[:3, :3].T for Rk, c in zip(rotations, cam2world))
        U, _, Vt = np.linalg.svd(M)
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    else:
        U, _, Vt = np.linalg.svd(Y.T @ X)
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    s = float((Y * (X @ R.T)).sum() / max((X ** 2).sum(), 1e-12))
    return {"s": s, "R": R, "t": mu_d - s * (R @ mu_s)}


def refine_to_target(views: list, sim: dict, target_solid: np.ndarray, masks=None,
                     yaws=np.arange(-6.0, 6.01, 1.0), scales=np.arange(0.92, 1.081, 0.02), sample: int = 60000,
                     coarse_scales=None, fast: bool = False) -> dict:
    """`sim` adjusted so the reconstruction's points best overlap the target's solid: a small
    turn about the vertical axis and a scale (both about the points' centre), then the integer
    shift that best overlaps the points' occupancy with the target's solid (FFT). This uses the
    target, as the TRELLIS input's placement does: for training pairs (input and target must
    line up) and for scoring the reconstruction's shape apart from its global placement.
    `coarse_scales`: first the best of these scales (no turn), then `yaws` x `scales` times it
    (for a model whose scale relative to its cameras is less sure). `fast`: the turn and scale
    on a coarse grid (3 degrees, 4%), then 1 degree and 1% around the best: 50 overlaps
    instead of 117, for big builds' training pairs."""
    from scipy import fft as sfft
    s, R, t = sim["s"], np.asarray(sim["R"]), np.asarray(sim["t"])
    pts = np.concatenate([(v["pts"][(v["mask"] if masks is None else (np.asarray(masks[k], bool) & v["mask"]))])
                          for k, v in enumerate(views)])
    if len(pts) > sample:
        pts = pts[np.random.default_rng(0).choice(len(pts), sample, replace=False)]
    g = s * (pts @ R.T) + t
    c = g.mean(0)
    dims = np.asarray(target_solid.shape)
    pad = 8
    box = dims + 2 * pad
    full = tuple(int(sfft.next_fast_len(int(a + b - 1), real=True)) for a, b in zip(dims, box))
    tgt_f = sfft.rfftn(target_solid.astype(np.float32), full, workers=FFT_WORKERS)

    def overlap(Ry, k):
        q = (g - c) @ (k * Ry).T + c
        idx = np.floor(q).astype(np.int64) + pad
        ok = ((idx >= 0) & (idx < box)).all(1)
        occ = np.zeros(tuple(box), np.float32)
        occ[tuple(idx[ok].T)] = 1.0
        # correlation = convolution with the flipped occupancy
        corr = sfft.irfftn(tgt_f * sfft.rfftn(occ[::-1, ::-1, ::-1], full, workers=FFT_WORKERS), full, workers=FFT_WORKERS)
        corr = corr[tuple(slice(0, int(a + b - 1)) for a, b in zip(dims, box))]
        j = np.unravel_index(int(np.argmax(corr)), corr.shape)
        # occ index y holds grid position y - pad; the correlation moves y to y + d
        return float(corr[j]), np.array([int(i) - (n - 1) for i, n in zip(j, occ.shape)]) + pad

    def turn(yaw):
        a = np.radians(yaw)
        return np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    base = 1.0
    if coarse_scales is not None:
        base = float(max(coarse_scales, key=lambda k: overlap(np.eye(3), float(k))[0]))
    best = None

    def search(ys, ks):
        nonlocal best
        for yaw in ys:
            Ry = turn(yaw)
            for k in ks:
                key, shift = overlap(Ry, base * float(k))
                if best is None or key > best[0]:
                    best = (key, Ry, base * float(k), shift, float(yaw), float(k))
    if fast:
        search(np.arange(-6.0, 6.01, 3.0), np.arange(0.92, 1.081, 0.04))
        y0, k0 = best[4], best[5]
        search(np.arange(y0 - 2.0, y0 + 2.01, 1.0), k0 * np.arange(0.98, 1.0201, 0.01))
    else:
        search(yaws, scales)
    _, Ry, k, shift = best[:4]
    # grid' = k Ry (s R w + t - c) + c + shift  ->  s' R' w + t'
    R2 = Ry @ R
    return {"s": s * k, "R": R2, "t": k * (Ry @ (t - c)) + c + shift, "refine": {"yaw": float(np.degrees(np.arctan2(Ry[0, 2], Ry[0, 0]))),
                                                                              "scale": k, "shift": shift.tolist()}}


def march(start: np.ndarray, d: np.ndarray, t0: float, step: float, n: int, dims, out: np.ndarray,
          chunk: int = 8192) -> None:
    """Mark every cell of `out` that the rays start + t d pass through, t = t0 + k step for
    k < n (`front._march` on that grid of t), sampling only where a ray is inside the grid."""
    dims = np.asarray(dims, np.float64)
    for c in range(0, len(start), chunk):
        s0 = np.asarray(start[c:c + chunk], np.float64)
        dd = np.asarray(d[c:c + chunk], np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(dd != 0, (0.0 - s0) / dd, -np.inf)
            t2 = np.where(dd != 0, (dims - s0) / dd, np.inf)
        flat = dd == 0
        outside = (flat & ((s0 < 0) | (s0 >= dims))).any(1)
        lo = np.where(flat, -np.inf, np.minimum(t1, t2)).max(1)
        hi = np.where(flat, np.inf, np.maximum(t1, t2)).min(1)
        k_lo = np.clip(np.ceil((lo - 1e-6 - t0) / step), 0, n).astype(np.int64)      # a hair wider: the
        k_hi = np.clip(np.floor((hi + 1e-6 - t0) / step) + 1, 0, n).astype(np.int64)  # cell test decides
        k_hi[outside | (hi < lo)] = 0
        count = np.maximum(k_hi - k_lo, 0)
        if not count.any():
            continue
        ray = np.repeat(np.arange(len(s0)), count)
        k = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count) + k_lo[ray]
        q = np.floor(s0[ray] + (t0 + k * step)[:, None] * dd[ray]).astype(np.int64)
        ok = ((q >= 0) & (q < dims.astype(np.int64))).all(1)
        q = q[ok]
        out[q[:, 0], q[:, 1], q[:, 2]] = True


def to_grid(views: list, sim: dict, dims, masks=None, min_points: int = 2, air_step: float = 0.5,
            air_every: int = 2, push: float = 0.5, res: int = 1, carve: bool = False, margin: int = 1) -> np.ndarray:
    """Per-photo reconstructions (`run_mapanything`'s output), a similarity into the block grid
    and the grid's `dims` -> the input (X, Y, Z, 5) uint8. `masks`: per photo, the building's
    pixels at the model's resolution (default: the model's own mask). `res` cells per block:
    2 is the model's input grid itself, with what the points show below a block (roof slopes,
    slabs, steps); 1 the block grid (`data.upsample` it for the model). `push`, `air_step` and
    `min_points` are per cell. `carve` (space carving, as `data.fuse_views`): a photo's points
    are dropped from cells another photo saw clear through, more than `margin` blocks from that
    photo's own surface (near it, its air depends on its depth)."""
    dims = np.asarray(dims, np.int64) * int(res)
    n = int(np.prod(dims))
    s, R, t = sim["s"] * res, np.asarray(sim["R"]), np.asarray(sim["t"]) * res
    count = np.zeros(n)
    per_view = []                                        # (cells, air) of each photo, for carving
    best_rgb = np.zeros((n, 3))
    best_lum = np.full(n, -1.0)
    air = np.zeros(tuple(dims), bool)
    for k, v in enumerate(views):
        m = v["mask"] if masks is None else (np.asarray(masks[k], bool) & v["mask"])
        pts = s * (v["pts"] @ R.T) + t                       # (H, W, 3) in blocks
        n_map, ok = map_normals(pts, m)
        nrm = np.zeros_like(pts)
        nrm[1:-1, 1:-1] = np.where(ok[..., None], n_map, 0.0)
        cam = s * (R @ v["cam2world"][:3, 3]) + t
        P, N, C = pts[m], nrm[m], v["rgb"][m].astype(np.float64)
        N = np.where(((N * (cam - P)).sum(1) < 0)[:, None], -N, N)   # outward: towards the camera
        idx = np.floor(P - push * N).astype(np.int64)
        keep = ((idx >= 0) & (idx < dims)).all(1)
        flat = np.ravel_multi_index(idx[keep].T, dims)
        c_k = np.bincount(flat, minlength=n)
        rgb_k = np.stack([np.bincount(flat, weights=C[keep, c], minlength=n) for c in range(3)], 1)
        seen = c_k > 0
        mean_k = rgb_k[seen] / c_k[seen][:, None]
        lum = mean_k @ np.array([0.299, 0.587, 0.114])
        better = lum > best_lum[seen]
        sel = np.nonzero(seen)[0][better]
        best_rgb[sel], best_lum[sel] = mean_k[better], lum[better]
        count += c_k
        air_all = air
        if carve:
            air = np.zeros(tuple(dims), bool)
        # seen air: from each point towards its camera, and along the background pixels' rays
        start = P[::air_every]
        d = cam[None, :] - start
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        n_t = len(np.arange(0.75, float(np.linalg.norm(dims)) + 1.0, air_step))
        march(start, d, 0.75, air_step, n_t, dims, air)
        bg = ~m
        rays = (v["ray"][bg][::4] @ v["cam2world"][:3, :3].T) @ R.T
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
        corners = np.array([[x, y, z] for x in (0, dims[0]) for y in (0, dims[1]) for z in (0, dims[2])], float)
        dist = np.linalg.norm(corners - cam, axis=1)
        t0 = max(float(dist.min()) - float(np.linalg.norm(dims)), 0.0)
        march(np.broadcast_to(cam, rays.shape), rays, t0, air_step, len(np.arange(t0, float(dist.max()) + 1.0, air_step)),
              dims, air)
        if carve:
            per_view.append((c_k.astype(np.int32), air))
            air = air_all | air
    if carve and len(per_view) > 1:
        from scipy.ndimage import binary_dilation
        cube = np.ones((3, 3, 3), bool)
        deep = [a & ~binary_dilation((c > 0).reshape(tuple(dims)), cube, iterations=int(margin * res))
                for c, a in per_view]
        count = np.zeros(n)
        for k, (c, _) in enumerate(per_view):
            others = np.logical_or.reduce([d for j, d in enumerate(deep) if j != k]).ravel()
            count += np.where(others, 0, c)
    solid = (count >= min_points).reshape(tuple(dims))
    out = np.zeros(tuple(dims) + (5,), np.uint8)
    out[..., 0] = solid
    out[..., 1] = air & ~solid
    out[..., 2:] = np.where(solid[..., None], np.clip(best_rgb.reshape(tuple(dims) + (3,)), 0, 255), 0).astype(np.uint8)
    return out


def to_blocks(inp: np.ndarray, dims) -> np.ndarray:
    """A `to_grid` input at 2 cells per block -> the block grid: solid where any of a block's
    cells is, seen air where all eight are; colour the mean of its solid cells."""
    dims = tuple(int(d) for d in dims)
    if inp.shape[:3] == dims:
        return inp
    c = inp[:2 * dims[0], :2 * dims[1], :2 * dims[2]].reshape(dims[0], 2, dims[1], 2, dims[2], 2, 5)
    solid = c[..., 0].astype(bool)
    n = solid.sum((1, 3, 5))
    out = np.zeros(dims + (5,), np.uint8)
    out[..., 0] = n > 0
    out[..., 1] = c[..., 1].astype(bool).all((1, 3, 5)) & (n == 0)
    rgb = (c[..., 2:].astype(np.float64) * solid[..., None]).sum((1, 3, 5)) / np.maximum(n, 1)[..., None]
    out[..., 2:] = np.where((n > 0)[..., None], np.clip(rgb, 0, 255), 0).astype(np.uint8)
    return out


def model_input(inp: np.ndarray, dims) -> np.ndarray:
    """A `to_grid` input (1 or 2 cells per block) -> the model's 2x input grid."""
    from .data import upsample
    dims = tuple(int(d) for d in dims)
    return upsample(inp) if inp.shape[:3] == dims else inp


def target_into(ts: np.ndarray, tm: np.ndarray, info: dict, dims) -> tuple:
    """The target (shapes, materials) moved into a free placement's grid `dims`: the inverse
    of `onto_target`'s alignment `info` (the input turned `turns` quarter turns about +Y, then
    shifted by `shift` onto the target), the shapes turned with their facings
    (`mcvoxel.blockstates.rotate_y`), air past the target's grid."""
    from mcvoxel.blockstates import rotate_y
    from training.score import _place
    dims = tuple(int(d) for d in dims)
    k = int(info["turns"]) % 4
    rdims = (dims[2], dims[1], dims[0]) if k % 2 else dims
    back = [-int(v) for v in info["shift"]]
    t_s, t_m = _place(np.asarray(ts), back, rdims, 0), _place(np.asarray(tm), back, rdims, 0)
    if k:
        t_s, t_m = rotate_y(t_s, (4 - k) % 4, t_m)
    return np.ascontiguousarray(t_s), np.ascontiguousarray(t_m)


def onto_target(inp: np.ndarray, dims, ts: np.ndarray, tm: np.ndarray) -> tuple:
    """A `to_grid` input placed without the target (`place.free_placement`, grid `dims`) ->
    (the input in the target's frame, alignment): the quarter turn about +Y and the whole-block
    shift that best overlap its seen blocks with the target's solid (`training.score.align`),
    applied to the input itself at its own resolution (lossless: a turn and a shift). The
    placement's own small errors (tilt, a few degrees of yaw) stay, as they will be on real
    photos."""
    from training.score import align
    from .front import place as put
    dims = tuple(int(d) for d in dims)
    blk = to_blocks(inp, dims)
    _, _, info = align(blk[..., 0].astype(np.uint8), np.zeros(dims, np.uint16), ts, tm)
    f = 2 if tuple(inp.shape[:3]) != dims else 1
    out = put(inp, info["turns"], [f * v for v in info["shift"]], tuple(f * d for d in ts.shape))
    return out, info
