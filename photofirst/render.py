"""Perspective photos of the real-TRELLIS evaluation builds, for testing the front end.

The smooth set's photos (BuildDataCollector `schemset.real_trellis`) are orthographic, and a
monocular depth model reads depth from perspective: on them MoGe-2 flattens the relief to
0.76 of the truth. These are the same photos (same surface, dark glass, smoothing and light:
`render_set_photo`'s smooth branch) through a perspective camera with `FOV_DEG` from the
same direction (yaw 35, pitch 12), the build filling about 85% of the frame. Runs in
BuildDataCollector's environment, from this repository's root:

    ..\\BuildDataCollector\\.venv\\Scripts\\python -m photofirst.render --dataset <BDC>/data/real-schems-v1

Writes <dataset>/photofirst/persp/<id>.png, <id>.seen.npy ([seen solid, seen air] packed, from
that camera: `visibility.cast_from`) and cameras.json (centre in grid coordinates, axes,
focal length in pixels, field of view, image size).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np

from .visibility import EVAL_VIEW, cast_from, view_axes
from .views import OUT_DIR

FOV_DEG = 50.0
SIZE = 768
FILL = 0.85
SAMPLES = 6_000_000


def camera_for(points, yaw=EVAL_VIEW[0], pitch=EVAL_VIEW[1], fov_deg=FOV_DEG, size=SIZE, fill=FILL) -> dict:
    """A camera looking at the middle of `points` (the build's surface) from (yaw, pitch), at
    the distance where the farthest point from the image centre reaches `fill` of the
    half-frame."""
    right, down, near = view_axes(yaw, pitch)
    corners = np.asarray(points, float)
    c0 = (corners.min(0) + corners.max(0)) / 2.0
    half = np.radians(fov_deg) / 2.0
    focal = (size / 2.0) / np.tan(half)

    def reach(dist):
        v = corners - (c0 + dist * near)
        z = -(v @ near)
        return float(np.max(np.abs(np.stack([v @ right, v @ down], 1)) / z[:, None])) * focal / (size / 2.0)
    a, b = 1e-3, 1e6
    for _ in range(100):                                         # reach falls as the camera backs off
        m = (a + b) / 2.0
        a, b = (m, b) if reach(m) > fill or not np.isfinite(reach(m)) else (a, m)
    dist = b
    return {"centre": (c0 + dist * near).tolist(), "right": right.tolist(), "down": down.tolist(),
            "near": near.tolist(), "focal_px": float(focal), "fov_deg": float(fov_deg), "size": int(size),
            "yaw": float(yaw), "pitch": float(pitch)}


SUN = (0.35, 0.85, 0.4)


def render(scene, cam: dict, light, samples: int = SAMPLES, sun=SUN):
    """A perspective RGBA render of `scene` (`synth.trellis_compare.render_photo`'s lighting and
    point splatting, with a perspective projection)."""
    from PIL import Image
    from mcvoxel.scene import scene_color_fns
    from synth.preview import sample_surface
    _, point_colors = scene_color_fns(scene)
    pts, cols, normals = sample_surface(scene.verts, scene.faces, point_colors, samples, seed=1)
    c, right, down, near = (np.asarray(cam[k], float) for k in ("centre", "right", "down", "near"))
    v = pts - c
    x, y, z = v @ right, v @ down, -(v @ near)
    keep = z > 1e-3
    sun = np.asarray(sun, float)
    sun /= np.linalg.norm(sun)
    lam = np.clip(np.abs(normals @ sun), 0, 1)
    rgb = np.clip(cols * (light[0] + light[1] * lam)[:, None], 0, 255).astype(np.uint8)
    big, f2 = 2 * cam["size"], 2 * cam["focal_px"]
    px = np.rint(f2 * x[keep] / z[keep] + big / 2.0).astype(np.int64)
    py = np.rint(f2 * y[keep] / z[keep] + big / 2.0).astype(np.int64)
    rgb, depth = rgb[keep], z[keep]
    canvas = np.zeros((big, big, 4), np.uint8)
    order = np.argsort(-depth, kind="stable")                     # far first, near last
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xx, yy = px[order] + dx, py[order] + dy
        ok = (xx >= 0) & (xx < big) & (yy >= 0) & (yy < big)
        canvas[yy[ok], xx[ok], :3] = rgb[order][ok]
        canvas[yy[ok], xx[ok], 3] = 255
    return Image.fromarray(canvas, "RGBA").resize((cam["size"], cam["size"]), Image.LANCZOS)


TRAIN_DIR = "persp_train"


def train_views(views: list, n: int) -> list:
    """`n` of a build's (yaw, pitch) views (`views.build`'s, one per yaw sector), spread round
    it: every len/n-th in yaw order."""
    order = sorted(views, key=lambda v: v[0])
    return [order[int(i * len(order) / n)] for i in range(n)]


TRAIN_SAMPLES = 3_000_000


def render_build(dataset, bid: str, views: list, seed: int = 0, seen: bool = False,
                 samples: int = TRAIN_SAMPLES) -> int:
    """Perspective photos of one build from each (yaw, pitch) of `views`, for training pairs:
    <dataset>/photofirst/persp_train/<bid>/<k>.png, cameras.json (by view index, written
    last: a build with it is done) and with `seen` <k>.seen.npy (what the camera truly sees;
    only for scoring the front end: the training target is the whole build). The build's light
    varies (sun direction, exposure), so the model does not learn one shading. Runs in
    BuildDataCollector's environment."""
    dataset = Path(dataset)
    if str(dataset.parents[1]) not in sys.path:
        sys.path.insert(0, str(dataset.parents[1]))
    from schemset.meshify import meshify_npz
    from schemset.pairs import glass_look
    from schemset.real_trellis import PHOTO_EXPOSURE, PHOTO_GLASS, PHOTO_LIGHT, SMOOTH_SIGMA, smooth_surface
    d = dataset / OUT_DIR / TRAIN_DIR / bid
    if (d / "cameras.json").is_file():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, *bid.encode()])
    mesh, _, _ = meshify_npz(dataset / "normalized" / f"{bid}.npz")
    surf = smooth_surface(glass_look(mesh, PHOTO_GLASS), SMOOTH_SIGMA)
    ts = np.load(dataset / "normalized" / f"{bid}.npz")["target_shape"]
    az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(35, 70))
    sun = (np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az))
    gain = PHOTO_EXPOSURE * rng.uniform(0.85, 1.25)
    a, s = PHOTO_LIGHT
    pts = np.asarray(mesh.verts)[:: max(1, len(mesh.verts) // 50000)]
    cams = {}
    for k, (yaw, pitch) in enumerate(views):
        cam = camera_for(pts, yaw=float(yaw), pitch=float(pitch))
        render(surf, cam, (a * gain, s * gain), samples=samples, sun=sun).save(d / f"{k}.png")
        if seen:
            solid, air = cast_from(ts, cam["centre"], cam["right"], cam["down"], cam["near"], cam["focal_px"],
                                   cam["size"])
            np.save(d / f"{k}.seen.npy", np.stack([np.packbits(solid.ravel()), np.packbits(air.ravel())]))
        cams[str(k)] = {**cam, "build": bid, "view": k, "sun": list(sun), "gain": gain, "dims": [int(v) for v in ts.shape]}
    (d / "cameras.json").write_text(json.dumps(cams, indent=1), encoding="utf-8", newline="\n")
    return len(views)


def _limit_threads(n: int = 3):
    """Each render worker on a few threads: several workers with all cores each (numpy's
    BLAS, scipy's FFT) slow one another down many times over."""
    import os
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass


def _render_job(job):
    dataset, bid, views = job
    import time
    t = time.time()
    try:
        n = render_build(dataset, bid, views)
    except MemoryError as e:                 # one build too big for the free memory: the rest go on
        return bid, f"failed ({type(e).__name__}); rerun to retry", time.time() - t
    return bid, n, time.time() - t


# --------------------------------------------------------------------------- street-level photos

STREET_DIR = "persp_street"
FRONT_ARC = (-80.0, -40.0, 0.0, 40.0, 80.0)  # views 0-4: degrees from the build's street front
BACK_ARC = (135.0, 180.0, 225.0)             # views 5-7: round the back, for 4-photo sets
STREET_FOV = (50.0, 70.0)                     # a phone's main camera, square crop (degrees)
STREET_FILL = (0.55, 0.9)                     # how much of the frame the build takes
STREET_PITCH = (-35.0, 10.0)                  # this module's pitch (positive looks down): tilt-up limit
EYE_M = 1.6
STOREY_M = 3.0
ELEVATED = 0.15                               # share of photos from a raised spot across the street
SMOOTH_SHARE = 0.5                            # builds drawn with the TRELLIS-like smoothed surface


def street_views(bid: str, seed: int = 0) -> list:
    """The eight photos of one build, seeded by its id: the street front's direction (any),
    five views across the front arc and three round the back, each with its lens, framing
    and whether it is taken from a raised spot. Val and test builds get the same recipe, so
    their street photos are fixed too."""
    rng = np.random.default_rng([seed, zlib.crc32(bid.encode())])
    front = float(rng.uniform(0.0, 360.0))
    out = []
    for arc, offs in (("front", FRONT_ARC), ("back", BACK_ARC)):
        for o in offs:
            out.append({"arc": arc, "yaw": round((front + o + rng.uniform(-10.0, 10.0)) % 360.0, 3),
                        "fov": round(float(rng.uniform(*STREET_FOV)), 2),
                        "fill": round(float(rng.uniform(*STREET_FILL)), 3),
                        "raised_m": round(float(rng.uniform(2.0, 8.0)), 2) if rng.random() < ELEVATED else 0.0})
    return out


def eye_blocks(record: dict) -> float:
    """A standing person's eye height in this build's blocks: 1.6 m at its storey height
    (`schemset.storey`, 3 m a storey) when that is known, else at 4 blocks a storey."""
    st = record.get("storey") or {}
    bps = st.get("blocks_per_storey") if st.get("confidence") not in (None, "low", "unknown") else None
    if not isinstance(bps, (int, float)) or bps <= 0:          # "unknown", or no estimate
        bps = 4.0
    return float(np.clip(EYE_M / STOREY_M * float(bps), 1.5, 3.5))


def street_camera(points, ground_y: float, eye: float, yaw: float, fov_deg: float, fill: float,
                  size: int = SIZE) -> dict:
    """A camera standing on the build's ground (`eye` blocks above the ground surface at
    `ground_y`), facing it from `yaw`, tilted to centre it vertically (within
    `STREET_PITCH`), at the distance where it fills `fill` of the half-frame."""
    pts = np.asarray(points, float)
    c0 = (pts.min(0) + pts.max(0)) / 2.0
    focal = (size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    _, _, h = view_axes(yaw, 0.0)                               # horizontal, towards the camera
    y_cam = ground_y + eye

    def setup(dist):
        centre = np.array([c0[0], y_cam, c0[2]]) + dist * h
        v = pts - centre
        horiz = -(v @ h)
        el = np.degrees(np.arctan2(v[:, 1], np.maximum(horiz, 1e-6)))
        pitch = float(np.clip(-(el.max() + el.min()) / 2.0, *STREET_PITCH))
        right, down, near = view_axes(yaw, pitch)
        z = -(v @ near)
        if (z <= 1e-3).any():
            return centre, pitch, (right, down, near), np.inf
        reach = float(np.max(np.abs(np.stack([v @ right, v @ down], 1)) / z[:, None])) * focal / (size / 2.0)
        return centre, pitch, (right, down, near), reach

    a, b = 1e-2, 1e5
    for _ in range(80):                                        # the reach falls as the camera backs off
        m = np.sqrt(a * b)
        a, b = (m, b) if setup(m)[3] > fill else (a, m)
    centre, pitch, (right, down, near), _ = setup(b)
    return {"centre": centre.tolist(), "right": right.tolist(), "down": down.tolist(), "near": near.tolist(),
            "focal_px": float(focal), "fov_deg": float(fov_deg), "size": int(size), "yaw": float(yaw),
            "pitch": pitch, "eye_blocks": float(eye), "ground_y": float(ground_y)}


def photo_effects(im, rng):
    """What a phone photo cut out by background removal looks like, on an RGBA render:
    white balance, the building's base hidden by things in front of it (parked cars, bushes,
    a fence: removed with the background, so the base is missing from the cutout), a ragged
    cutout edge, blur, sensor noise and JPEG. Returns (image, what was applied)."""
    from PIL import Image, ImageFilter
    from scipy import ndimage
    arr = np.asarray(im).astype(np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3] > 127
    info = {"tint": np.round(1.0 + rng.normal(0.0, 0.04, 3), 3).tolist()}
    rgb = np.clip(rgb * np.asarray(info["tint"], np.float32), 0, 255)
    info["occluded"] = bool(rng.random() < 0.5 and alpha.any())
    if info["occluded"]:                                       # things in front of the base
        H, W = alpha.shape
        cols = np.flatnonzero(alpha.any(0))
        rows = np.flatnonzero(alpha.any(1))
        hgt = rows[-1] - rows[0] + 1
        bottom = H - 1 - np.argmax(alpha[::-1], axis=0)       # lowest building pixel per column
        yy, xx = np.mgrid[:H, :W]
        for _ in range(int(rng.integers(1, 5))):
            cx = float(rng.uniform(cols[0], cols[-1]))
            cy = float(bottom[int(cx)]) - rng.uniform(0.0, 0.04) * hgt
            rx, ry = rng.uniform(0.05, 0.2) * (cols[-1] - cols[0] + 1), rng.uniform(0.03, 0.12) * hgt
            alpha &= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 > 1.0
    info["eroded"] = bool(rng.random() < 0.3)
    if info["eroded"]:                                         # a cutout a pixel or two tight
        alpha = ndimage.binary_erosion(alpha, iterations=int(rng.integers(1, 3)))
    out = Image.fromarray(rgb.astype(np.uint8), "RGB")
    info["blur"] = round(float(rng.uniform(0.0, 0.8)), 2)
    if info["blur"] > 0.1:
        out = out.filter(ImageFilter.GaussianBlur(info["blur"]))
    info["noise"] = round(float(rng.uniform(0.0, 3.0)), 2)
    noisy = np.asarray(out).astype(np.float32) + rng.normal(0.0, info["noise"], (*alpha.shape, 3))
    out = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")
    info["jpeg"] = int(rng.integers(60, 96))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=info["jpeg"])
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    rgba = np.dstack([np.asarray(out), np.where(alpha, 255, 0).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA"), info


def render_street_build(dataset, bid: str, seed: int = 0, samples: int = TRAIN_SAMPLES, seen: bool = False) -> int:
    """`street_views`' eight photos of one build from where people take them: standing on
    its ground at eye height (`eye_blocks`), sometimes from a raised spot, tilted up to fit
    it (so roofs mostly go unseen), with a phone's lens, varied sun, overcast days, exposure,
    smoothed or blocky surfaces and `photo_effects`. Writes <dataset>/photofirst/persp_street/
    <bid>/<k>.png and cameras.json (by view index, written last: a build with it is done;
    each camera records its arc, eye height and ground); with `seen` (the evaluation builds)
    also <k>.seen.npy, what the camera truly sees (`visibility.cast_from`). Runs in
    BuildDataCollector's environment."""
    dataset = Path(dataset)
    if str(dataset.parents[1]) not in sys.path:
        sys.path.insert(0, str(dataset.parents[1]))
    from schemset.meshify import meshify_npz
    from schemset.pairs import glass_look
    from schemset.real_trellis import PHOTO_EXPOSURE, PHOTO_GLASS, PHOTO_LIGHT, SMOOTH_SIGMA, smooth_surface
    d = dataset / OUT_DIR / STREET_DIR / bid
    if (d / "cameras.json").is_file():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed + 1, zlib.crc32(bid.encode())])
    npz = dataset / "normalized" / f"{bid}.npz"
    record = json.loads((dataset / "normalized" / "records" / f"{bid}.json").read_text(encoding="utf-8"))
    mesh, _, _ = meshify_npz(npz)
    smooth = bool(rng.random() < SMOOTH_SHARE)
    surf = glass_look(mesh, PHOTO_GLASS)
    if smooth:
        surf = smooth_surface(surf, SMOOTH_SIGMA)
    with np.load(npz) as z:
        ts = z["target_shape"]
        gy = int(z["ground_y"])
    solid = (ts != 0) & (ts != 255)
    ground = float(gy + 1) if gy >= 0 else float(np.flatnonzero(solid.any((0, 2)))[0])
    eye = eye_blocks(record)
    m_per_block = EYE_M / eye
    az, el = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(20, 70))
    sun = (np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az))
    overcast = bool(rng.random() < 0.3)
    a, s = PHOTO_LIGHT
    if overcast:                                               # flat light: little sun, more sky
        a, s = a * 2.2, s * 0.35
    gain = PHOTO_EXPOSURE * rng.uniform(0.8, 1.3)
    pts = np.asarray(mesh.verts)[:: max(1, len(mesh.verts) // 50000)]
    cams = {}
    for k, v in enumerate(street_views(bid, seed)):
        cam = street_camera(pts, ground, eye + v["raised_m"] / m_per_block, v["yaw"], v["fov"], v["fill"])
        im = render(surf, cam, (a * gain, s * gain), samples=samples, sun=sun)
        im, fx = photo_effects(im, rng)
        im.save(d / f"{k}.png")
        if seen:
            sol, air = cast_from(ts, cam["centre"], cam["right"], cam["down"], cam["near"], cam["focal_px"],
                                 cam["size"])
            np.save(d / f"{k}.seen.npy", np.stack([np.packbits(sol.ravel()), np.packbits(air.ravel())]))
        cams[str(k)] = {**cam, **v, **fx, "build": bid, "view": k, "sun": list(sun), "gain": gain,
                        "overcast": overcast, "smooth": smooth, "dims": [int(n) for n in ts.shape]}
    (d / "cameras.json").write_text(json.dumps(cams, indent=1), encoding="utf-8", newline="\n")
    return len(cams)


def _street_job(job):
    dataset, bid, seen = job
    import time
    t = time.time()
    try:
        n = render_street_build(dataset, bid, seen=seen)
    except MemoryError as e:
        return bid, f"failed ({type(e).__name__}); rerun to retry", time.time() - t
    return bid, n, time.time() - t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--views", type=int, nargs="*", default=[0],
                    help="corners: view k looks from yaw 35 + 90 k (0 is the evaluation photos' corner)")
    ap.add_argument("--train-views", type=int, default=0,
                    help="instead: this many views of every build of --splits (training pairs), into persp_train/")
    ap.add_argument("--street", action="store_true",
                    help="instead: `street_views`' eight street-level photos of every build of --splits, into persp_street/")
    ap.add_argument("--ids", nargs="*", default=None, help="with --street: only these builds")
    ap.add_argument("--splits", nargs="*", default=["train"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    dataset = args.dataset.resolve()
    if args.street:
        from concurrent.futures import ProcessPoolExecutor
        rows = json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]
        jobs = [(dataset, r["id"], r["split"] != "train") for r in rows if r["split"] in args.splits
                and (args.ids is None or r["id"] in args.ids)
                and (dataset / "normalized" / f"{r['id']}.npz").is_file()]
        import os
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = "3"
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_limit_threads) as ex:
            for i, (bid, n, dt) in enumerate(ex.map(_street_job, jobs), 1):
                print(f"[{i}/{len(jobs)}] {bid}: {n} views, {dt:.0f} s", flush=True)
        return 0
    if args.train_views:
        from concurrent.futures import ProcessPoolExecutor
        manifest = json.loads((dataset / OUT_DIR / "manifest.json").read_text(encoding="utf-8"))["builds"]
        jobs = [(dataset, bid, train_views(e["views"], args.train_views)) for bid, e in sorted(manifest.items())
                if e["split"] in args.splits]
        import os
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # inherited by the workers,
            os.environ[k] = "3"                                                     # before their numpy loads
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_limit_threads) as ex:
            for i, (bid, n, dt) in enumerate(ex.map(_render_job, jobs), 1):
                print(f"[{i}/{len(jobs)}] {bid}: {n} views, {dt:.0f} s", flush=True)
        return 0
    sys.path.insert(0, str(dataset.parents[1]))                 # BuildDataCollector
    from schemset.meshify import meshify_npz
    from schemset.pairs import glass_look
    from schemset.real_trellis import PHOTO_EXPOSURE, PHOTO_GLASS, PHOTO_LIGHT, SMOOTH_SIGMA, smooth_surface
    out = dataset / OUT_DIR / "persp"
    out.mkdir(parents=True, exist_ok=True)
    cams_path = out / "cameras.json"
    cams = json.loads(cams_path.read_text(encoding="utf-8")) if cams_path.is_file() and not args.force else {}
    smooth = json.loads((dataset / "pairs" / "real_trellis_smooth" / "index.json").read_text(encoding="utf-8"))
    for b in smooth["builds"]:
        bid = b["id"]
        keys = {k: view_key(bid, k) for k in args.views}
        todo = [k for k, key in keys.items() if args.force or key not in cams or not (out / f"{key}.png").is_file()]
        if b.get("status") != "ok" or not todo:
            continue
        mesh, _, _ = meshify_npz(dataset / "normalized" / f"{bid}.npz")
        surf = smooth_surface(glass_look(mesh, PHOTO_GLASS), SMOOTH_SIGMA)   # once per build, for every view
        ts = np.load(dataset / "normalized" / f"{bid}.npz")["target_shape"]
        a, s = PHOTO_LIGHT
        for k in todo:
            cam = camera_for(np.asarray(mesh.verts)[:: max(1, len(mesh.verts) // 50000)],
                             yaw=(EVAL_VIEW[0] + 90.0 * k) % 360.0)
            render(surf, cam, (a * PHOTO_EXPOSURE, s * PHOTO_EXPOSURE)).save(out / f"{keys[k]}.png")
            solid, air = cast_from(ts, cam["centre"], cam["right"], cam["down"], cam["near"], cam["focal_px"], cam["size"])
            np.save(out / f"{keys[k]}.seen.npy", np.stack([np.packbits(solid.ravel()), np.packbits(air.ravel())]))
            cams[keys[k]] = {**cam, "split": b["split"], "dims": [int(d) for d in ts.shape], "build": bid, "view": k}
            cams_path.write_text(json.dumps(cams, indent=1), encoding="utf-8", newline="\n")
            print(f"  {keys[k]}: rendered, {int(solid.sum())} blocks seen", flush=True)
    return 0


def view_key(bid: str, k: int) -> str:
    """File stem and cameras.json key of a build's view k (view 0 keeps the plain id)."""
    return bid if k == 0 else f"{bid}_v{k}"


if __name__ == "__main__":
    sys.exit(main())
