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
import json
import sys
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


def render(scene, cam: dict, light, samples: int = SAMPLES):
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
    sun = np.array([0.35, 0.85, 0.4])
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--views", type=int, nargs="*", default=[0],
                    help="corners: view k looks from yaw 35 + 90 k (0 is the evaluation photos' corner)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    dataset = args.dataset.resolve()
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
