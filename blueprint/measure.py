"""A building's size from its photo: MapAnything (metric, no hints) on the cut-out photo,
levelled and turned so the walls run along the axes (`photofirst.place.free_placement`),
then the extent of the building's points along the facade (width), away from it (depth)
and up (height), in metres. Runs in `r2m_train` (MapAnything), from this repository's root:

    r2m_train\\Scripts\\python -m blueprint.measure outputs\\vlm\\test50 --out outputs\\vlm\\measure

The cut-out is `refine prepare`'s segmentation when that ran into the same --out (the main
building's pixels only), else the app's background removal.

Single photos give the scale only roughly (MapAnything's metric depth), and only the depth
the photo shows; the ratios (height / width) are the more reliable part. Per photo:
<out>/<id>.measure.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LO, HI = 2.0, 98.0          # percentiles of the building's points: clipped against stray points


def measure_views(views: list, masks: list) -> dict:
    """Width / depth / height (reconstruction units: metres without hints) of the masked
    points, the facade being the wall axis that faces the first camera."""
    from photofirst import place as PL
    sim, dims, info = PL.free_placement(views, masks, metres_per_block=1.0)
    P, _ = PL.fused_points(views, masks)
    q = P @ sim["R"].T
    lo, hi = np.percentile(q, LO, axis=0), np.percentile(q, HI, axis=0)
    cam = np.asarray(views[0]["cam2world"])[:3, 3] @ sim["R"].T
    d = cam - (lo + hi) / 2.0
    front = 0 if abs(d[2]) >= abs(d[0]) else 2          # the facade runs across the view
    side = 2 if front == 0 else 0
    ext = hi - lo
    # the camera in the build's frame (facade along x, the camera on the +z side): the yaw and
    # pitch `blueprint.render` takes to see the build as the photo does
    dx, dz = (d[0], d[2]) if front == 0 else (d[2], -d[0])
    if dz < 0:
        dx, dz = -dx, -dz
    yaw = float(np.degrees(np.arctan2(-dx, dz)))
    pitch = float(np.degrees(np.arctan2(d[1], np.hypot(dx, dz))))
    return {"width_m": float(ext[front]), "depth_m": float(ext[side]), "height_m": float(ext[1]),
            "camera_yaw_deg": yaw, "camera_pitch_deg": pitch,
            "distance_m": float(np.hypot(d[0], d[2])), "corner_view_deg": float(np.degrees(np.arctan2(
                abs(d[side]), abs(d[front])))), "levelling_deg": info["levelling_deg"],
            "wall_share": info["wall_share"], "points": int(len(P))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", type=Path, help="photos, or folders of <id>/photo.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutouts-only", action="store_true", help="only the background removal (no GPU)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    from PIL import Image
    from photofirst import front_mv as M
    from photofirst.photos_to_schem import prepare_photos
    from .refine import photos_of
    photos = photos_of(args.photos)
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in photos
            if not (args.out / f"{p.parent.name if p.name == 'photo.jpg' else p.stem}.measure.json").is_file()]
    work = args.out / "_work"
    prepared = []
    for p in todo:                                      # cutouts first (the app's environment), all at once
        stem = p.parent.name if p.name == "photo.jpg" else p.stem
        d = work / stem
        d.mkdir(parents=True, exist_ok=True)
        if (d / "seg.json").is_file() and (d / "mask.png").is_file():
            # `refine prepare`'s segmentation: the main building only, trees and cars in front left out
            a = np.array(Image.open(d / "mask.png")) > 127
            ok = bool(json.loads((d / "seg.json").read_text())["ok"])
        elif (d / "mask.png").is_file() and (d / "cutout.json").is_file():
            a = np.array(Image.open(d / "mask.png")) > 127
            ok = json.loads((d / "cutout.json").read_text())["ok"]
        else:
            rgba, _, oks = prepare_photos([p], d)
            a, ok = rgba[0][..., 3] > 127, bool(oks[0])
            Image.fromarray(np.where(a[..., None], rgba[0][..., :3], 255).astype(np.uint8)).save(d / "white.png")
            Image.fromarray((a * 255).astype(np.uint8)).save(d / "mask.png")
            (d / "cutout.json").write_text(json.dumps({"ok": ok}))
        prepared.append((stem, d / "white.png", a, ok))
    if args.cutouts_only:
        print(f"{len(prepared)} cutouts ready", flush=True)
        return 0
    model = M.load_mapanything()
    for stem, white, a, ok in prepared:
        t = time.time()
        views = M.run_mapanything(model, [white])
        h, w = views[0]["pts"].shape[:2]
        mask = np.array(Image.fromarray(a).resize((w, h), Image.NEAREST))
        rec = {"id": stem, "cutout": ok, "mask_share": float(a.mean()), **measure_views(views, [mask]),
               "seconds": round(time.time() - t, 1)}
        (args.out / f"{stem}.measure.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{stem}: {rec['width_m']:.1f} x {rec['depth_m']:.1f} x {rec['height_m']:.1f} m "
              f"(w x d x h), cutout {'yes' if ok else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
