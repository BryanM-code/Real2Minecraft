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


def opening_positions(view: dict, mask: np.ndarray, R: np.ndarray, openings: list) -> list:
    """Each of the photo's window, door and garage door boxes (0-1000 coordinates) placed on
    the building, from MapAnything's points of the box's building pixels in the levelled frame
    turned to the walls: the wall it is on ("front", or the "left" / "right" side as seen from
    the camera), how far along it ("along": across the front from its left end, or along a side
    from its front corner) and how high ("up", from the building's foot to its top), its width
    and height ("w", "h"), as fractions of the whole building; and the same against its own
    wall ("along_wall", "up_wall", "w_wall", "h_wall"): along the wall's plane as far as the
    photo shows it, and up from the wall's foot (its lowest point anywhere along it: bushes
    hide the foot of many walls in places) to its top: on a side wall the eave (the lower of
    its two ends' tops: a gable end's gable or a hip roof does not move its openings), across
    the front the top of the opening's column (the eave, or a gable facing the front; the
    eave of the front's ends failed where two parts' fronts share a plane: the lower part's
    eave put the taller one's upper windows above it). Also
    how far behind the frontmost wall its wall is ("into", a fraction of the depth) and its
    size in metres (rough from one photo; its width over its height is the reliable part).
    Fractions, so neither the photo's perspective nor MapAnything's rough scale matters."""
    pts = np.asarray(view["pts"], np.float64)
    h, w = pts.shape[:2]
    q = pts @ R.T
    m = mask & np.asarray(view["mask"], bool)
    if m.sum() < 50:
        return []
    allq = q[m]
    lo, hi = np.percentile(allq, LO, axis=0), np.percentile(allq, HI, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    cam = np.asarray(view["cam2world"])[:3, 3] @ R.T
    d = cam - (lo + hi) / 2.0
    front = 0 if abs(d[2]) >= abs(d[0]) else 2         # the facade runs along this axis
    depth = 2 if front == 0 else 0                     # the camera looks along this one
    face = hi[depth] if d[depth] > 0 else lo[depth]    # the facade's plane
    inward = -1.0 if d[depth] > 0 else 1.0
    cols = np.nonzero(m)[1].astype(float)
    right = np.corrcoef(cols, allq[:, front])[0, 1] >= 0     # "along" grows to the camera's right
    tol = max(0.25, 0.04 * float(span.max()))          # a wall's plane: points this close to it
    out = []
    for o in openings:
        x0, y0, x1, y1 = (int(round(float(v) / 1000.0 * s)) for v, s in zip(o["box"], (w, h, w, h)))
        sel = np.zeros((h, w), bool)
        sel[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        pick = sel & m
        if pick.sum() < 6:                             # a window seen through a tree: its pixels anyway
            pick = sel & np.asarray(view["mask"], bool)
        if pick.sum() < 6:
            continue
        P = q[pick]
        c = np.median(P, axis=0)
        into = float((c[depth] - face) * inward) / span[depth]
        a = float((c[front] - lo[front]) / span[front])
        a = a if right else 1.0 - a
        lo_p, hi_p = np.percentile(P, 10, axis=0), np.percentile(P, 90, axis=0)
        side = into > 0.2 and (a < 0.12 or a > 0.88)   # deep behind the facade, at one of its ends: a side wall
        normal, tangent = (front, depth) if side else (depth, front)
        plane = np.abs(allq[:, normal] - c[normal]) < tol
        half = max(tol, (hi_p[tangent] - lo_p[tangent]) / 2.0)
        column = plane & (np.abs(allq[:, tangent] - c[tangent]) < half)
        foot, top = lo[1], hi[1]
        if plane.sum() >= 30:
            foot = np.percentile(allq[plane, 1], LO)
        if column.sum() >= 30:
            top = np.percentile(allq[column, 1], HI)
        if side and plane.sum() >= 30:                 # a side wall: up to its eave, at its two ends
            t = allq[plane, tangent]
            t0, t1 = np.percentile(t, LO), np.percentile(t, HI)
            ends = [plane & (np.abs(allq[:, tangent] - e) < max(tol, 0.08 * (t1 - t0))) for e in (t0, t1)]
            tops = [np.percentile(allq[e, 1], 90) for e in ends if e.sum() >= 10]
            if tops:
                top = min(tops)
        tall = max(float(top - foot), 1e-6)
        if side:                                       # along a side wall: from its front corner inwards
            v, vc = (allq[plane, depth] - face) * inward, (c[depth] - face) * inward
        else:                                          # across the front: from its left end as seen
            v, vc = (allq[:, front], c[front]) if right else (-allq[:, front], -c[front])
            v = v[plane]
        v0, v1 = (np.percentile(v, LO), np.percentile(v, HI)) if len(v) >= 30 else (vc - 1e-6, vc + 1e-6)
        long_ = max(float(v1 - v0), 1e-6)
        wm, hm = float(hi_p[tangent] - lo_p[tangent]), float(hi_p[1] - lo_p[1])
        rec = {"label": o.get("label"), "wall": ("right" if a > 0.5 else "left") if side else "front",
               "along": round(min(1.0, max(0.0, into if side else a)), 3),
               "up": round(float((c[1] - lo[1]) / span[1]), 3),
               "w": round(wm / span[tangent], 3), "h": round(hm / span[1], 3),
               "along_wall": round(min(1.0, max(0.0, float(vc - v0) / long_)), 3),
               "up_wall": round(min(1.6, max(0.0, float(c[1] - foot) / tall)), 3),
               "w_wall": round(min(1.0, wm / long_), 3), "h_wall": round(min(1.6, hm / tall), 3),
               "into": round(max(0.0, into), 3), "w_m": round(wm, 2), "h_m": round(hm, 2)}
        out.append(rec)
    return out


def measure_views(views: list, masks: list, openings: list | None = None) -> dict:
    """Width / depth / height (reconstruction units: metres without hints) of the masked
    points, the facade being the wall axis that faces the first camera; with `openings` (the
    photo's boxes) also where each is on the building (`opening_positions`)."""
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
    rec = {"width_m": float(ext[front]), "depth_m": float(ext[side]), "height_m": float(ext[1]),
           "camera_yaw_deg": yaw, "camera_pitch_deg": pitch,
           "distance_m": float(np.hypot(d[0], d[2])), "corner_view_deg": float(np.degrees(np.arctan2(
               abs(d[side]), abs(d[front])))), "levelling_deg": info["levelling_deg"],
           "wall_share": info["wall_share"], "points": int(len(P))}
    if openings:
        rec["openings"] = opening_positions(views[0], np.asarray(masks[0], bool), sim["R"], openings)
    return rec


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
        facts = args.out / f"{stem}.photo.json"       # `refine prepare`'s window and door boxes
        openings = json.loads(facts.read_text(encoding="utf-8")).get("openings") if facts.is_file() else None
        rec = {"id": stem, "cutout": ok, "mask_share": float(a.mean()), **measure_views(views, [mask], openings),
               "seconds": round(time.time() - t, 1)}
        (args.out / f"{stem}.measure.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{stem}: {rec['width_m']:.1f} x {rec['depth_m']:.1f} x {rec['height_m']:.1f} m "
              f"(w x d x h), cutout {'yes' if ok else 'no'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
