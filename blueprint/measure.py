"""A building's size from its photo, and where its windows and doors are on its walls.

MapAnything (`facebook/map-anything-apache`, Apache 2.0) turns the photo into a 3D point per
pixel, in metres. The building's points (its pixels from `refine prepare`'s segmentation)
are levelled and turned so the walls run along the axes; their extent along the front is the
width, away from it the depth, and upwards the height. Each window and door box the vision
model found is placed on its wall (`opening_positions`).

From the repository's root, in the measuring environment (MapAnything), after `refine prepare`
has run into the same --out:

    r2m_measure\\Scripts\\python -m blueprint.measure photos... --out DIR

One photo gives the scale only roughly, and only the depth it shows; the proportions
(height / width) are the reliable part. Writes <out>/<id>.measure.json per photo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MAPANYTHING = "facebook/map-anything-apache"
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

# ── MapAnything ───────────────────────────────────────────────────────────────
def load_model(device="cuda"):
    from mapanything.models import MapAnything
    return MapAnything.from_pretrained(MAPANYTHING).to(device).eval()


def reconstruct(model, image_paths: list) -> list:
    """Per photo: {"pts" (H, W, 3) world, "ray" (H, W, 3) camera-frame unit rays, "cam2world"
    (4, 4) OpenCV, "mask" (H, W) bool, "rgb" (H, W, 3) uint8} at the model's resolution."""
    import torch
    from mapanything.utils.image import load_images
    views = load_images([str(p) for p in image_paths])
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


# ── Levelling and turning the points onto the walls ───────────────────────────
def map_normals(points: np.ndarray, valid: np.ndarray) -> tuple:
    """Normals of a point map from its pixel neighbours: (normals (H-2, W-2, 3), valid)."""
    du = points[1:-1, 2:] - points[1:-1, :-2]
    dv = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    ok = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & (norm > 0)
    return n / np.maximum(norm, 1e-12)[..., None], ok


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


def level_frame(views: list, normals: np.ndarray, max_tilt: float = 15.0) -> tuple:
    """(R, levelling in degrees, share of normals on walls). R's rows are the x, up and z axes
    in the reconstruction's frame. Up: perpendicular to every camera's right axis (a hand-held
    camera is level), for a single photo then levelled on its surfaces (walls upright, roofs and
    ground along); x and z: along the walls."""
    up_cam = up_from_cameras(views)
    up = level_up(normals, up_cam, max_tilt) if len(views) == 1 else up_cam
    x_axis, z_axis, wall_share = wall_axes(normals, up)
    R = np.stack([x_axis, up, z_axis])                         # rows: grid axes in world coordinates
    tilt = float(np.degrees(np.arccos(np.clip(up @ up_cam, -1, 1))))
    return R, tilt, wall_share


def measure_views(views: list, masks: list, openings: list | None = None) -> dict:
    """Width / depth / height (reconstruction units: metres without hints) of the masked
    points, the facade being the wall axis that faces the first camera; with `openings` (the
    photo's boxes) also where each is on the building (`opening_positions`)."""
    P, N = fused_points(views, masks)
    R, levelling_deg, wall_share = level_frame(views, N)
    q = P @ R.T
    lo, hi = np.percentile(q, LO, axis=0), np.percentile(q, HI, axis=0)
    cam = np.asarray(views[0]["cam2world"])[:3, 3] @ R.T
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
               abs(d[side]), abs(d[front])))), "levelling_deg": levelling_deg,
           "wall_share": wall_share, "points": int(len(P))}
    if openings:
        rec["openings"] = opening_positions(views[0], np.asarray(masks[0], bool), R, openings)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", type=Path, help="photos, or folders of <id>/photo.jpg")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    from PIL import Image
    from .refine import photos_of, stem_of
    args.out.mkdir(parents=True, exist_ok=True)
    prepared = []
    for p in photos_of(args.photos):
        stem = stem_of(p)
        if (args.out / f"{stem}.measure.json").is_file():
            continue
        work = args.out / "_work" / stem
        if not ((work / "seg.json").is_file() and (work / "mask.png").is_file()):
            print(f"{stem}: no segmentation in {work}: run `blueprint.refine prepare` first", file=sys.stderr)
            continue
        mask = np.array(Image.open(work / "mask.png")) > 127          # the main building only
        ok = bool(json.loads((work / "seg.json").read_text())["ok"])
        prepared.append((stem, work / "white.png", mask, ok))
    if not prepared:
        return 0
    model = load_model()
    for stem, white, a, ok in prepared:
        t = time.time()
        views = reconstruct(model, [white])
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
