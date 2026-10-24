"""Computer measurements of a build against its photo: repeatable where the vision model's
judgement is not, though only as good as the photo's segmentation (`segment`) and the camera
angle (MapAnything's, its left / right from `vision.locate`).

Each side is cut to its frame first (the photo's building by `segment.main_building`, the
build by what its render draws) and the measures compare shapes, not sizes:

- outline: the overlap (IoU) of the two silhouettes fitted into the same square, the photo's
  occluded pixels (trees, cars in front) left out of both;
- proportions: the log ratio of the two frames' height / width;
- roof line: the top edge's height across the building (each frame stretched to a square),
  the mean difference, columns whose top a tree hides left out;
- windows and doors: the share of window area in each cell of a 6 x 6 grid over the frame
  (the photo's from the vision model's boxes), the difference of the two distributions, and
  how far apart the doors are across the front;
- colours: the walls' and the top's median colour (CIE76 delta E).

`photo_facts` measures a photo once; `build_facts` measures each candidate build from its
renders; `compare` scores one against the other (1 = the same, 0 = far off).
"""
from __future__ import annotations

import math

import numpy as np

N = 96                   # the square the outlines are fitted into
G = 6                    # the window grid
# the renderer darkens the walls it draws (side faces x 0.635 on average, BuildDataCollector's
# thumbs.SHADE_SIDE_BASE): undone before a build's colours are compared with a photo's
RENDER_GAIN = 1 / 0.65
WEIGHTS = {"outline": 2.0, "proportions": 1.0, "roof_line": 1.0, "windows": 0.5, "doors": 0.25, "colour": 0.5}


def _resize(a: np.ndarray, w: int, h: int, nearest: bool = True) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(a.astype(np.uint8) * (255 if a.dtype == bool else 1))
    im = im.resize((max(1, w), max(1, h)), Image.NEAREST if nearest else Image.BILINEAR)
    out = np.array(im)
    return out > 127 if a.dtype == bool else out


def fit_square(a: np.ndarray) -> np.ndarray:
    """A frame-cropped boolean mask scaled so its longer side is N, bottom-aligned and
    centred in an N x N square (the outline measure)."""
    h, w = a.shape
    s = N / max(h, w)
    b = _resize(a, round(w * s), round(h * s))
    out = np.zeros((N, N), bool)
    ox = (N - b.shape[1]) // 2
    out[N - b.shape[0]:, ox:ox + b.shape[1]] = b
    return out


def roof_line(a: np.ndarray, hidden: np.ndarray | None = None) -> np.ndarray:
    """(N,) heights (0 = bottom, 1 = top of the frame) of a frame-cropped mask's top edge,
    the frame stretched to N x N; NaN where nothing is drawn, or where the column is hidden
    from its top to its foot. A tree just above the roof is not taken as hiding it: trees
    behind a roof are far more common in the photos than trees in front of one."""
    m = _resize(a, N, N)
    hid = _resize(hidden, N, N) if hidden is not None else np.zeros_like(m)
    out = np.full(N, np.nan)
    for x in range(N):
        ys = np.flatnonzero(m[:, x])
        if len(ys) == 0 or hid[ys[0]:, x].mean() > 0.6:
            continue
        out[x] = 1.0 - ys[0] / N
    return out


def window_grid(win: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """(G, G) share of the building's pixels in each cell that are windows; None without any."""
    if not win.any():
        return None
    h, w = mask.shape
    g = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            ys, ye = i * h // G, (i + 1) * h // G
            xs, xe = j * w // G, (j + 1) * w // G
            area = max(1, mask[ys:ye, xs:xe].sum())
            g[i, j] = win[ys:ye, xs:xe].sum() / area
    return g


def _crop(a, frame):
    x0, y0, x1, y1 = frame
    return a[y0:y1, x0:x1]


def frame_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def facts(rgb: np.ndarray, mask: np.ndarray, frame, hidden=None, win=None, door=None) -> dict:
    """The measures of one side (photo or build), all in its own frame."""
    from .segment import zone_colours
    m = _crop(mask, frame)
    hid = _crop(hidden, frame) if hidden is not None else None
    out = {"frame": [int(v) for v in frame], "aspect": float(m.shape[0] / max(1, m.shape[1])),
           "square": fit_square(m), "hidden": fit_square(hid) if hid is not None else None,
           "roof_line": roof_line(m, hid)}
    if win is not None:
        out["windows"] = window_grid(_crop(win, frame), m | _crop(win, frame))
    if door is not None and door.any():
        xs = np.nonzero(_crop(door, frame))[1]
        out["door_x"] = float(xs.mean() / max(1, m.shape[1])) if len(xs) else None
    out["colours"] = zone_colours(rgb, mask, frame, skip=win)
    return out


def photo_facts(rgb: np.ndarray, seg: dict, openings: list | None = None) -> dict:
    """The photo's measures: `seg` from `segment.main_building`, `openings` the vision
    model's window / door boxes (0-1000 coordinates) or None."""
    from .segment import box_pixels
    h, w = seg["mask"].shape
    win = door = None
    if openings:
        win, door = np.zeros((h, w), bool), np.zeros((h, w), bool)
        for o in openings:
            x0, y0, x1, y1 = box_pixels(o["box"], w, h)
            (door if o.get("label") == "door" else win)[y0:y1, x0:x1] = True
        win &= seg["mask"] | seg["occluders"]
    return facts(rgb, seg["mask"], seg["frame"], seg["occluders"], win, door)


def build_facts(rgba: np.ndarray, labels_rgb: np.ndarray | None = None) -> dict | None:
    """A build's measures from its render (RGBA, transparent background) and, if given, the
    same view drawn with windows black and doors red (`render_labels`)."""
    mask = rgba[..., 3] > 0
    frame = frame_of(mask)
    if frame is None:
        return None
    win = door = None
    if labels_rgb is not None:
        r, g, b = (labels_rgb[..., k].astype(int) for k in range(3))
        win = mask & (np.maximum(np.maximum(r, g), b) < 70)
        door = mask & (r > g + 60) & (r > b + 60)
    rgb = np.clip(rgba[..., :3].astype(float) * RENDER_GAIN, 0, 255)
    return facts(rgb, mask, frame, None, win, door)


def compare(photo: dict, build: dict) -> dict:
    """{measure: (score 0-1, raw value)} and "total": the weighted mean of the measures both
    sides have."""
    out = {}
    p, b = photo["square"], build["square"]
    keep = ~photo["hidden"] if photo.get("hidden") is not None else np.ones_like(p)
    inter, union = (p & b & keep).sum(), ((p | b) & keep).sum()
    iou = float(inter / max(1, union))
    out["outline"] = (iou, iou)
    # proportions and roof line fall off smoothly, never flat at 0: scored linearly, a build 6
    # times too tall scored like one 2 times too tall, and no step towards the photo counted
    err = abs(math.log(max(build["aspect"], 1e-3) / max(photo["aspect"], 1e-3)))
    out["proportions"] = (math.exp(-err / 0.5), err)
    pl, bl = photo["roof_line"], build["roof_line"]
    ok = ~np.isnan(pl) & ~np.isnan(bl)
    if ok.sum() >= N // 4:
        e = float(np.abs(pl[ok] - bl[ok]).mean())
        out["roof_line"] = (math.exp(-e / 0.15), e)
    pw, bw = photo.get("windows"), build.get("windows")
    if pw is not None and bw is not None:
        e = float(0.5 * np.abs(pw / pw.sum() - bw / bw.sum()).sum())
        out["windows"] = (1.0 - e, e)
    if photo.get("door_x") is not None and build.get("door_x") is not None:
        e = abs(photo["door_x"] - build["door_x"])
        out["doors"] = (max(0.0, 1.0 - 2 * e), e)
    from .segment import palette_distance
    pc, bc = photo.get("colours", {}), build.get("colours", {})
    des = [palette_distance(pc[k], bc[k]) for k in ("wall", "top") if k in pc and k in bc]
    if des:
        e = float(np.mean(des))
        out["colour"] = (max(0.0, 1.0 - e / 40.0), e)
    wsum = sum(WEIGHTS[k] for k in out)
    out["total"] = (sum(WEIGHTS[k] * v[0] for k, v in out.items()) / wsum, None)
    return out


def mirrored_outline(photo: dict, build: dict) -> float:
    """The outline score with the build flipped left-right: well above the plain one, the
    build has its parts the wrong way round."""
    flipped = dict(build, square=build["square"][:, ::-1])
    return compare(photo, flipped)["outline"][0]


def summary(scores: dict) -> dict:
    """The scores as plain JSON numbers."""
    return {k: round(v[0], 3) for k, v in scores.items()}


def diagnose(photo: dict, build: dict) -> dict:
    """Which way each measure is off, for the evaluator's corrections: "taller" (the build
    is taller for its width than the photo), "wider", "roof_higher" (the build's roof line
    rises further above its eaves than the photo's), per thirds of the front, window share
    "more" / "fewer", and the door's side."""
    out = {}
    r = math.log(max(build["aspect"], 1e-3) / max(photo["aspect"], 1e-3))
    out["aspect_ratio"] = round(math.exp(r), 3)        # the build's height / width over the photo's
    if abs(r) > 0.12:
        out["shape"] = "taller" if r > 0 else "wider"
    pl, bl = photo["roof_line"], build["roof_line"]
    thirds = []
    for k in range(3):
        a, b = k * N // 3, (k + 1) * N // 3
        ok = ~np.isnan(pl[a:b]) & ~np.isnan(bl[a:b])
        if ok.sum() < 4:
            thirds.append(None)
            continue
        # the rise of the roof line above its lowest point in this third, build minus photo
        d = (bl[a:b][ok] - np.nanmin(bl)) .mean() - (pl[a:b][ok] - np.nanmin(pl)).mean()
        thirds.append("higher" if d > 0.08 else "lower" if d < -0.08 else "same")
    out["roof_thirds"] = thirds
    pw, bw = photo.get("windows"), build.get("windows")
    if pw is not None and bw is not None:
        ratio = (bw.mean() + 1e-3) / (pw.mean() + 1e-3)
        out["windows"] = "more" if ratio > 1.5 else "fewer" if ratio < 0.67 else "same"
    if photo.get("door_x") is not None and build.get("door_x") is not None:
        out["door"] = "left" if build["door_x"] < photo["door_x"] - 0.25 else (
            "right" if build["door_x"] > photo["door_x"] + 0.25 else "same")
    out["mirror_gain"] = round(mirrored_outline(photo, build) - compare(photo, build)["outline"][0], 3)
    from .segment import colour_word, palette_distance
    for zone, role in (("wall", "wall"), ("top", "roof")):
        pz, bz = photo.get("colours", {}).get(zone), build.get("colours", {}).get(zone)
        if pz and bz and palette_distance(pz, bz) > 15:     # the photo's main colour there, as a word
            out[f"{role}_colour"] = colour_word(pz["centres"][int(np.argmax(pz["weights"]))])
    return out
