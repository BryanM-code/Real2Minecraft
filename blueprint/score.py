"""Computer measures of a build against its photo: repeatable where the vision model's
judgement is not, though only as good as the photo's segmentation (`segment`), the vision
model's window and door boxes (`vision.openings`) and the camera angle.

Each side is cut to its frame first (the photo's building by `segment.main_building`, the
build by what its render draws) and positions are fractions of the frame, so the measures
compare shapes, not sizes:

- outline: the overlap (IoU) of the two silhouettes fitted into the same square, the photo's
  occluded pixels (trees, cars in front) left out of both, except above the building, where a
  tree is far more often behind the roof than in front of it (a roof built too tall counts);
- proportions: the log ratio of the two frames' height / width;
- roof line: the top edge's height across the building;
- sections: the roof line over each part of the build (its columns in the label view), so a
  wing too tall or with the wrong roof shows on its own;
- windows, doors, garage doors: each of the photo's matched to one of the build's of the same
  kind, by position; the score counts the matches and how close they are, so a missing, extra
  or misplaced opening costs. On the building when MapAnything placed the photo's (`match3d`:
  the same wall, how far along it and how high up it, `facade.fractions` for the build's), else
  in the frame (the build's from its label view: what each block is, `build`'s "labels");
- colours: the walls' and the top's main colours (`segment.palette_distance`);
- plan: the footprint's depth over its width against MapAnything's, when the photo shows a
  side wall well enough to measure the depth (an outline cannot tell a shallow house from a
  deep one).

`photo_facts` measures a photo once; `build_facts` measures a build from its render and its
label and part views; `compare` scores one against the other (1 = the same) and `diagnose`
says which way each part is off, for the evaluator's corrections.
"""
from __future__ import annotations

import math

import numpy as np

N = 96                   # the square the outlines are fitted into
# proportions at half weight: the outline already holds the frame's height / width, and at full
# weight a split-level house built level (taller, so closer in aspect) tied with the right one
WEIGHTS = {"outline": 2.0, "proportions": 0.5, "roof_line": 0.5, "sections": 1.0, "windows": 1.0, "doors": 0.5,
           "garages": 1.0, "colour": 0.5, "plan": 0.5}
# the renderer darkens the walls it draws (side faces x 0.635 on average, BuildDataCollector's
# thumbs.SHADE_SIDE_BASE): undone before a build's colours are compared with a photo's
RENDER_GAIN = 1 / 0.65
MATCH = 0.15             # farther apart than this (fraction of the frame), two openings do not match
OPENING_KINDS = ("window", "door", "garage")


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


def _crop(a, frame):
    x0, y0, x1, y1 = frame
    return a[y0:y1, x0:x1]


def frame_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _norm_box(box_px, frame) -> list:
    """A pixel box (x0, y0, x1, y1) as fractions of the frame."""
    x0, y0, x1, y1 = frame
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    return [(box_px[0] - x0) / w, (box_px[1] - y0) / h, (box_px[2] - x0) / w, (box_px[3] - y0) / h]


def boxes_of(mask: np.ndarray, frame, min_share: float = 0.0004, join: int = 0) -> list:
    """The boxes (fractions of the frame) of a mask's pieces; `join` pixels of horizontal
    dilation first merges a window's panes split by mullions."""
    from .segment import _components
    m = mask.copy()
    if join:
        for d in range(1, join + 1):
            m[:, d:] |= mask[:, :-d]
            m[:, :-d] |= mask[:, d:]
    comp = _components(m)
    out = []
    area = max(1, (frame[2] - frame[0]) * (frame[3] - frame[1]))
    for k in range(1, int(comp.max()) + 1):
        ys, xs = np.nonzero((comp == k) & mask)
        if len(xs) < max(4, min_share * area):
            continue
        out.append(_norm_box((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), frame))
    return out


def facts(rgb: np.ndarray, mask: np.ndarray, frame, hidden=None, skip=None) -> dict:
    """The outline, roof line and colour measures of one side (photo or build), in its frame."""
    from .segment import zone_colours
    m = _crop(mask, frame)
    hid = _crop(hidden, frame) if hidden is not None else None
    return {"frame": [int(v) for v in frame], "aspect": float(m.shape[0] / max(1, m.shape[1])),
            "square": fit_square(m), "hidden": fit_square(hid) if hid is not None else None,
            "roof_line": roof_line(m, hid), "colours": zone_colours(rgb, mask, frame, skip=skip)}


def photo_openings(openings: list, seg: dict) -> dict:
    """The vision model's boxes (0-1000) as {kind: [boxes as fractions of the frame]}; a
    "door" as wide as it is tall and big is a garage door (older answers had no "garage")."""
    from .segment import box_pixels
    h, w = seg["mask"].shape
    fx0, fy0, fx1, fy1 = seg["frame"]
    out = {k: [] for k in OPENING_KINDS}
    for o in openings or []:
        x0, y0, x1, y1 = box_pixels(o["box"], w, h)
        if x1 <= fx0 or x0 >= fx1 or y1 <= fy0 or y0 >= fy1:
            continue                                   # outside the building
        b = _norm_box((x0, y0, x1, y1), seg["frame"])
        label = str(o.get("label", "window")).lower()
        if "garage" in label or (label == "door" and (b[2] - b[0]) >= 0.9 * (b[3] - b[1]) and b[2] - b[0] > 0.06):
            label = "garage"
        out[label if label in OPENING_KINDS else "window"].append(b)
    return out


def photo_facts(rgb: np.ndarray, seg: dict, openings: list | None = None) -> dict:
    """The photo's measures: `seg` from `segment.main_building`, `openings` the vision
    model's window / door boxes (0-1000 coordinates) or None (not measured)."""
    from .segment import box_pixels
    skip = None
    if openings:
        h, w = seg["mask"].shape
        skip = np.zeros((h, w), bool)
        for o in openings:
            x0, y0, x1, y1 = box_pixels(o["box"], w, h)
            skip[y0:y1, x0:x1] = True
    out = facts(rgb, seg["mask"], seg["frame"], seg["occluders"], skip)
    out["openings"] = photo_openings(openings, seg) if openings else None
    return out


def build_facts(rgba: np.ndarray, labels_view: np.ndarray | None = None, parts_view: np.ndarray | None = None,
                n_parts: int = 0) -> dict | None:
    """A build's measures from its render (RGBA, transparent background), its label view
    (`render.render_codes` of `build`'s "labels": what each pixel shows) and its part view
    (which part of the build each pixel belongs to: 1 = the first part, ...)."""
    from .build import LABELS
    mask = rgba[..., 3] > 0
    frame = frame_of(mask)
    if frame is None:
        return None
    skip = None
    out_open = None
    if labels_view is not None:
        code = {n: k for k, n in enumerate(LABELS)}
        skip = np.isin(labels_view, [code["window"], code["skylight"]])
        width = frame[2] - frame[0]
        out_open = {"window": boxes_of(labels_view == code["window"], frame, join=max(1, round(width * 0.012))),
                    "door": boxes_of(labels_view == code["door"], frame),
                    "garage": boxes_of(labels_view == code["garage"], frame, join=1)}
    rgb = np.clip(rgba[..., :3].astype(float) * RENDER_GAIN, 0, 255)
    out = facts(rgb, mask, frame, None, skip)
    out["openings"] = out_open
    if parts_view is not None and n_parts:
        pv = _crop(parts_view, frame)
        cols = _resize((pv + 1).astype(np.uint8), N, N).astype(int) - 1     # nearest: the part codes survive
        owner = {}
        for x in range(N):                             # a column is the part highest in it
            hits = np.flatnonzero((cols[:, x] >= 1) & (cols[:, x] <= n_parts))
            if len(hits):
                owner.setdefault(int(cols[hits[0], x]), []).append(x)
        out["sections"] = {k: xs for k, xs in owner.items() if len(xs) >= 3}
    return out


def match(photo_boxes: list, build_boxes: list) -> tuple:
    """(score, pairs, missing, extra): photo boxes paired with build boxes, nearest centres
    first, none farther apart than MATCH; the score is 2 x the summed closeness of the pairs
    over the boxes on both sides (1: every box paired, in place)."""
    def centre(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    cand = []
    for i, p in enumerate(photo_boxes):
        for j, b in enumerate(build_boxes):
            (px, py), (bx, by) = centre(p), centre(b)
            d = math.hypot(px - bx, py - by)
            if d <= MATCH:
                cand.append((d, i, j))
    cand.sort()
    used_p, used_b, pairs, q = set(), set(), [], 0.0
    for d, i, j in cand:
        if i in used_p or j in used_b:
            continue
        used_p.add(i)
        used_b.add(j)
        pairs.append((i, j, d))
        q += 1.0 - d / MATCH
    n = len(photo_boxes) + len(build_boxes)
    missing = [photo_boxes[i] for i in range(len(photo_boxes)) if i not in used_p]
    extra = [build_boxes[j] for j in range(len(build_boxes)) if j not in used_b]
    return (2 * q / n if n else None), pairs, missing, extra


def match3d(photo_ops: list, build_ops: list) -> tuple:
    """`match` for openings placed on the building (`facade.photo_fractions` and
    `facade.fractions`): only on the same wall, by their along / up fractions."""
    cand = []
    for i, p in enumerate(photo_ops):
        for j, b in enumerate(build_ops):
            if p["wall"] != b["wall"]:
                continue
            d = math.hypot(p["along"] - b["along"], p["up"] - b["up"])
            if d <= MATCH:
                cand.append((d, i, j))
    cand.sort()
    used_p, used_b, pairs, q = set(), set(), [], 0.0
    for d, i, j in cand:
        if i in used_p or j in used_b:
            continue
        used_p.add(i)
        used_b.add(j)
        pairs.append((i, j, d))
        q += 1.0 - d / MATCH
    n = len(photo_ops) + len(build_ops)
    missing = [photo_ops[i] for i in range(len(photo_ops)) if i not in used_p]
    extra = [build_ops[j] for j in range(len(build_ops)) if j not in used_b]
    return (2 * q / n if n else None), pairs, missing, extra


def _openings_by_kind(photo: dict, build: dict):
    """{kind: (photo's, build's)} on the building (3D: `openings3d`) when both sides have them,
    else in the frame (`openings`); None when either side has neither."""
    p3, b3 = photo.get("openings3d"), build.get("openings3d")
    if p3 is not None and b3 is not None:
        return {k: ([o for o in p3 if o["kind"] == k], [o for o in b3 if o["kind"] == k]) for k in OPENING_KINDS}, True
    po, bo = photo.get("openings"), build.get("openings")
    if po is not None and bo is not None:
        return {k: (po[k], bo[k]) for k in OPENING_KINDS}, False
    return None, False


def compare(photo: dict, build: dict) -> dict:
    """{measure: (score 0-1, raw value)} and "total": the weighted mean of the measures both
    sides have."""
    out = {}
    p, b = photo["square"], build["square"]
    keep = _kept(photo)
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
    secs = build.get("sections") or {}
    errs = []
    for xs in secs.values():
        xs = [x for x in xs if not np.isnan(pl[x]) and not np.isnan(bl[x])]
        if len(xs) >= 3:
            errs.append(float(np.abs(pl[xs] - bl[xs]).mean()))
    if errs:
        e = float(np.mean(errs))
        out["sections"] = (math.exp(-e / 0.12), e)
    ops, on_building = _openings_by_kind(photo, build)
    if ops is not None:
        for kind, key in (("window", "windows"), ("door", "doors"), ("garage", "garages")):
            pk, bk = ops[kind]
            if not pk and not bk:
                continue                               # neither has any: nothing to score
            s, pairs, missing, extra = (match3d if on_building else match)(pk, bk)
            out[key] = (s, {"photo": len(pk), "build": len(bk), "matched": len(pairs)})
    if photo.get("plan") and build.get("plan"):
        e = abs(math.log(max(build["plan"], 1e-3) / max(photo["plan"], 1e-3)))
        out["plan"] = (math.exp(-e / 0.4), e)
    from .segment import palette_distance
    pc, bc = photo.get("colours", {}), build.get("colours", {})
    des = [palette_distance(pc[k], bc[k]) for k in ("wall", "top") if k in pc and k in bc]
    if des:
        e = float(np.mean(des))
        out["colour"] = (max(0.0, 1.0 - e / 40.0), e)
    wsum = sum(WEIGHTS[k] for k in out)
    out["total"] = (sum(WEIGHTS[k] * v[0] for k, v in out.items()) / wsum, None)
    return out


def _kept(photo: dict) -> np.ndarray:
    """The square's pixels the outline is scored on: all but the photo's hidden ones (trees,
    cars in front), and of those, the ones above the photo's building in their column count
    again: a tree there is far more often behind the roof than in front of it (as the roof
    line takes it), so a build that rises into it is too tall, not hidden."""
    p = photo["square"]
    if photo.get("hidden") is None:
        return np.ones_like(p)
    hid = photo["hidden"].copy()
    rows = np.arange(p.shape[0])[:, None]
    has = p.any(0)
    top = np.where(has, p.argmax(0), p.shape[0])           # the first building row in each column
    hid &= ~((rows < top[None, :]) & has[None, :])
    return ~hid


def mirrored_outline(photo: dict, build: dict) -> float:
    """The outline score with the build flipped left-right: well above the plain one, the
    build has its parts the wrong way round."""
    flipped = dict(build, square=build["square"][:, ::-1])
    p, b = photo["square"], flipped["square"]
    keep = _kept(photo)
    return float((p & b & keep).sum() / max(1, ((p | b) & keep).sum()))


def summary(scores: dict) -> dict:
    """The scores as plain JSON numbers."""
    return {k: round(v[0], 3) for k, v in scores.items()}


def diagnose(photo: dict, build: dict) -> dict:
    """Which way each measure is off, for the evaluator's corrections: the shape ("taller":
    the build is taller for its width than the photo), the roof line per third of the front
    and per part ("sections": {part: (height difference, shape difference)}), the openings
    per kind (counts, and the photo's ones the build lacks with the part they fall on), the
    mirror gain, and the photo's main wall and roof colours where the build's differ."""
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
        d = (bl[a:b][ok] - np.nanmin(bl)).mean() - (pl[a:b][ok] - np.nanmin(pl)).mean()
        thirds.append("higher" if d > 0.08 else "lower" if d < -0.08 else "same")
    out["roof_thirds"] = thirds
    secs = {}
    for k, xs in (build.get("sections") or {}).items():
        xs = [x for x in xs if not np.isnan(pl[x]) and not np.isnan(bl[x])]
        if len(xs) < 3:
            continue
        d = bl[xs] - pl[xs]
        # height: the mean difference; shape: how much more the build's line rises and falls
        secs[k] = (round(float(d.mean()), 3), round(float(np.ptp(bl[xs]) - np.ptp(pl[xs])), 3))
    out["sections"] = secs
    by_kind, on_building = _openings_by_kind(photo, build)
    if by_kind is not None:
        ops = {}
        for kind in OPENING_KINDS:
            pk, bk = by_kind[kind]
            s, pairs, missing, extra = (match3d if on_building else match)(pk, bk)
            ops[kind] = {"photo": len(pk), "build": len(bk), "missing": missing, "extra": extra}
        out["openings"] = ops
        out["openings_on_building"] = on_building
    if photo.get("plan") and build.get("plan"):
        p = build["plan"] / max(photo["plan"], 1e-3)
        out["plan_ratio"] = round(p, 3)
        if abs(math.log(max(p, 1e-3))) > 0.22:
            out["plan"] = "deeper" if p > 1 else "shallower"
    out["mirror_gain"] = round(mirrored_outline(photo, build) - compare(photo, build)["outline"][0], 3)
    from .segment import colour_word, palette_distance
    for zone, role in (("wall", "wall"), ("top", "roof")):
        pz, bz = photo.get("colours", {}).get(zone), build.get("colours", {}).get(zone)
        if pz and bz and palette_distance(pz, bz) > 15:     # the photo's main colour there, as a word
            out[f"{role}_colour"] = colour_word(pz["centres"][int(np.argmax(pz["weights"]))])
    return out
