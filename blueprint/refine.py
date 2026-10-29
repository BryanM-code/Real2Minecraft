"""The evaluator: a build is checked against its photo and corrected, one change at a time,
until it is close enough or nothing more helps.

Every build the loop measures has the photo's windows, doors and garage doors where the photo
has them (`facade.place_openings_3d`, from MapAnything's placing of the vision model's boxes),
and lists what its form asked that it could not build ("conflicts").

Two evaluations, combined (`combined`, 0-1):
- the computer measures (`score`): outline, proportions, roof line, each part's roof line,
  windows, doors and garage doors placed on the building, colours. Repeatable, but only as
  good as the segmentation and the camera angle;
- the vision model (`vision`): it looks at a render of the build beside the photo, says what
  is wrong in the builder's own terms (style, kind, proportions, each part's size, level, roof
  and garage doors, missing or extra features, mirrored, windows, colours) and how alike they
  are (0-10).

Before the loop, `prepare` asks each part's roof its concrete questions (`roof_pass`; `roofs`
adds them to older readings). The loop changes no pitch: on real photos neither the measures nor
the critique told a right pitch from a wrong one.

First, the photo-only check (`check_round`): the vision model answers about the photo alone,
for each of the build's parts (whether the building has it, its roof's ends and ridge, its
floor against the main block's, its storeys, its garage doors, its base band, which part is
tallest), twice with the choices reversed, and the code compares the answers both give with
the build; each correction they ask for is kept unless the measures find it much worse. (Shown
the photo and a render together, the model described the render: on a house built with ten
deliberate mistakes, the 9B and 27B models rated 8 of them 9 of 10, like the right build, and
named 3 and 4. The measures tell right from wrong on renders standing in for photos, but on
real photos hardly: see CHECK_GAIN.)

Then each round every correction the critique asks for, every one the measures point to and
every one that would let the build do what its form asked (its conflicts: a part too narrow
for its windows) (`edits`) is built and measured (a build and its renders take about a
second). Three are tried in turn, the model's own first; a conflict's is kept unless the
measures find it worse. A change of proportions the measures found is kept on
the measures; any other is checked by critiquing the candidate build on its own (`_verify`):
kept when the model's complaint is gone, nothing in the new critique asks to undo it and its
rating does not drop. (Asked instead which of two builds is closer, the 9B model picked the
image shown last 59 times in 70, so `compare` decides nothing here.) A correction the next
critique contradicts is undone. The loop stops at the target score, after `rounds` rounds,
or when no correction helps, and returns the best build it rated.

Corrections only exist in the builder's terms, so the loop cannot ask for what the builder
cannot draw. What it could not do is logged per photo: what the model asked for outside the
list ("gaps"), corrections that hit a limit (no roof steeper than 72 degrees: "limits") and
corrections that changed no block ("no_effect"); `summary` counts them over a run, which
says what to build next.

Stages, from this repository's root, MapAnything (r2m_train) measuring in between:

    r2m_vlm\\Scripts\\python -m blueprint.refine prepare photos... --out DIR [--model qwen3.8-27b]
    r2m_vlm\\Scripts\\python -m blueprint.refine roofs photos... --out DIR   (the roof pass on older readings)
    r2m_train\\Scripts\\python -m blueprint.measure photos... --out DIR
    r2m_vlm\\Scripts\\python -m blueprint.refine run photos... --out DIR [--rounds 4] [--model qwen3.8-27b]
    r2m_vlm\\Scripts\\python -m blueprint.refine judge photos... --out DIR
    r2m_vlm\\Scripts\\python -m blueprint.refine summary --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

TARGET = 0.8            # a combined score this good ends the loop, with the measures alone at least
MEASURED_TARGET = 0.7   # this (the model's resemblance is lenient: 9 of 10 for a first build measured 0.49)
SHORTLIST = 3           # corrections the vision model judges per round, at most
# the measured gain that keeps a correction the measures found (and the model does not
# object to): low for a change of proportions (what the measures judge best), high for a
# change of what the building is (a feature, the roof's type: what they cannot see), between
# for the rest; a storey, the style, the kind or a mirror image of theirs only when the model
# rates the result higher (a flip the measures wanted took a chalet's rating from 6 to 2)
SHAPE_GAIN, CLEAR_GAIN, MEANING_GAIN = 0.02, 0.05, 0.08
SHAPE_OPS = ("width", "height", "depth", "footprint", "share", "part_height", "part_depth", "part_level", "part_forward")
MEANING_OPS = ("roof", "feature", "add_part", "remove_part", "units", "garage_wall", "skylights_on", "window_style",
               "wall_material", "garage_doors")
MODEL_ONLY = ("storeys", "style", "kind", "mirror")
# what a building is shaped like, which the measures do see: the critique's corrections of these
# need the measures not to find them worse too (on its own, the critique of a candidate confirms
# little: shown a render, the model describes it)
STRUCTURE_OPS = SHAPE_OPS + ("roof", "pitch", "ridge", "storeys", "mirror", "add_part", "remove_part", "garage_doors")
WORSE = -0.03           # a correction the measures find this much worse is not shown
# a correction the photo-only check asks for (`vision.check`, asked twice, its answers kept where
# both agree) is kept unless the measures find it this much worse. On builds rendered as their
# own photos (`calibrate synthetic`, 50 buildings of 21 kinds) the measures told right corrections
# from wrong ones well; on the 50 real photos, measured one by one from round 3's builds, hardly
# (losing at most 0.005 they kept 10 of 16 right corrections and 69 of 117 wrong ones; at most
# 0.05, 14 and 96): a stricter rule would only drop right corrections, the answers' own accuracy
# decides. A drastic one (a part removed, the build mirrored) must not lose more than STRICT_GAIN
CHECK_GAIN = -0.05
STRICT_GAIN = -0.005
# ... and a part's level the photo's answer and its doors agree on (`facade.levels`: two signals
# from the photo) unless they find it this much worse: on real photos a silhouette judges levels
# worst (bushes hide the walls' feet, the ground slopes, the photo's perspective is not the
# render's; a split-level house's right level measured 0.065 worse), against a camera error
# alone (right level changes measured no worse in 96% of the buildings above)
DOORS_GAIN = -0.1
# kinds whose windows and doors the vision model can box one by one (elsewhere they are too
# many, or not windows: on a 9-storey pagoda it drew a grid that is not there, in 80 s)
BOXED_KINDS = ("house", "mansion", "cottage", "townhouse", "shop", "restaurant", "barn", "gas_station", "warehouse",
               "factory", "greenhouse", "other")


def stem_of(p: Path) -> str:
    return p.parent.name if p.name == "photo.jpg" else p.stem


def photos_of(paths) -> list:
    """Photos, a folder holding photo.jpg, or a folder of <id>/photo.jpg folders."""
    out = []
    for p in paths:
        if p.is_dir():
            out += [p / "photo.jpg"] if (p / "photo.jpg").is_file() else sorted(p.glob("*/photo.jpg"))
        else:
            out.append(p)
    return out


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --- prepare: the photo's reading, its building's pixels, windows and doors --------------------

# what the roof pass shows of each part (`vision.part_view`). The whole photo: on the 50 test
# photos a close-up of each part read the main blocks' front tops 38 of 48 (the photo 37) and
# their pitch 13 of 21 (14), and the parts it read differently no better (10 of 18 tops, 2 of 8
# pitches against 10 and 3); a red box round the part, 38 and 14, 11 and 5, but more gables rising
# from an eave that is not there; the builds came out the same (roofs 138, 137, 137 of 157)
ROOF_VIEW = "full"


def prepare(path: Path, out: Path, vlm, seg, roof_view: str = ROOF_VIEW) -> dict:
    """The local model's reading (`describe.run_read`), the main building's box and visible
    side, its windows and doors (BOXED_KINDS only), and its pixels;
    saved as <stem>.read.json, <stem>.photo.json and _work/<stem>/ (photo, masks, white.png
    for MapAnything)."""
    from PIL import Image

    from . import segment, vision
    from .compose import kind_of
    from .describe import photo, run_read
    stem = stem_of(path)
    work = out / "_work" / stem
    work.mkdir(parents=True, exist_ok=True)
    image = photo(path)
    read = _json(out / f"{stem}.read.json")
    if read is None:
        read = {"photo": str(path), "mode": "read", **run_read(*vlm, image)}
        (out / f"{stem}.read.json").write_text(json.dumps(read, indent=1, ensure_ascii=False), encoding="utf-8")
    t = time.time()
    loc = vision.locate(vlm, image)
    ops = {"boxes": [], "complete": False, "text": ""}
    if kind_of(read.get("kind")) in BOXED_KINDS:
        ops = vision.openings(vlm, image, 1000)
    lab = segment.labels(seg, image)
    sb = segment.main_building(lab, loc["box"])
    rgb = np.array(image)
    image.save(work / "photo.png")
    Image.fromarray(lab.astype(np.uint8)).save(work / "labels.png")
    Image.fromarray((sb["mask"] * 255).astype(np.uint8)).save(work / "mask.png")
    Image.fromarray((sb["occluders"] * 255).astype(np.uint8)).save(work / "occluders.png")
    Image.fromarray(np.where(sb["mask"][..., None], rgb, 255).astype(np.uint8)).save(work / "white.png")
    info = {k: sb.get(k) for k in ("box", "frame", "share", "ok", "occluded")}
    (work / "seg.json").write_text(json.dumps(info), encoding="utf-8")
    facts = {"id": stem, "photo": str(path), "size": [image.width, image.height], "locate": loc, **info,
             "openings": ops["boxes"], "openings_complete": ops["complete"], "openings_text": ops["text"][-300:],
             "seconds": round(time.time() - t, 1)}
    (out / f"{stem}.photo.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    if "roofs" not in read:                            # after the segmentation: a part's view needs its pixels
        add_roofs(read, vlm, image, out / f"{stem}.read.json", sb["mask"], roof_view)
    return facts


# --- run: the loop ------------------------------------------------------------------------------

def add_roofs(read: dict, vlm, image, path: Path, mask=None, view: str = "full") -> dict:
    """The roof pass on a reading, saved with it."""
    t = time.time()
    read["roofs"] = roof_pass(vlm, image, read, mask, view)
    read["roofs_view"] = view if mask is not None else "full"
    read["seconds_roofs"] = round(time.time() - t, 1)
    path.write_text(json.dumps(read, indent=1, ensure_ascii=False), encoding="utf-8")
    return read


PITCHED_TOPS = ("triangle", "eave", "both")


def form_front(p: dict) -> str | None:
    """What a form part says tops its front wall, in the roof pass's words."""
    roof = str(p.get("roof") or "").lower()
    if roof == "flat":
        return "flat"
    if roof == "cross_gable":
        return "both"
    if roof in ("gable", "hip", "gambrel", "mansard", "shed", "barrel", "butterfly", "clipped_gable"):
        return "triangle" if p.get("ridge") == "perpendicular" else "eave"
    return "other" if roof else None


def settle_front(vlm, photo, which: str, seen: str | None, form: str | None) -> str | None:
    """A part's front top where the roof pass and the form disagree on a flat roof against a
    pitched one: a yes / no question decides (`vision.ROOF_CHECKS`); where its two askings
    differ, the pass's answer stands. (The same for a gable rising from the eave fixed two of the
    test photos' main blocks and broke two.)"""
    from . import vision
    if (seen in PITCHED_TOPS and form == "flat") or (seen == "flat" and form in PITCHED_TOPS):
        yes = vision.roof_yes(vlm, photo, which, "slope")
        if yes is not None:
            return (seen if seen != "flat" else form) if yes else "flat"
    return seen


def roof_pass(vlm, photo, read: dict, mask=None, view: str = "full") -> list:
    """Each part's roof as the photo shows it, one concrete question at a time
    (`vision.ROOF_QUESTIONS`): what tops its front wall, what closes its ends, how steep its
    slopes are, where a gable rising from its eave stands; a tower's top (`settle_front` where
    the front top contradicts the form's), each part shown as `view` says (`vision.part_view`:
    the whole photo, a close-up of the part, or the photo with a box round it; `mask` is the
    building's pixels). Written into the reading's form parts as "roof_seen"
    (compose takes what `compose.ROOF_TRUST` says it reads better than the form; "front_first":
    the pass's own answer where the tie-break changed it) and returned, a dict a part."""
    from . import vision
    form = read.get("form") if isinstance(read.get("form"), dict) else {}
    parts = [p for p in form.get("parts") or [] if isinstance(p, dict)]
    out = []
    for k, p in enumerate(parts):
        role = str(p.get("part") or "wing").lower()
        image, which = vision.part_view(photo, mask, parts, k, role, view)
        seen = {}
        if role == "tower":
            seen["tower_top"] = vision.roof_question(vlm, image, which, "tower_top")
        else:
            first = vision.roof_question(vlm, image, which, "front_top")
            seen["front_top"] = settle_front(vlm, image, which, first, form_front(p))
            if seen["front_top"] != first:
                seen["front_first"] = first
            if seen["front_top"] in PITCHED_TOPS:
                seen["ends"] = vision.roof_question(vlm, image, which, "ends")
                seen["pitch"] = vision.roof_question(vlm, image, which, "pitch")
            if seen["front_top"] == "both":
                seen["cross_at"] = vision.roof_question(vlm, image, which, "cross_at")
        p["roof_seen"] = {key: v for key, v in seen.items() if v}
        out.append(p["roof_seen"])
    return out


def trusted_measure(out: Path, stem: str, facts: dict | None) -> dict | None:
    """MapAnything's measurement when the building was segmented cleanly, with the frame's
    height / width for when its size fails; else None (typical sizes)."""
    m = _json(out / f"{stem}.measure.json")
    if not m or not m.get("cutout") or not 2.0 <= m.get("height_m", 0) <= 600:
        return None
    if facts and facts.get("frame"):
        x0, y0, x1, y1 = facts["frame"]
        m["mask_aspect"] = (y1 - y0) / max(1, x1 - x0)
    return m


def camera(facts: dict, measure: dict | None) -> tuple:
    """(yaw, pitch) to render the build as the photo sees it: which side from the vision
    model (MapAnything's left and right agreed with the better outline on only 26 of 46
    photos), how far round and how high from MapAnything (else 30 and 12 degrees). The
    renderer's positive yaw looks from the front left."""
    side = ((facts or {}).get("locate") or {}).get("side", "none")
    yaw_m = abs(float(measure.get("camera_yaw_deg", 30.0))) if measure else 30.0
    pitch = float(measure.get("camera_pitch_deg", 12.0)) if measure else 12.0
    yaw = 0.0 if side == "none" else max(12.0, min(60.0, yaw_m)) * (1 if side == "left" else -1)
    return yaw, max(-25.0, min(50.0, pitch))


# the label view's colours (`render.render_codes`): build.LABELS index -> hue slot, None = white;
# windows, doors, garage doors and skylights each their own
LABEL_COLOURS = {1: None, 11: None, 2: 1, 12: 1, 3: 6, 4: 0, 5: 5, 6: 3, 7: 4, 8: 7, 9: 2, 10: 2, 13: 2}


def evaluate(state: dict, ctx: dict) -> dict:
    """Compose a state, put the photo's windows, doors and garage doors where the photo has
    them (`facade.place_openings_3d`, when MapAnything placed them), build, render (the build,
    what each block is, which part it belongs to) and measure; the conflicts are what the
    form asked that the build could not do."""
    from .compose import compose
    bp = compose(state["form"], state["style"], state["kind"], ctx["measure"], adjust=state["adjust"])
    return evaluate_bp(bp, ctx, state)


def evaluate_bp(bp: dict, ctx: dict, state: dict | None = None, place: bool = True) -> dict:
    """`evaluate` of a blueprint: with `place` the photo's openings replace its own."""
    from . import score
    from .build import build
    from .facade import fractions, place_openings_3d
    from .render import render_codes, render_rgba
    conflicts = list((bp.get("meta") or {}).get("conflicts") or [])
    if place and ctx.get("openings3d"):
        meta = bp.get("meta")
        bp, notes = place_openings_3d(bp, ctx["openings3d"], ctx.get("side"), ctx.get("garages", True))
        bp["meta"] = meta
        conflicts += notes["grown"] + notes["lost"]
    b = build(bp)
    key = hashlib.md5(b["shape"].tobytes() + b["material"].tobytes() + str(b["dims"]).encode()).hexdigest()
    cam = (ctx["yaw"], ctx["pitch"], ctx["size"])
    rgba = render_rgba(b["shape"], b["material"], *cam)
    bf = None
    if rgba is not None:
        lab = render_codes(b["shape"], b["material"], b["labels"], LABEL_COLOURS, *cam)
        n = min(8, len(b["parts"]))
        parts = render_codes(b["shape"], b["material"], b["part"], {0: None, **{k: k - 1 for k in range(1, n + 1)}}, *cam)
        bf = score.build_facts(rgba, lab, parts, n)
    pf = ctx["photo_facts"]
    if bf is not None and pf is not None and pf.get("openings3d") is not None:
        bf["openings3d"] = fractions(bp, b, ctx.get("side"))
    if bf is not None and pf is not None and pf.get("plan"):
        boxes = [m for m in bp["masses"] if m.get("shape", "box") == "box"] or bp["masses"]
        wx = max(m["x"] + m["w"] for m in boxes) - min(m["x"] for m in boxes)
        dz = max(m["z"] + m["d"] for m in boxes) - min(m["z"] for m in boxes)
        bf["plan"] = dz / max(wx, 1)
    scores = score.compare(pf, bf) if bf is not None and pf is not None else {}
    return {"state": state, "bp": bp, "build": b, "key": key, "rgba": rgba, "scores": scores,
            "measured": scores["total"][0] if scores else None, "conflicts": conflicts,
            "diag": score.diagnose(pf, bf) if scores else {}}


def view(res: dict):
    """The render the vision model sees: the build on a plain background, cropped."""
    from PIL import Image

    from .render import finish
    from .vision import SIDE
    return finish(Image.fromarray(res["rgba"]), SIDE)


def combined(res: dict):
    """The measures' total and the vision model's resemblance (out of 10), averaged; either
    alone when the other is missing."""
    parts = [v for v in (res.get("measured"), None if res.get("resemblance") is None else res["resemblance"] / 10)
             if v is not None]
    return round(float(np.mean(parts)), 3) if parts else None


def _kept(hist: list) -> list:
    """The corrections the final build has: those kept, less those undone."""
    out = []
    for h in hist:
        if h.get("reverted") and h["reverted"] in out:
            out.remove(h["reverted"])
        acc = h.get("accepted")
        out += acc if isinstance(acc, list) else [acc] if acc else []
    return out


def _verify(edit: dict, before: dict, after: dict, gain: float) -> str:
    """Whether a correction is kept, judged from the critique of the build before it and of
    the candidate on its own: "kept", or why not. Not when the new critique asks to undo it or
    rates the build lower; a correction the model asked for is kept when its complaint is
    gone; one the measures found when they gain enough (a storey, style, kind or mirror image
    of theirs only when the model rates the result higher)."""
    from .edits import _feature_name
    if _contradicts(after, edit):
        return "contradicted"
    ra, rb = after.get("resemblance"), before.get("resemblance")
    if ra is not None and rb is not None and ra < rb:
        return f"rated lower ({rb:g} -> {ra:g})"
    op = edit.get("op")
    if edit.get("why") != "vision":
        if op in MODEL_ONLY:
            return "kept" if ra is not None and rb is not None and ra > rb else "not rated higher"
        need = MEANING_GAIN if op in MEANING_OPS else CLEAR_GAIN
        return "kept" if gain >= need else "too small a gain"
    still = False                                      # the complaint that asked for it, still there?
    if op == "feature":
        words = after.get("missing" if edit.get("on", True) else "extra") or []
        still = edit.get("name") in {_feature_name(w) for w in (words if isinstance(words, list) else [words])}
    elif op in ("roof", "pitch"):
        want = {"pitch": "steeper" if edit.get("delta", 0) > 0 else "flatter"}.get(op)
        still = any(str(r.get("pitch")).lower() == want if want else
                    (r.get("type_ok") is False and str(r.get("type")).lower() == edit.get("value"))
                    for r in _roofs(after))
    elif op in ("style", "kind"):
        still = after.get(f"{op}_ok") is False
    elif op == "mirror":
        still = after.get("mirrored") is True
    elif op in ("width", "height", "depth", "storeys"):
        axis = "height" if op == "storeys" else op
        still = str((after.get("proportions") or {}).get(axis, "ok")).lower() != "ok"
    elif op in ("part_height", "part_depth", "part_level", "part_forward", "garage_doors") or (
            op == "share" and edit.get("part") is not None):
        key = {"part_height": "height", "part_depth": "depth", "part_level": "level", "part_forward": "forward",
               "share": "width", "garage_doors": "garage_doors"}[op]
        entry = _part_entry(after, edit.get("shown"))
        if entry is not None:
            v = entry.get(key)
            still = (v is not None and int(_num(v, -1)) != int(edit.get("value", -2))) if key == "garage_doors" \
                else str(v or "ok").lower() != "ok"
    elif op == "windows":
        still = str(after.get("windows") or "ok").lower() != "ok"
    elif op == "colour":
        still = str(after.get(f"{edit.get('role')}_colour") or "ok").lower() not in ("ok", "same", "correct", "none")
    elif op == "add_part":
        still = str(after.get("missing_part") or "none").lower() in ("left", "right")
    elif op == "garage_wall":
        still = str(after.get("garage") or "ok").lower() == "other_side"
    return "complaint still there" if still else "kept"


def _roofs(crit: dict) -> list:
    """The critique's roof answers, from its "parts" (or an older answer's "roofs"), as
    {"part", "type_ok", "type", "pitch"}."""
    out = [r for r in (crit.get("roofs") or []) if isinstance(r, dict)]
    for p in crit.get("parts") or []:
        if isinstance(p, dict):
            out.append({"part": p.get("part"), "type_ok": p.get("roof_ok"), "type": p.get("roof_type"),
                        "pitch": p.get("pitch")})
    return out


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _part_entry(crit: dict, shown) -> dict | None:
    """The critique's entry for the build's part shown as number `shown`."""
    for p in crit.get("parts") or []:
        if isinstance(p, dict) and shown is not None and int(_num(p.get("part"), -1)) == int(shown):
            return p
    return None


def _contradicts(crit: dict, edit: dict) -> bool:
    """Whether a critique asks to undo `edit`: a feature just added now called extra (or
    removed, now missing), a pitch just raised now to be flatter, a style or kind just set now
    called wrong, a mirror just made now called mirrored."""
    from .edits import _feature_name
    op = edit.get("op")
    if op == "feature":
        words = crit.get("extra" if edit.get("on", True) else "missing") or []
        return edit.get("name") in {_feature_name(w) for w in (words if isinstance(words, list) else [words])}
    if op == "pitch":
        want = "flatter" if edit.get("delta", 0) > 0 else "steeper"
        return any(str(r.get("pitch")).lower() == want for r in _roofs(crit))
    if op in ("style", "kind"):
        return crit.get(f"{op}_ok") is False
    if op == "mirror":
        return crit.get("mirrored") is True
    opposite = {"part_height": ("height", "too_tall", "too_low"), "part_depth": ("depth", "too_deep", "too_shallow"),
                "part_level": ("level", "should_be_higher", "should_be_lower"),
                "part_forward": ("forward", "should_go_back", "should_come_forward")}
    if op in opposite:                                 # raised, and now called too tall
        key, after_up, after_down = opposite[op]
        entry = _part_entry(crit, edit.get("shown"))
        up = _num(edit.get("factor", edit.get("delta", 0)), 0) > (1 if "factor" in edit else 0)
        return entry is not None and str(entry.get(key) or "").lower() == (after_up if up else after_down)
    return False


def _scores(res: dict) -> dict:
    from .score import summary
    return summary(res["scores"]) if res.get("scores") else {}


def context(stem: str, out: Path) -> tuple:
    """(ctx, photo, read): what `evaluate` measures builds of one prepared photo with: its
    measures (`score.photo_facts`, and its openings on the building when MapAnything placed
    them), its measurement, and the camera to render builds from."""
    from PIL import Image

    from . import score
    read = _json(out / f"{stem}.read.json")
    facts = _json(out / f"{stem}.photo.json") or {}
    work = out / "_work" / stem
    photo = Image.open(work / "photo.png").convert("RGB")
    measure = trusted_measure(out, stem, facts)
    pf = None
    if facts.get("ok"):
        seg = {"mask": np.array(Image.open(work / "mask.png")) > 127,
               "occluders": np.array(Image.open(work / "occluders.png")) > 127, "frame": facts["frame"]}
        boxes = facts.get("openings") or []
        pf = score.photo_facts(np.array(photo), seg, boxes if (facts.get("openings_complete") or len(boxes) >= 3)
                               else None)
    yaw, pitch = camera(facts, measure)
    side = ((facts or {}).get("locate") or {}).get("side", "none")
    openings3d = (measure or {}).get("openings") or None
    garages = has_garages((read or {}).get("form") or {})
    if pf is not None and openings3d:
        from .facade import photo_fractions
        pf["openings3d"] = photo_fractions(openings3d, garages)
    if pf is not None and measure and side in ("left", "right") and float(measure.get("corner_view_deg", 0)) >= 15 \
            and float(measure.get("depth_m", 0)) <= 1.3 * float(measure.get("width_m", 1)):
        # the photo sees a side wall: MapAnything's depth over width (compose keeps its front then)
        pf["plan"] = float(measure["depth_m"]) / max(float(measure["width_m"]), 1e-3)
    ctx = {"measure": measure, "photo_facts": pf, "yaw": yaw, "pitch": pitch, "size": 512,
           "openings3d": openings3d, "side": side if side in ("left", "right") else None, "garages": garages}
    return ctx, photo, read


def has_garages(form: dict) -> bool:
    """Whether a reading says the building has garage doors (or big doors), anywhere."""
    def n(v):
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0
    g = form.get("garage") if isinstance(form.get("garage"), dict) else {}
    parts = [p for p in form.get("parts") or [] if isinstance(p, dict)]
    return n(g.get("doors")) + n(form.get("big_doors")) + sum(
        n((p.get("front") or {}).get("garage_doors") if isinstance(p.get("front"), dict) else 0) +
        n(p.get("side_garage_doors")) for p in parts) > 0


def check_round(cur: dict, ctx: dict, vlm, photo, log=print, mode: str = "agree") -> tuple:
    """(build, log entry): the photo-only check (`vision.check`: each of the build's parts as
    the photo shows it, asked of the photo alone) against the build, and each correction its
    answers ask for (`edits.from_check`) kept in turn unless the measures find it CHECK_GAIN
    worse; a mirror image last (it renumbers the parts). `mode` "agree" (the default) acts only on
    the answers the form's reading agrees with (`edits.agreed_with_form`): on the 50 test photos,
    acting on all of them fixed 7 part-level answers and broke 7 (146 of 185 either way), since
    the model's two readings, where they disagree, are right about as often as wrong; "full" acts
    on all (for a stronger model, or a person's answers: with right answers the check took round
    3's builds from 144 to 155)."""
    from . import edits as E
    from . import vision
    from .spec import BlueprintError
    from .facade import levels
    descs, numbers = E.check_parts(cur["bp"], cur["state"])
    chk = vision.check(vlm, photo, descs)
    door_levels = levels(cur["bp"], ctx["openings3d"]) if ctx.get("openings3d") else {}
    use = chk if mode == "full" else E.agreed_with_form(chk, cur["state"]["form"], numbers)
    cands = E.from_check(use, cur["state"], cur["bp"], numbers, door_levels)
    # per-part corrections first, then removing parts (the highest first: the others keep their
    # numbers), a mirror image last; after either, the part numbers the rest name have changed
    cands.sort(key=lambda e: (e["op"] == "mirror", e["op"] == "remove_part",
                              -int(e.get("part", 0)) if e["op"] == "remove_part" else 0))
    entry = {"round": "check", "mode": mode, "check": {k: v for k, v in chk.items() if k not in ("text", "text2")},
             "parts": descs,
             "door_levels": door_levels, "measured": _scores(cur), "judged": []}
    todo = list(cands)
    for _ in range(3):                                 # again after a change: a split level needs its storey first
        left, changed = [], False
        for e in todo:
            new, note = E.apply(cur["state"], e, cur["bp"])
            if note or new == cur["state"]:
                entry["judged"].append({"edit": E.label(e), "verdict": note or "already so"})
                continue
            try:
                res = evaluate(new, ctx)
            except (BlueprintError, ValueError, KeyError, TypeError, IndexError) as ex:
                entry["judged"].append({"edit": E.label(e), "verdict": f"invalid: {type(ex).__name__}: {ex}"[:200]})
                continue
            gain = (res["measured"] - cur["measured"]) if res["measured"] is not None and cur["measured"] is not None                 else 0.0
            need = DOORS_GAIN if e.get("doors_agree") else STRICT_GAIN if e["op"] in ("remove_part", "mirror") \
                else CHECK_GAIN
            if res["key"] != cur["key"] and gain >= need:
                entry["judged"].append({"edit": E.label(e), "verdict": "kept", "gain": round(gain, 3)})
                res["edit"] = e
                cur, changed = res, True
                log(f"  check: kept {E.label(e)} (measured gain {gain:+.3f})")
                if e["op"] in ("remove_part", "mirror"):  # the parts are renumbered: no retries by number
                    left = [(x, g) for x, g in left if x["op"] in ("remove_part", "mirror")]
            elif res["key"] == cur["key"]:
                entry["judged"].append({"edit": E.label(e), "verdict": "no change"})
            else:
                left.append((e, gain))
        todo = [e for e, _ in left]
        if not changed or not todo:
            break
    entry["judged"] += [{"edit": E.label(e), "verdict": "measured worse", "gain": round(g, 3)} for e, g in left]
    entry["accepted"] = [j["edit"] for j in entry["judged"] if j["verdict"] == "kept"]
    return cur, entry


def refine_one(stem: str, out: Path, vlm, rounds: int = 4, log=print, check: str = "agree") -> dict:
    from . import edits as E
    from . import vision
    from .spec import BlueprintError
    t0 = time.time()
    ctx, photo, read = context(stem, out)
    measure, yaw, pitch, pf = ctx["measure"], ctx["yaw"], ctx["pitch"], ctx["photo_facts"]
    cur = evaluate(E.initial(read), ctx)
    first = cur
    hist, gaps, no_effect, limits, invalid = [], [], [], [], []
    if vlm is not None and check != "off":
        cur, entry = check_round(cur, ctx, vlm, photo, log, check)
        hist.append(entry)
    tried = set()
    prev = None                                        # (the build before the last kept correction, that correction)
    scored = []                                        # every build the model rated: (round, build, its corrections)
    for rnd in range(rounds + 1):
        if cur.get("critique") is not None:            # critiqued as a candidate last round
            text, numbers, crit = cur["critique"]
        else:
            text, numbers = E.describe_state(cur["state"], cur["bp"])
            crit = vision.critique(vlm, photo, view(cur), text)
            cur["critique"] = (text, numbers, crit)
        cur["resemblance"] = crit.get("resemblance")
        entry = {"round": rnd, "measured": _scores(cur), "resemblance": cur["resemblance"], "combined": combined(cur),
                 "critique": {k: v for k, v in crit.items() if k != "text"}, "conflicts": cur.get("conflicts") or []}
        scored.append((rnd, cur, _kept(hist)))
        hist.append(entry)
        log(f"  round {rnd}: combined {entry['combined']}, measured {entry['measured'].get('total')}, "
            f"resemblance {cur['resemblance']}")
        if prev is not None and _contradicts(crit, prev[1]):
            # the model now asks to undo what it just asked for (a porch added, then called extra)
            entry["reverted"] = E.label(prev[1])
            log(f"  undid {entry['reverted']}: the next critique contradicts it")
            cur, prev = prev[0], None
            if rnd == rounds:
                break
            continue
        good = entry["combined"] is not None and entry["combined"] >= TARGET and (
            cur["measured"] is None or cur["measured"] >= MEASURED_TARGET)
        if rnd == rounds or good:
            entry["stop"] = "rounds" if rnd == rounds else "target"
            break
        cands, g = E.from_critique(crit, cur["state"], numbers)
        gaps += [{"round": rnd, "gap": x} for x in g]
        cands += E.from_measures(cur["diag"], cur["state"], cur["bp"]) if cur["diag"] else []
        cands += E.from_conflicts(cur.get("conflicts"), cur["state"], cur["bp"])
        evaluated = []
        for e in cands:
            name = E.label(e)
            if (cur["key"], name) in tried:
                continue
            tried.add((cur["key"], name))
            new, note = E.apply(cur["state"], e, cur["bp"])
            if note:
                (invalid if note.startswith("invalid") else limits).append({"round": rnd, "edit": name, "note": note})
                continue
            if new == cur["state"]:                    # already so (a chimney it has): nothing to try
                continue
            try:
                res = evaluate(new, ctx)
            except (BlueprintError, ValueError, KeyError, TypeError, IndexError) as ex:
                invalid.append({"round": rnd, "edit": name, "note": f"{type(ex).__name__}: {ex}"[:200]})
                continue
            if res["key"] == cur["key"]:
                no_effect.append({"round": rnd, "edit": name, "why": e.get("why")})
                continue
            res["edit"] = e
            res["gain"] = (res["measured"] - cur["measured"]) if res["measured"] is not None and cur["measured"] is not None else 0.0
            evaluated.append(res)
        evaluated.sort(key=lambda r: -r["gain"])
        entry["candidates"] = [{"edit": E.label(r["edit"]), "why": r["edit"].get("why"), "gain": round(r["gain"], 3)}
                               for r in evaluated]
        # the model's own corrections and the measures' in turn, the model's first: led by the
        # measured gain alone, small changes of proportion always came first and its asks
        # (dormers, skylights) were never shown to it
        fine = [r for r in evaluated if r["gain"] > WORSE]
        short = []
        for pair in itertools.zip_longest([r for r in fine if r["edit"].get("why") == "vision"],
                                          [r for r in fine if r["edit"].get("why") != "vision"]):
            short += [r for r in pair if r is not None]
        short = short[:SHORTLIST]
        accepted = None
        for r in short:
            op, why = r["edit"]["op"], r["edit"].get("why")
            if why == "conflict":
                # what the form asked and the build could not do: kept unless the measures object
                verdict = "kept" if r["gain"] >= CHECK_GAIN else "measured worse"
            elif op in SHAPE_OPS and why != "vision":
                # proportions the measures found are kept on the measures: the next critique and
                # the best-build rule catch one the model finds worse
                verdict = "kept" if r["gain"] >= SHAPE_GAIN else "too small a gain"
            else:
                # the rest is checked by critiquing the candidate on its own: asked which of two
                # builds is closer, the model picked the image shown last 59 times in 70
                text_c, numbers_c = E.describe_state(r["state"], r["bp"])
                crit_c = vision.critique(vlm, photo, view(r), text_c)
                r["critique"] = (text_c, numbers_c, crit_c)
                verdict = _verify(r["edit"], crit, crit_c, r["gain"])
                if verdict == "kept" and why == "vision" and op in STRUCTURE_OPS and r["gain"] < STRICT_GAIN:
                    verdict = "measured worse"             # the critique's own say-so is weak (it describes the render)
            entry.setdefault("judged", []).append({"edit": E.label(r["edit"]), "verdict": verdict,
                                                   "gain": round(r["gain"], 3),
                                                   "resemblance": (r.get("critique") or (0, 0, {}))[2].get("resemblance")})
            if verdict == "kept":
                accepted = r
                break
        if accepted is None:
            entry["stop"] = "no correction helped"
            break
        entry["accepted"] = E.label(accepted["edit"])
        log(f"  kept {entry['accepted']} (measured gain {accepted['gain']:+.3f})")
        prev = (cur, accepted["edit"])
        cur = accepted
    # the best build the loop saw, not just its last: each correction looked better at the
    # time, yet a run of them can end worse (a chalet widened three times and mirrored)
    best_round, cur, kept = max(scored, key=lambda t: (combined(t[1]) or -1.0, t[1]["measured"] or -1.0))
    d = out / stem
    d.mkdir(parents=True, exist_ok=True)
    from photofirst.schem import write_schem
    for tag, res in (("first", first), ("final", cur)):
        view(res).save(d / f"{tag}.png")
        (d / f"{tag}.blueprint.json").write_text(json.dumps(res["bp"], indent=1), encoding="utf-8")
        try:
            write_schem(res["build"]["shape"], res["build"]["material"], d / f"{tag}.schem")
        except ImportError:                            # no nbtlib here (r2m_vlm): `python -m blueprint` writes it
            pass
    rec = {"id": stem, "camera": [yaw, pitch], "measured": measure is not None, "photo_measures": pf is not None,
           "first": {"measured": _scores(first), "resemblance": first.get("resemblance"), "combined": combined(first)},
           "final": {"measured": _scores(cur), "resemblance": cur.get("resemblance"), "combined": combined(cur),
                     "state": cur["state"], "round": best_round},
           "kept": kept, "rounds": hist, "gaps": gaps, "conflicts": {"first": first.get("conflicts") or [],
                                                                     "final": cur.get("conflicts") or []},
           "limits": limits, "no_effect": no_effect, "invalid": invalid, "seconds": round(time.time() - t0, 1)}
    (out / f"{stem}.refine.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


# --- judge: the first build against the final one, both orders -----------------------------

def judge(stem: str, out: Path, vlm) -> dict:
    """The vision model's verdict, first build against final, asked twice with the order
    swapped: "final" or "first" only when both answers agree (with Qwen3.5-9B mostly "tie": it
    picks the image shown last; kept to test other models)."""
    from PIL import Image

    from .describe import ask, parse_json
    from .vision import COMPARE, _small
    d = out / stem
    rec = _json(out / f"{stem}.refine.json") or {}
    if not rec.get("kept"):                            # the loop changed nothing: nothing to judge
        rec = {"id": stem, "votes": [], "verdict": "unchanged"}
        (out / f"{stem}.judge.json").write_text(json.dumps(rec), encoding="utf-8")
        return rec
    photo = Image.open(out / "_work" / stem / "photo.png").convert("RGB")
    a, b = Image.open(d / "first.png"), Image.open(d / "final.png")
    votes = []
    for first_shown in ("first", "final"):
        ims = [a, b] if first_shown == "first" else [b, a]
        js = parse_json(ask(*vlm, [_small(photo), _small(ims[0]), _small(ims[1])], COMPARE, 60)) or {}
        pick = js.get("closer")
        votes.append(None if pick not in (2, 3) else (first_shown if pick == 2 else
                                                      ("final" if first_shown == "first" else "first")))
    verdict = votes[0] if votes[0] == votes[1] else "tie"
    rec = {"id": stem, "votes": votes, "verdict": verdict}
    (out / f"{stem}.judge.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def summary(out: Path) -> dict:
    """Over a run: first and final scores, the corrections kept, and the capability gaps."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.refine.json"))]
    judged = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.judge.json"))]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    s = {"photos": len(recs)}
    for tag in ("first", "final"):
        s[tag] = {"combined": mean(r[tag]["combined"] for r in recs),
                  "resemblance": mean(r[tag]["resemblance"] for r in recs),
                  "measured": mean(r[tag]["measured"].get("total") for r in recs),
                  "outline": mean(r[tag]["measured"].get("outline") for r in recs),
                  "proportions": mean(r[tag]["measured"].get("proportions") for r in recs)}
    s["changed"] = sum(1 for r in recs if r["kept"])
    s["kept"] = Counter(k.split("(")[0] for r in recs for k in r["kept"]).most_common()
    s["no_effect"] = Counter(n["edit"] for r in recs for n in r["no_effect"]).most_common(25)
    s["limits"] = Counter(n["note"] for r in recs for n in r["limits"]).most_common(10)
    s["gaps"] = Counter(g["gap"].lower() for r in recs for g in r["gaps"]).most_common(40)
    s["judge"] = dict(Counter(j["verdict"] for j in judged))
    s["seconds"] = mean(r["seconds"] for r in recs)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("prepare", "roofs", "run", "judge", "summary"))
    ap.add_argument("photos", nargs="*", type=Path, help="photos, or folders of <id>/photo.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="run photos that already have a result again")
    ap.add_argument("--model", default=None, help="the vision model (describe.load_model; default its MODEL)")
    ap.add_argument("--roof-view", default=ROOF_VIEW, choices=("full", "crop", "box"),
                    help="what the roof pass shows of each part (vision.part_view)")
    ap.add_argument("--check", default="agree", choices=("agree", "full", "off"),
                    help="the photo-only check: act on the answers the form agrees with, on all, or skip it")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        print(json.dumps(summary(out), indent=1))
        return 0
    photos = photos_of(args.photos)
    suffix = {"prepare": "photo", "run": "refine", "judge": "judge", "roofs": "read"}[args.stage]
    if args.stage == "roofs":                          # readings that have no roof pass yet
        todo = [p for p in photos if (out / f"{stem_of(p)}.read.json").is_file() and (
            args.redo or "roofs" not in (_json(out / f"{stem_of(p)}.read.json") or {}))]
    else:
        todo = [p for p in photos if args.redo or not (out / f"{stem_of(p)}.{suffix}.json").is_file()]
    if not todo:
        return 0
    from .describe import MODEL, SERVER_MODELS, load_model
    args.model = args.model or MODEL
    t = time.time()
    vlm = load_model(args.model)
    seg = None
    if args.stage == "prepare":
        from . import segment
        seg = segment.load("cpu" if args.model in SERVER_MODELS else "cuda")
    print(f"models loaded in {time.time() - t:.0f} s", flush=True)
    for p in todo:
        stem = stem_of(p)
        t = time.time()
        try:
            if args.stage == "roofs":
                from PIL import Image

                from .describe import photo as load_photo
                mf = out / "_work" / stem / "mask.png"
                mask = np.array(Image.open(mf)) > 127 if mf.is_file() else None
                r = add_roofs(_json(out / f"{stem}.read.json"), vlm, load_photo(p), out / f"{stem}.read.json", mask,
                              args.roof_view)
                print(f"{stem}: {r['seconds_roofs']} s, {r['roofs']}", flush=True)
            elif args.stage == "prepare":
                f = prepare(p, out, vlm, seg, args.roof_view)
                print(f"{stem}: {time.time() - t:.0f} s, building {f['share']:.0%} of the photo, side {f['locate']['side']}, "
                      f"{len(f['openings'])} openings", flush=True)
            elif args.stage == "run":
                print(f"{stem}:", flush=True)
                r = refine_one(stem, out, vlm, args.rounds, check=args.check)
                print(f"{stem}: {r['first']['combined']} -> {r['final']['combined']} in {r['seconds']} s, "
                      f"kept {r['kept']}", flush=True)
            else:
                r = judge(stem, out, vlm)
                print(f"{stem}: {r['verdict']} {r['votes']}", flush=True)
        except Exception as ex:  # noqa: BLE001  one photo's failure must not stop the run
            import traceback
            print(f"{stem}: FAILED {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
