"""What the local vision model (`describe.load_model`) says about a photo beyond the form:
where the main building is and which of its sides the photo shows (`locate`), its windows'
and doors' boxes (`openings`), and, for the evaluator, what is wrong with a build against the
photo (`critique`: the whole, and each part's size, level, roof and garage doors), the photo's
own answers about each of the build's parts for the code to compare (`check`), and which of
two builds is closer to it (`compare`). Boxes come in the model's 0-1000 coordinates. Every
answer is parsed into fixed choices; the critique offers only the roofs, features, styles and
kinds the builder has (`edits`), and anything else goes in its "other" answer, which the
evaluator logs as a capability gap.
"""
from __future__ import annotations

import random
import re

from .describe import ask, parse_json

SIDE = 768                   # images for the comparisons are this size at most

LOCATE = """Find the main building in this photo: the one the photo is about, not its neighbours or the buildings
behind it. Reply with JSON only:
{"bbox_2d": [x1, y1, x2, y2] (the whole main building, including any part hidden behind trees, cars or people),
 "side": "left|right|none" (besides its front, which of its side walls the photo also shows: the one on the left
 of the photo, the one on the right, or none when the photo looks straight at the front)}"""

OPENINGS = """Locate every window, door and garage door on the main building's visible walls.
Output JSON only: a list of {"bbox_2d": [x1, y1, x2, y2], "label": "window", "door" or "garage_door"}."""

CRITIQUE = """Image 1 is a photo of a real building. Image 2 is a Minecraft build made from the photo, seen from
about the same angle: blocky and simplified on purpose, on a plain background. Ignore the photo's trees, cars,
people and surroundings and judge the building only.

The build is:
STATE

Compare the build with the building in the photo. Reply with JSON only:
{"resemblance": <0-10: how much the build looks like the building, overall>,
 "style_ok": true|false, "style": "<if wrong, the right one of: STYLES>",
 "kind_ok": true|false, "kind": "<if wrong, the right one of: KINDS>",
 "proportions": {"width": "ok|too_narrow|too_wide", "height": "ok|too_low|too_tall", "depth": "ok|too_shallow|too_deep"},
 "parts": [one entry per part of the build, against the same part of the building in the photo:
           {"part": <its number>,
            "height": "ok|too_low|too_tall" (its walls),
            "width": "ok|too_narrow|too_wide",
            "depth": "ok|too_shallow|too_deep",
            "level": "ok|should_be_lower|should_be_higher" (its floor: a split level's lower part sits lower),
            "forward": "ok|should_come_forward|should_go_back" (its front wall, against the other parts'),
            "roof_ok": true|false, "roof_type": "<if wrong, the photo's: ROOFS>",
            "pitch": "ok|steeper|flatter" (should its roof be steeper or flatter),
            "garage_doors": <integer: how many garage doors the photo's part has>}],
 "extra_parts": [the numbers of the build's parts the building does not have],
 "missing_part": "none|left|right" (a wing or block the building has and the build lacks, on that side),
 "missing": [features the photo shows that the build lacks, from: FEATURES],
 "extra": [features the build has that the photo does not, from the same list],
 "mirrored": true|false (the build's parts are in the opposite left-to-right order from the photo's, e.g. its
  garage or tower is on the other side),
 "garage": "ok|other_side|missing|not_in_photo|none" (the build's garage doors against the photo's; none: neither
  has a garage),
 "windows": "ok|too_few|too_many",
 "wall_colour": "ok|<the photo's wall colour, a word or two>",
 "roof_colour": "ok|<the photo's roof colour>",
 "other": "<anything else clearly wrong that these answers cannot say, in a few words; empty if nothing>"}"""

CHECK = """Look at the main building in this photo (ignore trees, cars and other buildings). Seen from the
photo, it is made of these parts, left to right: PARTS.
For each part, answer from the photo only. Reply with JSON only:
{"parts": [{"part": <its number>,
            "exists": true|false (false when the photo's building has no such part: a neighbouring building, or a
             part of the same block counted twice),
            "roof_ends": "triangle|slope|flat|not_visible" (what closes the short ends of its roof: a triangular
             wall, as on a gable roof; a sloping roof face, as on a hip roof; or a flat roof),
            "ridge": "along_front|front_to_back|none" (the top line of its roof runs along its front wall, the
             wall facing the camera most, or from that wall to its back),
            "floor": "same|lower|higher" (its lowest floor against the main block's: lower when part of it sits
             below the main block's ground floor, as in a split level),
            "storeys": <integer: its floors, counting one partly below the ground>,
            "garage_doors": <integer>, "garage_wall": "front|end|none" (the wall facing the camera, or its end wall),
            "base_m": <metres a band of brick, stone or concrete rises at the foot of its front wall; 0 if none>,
            "base_end_m": <the same on its end wall the photo shows>}],
 "tallest": <the number of the tallest part>}"""

COMPARE = """Image 1 is a photo of a real building. Images 2 and 3 are two Minecraft builds made from it, seen from
about the same angle (blocky and simplified on purpose). Which build looks more like the building in the photo:
its overall shape and proportions, its roofs, its parts and their left-to-right order, its windows and its
colours? Reply with JSON only: {"closer": 2 or 3, "sure": true|false}"""


def _small(image, side: int = SIDE):
    im = image.convert("RGB")
    im.thumbnail((side, side))
    return im


def boxes(text: str) -> list:
    """[{"box": [x0, y0, x1, y1], "label": str}] of every "bbox_2d" in an answer, a cut-off
    answer's complete ones included."""
    out = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[([^\]]+)\]\s*(?:,\s*"label"\s*:\s*"([^"]*)")?', text):
        try:
            v = [float(t) for t in m.group(1).split(",")]
        except ValueError:
            continue
        if len(v) == 4 and v[2] > v[0] and v[3] > v[1]:
            out.append({"box": v, "label": (m.group(2) or "").lower()})
    return out


def locate(vlm, image) -> dict:
    """{"box": 0-1000 box or None, "side": "left" | "right" | "none", "text"}."""
    text = ask(*vlm, image, LOCATE, 200)
    js = parse_json(text) or {}
    found = boxes(text)
    side = str(js.get("side", "none")).lower()
    return {"box": found[0]["box"] if found else None, "side": side if side in ("left", "right") else "none",
            "text": text}


def openings(vlm, image, limit: int = 1200) -> dict:
    """{"boxes": [...], "complete": bool (the answer was not cut off), "text"}."""
    text = ask(*vlm, image, OPENINGS, limit)
    return {"boxes": boxes(text), "complete": text.rstrip().rstrip("`").rstrip().endswith("]"), "text": text}


# the choices of the check, each also asked in the reverse order (`check`)
CHECK_CHOICES = ("triangle|slope|flat|not_visible", "along_front|front_to_back|none", "same|lower|higher",
                 "front|end|none", "true|false")


def _flipped(prompt: str) -> str:
    for c in CHECK_CHOICES:
        prompt = prompt.replace(c, "|".join(reversed(c.split("|"))))
    return prompt


def check(vlm, photo, parts: list, twice: bool = True) -> dict:
    """The model's answers about the photo alone for each of the build's parts (`CHECK`):
    `parts` are their descriptions left to right ("the main block", "a wing", ...). Asked of
    one image, so the answers are the photo's: given a photo and a render together, the model
    described the render (a hip roof the photo does not have called right, a mirrored build's
    garage doors given to the wrong part). With `twice` it is asked again with every list of
    choices reversed, and only the answers both give are kept ("disagreed" counts the others):
    on real photos the measures cannot tell a right correction from a wrong one (on 50 photos
    they kept 63% of right ones and 59% of wrong ones), so the answers must be right."""
    listed = "; ".join(f"{k} = {p}" for k, p in enumerate(parts, 1))
    prompt = CHECK.replace("PARTS", listed)
    text = ask(*vlm, photo, prompt, 500)
    js = parse_json(text) or {}
    js["text"] = text
    if not twice:
        return js
    text2 = ask(*vlm, photo, _flipped(prompt), 500)
    js2 = parse_json(text2) or {}
    second = {p.get("part"): p for p in js2.get("parts") or [] if isinstance(p, dict)}
    agreed, disagreed = [], 0
    for p in js.get("parts") or []:
        if not isinstance(p, dict):
            continue
        q = second.get(p.get("part")) or {}
        keep = {"part": p.get("part")}
        for k, v in p.items():
            if k == "part":
                continue
            if k in q and str(q[k]).lower() == str(v).lower():
                keep[k] = v
            else:
                disagreed += 1
        agreed.append(keep)
    out = {"parts": agreed, "disagreed": disagreed, "text": text, "text2": text2}
    if js.get("tallest") is not None and str(js.get("tallest")) == str(js2.get("tallest")):
        out["tallest"] = js["tallest"]
    return out


# --- the roofs, one concrete question at a time -------------------------------------------------------
# Asked "which way does the ridge run", both models were right on about 6 of 10 main roofs; asked
# what tops the front wall (a triangle or a level eave), the 27B was right on 20 of 24: questions
# about what is seen, not about what it means, and one at a time.
# (round 5: "a pediment is decoration, not the roof" read a temple-fronted church's gable as an
# eave; this wording took the main blocks' front tops from 35 to 38 of the 48 graded photos)
PEDIMENT = ("A triangular pediment over columns or over an entrance porch is the roof's own gable only where the main "
            "roof's ridge runs back from its peak, away from the camera; where the roof behind it is flat or runs "
            "along the front, the pediment is decoration: look at the roof itself.")
ROOF_QUESTIONS = {
    "front_top": (f"""Look at the building in this photo: WHICH. Find its front wall: the wall that faces the camera most.
What is at the top of that front wall? {PEDIMENT}
- "triangle": the wall rises to a point in the middle, a triangular gable, with the roof sloping down to its left and right
- "eave": the roof starts along a level edge across the top of the wall and slopes back, away from the camera
- "both": a level eave along most of the wall, with a triangular gable rising from it over part of the wall, the
  gable's wall part of the front wall (a dormer window set back on the roof slope does not count)
- "flat": a flat roof or a parapet, no slope
- "other": a dome, a spire, a tower, stacked roofs, or something else
Reply with JSON only: {{"answer": "triangle|eave|both|flat|other"}}""", ("triangle", "eave", "both", "flat", "other")),
    "ends": (f"""Look at the building in this photo: WHICH. Look at the two short ends of its roof (its left and right
ends if the roof runs along the front, its front and back if it runs away from you). What closes them? {PEDIMENT}
- "triangle": a triangular wall, the wall itself rising to the roof's peak (a gable roof)
- "slope": a sloping face of the roof, so the roof slopes down on every side (a hip roof)
- "none": the roof is flat, a dome, or a spire
- "not_visible": you cannot see its ends
Reply with JSON only: {{"answer": "triangle|slope|none|not_visible"}}""", ("triangle", "slope", "none", "not_visible")),
    "pitch": ("""Look at the building in this photo: WHICH. How steep are the slopes of its roof?
- "low": gentle, clearly flatter than a staircase, under about 25 degrees
- "medium": about 30 to 40 degrees
- "steep": as steep as a staircase or steeper, 45 degrees or more
- "none": the roof is flat, or you cannot see its slopes
Reply with JSON only: {"answer": "low|medium|steep|none"}""", ("low", "medium", "steep", "none")),
    "rise": ("""Look at the building in this photo: WHICH. Compare the height of its roof (from the eave, where the
roof starts, up to its highest point) with the height of the walls below it (from the ground up to the eave).
- "small": the roof is much less than half as tall as the walls
- "half": the roof is about half as tall as the walls
- "equal": the roof is about as tall as the walls
- "taller": the roof is taller than the walls
- "none": the roof is flat, or you cannot see it
Reply with JSON only: {"answer": "small|half|equal|taller|none"}""", ("small", "half", "equal", "taller", "none")),
    "tower_top": ("""Look at the building in this photo: WHICH. What is on top of it?
- "spire": a tall, thin pointed roof
- "cone": a pointed round roof, about as tall as it is wide
- "pyramid": a pointed roof with four flat sides
- "dome": a rounded dome
- "lantern": a small open or glazed room with columns or windows all round, with a little roof or dome on it
- "battlements": a flat top behind a notched parapet
- "flat": a plain flat top
- "onion": a bulb-shaped dome that swells out wider than its base and ends in a point
Reply with JSON only: {"answer": "spire|cone|pyramid|dome|lantern|battlements|flat|onion"}""",
                  ("spire", "cone", "pyramid", "dome", "lantern", "battlements", "flat", "onion")),
    "cross_at": ("""Look at the building in this photo: WHICH. Along the top of its front wall a triangular gable rises
from the roof's level edge over part of the wall. Where along the front wall is that gable?
Reply with JSON only: {"answer": "left|centre|right"}""", ("left", "centre", "right")),
}


def roof_question(vlm, photo, which: str, key: str) -> str | None:
    """One roof question (`ROOF_QUESTIONS`) about one part of the building, `which` describing
    it ("its main block, the part at the left of the 2 parts it is made of"); the answer if it
    is one of the choices, else None."""
    prompt, choices = ROOF_QUESTIONS[key]
    js = parse_json(ask(*vlm, photo, prompt.replace("WHICH", which), 60)) or {}
    a = str(js.get("answer") or "").lower().strip()
    return a if a in choices else None


# yes / no questions that settle a part's front top where the reading's form and the roof pass
# disagree (`refine.settle_front`): a flat roof against a pitched one
ROOF_CHECKS = {
    "slope": """Look at the building in this photo: WHICH. Look just above the top of its walls. Can you see the roof's
own surface sloping up and back from the walls: rows of shingles, tiles or slates, or metal sheets, rising towards a
ridge? A flat roof behind a straight parapet or a cornice, or a flat roof that juts out over the walls as a thin
level slab, shows no sloping surface; an awning, a sign or a porch roof is not the building's roof.
Reply with JSON only: {"answer": "ORDER"}""",
}


def roof_yes(vlm, photo, which: str, key: str) -> bool | None:
    """A `ROOF_CHECKS` question asked twice, yes and no the other way round the second time (the
    model leans to whichever comes first); the answer both askings give, else None."""
    got = set()
    for order in ("yes|no", "no|yes"):
        js = parse_json(ask(*vlm, photo, ROOF_CHECKS[key].replace("WHICH", which).replace("ORDER", order), 40)) or {}
        got.add(str(js.get("answer") or "").lower().strip())
    return {"yes": True, "no": False}.get(got.pop()) if len(got) == 1 else None


def part_words(n: int, k: int, role: str = "wing") -> str:
    """How a question names part k (0 = leftmost) of n."""
    if n <= 1:
        return "the whole building"
    name, pos = _part_name(role), _part_pos(n, k)
    return f"{name}, the part at {pos} of the {n} parts it is made of"


def _part_name(role: str) -> str:
    return {"main": "its main block", "tower": "its tower", "annex": "its low annex"}.get(role, "the wing")


def _part_pos(n: int, k: int) -> str:
    return "the left" if k == 0 else "the right" if k == n - 1 else ("the middle" if n == 3 else
                                                                      f"number {k + 1} from the left")


def part_box(mask, parts: list, k: int) -> tuple | None:
    """Part k's box in the photo, (x0, y0, x1, y1): the building's pixels (`mask`) split left to
    right in proportion to the parts' widths (the form's metres, else their shares), widened by
    a margin on each side; from a little above the part's highest pixel to the building's foot.
    At least half as wide as tall, so a tower's top keeps its surroundings. None without pixels."""
    import numpy as np
    cols = np.flatnonzero(mask.any(axis=0))
    if len(cols) == 0:
        return None
    h_img, w_img = mask.shape
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    widths = [float(p.get("width_m") or 0) for p in parts]
    if not all(w > 0 for w in widths):
        widths = [float(p.get("share") or 0) or 1.0 for p in parts]
    total = sum(widths) or 1.0
    a = x0 + (x1 - x0) * sum(widths[:k]) / total
    b = x0 + (x1 - x0) * sum(widths[:k + 1]) / total
    margin = max(0.06 * (x1 - x0), 0.2 * (b - a))
    pa, pb = max(0, int(a - margin)), min(w_img, int(b + margin) + 1)
    rows = np.flatnonzero(mask[:, int(a):max(int(a) + 1, int(b))].any(axis=1))
    rows_all = np.flatnonzero(mask.any(axis=1))
    top = int(rows[0]) if len(rows) else int(rows_all[0])
    foot = int(rows_all[-1]) + 1
    top = max(0, int(top - 0.12 * (foot - top)))
    foot = min(h_img, int(foot + 0.04 * (foot - top)))
    need = (foot - top) // 2 - (pb - pa)                # at least half as wide as tall
    if need > 0:
        pa, pb = max(0, pa - need // 2 - 1), min(w_img, pb + need // 2 + 1)
    return pa, top, pb, foot


def part_view(photo, mask, parts: list, k: int, role: str, view: str = "full"):
    """What a part's question is shown and how it names the part: "full" the whole photo
    (`part_words`); "crop" a close-up cut out around the part (`part_box`); "box" the whole
    photo with a red box drawn round the part. Falls back to "full" without the building's pixels
    or a box; a one-part building's close-up is the building itself."""
    n = len(parts)
    box = part_box(mask, parts, k) if view != "full" and mask is not None else None
    if box is None:
        return photo, part_words(n, k, role)
    if view == "crop":
        if n <= 1:
            return photo.crop(box), "the building in this close-up"
        return photo.crop(box), (f"{_part_name(role)}, the part in the middle of this close-up (cut out of a "
                                 f"photo of the whole building, which is made of {n} parts; this is the one at "
                                 f"{_part_pos(n, k)})")
    from PIL import ImageDraw
    im = photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    d.rectangle(box, outline=(255, 0, 0), width=max(3, im.width // 250))
    if n <= 1:
        return im, "the whole building, inside the red box"
    return im, f"{_part_name(role)}, the part inside the red box ({_part_pos(n, k)} of the {n} parts it is made of)"


def critique(vlm, photo, render, state_text: str) -> dict:
    """The model's comparison of a build (render) with its photo, parsed (see CRITIQUE)."""
    from .edits import feature_menu, roof_menu
    from .styles import KINDS, STYLES
    prompt = (CRITIQUE.replace("STATE", state_text).replace("STYLES", ", ".join(k for k in STYLES if k != "other"))
              .replace("KINDS", ", ".join(k for k in KINDS if k != "other")).replace("ROOFS", roof_menu())
              .replace("FEATURES", ", ".join(feature_menu())))
    text = ask(*vlm, [_small(photo), _small(render)], prompt, 1100)
    js = parse_json(text) or {}
    js["text"] = text
    try:
        js["resemblance"] = max(0.0, min(10.0, float(js.get("resemblance"))))
    except (TypeError, ValueError):
        js["resemblance"] = None
    return js


def compare(vlm, photo, a, b, rng=None, order: str | None = None) -> dict:
    """Which of builds `a` and `b` (renders) is closer to the photo: {"closer": "a" | "b" |
    None (not sure), "order"}; the two are shown in a random order (the model may lean to one
    place), or in `order` ("ab": `a` first, "ba")."""
    rng = rng or random.Random()
    flip = (rng.random() < 0.5) if order is None else order == "ba"
    first, second = (b, a) if flip else (a, b)
    text = ask(*vlm, [_small(photo), _small(first), _small(second)], COMPARE, 60)
    js = parse_json(text) or {}
    try:
        pick = int(js.get("closer"))
    except (TypeError, ValueError):
        pick = None
    closer = None
    if pick in (2, 3) and js.get("sure", True) is not False:
        closer = ("a" if pick == 2 else "b") if not flip else ("b" if pick == 2 else "a")
    return {"closer": closer, "order": "ba" if flip else "ab", "text": text}
