"""What the local vision model (`describe.load_model`) says about a photo beyond the form:
where the main building is and which of its sides the photo shows (`locate`), its windows'
and doors' boxes (`openings`), and, for the evaluator, what is wrong with a build against the
photo (`critique`) and which of two builds is closer to it (`compare`). Boxes come in the
model's 0-1000 coordinates. Every answer is parsed into fixed choices; the critique offers
only the roofs, features, styles and kinds the builder has (`edits`), and anything else goes
in its "other" answer, which the evaluator logs as a capability gap.
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

OPENINGS = """Locate every window and every door on the main building's visible walls (garage doors count as doors).
Output JSON only: a list of {"bbox_2d": [x1, y1, x2, y2], "label": "window" or "door"}."""

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
 "roofs": [one entry per part of the build: {"part": <its number>, "type_ok": true|false,
           "type": "<if wrong, the photo's: ROOFS>", "pitch": "ok|steeper|flatter" (should it be steeper or flatter)}],
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


def critique(vlm, photo, render, state_text: str) -> dict:
    """The model's comparison of a build (render) with its photo, parsed (see CRITIQUE)."""
    from .edits import feature_menu, roof_menu
    from .styles import KINDS, STYLES
    prompt = (CRITIQUE.replace("STATE", state_text).replace("STYLES", ", ".join(k for k in STYLES if k != "other"))
              .replace("KINDS", ", ".join(k for k in KINDS if k != "other")).replace("ROOFS", roof_menu())
              .replace("FEATURES", ", ".join(feature_menu())))
    text = ask(*vlm, [_small(photo), _small(render)], prompt, 700)
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
