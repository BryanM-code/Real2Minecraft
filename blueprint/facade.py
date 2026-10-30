"""Windows, doors and garage doors where the photo has them. MapAnything's points place each
of the vision model's boxes on the building (`measure.opening_positions`: its wall, how far
along it and how high up it, as fractions); `place_openings_3d` puts each on the build's wall
at the same fractions (across the front by the building's width, on the part that is there;
along a side wall by that wall's length; up by its wall from foot to eave, or to a gable
facing the front), snapped to whole blocks at the heights a builder would use. The openings
compose spread evenly by count are replaced on every wall the photo shows; the walls it does
not show keep theirs. What cannot be placed (no wall there, a clash) is reported; garage doors
that do not fit a side wall make it longer rather than being dropped.

`fractions` gives a build's openings in the same terms, for the scores (`score.match3d`).
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .spec import load, wall_rows

def extents(bp: dict, b: dict) -> dict:
    """The build's box as the photo's fractions see it: x across the front, the front and
    back faces, the foot (the lowest mass's base) and the top (roofs and chimneys included)."""
    ms = bp["masses"]
    return {"x0": min(m["x"] for m in ms), "x1": max(m["x"] + m["w"] for m in ms),
            "front": max(m["z"] + m["d"] for m in ms), "back": min(m["z"] for m in ms),
            "foot": min(m["y"] for m in ms), "top": b["origin"][1] + b["dims"][1]}


def _column(b: dict, k: int, m: dict, side: str, t: float) -> tuple:
    """(foot, top) in world rows of a mass's wall column at t (wall units along the wall from
    its left end seen from outside): the mass's foot, and the top of the mass's own blocks in
    the wall's plane there (its eave, or the gable's edge on a gable end)."""
    length = m["w"] if side in ("south", "north") else m["d"]
    ti = int(min(length - 1, max(0, math.floor(t))))
    x, z = {"south": (m["x"] + ti, m["z"] + m["d"] - 1), "north": (m["x"] + m["w"] - 1 - ti, m["z"]),
            "east": (m["x"] + m["w"] - 1, m["z"] + m["d"] - 1 - ti), "west": (m["x"], m["z"] + ti)}[side]
    ox, oy, oz = b["origin"]
    xi, zi = x - ox, z - oz
    top = m["y"] + wall_rows(m)
    if 0 <= xi < b["dims"][0] and 0 <= zi < b["dims"][2]:
        ys = np.nonzero(b["part"][xi, :, zi] == k + 1)[0]
        if len(ys):
            top = max(top, oy + int(ys.max()) + 1)
    return m["y"], top


def _wall_top(b: dict, k: int, m: dict, side: str, t: float) -> tuple:
    """(foot, top) the opening's height is measured against, as `measure` measures the
    photo's: across the front its column's top, on a side wall the eave (the lower of its two
    ends' tops)."""
    if side == "south":
        return _column(b, k, m, side, t)
    length = m["w"] if side == "north" else m["d"]
    ends = [_column(b, k, m, side, u) for u in (0.5, length - 0.5)]
    return m["y"], min(e[1] for e in ends)


def _shown(bp: dict, side_seen: str | None) -> set:
    """(mass id, side) of the walls a photo from the front (and `side_seen`) shows: the front
    walls and that side's walls not wholly behind another mass."""
    ms = [m for m in bp["masses"] if m.get("shape", "box") == "box" and not m.get("open")]
    out = set()
    for m in ms:
        ahead = [o for o in ms if o is not m and o["z"] + o["d"] > m["z"] + m["d"]]
        cover = sum(max(0, min(o["x"] + o["w"], m["x"] + m["w"]) - max(o["x"], m["x"])) for o in ahead)
        if cover < m["w"]:
            out.add((m["id"], "south"))
        for side, sign in (("east", 1), ("west", -1)):
            if side_seen != ("right" if side == "east" else "left"):
                continue
            edge = m["x"] + m["w"] if sign > 0 else m["x"]
            beyond = [o for o in ms if o is not m and sign * ((o["x"] + o["w"] if sign > 0 else o["x"]) - edge) > 0]
            cover = sum(max(0, min(o["z"] + o["d"], m["z"] + m["d"]) - max(o["z"], m["z"])) for o in beyond)
            if cover < m["d"]:
                out.add((m["id"], side))
    return out


GARAGE_RATIO = 0.85       # a door this wide for its height (the photo's metres), at the wall's foot: a garage door


def kind_of(o: dict, garages: bool = True) -> str:
    """window / door / garage for one of `measure.opening_positions`'s records. With
    `garages` False (the reading says the building has no garage doors: right on 46 of the 50
    test photos, where the boxes' labels put garage doors on 5 houses that have none, from
    basement windows and porch openings) a garage door's box is a door when it is tall and at
    the foot of its wall, else a window."""
    label = str(o.get("label") or "window").lower()
    bottom = float(o.get("up_wall", o["up"])) - float(o.get("h_wall", o["h"])) / 2.0
    w_m, h_m = float(o.get("w_m", 0.0)), max(float(o.get("h_m", 1.0)), 1e-6)
    if "garage" in label:
        if garages:
            return "garage"
        return "door" if h_m >= 1.4 * w_m and bottom <= 0.12 else "window"
    if label == "door":
        return "garage" if garages and w_m >= GARAGE_RATIO * h_m and bottom <= 0.12 else "door"
    return "window"


def _locate(o: dict, bp: dict, b: dict, e: dict):
    """(mass, side, t, width) of one of the photo's openings on the build: across the front on
    the mass there (of those it is not above, the one whose front is at its depth; one just
    past a mass's end is that mass's), on a side wall the outermost mass's; t and width in
    blocks along the wall. None when no mass is there."""
    W, D, H = e["x1"] - e["x0"], max(1, e["front"] - e["back"]), max(1, e["top"] - e["foot"])
    index = {m["id"]: k for k, m in enumerate(bp["masses"])}
    boxes = [m for m in bp["masses"] if m.get("shape", "box") == "box" and not m.get("open")]
    if not boxes:                                      # round towers (or posts) only: no flat wall to put it on
        return None
    y_global = e["foot"] + float(o["up"]) * H
    if o["wall"] == "front":
        x = e["x0"] + float(o["along"]) * W
        wb = float(o["w"]) * W
        z_target = e["front"] - float(o.get("into", 0.0)) * D
        options = []
        for m in boxes:
            if not (m["x"] - wb / 2 - 1 <= x <= m["x"] + m["w"] + wb / 2 + 1):
                continue
            half = min(wb / 2 + 1, m["w"] / 2)
            xc = min(max(x, m["x"] + half), m["x"] + m["w"] - half)
            under = y_global <= _column(b, index[m["id"]], m, "south", xc - m["x"])[1] + 1
            options.append(((not under, abs(xc - x) > 0.5, abs(m["z"] + m["d"] - z_target), -(m["z"] + m["d"])), m, xc))
        if not options:
            return None
        _, m, xc = min(options, key=lambda t: t[0])
        return m, "south", xc - m["x"], wb
    side = "east" if o["wall"] == "right" else "west"
    sign = 1 if side == "east" else -1
    edge = max(sign * (m["x"] + (m["w"] if sign > 0 else 0)) for m in boxes)
    m = max((m for m in boxes if sign * (m["x"] + (m["w"] if sign > 0 else 0)) == edge), key=lambda m: m["z"] + m["d"])
    along = float(o.get("along_wall", o["along"]))
    t = along * m["d"] if side == "east" else (1.0 - along) * m["d"]
    return m, side, t, float(o.get("w_wall", o["w"])) * m["d"]


LEVEL_ROWS = 3            # a part's floor this many rows off what the doors say is moved


def levels(bp: dict, openings: list) -> dict:
    """{mass id: rows its floor should move} from where the photo's doors and garage doors
    stand (MapAnything places them in 3D): each mass's lowest door bottom against the
    entrance's mass's, as fractions of the building's height, in the build's rows; only where
    the build is LEVEL_ROWS or more off (an entrance has steps: a garage door on the same
    ground sits half a metre lower). A split level's garage doors stand well below the
    entrance, where a silhouette cannot tell (the photos' bushes and perspective)."""
    from .build import build
    bp = load(copy.deepcopy(bp))
    b = build(bp)
    e = extents(bp, b)
    H = max(1, e["top"] - e["foot"])
    bottoms, entrance = {}, None
    for o in openings or []:
        kind = kind_of(o)
        if kind not in ("door", "garage"):
            continue
        at = _locate(o, bp, b, e)
        if at is None:
            continue
        m = at[0]
        bottom = float(o["up"]) - float(o["h"]) / 2.0
        bottoms[m["id"]] = min(bottoms.get(m["id"], bottom), bottom)
        if kind == "door" and (entrance is None or bottom < entrance[1]):
            entrance = (m, bottom)
    if entrance is None:
        return {}
    ref, ref_bottom = entrance
    by_id = {m["id"]: m for m in bp["masses"]}
    out = {}
    for mid, bottom in bottoms.items():
        if mid == ref["id"]:
            continue
        want = (bottom - ref_bottom) * H                 # rows below (-) or above the entrance's floor
        have = by_id[mid]["y"] - ref["y"]
        if abs(want - have) >= LEVEL_ROWS:
            out[mid] = int(round(want - have))
    return out


def place_openings_3d(bp: dict, openings: list, side_seen: str | None = None, garages: bool = True) -> tuple:
    """(blueprint, notes): `bp` with the windows, doors and garage doors of the walls the photo
    shows taken from `openings` (`measure.opening_positions`), each where it is on its wall:
    across the front by its share of the building's width, on the mass that is there (the one
    whose front is at the opening's depth, of those it is not above), along a side wall by its
    share of that wall, and up by its share of its wall (`_wall_top`: foot to the eave, or to a
    front gable), so the roofs do not move it; then whole blocks at the heights a builder would use
    (`_snap`). Walls the photo shows no windows on keep compose's windows, walls it does not
    show keep all of compose's openings, and so do kinds the photo shows none of. `side_seen`
    ("left" / "right"): the side wall the photo shows; `garages`: whether the reading says the
    building has garage doors (`kind_of`). The notes list what was placed and what could not be
    (no wall there, clashes), for the conflicts report."""
    from .build import build
    bp = load(copy.deepcopy(bp))
    b = build(bp)
    e = extents(bp, b)
    index = {m["id"]: k for k, m in enumerate(bp["masses"])}
    new, lost = [], []
    for o in openings:
        kind = kind_of(o, garages)
        at = _locate(o, bp, b, e)
        if at is None:
            lost.append(f"{kind} on the {o['wall']} at {float(o['along']):.2f} of its length: no flat wall there")
            continue
        m, side, t, wb = at
        foot, top = _wall_top(b, index[m["id"]], m, side, t)
        up, hh = float(o.get("up_wall", o["up"])), float(o.get("h_wall", o["h"]))
        j = (foot - m["y"]) + up * (top - foot)
        hb = hh * (top - foot)
        length = m["w"] if side == "south" else m["d"]
        o_new = _snap(kind, m, side, t - wb / 2, t + wb / 2, j - hb / 2, j + hb / 2, length, wall_rows(m))
        if o_new:
            new.append(o_new)
        else:
            lost.append(f"{kind} on {m['id']}'s {side} wall at row {j:.0f}: above its walls (in the roof)")
    grown = _fit_garages(bp, new)
    if grown:
        b = build(bp)
    tidy = _tidy(new)
    for o in new:
        if o not in tidy:
            lost.append(f"{o['kind']} on {o['mass']}'s {o['side']} wall at {o['u']}: overlaps another")
    # compose's windows go from the walls the photo's windows landed on (a wall the photo shows
    # none on may be hidden behind trees, not blank); its doors and garage doors from every wall
    # the photo shows, when it shows any (a house has one entrance, not one per guess)
    hit = {(o["mass"], o["side"]) for o in tidy if o["kind"] == "window"}
    shown = _shown(bp, side_seen) | {(o["mass"], o["side"]) for o in tidy}
    kinds = {o["kind"] for o in tidy}
    keep = [o for o in bp["openings"] if not (o["kind"] == "window" and (o["mass"], o["side"]) in hit or
                                              o["kind"] in kinds - {"window"} and (o["mass"], o["side"]) in shown)]
    # an arcade's arches stay open: what the photo shows through one is the wall behind it (a window
    # framed in an arch filled a whitewashed loggia with trim)
    arches = [a for a in keep if a["kind"] == "arch"]
    clear = []
    for o in tidy:
        if any(a["mass"] == o["mass"] and a["side"] == o["side"] and o["u"] < a["u"] + a["w"] and a["u"] < o["u"] + o["w"]
               and o["y"] < a["y"] + a["h"] and a["y"] < o["y"] + o["h"] for a in arches):
            lost.append(f"{o['kind']} on {o['mass']}'s {o['side']} wall at {o['u']}: in an arch (the wall behind it)")
        else:
            clear.append(o)
    tidy = clear
    bp["openings"] = keep + tidy
    placed = {k: sum(1 for o in tidy if o["kind"] == k) for k in ("window", "door", "garage")}
    return bp, {"placed": placed, "lost": lost, "grown": grown}


def photo_fractions(openings: list, garages: bool = True) -> list:
    """The photo's openings (`measure.opening_positions`) as the scores compare them: kind,
    wall, along (across the front, or along its own side wall) and up (its wall's column)."""
    out = []
    for o in openings or []:
        side = o["wall"] != "front"
        out.append({"kind": kind_of(o, garages), "wall": o["wall"],
                    "along": float(o.get("along_wall", o["along"]) if side else o["along"]),
                    "up": float(o.get("up_wall", o["up"]))})
    return out


def fractions(bp: dict, b: dict, side_seen: str | None) -> list:
    """The build's windows, doors and garage doors on the walls a photo from the front (and
    `side_seen`) shows, in the photo's fractions (`photo_fractions`), each with its mass."""
    e = extents(bp, b)
    W = max(1, e["x1"] - e["x0"])
    index = {m["id"]: k for k, m in enumerate(bp["masses"])}
    by_id = {m["id"]: m for m in bp["masses"]}
    shown = _shown(bp, side_seen)
    out = []
    for o in bp["openings"]:
        if o["kind"] not in ("window", "door", "garage") or (o["mass"], o["side"]) not in shown:
            continue
        m = by_id[o["mass"]]
        w, h = o.get("w", 1), o.get("h", 2)
        t = o["u"] + w / 2
        foot, top = _wall_top(b, index[m["id"]], m, o["side"], t)
        up = (m["y"] + o["y"] + h / 2 - foot) / max(1, top - foot)
        if o["side"] == "south":
            wall, along = "front", (m["x"] + t - e["x0"]) / W
        else:                                          # along a side wall from its front corner
            wall = "right" if o["side"] == "east" else "left"
            along = t / m["d"] if o["side"] == "east" else 1.0 - t / m["d"]
        out.append({"kind": o["kind"], "wall": wall, "along": along, "up": up, "mass": o["mass"]})
    return out


def _fit_garages(bp: dict, new: list) -> list:
    """Garage doors side by side on one wall packed with a shared frame column between them,
    near where the photo has them; when they do not fit, narrowed to 3 blocks, and when they
    still do not fit on a side wall, that mass is made deeper (backwards: its front stays)
    rather than a garage door the photo shows being dropped. Returns what was grown."""
    by_id = {m["id"]: m for m in bp["masses"]}
    walls = {}
    for o in new:
        if o["kind"] == "garage":
            walls.setdefault((o["mass"], o["side"]), []).append(o)
    grown = []
    for (mid, side), gs in walls.items():
        if len(gs) < 2:
            continue
        m = by_id[mid]
        gs.sort(key=lambda o: o["u"])
        length = m["w"] if side in ("south", "north") else m["d"]
        need = lambda: sum(o["w"] for o in gs) + len(gs) + 1      # noqa: E731 - frames at both ends, one between
        while need() > length and any(o["w"] > 3 for o in gs):
            max(gs, key=lambda o: o["w"])["w"] -= 1
        if need() > length:
            if side in ("east", "west"):
                extra = need() - length
                m["z"] -= extra
                m["d"] += extra
                if side == "west":                    # u runs from the back corner there
                    for o in bp["openings"] + new:
                        if o["mass"] == mid and o["side"] == "west":
                            o["u"] += extra
                grown.append(f"{mid} made {extra} deeper to fit {len(gs)} garage doors on its {side} wall")
                length = m["d"]
            else:
                continue
        lo = 1
        for o in gs:                                  # left to right, each after the one before
            o["u"] = max(o["u"], lo)
            lo = o["u"] + o["w"] + 1
        hi = length - 1
        for o in reversed(gs):                        # and back inside the wall's right end
            o["u"] = min(o["u"], hi - o["w"])
            hi = o["u"] - 1
    return grown


def _snap(kind: str, m: dict, side: str, t0: float, t1: float, j0: float, j1: float, length: int, rows: int):
    """An opening on a wall from its extent in wall units (t along, j up): whole blocks,
    inside the wall, and at the heights a builder would put it: a window in the window band
    of the storey it is in, a door on the ground floor (one wholly on an upper floor, a
    balcony's, as a window there), a garage door at the foot of the wall (below the floor
    above it); None for a window above the wall's top (in a gable)."""
    sh = int(m.get("storey_h") or 5)
    floors = sorted(int(f) for f in (m.get("floors") or [0]))
    width = t1 - t0
    if kind == "garage":
        w = max(3, min(length - 2, int(round(width))))
        h = max(3, min(rows - 1, int(round(j1 - max(0.0, j0))), sh - 1 if len(floors) > 1 else sh))
        u = max(0, min(length - w, int(round((t0 + t1) / 2 - w / 2))))
        return {"mass": m["id"], "side": side, "kind": "garage", "u": u, "y": 0, "w": w, "h": h}
    jc = (j0 + j1) / 2
    floor = max([f for f in floors if f <= jc] or floors[:1])
    if kind == "door" and floor > floors[0] and j0 > floor:
        kind = "window"                                # a door upstairs (a balcony's): a tall window there
    if kind == "door":                                 # always 2 high, on the ground floor
        floor = floors[0]
        w = 2 if width >= 1.6 else 1
        u = max(1, min(length - w - 1, int(round((t0 + t1) / 2 - w / 2))))
        return {"mass": m["id"], "side": side, "kind": "door", "u": u, "y": max(1, min(rows - 3, floor + 1)), "w": w}
    sill = 2 if sh >= 6 else 1
    y = floor + sill
    h = max(2, min(int(round(j1 - j0)), sh - sill - 1, rows - 1 - y))
    if h < 1 or y + h > rows - 1:
        return None
    w = max(1, min(int(round(width)), length - 2))
    u = max(1, min(length - 1 - w, int(round((t0 + t1) / 2 - w / 2))))
    return {"mass": m["id"], "side": side, "kind": "window", "u": u, "y": y, "w": w, "h": h}


def _tidy(openings: list) -> list:
    """Drop openings that overlap one already kept on the same wall (frames included), the
    bigger first; a door or garage door before windows."""
    order = sorted(openings, key=lambda o: (o["kind"] == "window", -(o["w"] * o.get("h", 2))))
    kept = []
    for o in order:
        u0, u1 = o["u"] - 1, o["u"] + o["w"]
        y0, y1 = o["y"] - 1, o["y"] + o.get("h", 2)
        clash = any(k["mass"] == o["mass"] and k["side"] == o["side"] and u0 <= k["u"] + k["w"] and k["u"] - 1 <= u1
                    and y0 <= k["y"] + k.get("h", 2) and k["y"] - 1 <= y1
                    and not (o["kind"] == k["kind"] == "garage"          # garage doors share a frame column
                             and (o["u"] >= k["u"] + k["w"] + 1 or k["u"] >= o["u"] + o["w"] + 1))
                    for k in kept)
        if not clash:
            kept.append(o)
    return kept
