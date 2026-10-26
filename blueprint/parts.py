"""The architectural parts beyond walls and gable roofs: flat roofs (with battlements),
domes, sawtooth roofs, wall decor (half-timbering, quoins, string courses, pilasters,
buttresses, curtain walls) and the features attached to a mass (porticos, porches,
balconies, dormers, bay windows, smokestacks, canopies, awnings, windmill sails, galleries,
lanterns). Everything goes through `build`'s owner-tracked canvas.

Wall-attached parts use the wall's own frame (`build.local`): t along the wall from its left
end seen from outside, j rows above the mass's base, o blocks out from the wall's plane.
"""
from __future__ import annotations

import math

from .build import BACK, LEFT, OUT, RIGHT, Canvas, _empty, footprint, local, perimeter, _along
from .spec import mass_bounds, wall_length, wall_rows


def _free(c) -> bool:
    """Empty, or something a part may grow over (the ground, plants)."""
    return c is None or c[2][0] in ("site", "plant")


def _spread(n: int, lo: int, hi: int) -> list:
    """`n` positions from `lo` to `hi` inclusive, the ends included."""
    if n <= 1:
        return [(lo + hi) // 2]
    return sorted({round(lo + (hi - lo) * i / (n - 1)) for i in range(n)})


# --- roofs ------------------------------------------------------------------------------

def flat_roof(cv: Canvas, m: dict, mats: dict, ok) -> None:
    """A deck inside a parapet: capped, or crenellated with `battlements`; a cornice under
    the eave with an overhang."""
    r, mid = m["roof"], m["id"]
    e = m["y"] + wall_rows(m)
    cells = footprint(m)
    edge = perimeter(cells)
    p = int(r["parapet"])
    for (x, z) in edge:
        for y in range(e, e + p):
            cv.put((x, y, z), "full", mats["wall"], ("wall", mid), ok)
        if r.get("battlements"):
            if (x + z) % 2 == 0:
                cv.put((x, e + p, z), "full", mats["wall"], ("wall", mid), ok)
        else:
            cv.put((x, e + p, z), "slab_bottom", mats["cap"], ("roof", mid), ok)
    for x, z in cells - set(edge):
        cv.put((x, e, z), "slab_bottom", mats["roof"], ("roof", mid), ok)
    if int(r["overhang"]) > 0 and not r.get("battlements"):
        x0, x1, z0, z1 = mass_bounds(m)
        grown = footprint(dict(m, x=x0 - 1, z=z0 - 1, w=m["w"] + 2, d=m["d"] + 2))
        for x, z in set(perimeter(grown)) - cells:
            cv.put((x, e - 1, z), "slab_top", mats["roof_edge"], ("roof", mid), ok)


def dome(cv: Canvas, mats: dict, m: dict | None, f: dict, ok=None) -> None:
    """A dome shell: on top of mass `m` (centred, as wide as it allows) or at f's x, z, r
    and base row y; `drum` rows of round wall with windows under it, `lantern` on top."""
    ok = ok or (lambda c: c is None or c[2][0] in ("site", "plant", "roof"))
    if m is not None:
        x0, x1, z0, z1 = mass_bounds(m)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        R = min(m["w"], m["d"]) / 2 - (0 if f.get("kind") != "dome" else 1)
        y0 = f.get("y", m["y"] + wall_rows(m) + (1 if f.get("kind") == "dome" else 0))
        mid = m["id"]
    else:
        cx, cz, R, y0, mid = f["x"], f["z"], f["r"], f["y"], "dome"
    drum = int(f.get("drum", 0))
    squash = float(f.get("squash", 1.0))
    owner = ("roof", mid)
    tops = {}
    for x in range(math.floor(cx - R), math.ceil(cx + R) + 1):
        for z in range(math.floor(cz - R), math.ceil(cz + R) + 1):
            d = math.hypot(x - cx, z - cz)
            if d <= R:
                tops[(x, z)] = y0 + drum + int(round(math.sqrt(max(R * R - d * d, 0.0)) * squash))
    ring = {p for p in tops if any((p[0] + dx, p[1] + dz) not in tops for dx, dz in OUT.values())}
    if m is not None and f.get("kind") != "dome":         # a dome roof: a deck round its foot
        for x, z in footprint(m) - set(tops):
            cv.put((x, y0, z), "slab_bottom", mats["roof"], owner, ok)
    for (x, z), top in tops.items():
        low = min([tops.get((x + dx, z + dz), y0 + drum) for dx, dz in OUT.values()] + [top]) + 1
        low = min(low, top)
        if (x, z) in ring:
            low = y0                                      # the drum (or the dome's foot) is a wall
        for y in range(low, top + 1):
            in_drum = y < y0 + drum
            block, shape = mats["dome"], "full"
            if in_drum and (y - y0) % 4 in (1, 2) and (x + z) % 3 == 0:
                block, shape = mats["window"], "pane"     # drum windows
            elif in_drum:
                block = mats["wall"]
            cv.put((x, y, z), shape, block, owner, ok, label="window" if block == mats["window"] else None)
    if f.get("lantern") and tops:
        (x, z), top = max(tops.items(), key=lambda kv: kv[1])
        for y in range(top + 1, top + 3):
            cv.put((x, y, z), "full", mats["window"], owner, _empty, label="window")
        cv.put((x, top + 3, z), "slab_bottom", mats["dome"], owner, _empty)
        cv.put((x, top + 4, z), "fence", mats["railing"], owner, _empty)


def sawtooth(cv: Canvas, m: dict, mats: dict, ok, tooth: int = 6) -> None:
    """Teeth along x: each a slope rising east, then a glazed vertical face."""
    x0, x1, z0, z1 = mass_bounds(m)
    e, mid = m["y"] + wall_rows(m), m["id"]
    for x in range(x0, x1 + 1):
        k = (x - x0) % tooth
        for z in range(z0, z1 + 1):
            cv.put((x, e - 1 + k, z), "stairs_east", mats["roof"], ("roof", mid), ok)
            if k == 0 and x > x0:                          # the glass face over the low edge
                for y in range(e, e - 1 + tooth):
                    cv.put((x, y, z), "full", mats["window"], ("roof", mid), ok)
            if z in (z0, z1):                              # the end walls follow the teeth
                for y in range(e, e - 1 + k):
                    cv.put((x, y, z), "full", mats["wall"], ("wall", mid), _empty)


# --- wall decor -------------------------------------------------------------------------

def _wall_cells(cv: Canvas, m: dict):
    """(x, y, z, side, t, j) of a mass's plain wall blocks (not its base, trim or openings)."""
    x0, x1, z0, z1 = mass_bounds(m)
    mid = m["id"]
    for (x, z), sides in perimeter(footprint(m)).items():
        side = sides[0]
        t = {"south": x - x0, "north": x1 - x, "east": z1 - z, "west": z - z0}[side]
        for j in range(wall_rows(m)):
            p = (x, m["y"] + j, z)
            c = cv.cells.get(p)
            if c is not None and c[2] == ("wall", mid):
                yield p, c, sides, t, j


def decor(cv: Canvas, m: dict, mats: dict) -> None:
    mid = m["id"]
    owner = ("wall", mid)
    rows = wall_rows(m)
    floors = set(m["floors"])
    first = min((f for f in floors if f > 0), default=rows)
    for dec in m.get("decor", []):
        for p, c, sides, t, j in list(_wall_cells(cv, m)):
            plain = c[1] == mats["wall"]
            corner = len(sides) >= 2
            length = wall_length(m, sides[0])
            if dec == "half_timber" and plain and j >= first:
                if t % 3 == 0 or j in floors or j == rows - 1:
                    cv.put(p, "full", mats["trim"], owner, label="trim")
            elif dec == "quoins" and corner and j >= 1:
                cv.put(p, "full", mats["wall_base"] if mats["wall_base"] != mats["wall"] else mats["trim"], owner,
                       label="trim")
            elif dec == "quoins" and j % 2 == 0 and j >= 1 and (t in (1, length - 2)) and plain:
                cv.put(p, "full", mats["wall_base"] if mats["wall_base"] != mats["wall"] else mats["trim"], owner,
                       label="trim")
            elif dec == "bands" and plain and j in floors and j > 0:
                cv.put(p, "full", mats["trim"], owner, label="trim")
            elif dec == "pilasters" and plain and not corner and t % 4 == 2 and 1 < t < length - 2:
                cv.put(p, "full", mats["trim"], owner, label="trim")
            elif dec == "curtain" and plain and not corner:
                if t % 4 == 0 or j in floors:
                    cv.put(p, "full", mats["trim"], owner, label="trim")
                else:
                    cv.put(p, "full", mats["window"], owner, label="window")
        if dec == "buttresses" and m.get("shape", "box") == "box":
            top = max(2, rows * 2 // 3)
            for side in ("south", "north", "east", "west"):
                length = wall_length(m, side)
                for t in range(2, length - 2, 5):
                    for j in range(0, top):
                        cv.put(local(m, side, t, j, 1), "full", mats["wall_base"], ("out", mid), _free)
                    cv.put(local(m, side, t, top, 1), "stairs_" + BACK[side], mats["wall_base"], ("out", mid), _free)


# --- features attached to a wall -------------------------------------------------------

def portico(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """Columns standing out from a wall on a raised floor, an entablature, and a triangular
    pediment (or a flat top)."""
    side, u, w, D = f["side"], f["u"], f["w"], int(f["depth"])
    mid = m["id"]
    owner = ("portico", mid)
    H = int(f["height"]) or min(wall_rows(m) - 1, 2 * int(m["storey_h"]))
    n = int(f["columns"]) or max(2, w // 3 + 1)
    can = lambda c: c is None or c[2][0] in ("site", "plant", "roof", "portico")   # noqa: E731
    for t in range(u, u + w):
        for o in range(1, D + 1):
            cv.put(local(m, side, t, 0, o), "full", mats["wall_base"], owner, can)
        cv.put(local(m, side, t, 0, D + 1), "stairs_" + BACK[side], mats["wall_base"], owner, can)
    for t in _spread(n, u, u + w - 1):
        for j in range(1, H):
            cv.put(local(m, side, t, j, D), "full", mats["column"], owner, can)
    for t in range(u, u + w):
        for o in range(1, D + 1):
            cv.put(local(m, side, t, H, o), "full", mats["trim"], owner, can)
    if f["pediment"]:
        for t in range(u - 1, u + w + 1):
            k = min(t - (u - 1), (u + w) - t)
            jj = H + 1 + k // 2
            shape = "slab_bottom" if k % 2 == 0 else "slab_top"
            for o in range(0, D + 2):
                cv.put(local(m, side, t, jj, o), shape, mats["roof"], owner, can)
            for j in range(H + 1, jj + (1 if shape == "slab_top" else 0)):
                cv.put(local(m, side, t, j, D), "full", mats["trim"], owner, can)
    else:
        for t in range(u - 1, u + w + 1):
            for o in range(1, D + 2):
                cv.put(local(m, side, t, H + 1, o), "slab_bottom", mats["roof_edge"], owner, can)


def porch(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor, wrapping: bool = False) -> None:
    """A deck along a wall with posts, a railing and a lean-to roof; `wrap` carries it round
    the right-hand corner."""
    side, u, w, D, H = f["side"], f["u"], f["w"], int(f["depth"]), int(f["height"])
    mid = m["id"]
    owner = ("porch", mid)
    can = lambda c: c is None or c[2][0] in ("site", "plant", "porch")   # noqa: E731
    post = mats["railing"]
    for t in range(u, u + w):
        for o in range(1, D + 1):
            cv.put(local(m, side, t, 0, o), "full", mats["floor"], owner, can)
    posts = set(range(u, u + w, 3)) | {u + w - 1}
    middle = {u + w // 2 - 1, u + w // 2} if not wrapping else set()
    for t in range(u, u + w):
        if t in posts:
            for j in range(1, H):
                cv.put(local(m, side, t, j, D), "fence", post, owner, can)
        elif f["railing"] and t not in middle:
            cv.put(local(m, side, t, 1, D), "fence", post, owner, can)
    for t in middle:
        cv.put(local(m, side, t, 0, D + 1), "stairs_" + BACK[side], mats["wall_base"], owner, can)
    for t in range(u - 1, u + w + 1):
        for o in range(0, D + 2):
            k = D + 1 - o
            shape = "slab_bottom" if k % 2 == 0 else "slab_top"
            cv.put(local(m, side, t, H + k // 2, o), shape, mats["roof"], owner, can)
    if f.get("wrap") and not wrapping:              # round the corner, to the right unless wrap_side says left
        to = (LEFT if f.get("wrap_side") == "left" else RIGHT)[side]
        ww = min(wall_length(m, to), w)
        u2 = 0 if to == RIGHT[side] else wall_length(m, to) - ww       # starting at the front corner
        porch(cv, m, dict(f, side=to, u=u2, w=ww, wrap=False), mats, pieces, floor, wrapping=True)


def balcony(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """A slab platform outside the wall at row `y` (a floor's row) with a railing round it."""
    side, u, w, D, y = f["side"], f["u"], f["w"], int(f["depth"]), int(f["y"])
    owner = ("out", m["id"])
    for t in range(u, u + w):
        for o in range(1, D + 1):
            cv.put(local(m, side, t, y, o), "slab_top", mats["trim"], owner, _free)
            edge = o == D or t in (u, u + w - 1)
            if edge:
                cv.put(local(m, side, t, y + 1, o), "fence", mats["railing"], owner, _free)


def bay(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """Windows in a box standing out from the wall: glass on its front and sides, a slab
    roof and a sill under it."""
    side, u, w, D, y, h = f["side"], f["u"], f["w"], int(f["depth"]), int(f["y"]), int(f["h"])
    owner = ("out", m["id"])
    for t in range(u, u + w):
        for o in range(1, D + 1):
            cv.put(local(m, side, t, y - 1, o), "slab_top", mats["trim"], owner, _free)
            cv.put(local(m, side, t, y + h, o), "slab_bottom", mats["roof"], owner, _free)
            for j in range(y, y + h):
                corner = t in (u, u + w - 1) and o == D
                face = o == D or t in (u, u + w - 1)
                if corner:
                    cv.put(local(m, side, t, j, o), "full", mats["frame"], owner, _free)
                elif face:
                    cv.put(local(m, side, t, j, o), "pane", mats["window"], owner, _free, label="window")


def awning(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """A sloping canopy over shop windows: high at the wall, low at its edge."""
    side, u, w, y = f["side"], f["u"], f["w"], int(f["y"])
    owner = ("out", m["id"])
    for t in range(u, u + w):
        cv.put(local(m, side, t, y, 1), "slab_top", mats["awning"], owner, _free)
        if int(f["depth"]) > 1:
            cv.put(local(m, side, t, y, 2), "slab_bottom", mats["awning"], owner, _free)


def dormer(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """A gabled window standing out of the roof slope that faces `side`, its front just
    inside the wall line, `u` along the wall from its left end seen from outside."""
    if not pieces:
        return
    side, u, w, h = f["side"], f["u"], int(f["w"]), int(f["h"])
    mid = m["id"]
    owner = ("dormer", mid)
    can = lambda c: c is None or c[2][0] in ("roof", "dormer")   # noqa: E731
    xz_front = [local(m, side, t, 0, -1) for t in range(u, u + w)]
    rows = [pieces[(x, z)]["y"] for x, _, z in xz_front if (x, z) in pieces]
    if not rows:
        return
    base = max(rows) - m["y"]                       # the dormer's floor row, relative to the mass
    depth = h + 3
    top = base + h + 1                              # its head
    for t in range(u - 1, u + w + 1):
        for o in range(-depth, 0):
            for j in range(base, top + 1):
                cheek = t in (u - 1, u + w)
                front = o == -1
                if front and not cheek and base < j < top:
                    cv.put(local(m, side, t, j, o), "pane", mats["window"], owner, can, label="window")
                elif cheek or front:
                    cv.put(local(m, side, t, j, o), "full", mats["frame"] if front else mats["wall"], owner, can,
                           label="trim" if front else "wall")
                else:
                    cv.put(local(m, side, t, j, o), "air", None, None, can)
    for t in range(u - 2, u + w + 2):               # its own little gable, ridge running back
        k = min(t - (u - 2), (u + w + 1) - t)
        ridge = t - (u - 2) == (u + w + 1) - t
        facing = None if ridge else (RIGHT[side] if t - (u - 2) < (u + w + 1) - t else LEFT[side])
        for o in range(-depth, 1):
            p = local(m, side, t, top + k, o)
            if facing is None:
                cv.put(p, "slab_bottom", mats["roof"], owner, can)
            else:
                cv.put(p, "stairs_" + facing, mats["roof"], owner, can)
        if u - 1 <= t <= u + w:                     # the little gable's triangle over its front
            for j in range(top + 1, top + k):
                cv.put(local(m, side, t, j, -1), "full", mats["wall"], owner, can)


# --- free-standing and top-mounted features ----------------------------------------------

def stack(cv: Canvas, m, f: dict, mats: dict, pieces, floor) -> None:
    """A round industrial chimney."""
    r = int(f["r"])
    for x in range(f["x"] - r, f["x"] + r + 1):
        for z in range(f["z"] - r, f["z"] + r + 1):
            if (x - f["x"]) ** 2 + (z - f["z"]) ** 2 <= r * r + 0.5:
                for y in range(f.get("y", floor), f["top"]):
                    cv.put((x, y, z), "full", mats["chimney"], ("feature",))


def canopy(cv: Canvas, m, f: dict, mats: dict, pieces, floor) -> None:
    """A flat roof on posts (a fuel station's)."""
    x0, x1 = sorted((f["x0"], f["x1"]))
    z0, z1 = sorted((f["z0"], f["z1"]))
    top, step = f["top"], int(f["posts"])
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) or z in (z0, z1)
            cv.put((x, top, z), "full" if edge else "slab_top", mats["trim"], ("feature",), _free)
    for x in sorted(set(range(x0 + 1, x1, step)) | {x0 + 1, x1 - 1}):
        for z in sorted({z0 + 1, z1 - 1}):
            for y in range(f.get("y", floor), top):
                cv.put((x, y, z), "full", mats["column"], ("feature",), _free)


def sails(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """Windmill sails: a cross of lattice arms in front of a wall near the top."""
    side = f["side"]
    length = wall_length(m, side)
    tc, jc = length // 2, wall_rows(m) - 2
    L = int(f["length"]) or max(4, wall_rows(m) - 3)
    owner = ("out", m["id"])
    cv.put(local(m, side, tc, jc, 1), "full", mats["trim"], owner, _free)
    for i in range(1, L + 1):
        for st, sj in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            if jc + sj * i < 2:                            # the sweep stops above the ground
                continue
            cv.put(local(m, side, tc + st * i, jc + sj * i, 1), "fence", mats["railing"], owner, _free)
            if i >= 2:                                     # the sail cloth beside each arm
                cv.put(local(m, side, tc + st * i, jc + sj * (i - 1), 1), "slab_top", mats["wall_base"], owner, _free)


def gallery(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """A railed walkway round a mass at row `y`."""
    x0, x1, z0, z1 = mass_bounds(m)
    inner = footprint(m)
    outer = footprint(dict(m, x=x0 - 1, z=z0 - 1, w=m["w"] + 2, d=m["d"] + 2))
    y = m["y"] + int(f["y"])
    owner = ("out", m["id"])
    for x, z in outer - inner:
        cv.put((x, y, z), "slab_top", mats["trim"], owner, _free)
        cv.put((x, y + 1, z), "fence", mats["railing"], owner, _free)


def lantern(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """A small room on top of a mass under a cap: glass all round (a lighthouse's lantern), or
    `open`, posts at its corners and a rail at its foot (a belfry, a city hall tower's lantern).
    `cap`: pyramid (the default; a cone on a round mass), dome or spire; `shrink`: blocks in from
    the mass's sides (-1: one where the mass is 5 or more wide)."""
    from .build import _mass, _roof
    x0, x1, z0, z1 = mass_bounds(m)
    shrink = int(f.get("shrink", -1))
    if shrink < 0:
        shrink = 1 if min(m["w"], m["d"]) >= 5 else 0
    shrink = max(0, min(shrink, (min(m["w"], m["d"]) - 3) // 2))
    round_ = m.get("shape", "box") != "box"
    cap = str(f.get("cap") or "")
    mini = {"id": m["id"] + "_lantern", "x": x0 + shrink, "z": z0 + shrink, "w": m["w"] - 2 * shrink,
            "d": m["d"] - 2 * shrink, "y": m["y"] + wall_rows(m) + 1, "walls": int(f["h"]), "floors": [0],
            "plinth": {s: 0 for s in OUT}, "corner_trim": False, "shape": m.get("shape", "box"),
            "roof": {"type": "spire" if cap == "spire" else "cone" if round_ else "pyramid", "pitch": 1.0,
                     "overhang": 0 if cap == "spire" else 1, "ridge": "x", "high": "north", "parapet": 1}}
    owner = ("roof", m["id"])
    if f.get("open"):                                  # corner posts, a rail between them, a lintel ring
        mx0, mx1, mz0, mz1 = mass_bounds(mini)
        ring = perimeter(footprint(mini))
        corners = {(mx0, mz0), (mx0, mz1), (mx1, mz0), (mx1, mz1)}
        posts = corners if not round_ else {p for p in ring if (p[0] + p[1]) % 3 == 0}
        top = mini["y"] + mini["walls"] - 1
        for x, z in ring:
            cv.put((x, mini["y"], z), "fence", mats["railing"], owner, _empty)
            cv.put((x, top, z), "full", mats["trim"], owner, _empty)
            if (x, z) in posts:
                for y in range(mini["y"], top):
                    cv.put((x, y, z), "full", mats["trim"], owner)
        for x, z in footprint(mini):                   # its floor
            cv.put((x, mini["y"] - 1, z), "full", mats["trim"], owner)
    else:
        lmats = dict(mats, wall=mats["window"], wall_base=mats["window"])
        _mass(cv, mini, lmats)
    if cap == "dome":
        dome(cv, mats, mini, {"squash": 1.0})
    else:
        _roof(cv, mini, dict(mats, roof=mats["dome"], roof_edge=mats["dome"], wall=mats["trim"]))


def dome_feature(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    dome(cv, mats, m, dict(f, kind="dome"))


def finial(cv: Canvas, m: dict, f: dict, mats: dict, pieces, floor) -> None:
    """A mast on the middle of a mass's roof (a pagoda's spire): fence posts with two rings."""
    x0, x1, z0, z1 = mass_bounds(m)
    cx, cz = (x0 + x1) // 2, (z0 + z1) // 2
    top = max((y for (x, y, z) in cv.cells if x == cx and z == cz), default=m["y"] + wall_rows(m))
    h = int(f["h"]) or max(4, round(min(m["w"], m["d"]) * 0.7))
    owner = ("roof", m["id"])
    for j in range(1, h + 1):
        cv.put((cx, top + j, cz), "fence", mats["railing"], owner, _empty)
    for j in sorted({max(1, h // 3), max(2, 2 * h // 3)}):
        for dx, dz in OUT.values():
            cv.put((cx + dx, top + j, cz + dz), "slab_bottom", mats["roof_edge"], owner, _empty)


FEATURES = {"portico": portico, "porch": porch, "balcony": balcony, "bay": bay, "awning": awning, "dormer": dormer,
            "stack": stack, "canopy": canopy, "sails": sails, "gallery": gallery, "lantern": lantern,
            "dome": dome_feature, "finial": finial}
