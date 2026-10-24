"""Blueprint -> blocks: mcvox-v3 (shape, material) grids (X, Y, Z), the human builds'
representation, so a built blueprint is written by `photofirst.schem.write_schem` (which
derives stair corners, door halves and fence / pane arms) and scored like any other build.

Parts are drawn in a fixed order: the site's ground, every mass's walls and floors, the wall
decor, the roofs from the lowest eave up, the openings, the features, then the site's boxes
and plants. Each block remembers the part that placed it, so a roof never cuts into another
mass's wall, an opening only replaces its own wall, and plants only grow into air.
"""
from __future__ import annotations

import math

import numpy as np

from mcvoxel import vocab as V

from .spec import load, mass_bounds, material_index, resolve_materials, shape_ok, wall_rows

OUT = {"south": (0, 1), "north": (0, -1), "east": (1, 0), "west": (-1, 0)}        # outward (dx, dz)
BACK = {"south": "north", "north": "south", "east": "west", "west": "east"}
LEFT = {"south": "west", "north": "east", "east": "south", "west": "north"}       # seen from outside
RIGHT = {v: k for k, v in LEFT.items()}


class Canvas:
    """Sparse blocks: (x, y, z) -> (shape name, block name, owner); owner is a tuple such
    as ("wall", mass id), ("roof", mass id), ("site",)."""

    def __init__(self):
        self.cells = {}

    def put(self, p, shape, block, owner, when=None) -> bool:
        """Place (or with shape "air" clear) unless `when(current cell or None)` is false.
        A shape the block lacks falls back (stairs -> full, slab -> full, fence -> wall)."""
        if when is not None and not when(self.cells.get(p)):
            return False
        if shape == "air":
            self.cells.pop(p, None)
            return True
        shape = fit(shape, block)
        if shape is None:
            return False
        self.cells[p] = (shape, block, owner)
        return True


def fit(shape: str, block: str):
    """`shape` if `block` has it, else the nearest it has; None for a part that has no
    stand-in (a trapdoor shutter, a door)."""
    if shape_ok(shape, block):
        return shape
    if shape.startswith(("trapdoor", "door")):
        return None
    for alt in {"fence": ("wall", "full"), "wall": ("fence", "full"), "pane": ("full",)}.get(shape, ("full",)):
        if shape_ok(alt, block):
            return alt
    return shape                                   # grids() reports it


def _empty(c) -> bool:
    return c is None


# --- footprints -------------------------------------------------------------------------

def footprint(m: dict) -> set:
    """The (x, z) columns a mass stands on: its box, or the round / octagonal shape inside it."""
    x0, x1, z0, z1 = mass_bounds(m)
    shape = m.get("shape", "box")
    cells = set()
    cx, cz, rx, rz = (x0 + x1) / 2, (z0 + z1) / 2, m["w"] / 2, m["d"] / 2
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            u, v = abs(x - cx) / rx, abs(z - cz) / rz
            if shape == "box" or (shape == "round" and u * u + v * v <= 1.0) or (shape == "octagon" and u + v <= 1.45):
                cells.add((x, z))
    return cells


def perimeter(cells: set) -> dict:
    """{(x, z): sides facing out} of the columns at the edge of `cells`."""
    out = {}
    for x, z in cells:
        sides = tuple(s for s, (dx, dz) in OUT.items() if (x + dx, z + dz) not in cells)
        if sides:
            out[(x, z)] = sides
    return out


def _ring(x0, x1, z0, z1):
    """Perimeter columns (x, z, sides they face) of a box."""
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            sides = tuple(s for s, on in (("west", x == x0), ("east", x == x1), ("north", z == z0),
                                          ("south", z == z1)) if on)
            if sides:
                yield x, z, sides


def _along(m: dict, side: str, t: int):
    """(x, z) of the wall column `t` blocks from the wall's left end, seen from outside
    (round masses: the outermost column of their shape there), or None."""
    x0, x1, z0, z1 = mass_bounds(m)
    if m.get("shape", "box") == "box":
        return {"south": (x0 + t, z1), "north": (x1 - t, z0), "east": (x1, z1 - t), "west": (x0, z0 + t)}[side]
    cells = footprint(m)
    if side in ("south", "north"):
        x = x0 + t if side == "south" else x1 - t
        zs = [z for (cx, z) in cells if cx == x]
        return (x, max(zs) if side == "south" else min(zs)) if zs else None
    z = z1 - t if side == "east" else z0 + t
    xs = [x for (x, cz) in cells if cz == z]
    return (max(xs) if side == "east" else min(xs), z) if xs else None


def local(m: dict, side: str, t: int, j: int, o: int) -> tuple:
    """World (x, y, z) of a point on a box mass's wall: `t` along it from the left seen from
    outside, `j` rows above the mass's base, `o` blocks out from the wall's plane."""
    x0, x1, z0, z1 = mass_bounds(m)
    y = m["y"] + j
    return {"south": (x0 + t, y, z1 + o), "north": (x1 - t, y, z0 - o),
            "east": (x1 + o, y, z1 - t), "west": (x0 - o, y, z0 + t)}[side]


# --- masses -----------------------------------------------------------------------------

def _mass(cv: Canvas, m: dict, mats: dict) -> None:
    y0, rows, mid = m["y"], wall_rows(m), m["id"]
    cells = footprint(m)
    edge = perimeter(cells)
    box = m.get("shape", "box") == "box"
    for (x, z), sides in edge.items():
        base = max(m["plinth"][s] for s in sides)
        corner = box and len(sides) == 2 and m["corner_trim"]
        for y in range(y0, y0 + rows):
            role = "wall_base" if y - y0 < base else "trim" if corner else "wall"
            cv.put((x, y, z), "full", mats[role], ("wall", mid))
    for f in m["floors"]:
        for x, z in cells:
            if (x, z) not in edge:
                cv.put((x, y0 + f, z), "full", mats["floor"], ("floor", mid))


def _clear_insides(cv: Canvas, masses: list) -> None:
    """Inside every mass (above its base row, below its eave) nothing but its own floors:
    the ground it is dug into goes, and so do other masses' walls (masses join as a union)."""
    for m in masses:
        cells = footprint(m)
        inner = cells - set(perimeter(cells))
        y0, top, mid = m["y"], m["y"] + wall_rows(m), m["id"]
        for x, z in inner:
            for y in range(y0 + 1, top):
                c = cv.cells.get((x, y, z))
                if c is not None and (c[2] == ("site",) or (c[2][0] == "wall" and c[2][1] != mid)):
                    del cv.cells[(x, y, z)]


# --- roofs ------------------------------------------------------------------------------

def _slopes(roof: dict, xa: int, xb: int, za: int, zb: int) -> dict:
    """{(x, z): (k, facing)}: k rows up from the roof's outer edge; facing is the way the
    slope climbs (its stairs face that way), None on a ridge where two slopes meet."""
    t = roof["type"]
    if t in ("gable", "gambrel"):
        keys = ("north", "south") if roof["ridge"] == "x" else ("east", "west")
    elif t == "shed":
        keys = (roof["high"],)
    else:                                                     # hip, pyramid, mansard, pagoda, spire, cone
        keys = ("north", "south", "east", "west")
    out = {}
    for x in range(xa, xb + 1):
        for z in range(za, zb + 1):
            # climbing north starts from the south edge, and so on
            d = {"north": zb - z, "south": z - za, "east": x - xa, "west": xb - x}
            k = min(d[f] for f in keys)
            best = [f for f in keys if d[f] == k]
            ridge = any(BACK[a] == b for a in best for b in best)
            out[(x, z)] = (k, None if ridge else best[0])
    return out


def _radial(m: dict, o: int) -> dict:
    """{(x, z): (k, facing)} over a round / octagonal mass's shape grown by `o`: k rings in
    from the edge, facing towards the middle."""
    x0, x1, z0, z1 = mass_bounds(m)
    grown = dict(m, x=x0 - o, z=z0 - o, w=m["w"] + 2 * o, d=m["d"] + 2 * o)
    cells = footprint(grown)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    r = (m["w"] + m["d"]) / 4 + o
    out = {}
    for x, z in cells:
        dx, dz = cx - x, cz - z
        dist = math.hypot(dx, dz)
        k = max(0, int(r - dist - 0.5))
        if dist < 0.75:
            facing = None
        elif abs(dx) >= abs(dz):
            facing = "east" if dx > 0 else "west"
        else:
            facing = "south" if dz > 0 else "north"
        out[(x, z)] = (k, facing)
    return out


def _cross(m: dict, o: int, xa: int, xb: int, za: int, zb: int) -> dict:
    """A cross gable: the main ridge along x, a gable with its ridge along z over the
    middle third, each column taking the higher of the two."""
    main = _slopes({"type": "gable", "ridge": "x"}, xa, xb, za, zb)
    w = xb - xa + 1
    ca, cb = xa + w // 3, xb - w // 3
    wing = _slopes({"type": "gable", "ridge": "z"}, ca - o, cb + o, za, zb)
    out = dict(main)
    for p, (k, f) in wing.items():
        if p in out and k > out[p][0]:
            out[p] = (k, f)
    return out


def _pieces_for(pitch: float, k: int, facing, e: int, x: int, z: int, kind: str, kbreak: int):
    """[(p, shape), ...] of one roof column (its surface piece first), and that piece's row."""
    if kind == "mansard" and k > kbreak:                    # the flat top over the steep sides
        y = e + 2 * kbreak
        return [((x, y, z), "full")], y
    if kind == "gambrel" and k > kbreak:                    # gentle upper slopes over steep lower ones
        base = e + 2 * kbreak + 1
        kk = k - kbreak - 1
        y = base + kk // 2
        return [((x, y, z), "slab_bottom" if kk % 2 == 0 else "slab_top")], y
    if kind in ("mansard", "gambrel"):
        pitch = 2.0
    if pitch == 0.5:
        y = e - 1 + k // 2
        return [((x, y, z), "slab_bottom" if k % 2 == 0 else "slab_top")], y
    p = int(pitch)
    top = e - 2 + p * (k + 1)                               # the stair's row
    if facing is None:                                      # a ridge: blocks and a slab cap
        rows = range(e - 1 + p * (k - 1), e - 1 + p * k)
        parts = [((x, y, z), "full") for y in rows] + [((x, e - 1 + p * k, z), "slab_bottom")]
        return parts, rows[0] if len(rows) else e - 1 + p * k
    parts = [((x, top, z), "stairs_" + facing)] + [((x, y, z), "full") for y in range(top - p + 1, top)]
    return parts, top - p + 1


def _roof(cv: Canvas, m: dict, mats: dict) -> dict:
    """Draw a mass's roof; returns its pieces {(x, z): {"k", "facing", "y", "shape"}}, the
    surface block of each column (skylights and dormers are cut into them)."""
    from . import parts
    r = m["roof"]
    x0, x1, z0, z1 = mass_bounds(m)
    e, mid = m["y"] + wall_rows(m), m["id"]
    t = r["type"]
    shape = m.get("shape", "box")
    if shape != "box" and t in ("gable", "hip", "pyramid", "cross_gable", "mansard", "gambrel", "pagoda"):
        t = "cone"                                          # a round tower's roof is round

    def ok(c):
        return c is None or c[2][0] in ("site", "roof", "plant") or (c[2][0] in ("wall", "floor") and c[2][1] == mid)
    if t == "flat":
        parts.flat_roof(cv, m, mats, ok)
        return {}
    if t == "dome":
        parts.dome(cv, mats, m, {"squash": r.get("squash", 1.0)}, ok)
        return {}
    if t == "sawtooth":
        parts.sawtooth(cv, m, mats, ok)
        return {}
    o = int(r["overhang"]) + (1 if t == "pagoda" else 0)
    xa, xb, za, zb = x0 - o, x1 + o, z0 - o, z1 + o
    pitch = {"spire": 3.0, "cone": 2.0}.get(t, r["pitch"])
    if shape != "box":
        slopes = _radial(m, o)
    elif t == "cross_gable":
        slopes = _cross(m, o, xa, xb, za, zb)
    else:
        slopes = _slopes(dict(r, type={"pyramid": "hip", "spire": "hip", "cone": "hip", "pagoda": "hip"}.get(t, t)),
                         xa, xb, za, zb)
    kmax = max(k for k, _ in slopes.values())
    kbreak = o + (2 if t == "mansard" else max(1, (kmax - o) * 2 // 5))
    if o == 0:          # no overhang: the eave row would sit in the walls' top row, so lift it
        e += 1
    pieces = {}
    skirt = int(r.get("skirt", 0))
    for (x, z), (k, facing) in slopes.items():
        if skirt and k >= skirt:                            # a pagoda tier's eave: its outer rows only
            continue
        if t == "gable":
            rake = x in (xa, xb) if r["ridge"] == "x" else z in (za, zb)
        elif t == "shed":
            rake = (x in (xa, xb) if r["high"] in ("north", "south") else z in (za, zb)) or k == kmax
        else:
            rake = False
        block = mats["roof_edge" if k == 0 or rake else "roof"]
        if t == "spire" and r.get("material_spire"):
            block = r["material_spire"]
        cells, y = _pieces_for(pitch, k, facing, e, x, z, t, kbreak)
        placed = [cv.put(p, s, block, ("roof", mid), ok) for p, s in cells]
        if placed[0]:
            pieces[(x, z)] = {"k": k, "facing": facing, "y": cells[0][0][1], "low": y, "shape": cells[0][1],
                              "block": block}
    if t == "pagoda":                                       # the corners curl up, a block and a half
        for x, z in ((xa, za), (xa, zb), (xb, za), (xb, zb)):
            if (x, z) in pieces:
                cv.put((x, e - 1, z), "full", mats["roof_edge"], ("roof", mid))
            cv.put((x, e, z), "full", mats["roof_edge"], ("roof", mid), _empty)
            cv.put((x, e + 1, z), "slab_bottom", mats["roof_edge"], ("roof", mid), _empty)
    if t in ("spire", "cone"):                              # a finial on the tip
        tip = max(pieces.items(), key=lambda kv: (kv[1]["k"], kv[1]["facing"] is None, kv[1]["y"]), default=None)
        if tip:
            (x, z), pc = tip
            y = max(yy for (xx, yy, zz) in cv.cells if xx == x and zz == z)
            cv.put((x, y + 1, z), "fence", mats["railing"], ("roof", mid), _empty)
    walls = perimeter(footprint(m))
    for (x, z), pc in pieces.items():                       # the walls follow the roof up
        if (x, z) not in walls:
            continue
        if pc["shape"] == "slab_top":                       # no half-block slit over a wall
            cv.put((x, pc["y"], z), "full", pc["block"], ("roof", mid))
            pc["shape"] = "full"
        for y in range(m["y"] + wall_rows(m), pc["low"]):
            cv.put((x, y, z), "full", mats["wall"], ("wall", mid), _empty)
    if t == "gable" and r.get("gable_end") == "stepped":
        _stepped_gables(cv, m, pieces, mats)
    return pieces


def _stepped_gables(cv: Canvas, m: dict, pieces: dict, mats: dict) -> None:
    """Crow-stepped gables: each end wall rises above the roof in steps two blocks wide,
    every step one block above the roof at its inner column and capped with a slab."""
    x0, x1, z0, z1 = mass_bounds(m)
    along_x = m["roof"]["ridge"] == "z"                     # the gable ends face south and north
    for end in ((z0, z1) if along_x else (x0, x1)):
        cols = [c for c in (range(x0, x1 + 1) if along_x else range(z0, z1 + 1))
                if ((c, end) if along_x else (end, c)) in pieces]
        if len(cols) < 3:
            continue
        top = [pieces[(c, end) if along_x else (end, c)]["y"] for c in cols]
        n = len(cols)
        for i, c in enumerate(cols):
            d = min(i, n - 1 - i)
            inner = min((d // 2) * 2 + 1, (n - 1) // 2)
            h = top[inner if i <= (n - 1) / 2 else n - 1 - inner] + 1
            for y in range(top[i], h + 1):
                p = (c, y, end) if along_x else (end, y, c)
                cv.put(p, "full", mats["wall"], ("wall", m["id"]))
            cv.put((c, h + 1, end) if along_x else (end, h + 1, c), "slab_bottom", mats["cap"], ("roof", m["id"]),
                   _empty)


# --- openings ---------------------------------------------------------------------------

def _mine(mid):
    return lambda c: c is not None and c[2] in (("wall", mid), ("frame", mid))


def _window(cv: Canvas, m: dict, o: dict, mats: dict) -> None:
    mid, side, base = m["id"], o["side"], m["y"]
    dx, dz = OUT[side]
    u, y, w, h = o["u"], o["y"], o["w"], o["h"]
    mullions = set(o["mullions"])
    style = o.get("style", "rect")
    for t in range(u - 1, u + w + 1):
        xz = _along(m, side, t)
        if xz is None:
            continue
        x, z = xz
        for j in range(y - 1, y + h + 1):
            inside = u <= t < u + w and y <= j < y + h
            corner = t in (u, u + w - 1) and w >= 2
            if inside and style in ("arched", "lancet") and corner and j == y + h - 1:
                if style == "lancet" and w >= 3:
                    cv.put((x, base + j, z), "full", mats["frame"], ("frame", mid), _mine(mid))
                else:                                       # the arch's shoulders
                    facing = LEFT[side] if t == u else RIGHT[side]
                    cv.put((x, base + j, z), "stairs_top_" + facing, mats["frame"], ("frame", mid), _mine(mid))
            elif inside and style == "lancet" and corner and j == y + h - 2 and w >= 3:
                facing = LEFT[side] if t == u else RIGHT[side]
                cv.put((x, base + j, z), "stairs_top_" + facing, mats["frame"], ("frame", mid), _mine(mid))
            elif inside and t - u not in mullions:
                cv.put((x, base + j, z), "pane", mats["window"], ("open", mid), _mine(mid))
            elif inside or o["frame"]:
                cv.put((x, base + j, z), "full", mats["frame"], ("frame", mid), _mine(mid))
    if o["sill"]:
        for t in range(u - 1, u + w + 1):
            xz = _along(m, side, t)
            if xz is not None:
                cv.put((xz[0] + dx, base + y - 1, xz[1] + dz), "stairs_top_" + BACK[side], mats["sill"], ("out", mid),
                       _empty)
    if o["shutters"]:
        for t in (u - 1, u + w):
            xz = _along(m, side, t)
            if xz is None:
                continue
            for j in range(y, y + h):
                cv.put((xz[0] + dx, base + j, xz[1] + dz), "trapdoor_open_" + BACK[side], mats["shutter"],
                       ("out", mid), _empty)


def _arch(cv: Canvas, m: dict, o: dict, mats: dict) -> None:
    """An open arch (an arcade bay or a gateway): no glass, shoulders of stairs at the top."""
    mid, side, base = m["id"], o["side"], m["y"]
    u, y, w, h = o["u"], o["y"], o["w"], o["h"]
    for t in range(u - 1, u + w + 1):
        xz = _along(m, side, t)
        if xz is None:
            continue
        x, z = xz
        for j in range(y, y + h + 1):
            inside = u <= t < u + w and j < y + h
            if inside and j == y + h - 1 and t in (u, u + w - 1) and w >= 2:
                facing = LEFT[side] if t == u else RIGHT[side]
                cv.put((x, base + j, z), "stairs_top_" + facing, mats["frame"], ("frame", mid), _mine(mid))
            elif inside:
                cv.put((x, base + j, z), "air", None, None, _mine(mid))
            elif o["frame"]:
                cv.put((x, base + j, z), "full", mats["frame"], ("frame", mid), _mine(mid))


def _door(cv: Canvas, m: dict, o: dict, mats: dict) -> None:
    mid, side, base = m["id"], o["side"], m["y"]
    u, y, w = o["u"], o["y"], o["w"]
    for t in range(u - 1, u + w + 1):
        xz = _along(m, side, t)
        if xz is None:
            continue
        x, z = xz
        if u <= t < u + w:                                  # the upper half is derived on writing
            cv.put((x, base + y, z), "door_" + side, mats["door"], ("open", mid), _mine(mid))
            cv.put((x, base + y + 1, z), "air", None, None, _mine(mid))
            if o["frame"]:
                cv.put((x, base + y + 2, z), "full", mats["frame"], ("frame", mid), _mine(mid))
        elif o["frame"]:
            for j in range(y, y + 3):
                cv.put((x, base + j, z), "full", mats["frame"], ("frame", mid), _mine(mid))


def _garage(cv: Canvas, m: dict, o: dict, mats: dict) -> None:
    """A door recessed one block into the wall, framed in the wall's plane."""
    mid, side, base = m["id"], o["side"], m["y"]
    dx, dz = OUT[side]
    u, y, w, h = o["u"], o["y"], o["w"], o["h"]
    for t in range(u - 1, u + w + 1):
        xz = _along(m, side, t)
        if xz is None:
            continue
        x, z = xz
        for j in range(y, y + h + 1):
            if u <= t < u + w and j < y + h:
                if cv.put((x, base + j, z), "air", None, None, _mine(mid)):
                    cv.put((x - dx, base + j, z - dz), "full", mats["garage"], ("open", mid),
                           lambda c: c is None or c[2][0] in ("site", "floor"))
            elif o["frame"]:
                cv.put((x, base + j, z), "full", mats["frame"], ("frame", mid), _mine(mid))


def _panel(cv: Canvas, m: dict, o: dict, mats: dict) -> None:
    mid, base = m["id"], m["y"]
    block = o["material"] or mats["wall_base"]
    for t in range(o["u"], o["u"] + o["w"]):
        xz = _along(m, o["side"], t)
        if xz is None:
            continue
        for j in range(o["y"], o["y"] + o["h"]):
            cv.put((xz[0], base + j, xz[1]), "full", block, ("wall", mid), _mine(mid))


DRAW = {"window": _window, "door": _door, "garage": _garage, "panel": _panel, "arch": _arch}


# --- features ---------------------------------------------------------------------------

def _chimney(cv: Canvas, f: dict, mats: dict, floor: int) -> None:
    y0 = f.get("y", floor)
    for x in range(f["x"], f["x"] + f["w"]):
        for z in range(f["z"], f["z"] + f["d"]):
            for y in range(y0, f["top"]):
                cv.put((x, y, z), "full", mats["chimney"], ("feature",))
            if f["cap"]:
                cv.put((x, f["top"], z), "slab_bottom", mats["cap"], ("feature",))


def _skylight(cv: Canvas, m: dict, f: dict, mats: dict, pieces: dict) -> None:
    """Glass cut into the slope that faces `side`; `u` from that slope's left end seen from
    outside, `v` rows up from its outer edge."""
    mid, side = m["id"], f["side"]
    xs = [x for x, _ in pieces] or [0]
    zs = [z for _, z in pieces] or [0]
    xa, xb, za, zb = min(xs), max(xs), min(zs), max(zs)
    for (x, z), pc in pieces.items():
        if pc["facing"] != BACK[side]:
            continue
        t = {"south": x - xa, "north": xb - x, "east": zb - z, "west": z - za}[side]
        k = pc["k"]
        inside = f["u"] <= t < f["u"] + f["w"] and f["v"] <= k < f["v"] + f["h"]
        ring = f["u"] - 1 <= t <= f["u"] + f["w"] and f["v"] - 1 <= k <= f["v"] + f["h"]
        if inside:
            cv.put((x, pc["y"], z), "full", mats["skylight"], ("roof", mid))
        elif ring and f["frame"]:
            shape = pc["shape"] if shape_ok(pc["shape"], mats["frame"]) else "full"
            cv.put((x, pc["y"], z), shape, mats["frame"], ("roof", mid))


# --- site -------------------------------------------------------------------------------

def _ground(cv: Canvas, site: dict) -> None:
    """Pads of ground, later ones over earlier; a lower pad digs the higher one away."""
    for g in site["ground"]:
        ymax = max((p[1] for p in cv.cells), default=g["top"])
        for x in range(min(g["x0"], g["x1"]), max(g["x0"], g["x1"]) + 1):
            for z in range(min(g["z0"], g["z1"]), max(g["z0"], g["z1"]) + 1):
                for y in range(g["top"] + 1, ymax + 1):
                    cv.put((x, y, z), "air", None, None, lambda c: c is not None and c[2] == ("site",))
                for d in range(int(g["depth"])):
                    cv.put((x, g["top"] - d, z), "full", g["material"] if d == 0 else g["under"], ("site",))


def _ground_top(cv: Canvas, x: int, z: int, below: int = 200) -> int:
    for y in range(below, -200, -1):
        c = cv.cells.get((x, y, z))
        if c is not None and c[2][0] in ("site", "box"):
            return y
    return -1


def _boxes(cv: Canvas, site: dict) -> None:
    for b in site["boxes"]:
        for x in range(min(b["x0"], b["x1"]), max(b["x0"], b["x1"]) + 1):
            for y in range(min(b["y0"], b["y1"]), max(b["y0"], b["y1"]) + 1):
                for z in range(min(b["z0"], b["z1"]), max(b["z0"], b["z1"]) + 1):
                    if b["material"] == "air":
                        cv.put((x, y, z), "air", None, None)
                    else:
                        cv.put((x, y, z), b["shape"], b["material"], ("box",))


def _plants(cv: Canvas, site: dict) -> None:
    rng = np.random.default_rng(int(site["seed"]))

    def leaf(p, block):
        cv.put(p, "full", block, ("plant",), _empty)
    for b in site["bushes"]:
        x, z, r = b["x"], b["z"], b["r"]
        y = b.get("y", _ground_top(cv, x, z)) + 1
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                for dy in range(0, r + 1):
                    core = dx == 0 and dz == 0
                    if dx * dx + dz * dz + 1.6 * dy * dy < r * r + 1 and (core or rng.random() > 0.05):
                        block = b["material"]
                        if b.get("flowers") and rng.random() < 0.5:
                            block = "flowering_azalea_leaves"
                        leaf((x + dx, y + dy, z + dz), block)
    for t in site["trees"]:
        x, z, h, kind = t["x"], t["z"], int(t["h"]), t["kind"]
        y = t.get("y", _ground_top(cv, x, z)) + 1
        for yy in range(y, y + h):
            cv.put((x, yy, z), "full", f"{kind}_log", ("plant",), lambda c: c is None or c[2] == ("plant",))
        top = y + h - 1
        if kind == "spruce":                          # a cone of shrinking rings
            for i, yy in enumerate(range(top + 1, y + h // 3, -1)):
                r = min(1 + i // 2, 4)
                for dx in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if dx * dx + dz * dz <= r * r + (i % 2) and rng.random() > 0.04:
                            leaf((x + dx, yy, z + dz), f"{kind}_leaves")
            continue
        r = max(2, h // 3)
        for cx, cy, cz, rr in ((0, 0, 0, r), (1, -2, 1, r - 1), (-1, -2, -1, r - 1)):
            for dx in range(-rr, rr + 1):
                for dz in range(-rr, rr + 1):
                    for dy in range(-rr, rr + 1):
                        if dx * dx + dz * dz + 1.5 * dy * dy < rr * rr + 1 and rng.random() > 0.08:
                            leaf((x + cx + dx, top + cy + dy, z + cz + dz), f"{kind}_leaves")


# --- assembly ---------------------------------------------------------------------------

def build(src) -> dict:
    """{"shape", "material"} (X, Y, Z) grids of a blueprint (path, JSON text or dict), its
    "origin" (the blueprint coordinates of grid cell 0), "dims", "blocks" and "warnings"."""
    from . import parts
    bp = load(src)
    cv = Canvas()
    site, masses = bp["site"], bp["masses"]
    mats = {m["id"]: resolve_materials(bp["materials"], m.get("materials")) for m in masses}
    _ground(cv, site)
    for m in masses:
        _mass(cv, m, mats[m["id"]])
    _clear_insides(cv, masses)
    for m in masses:
        parts.decor(cv, m, mats[m["id"]])
    pieces = {}
    for m in sorted(masses, key=lambda m: m["y"] + wall_rows(m)):
        pieces[m["id"]] = _roof(cv, m, mats[m["id"]])
    by_id = {m["id"]: m for m in masses}
    for o in sorted(bp["openings"], key=lambda o: o["kind"] != "panel"):
        m = by_id[o["mass"]]
        DRAW[o["kind"]](cv, m, o, mats[m["id"]])
    floor = min(m["y"] for m in masses)
    glob = resolve_materials(bp["materials"])
    for f in bp["features"]:
        m = by_id.get(f.get("mass"))
        mm = mats[m["id"]] if m else glob
        if f["kind"] == "chimney":
            _chimney(cv, f, glob, floor)
        elif f["kind"] == "skylight":
            _skylight(cv, m, f, mm, pieces[m["id"]])
        else:
            parts.FEATURES[f["kind"]](cv, m, f, mm, pieces.get(m["id"]) if m else None, floor)
    _boxes(cv, site)
    _plants(cv, site)
    return grids(cv)


def grids(cv: Canvas) -> dict:
    if not cv.cells:
        raise ValueError("the blueprint built no blocks")
    pts = np.array(list(cv.cells), np.int64)
    lo, hi = pts.min(0), pts.max(0)
    dims = hi - lo + 1
    shape = np.zeros(dims, np.uint8)
    material = np.zeros(dims, np.uint16)
    warnings = set()
    for (x, y, z), (s, block, _) in cv.cells.items():
        mi = material_index(block)
        si = V.SHAPE_INDEX[s]
        if mi is None or not V.VALID[si, mi]:
            warnings.add(f"{block} has no {s}: drawn as a full block")
            si = V.SHAPE_INDEX["full"]
            if mi is None or not V.VALID[si, mi]:
                continue
        shape[x - lo[0], y - lo[1], z - lo[2]] = si
        material[x - lo[0], y - lo[1], z - lo[2]] = mi
    return {"shape": shape, "material": material, "origin": lo.tolist(), "dims": dims.tolist(),
            "blocks": int((shape != 0).sum()), "warnings": sorted(warnings)}
