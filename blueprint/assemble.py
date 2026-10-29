"""A filled-in form about a building (wings, storeys, roofs, openings, colours: what a vision
model reads reliably) -> a blueprint. The model never places a block or picks a coordinate:
the sizes, positions and window spacing come from the rules below, at 2 blocks per metre.

Form (`FORM` in `blueprint.describe`):

    {"wings": [{"position": "left|centre|right", "storeys": 2, "level": "ground|half_below",
                "front_offset": "flush|forward|back", "windows_across": 2,
                "roof": "gable|hip|flat|shed", "ridge": "parallel|perpendicular",
                "pitch": "low|medium|steep",
                "front": {"ground_windows": 1, "wide_window": false, "door": true, "garage_doors": 0},
                "side_garage_doors": 0, "chimney": true, "skylights": 0}, ...],   # left to right
     "wall": {"material": "wood_siding", "colour": "cream"}, "base": {"material": "brick"},
     "trim_colour": "dark brown", "roof_colour": "grey", "door_colour": "light wood"}
"""
from __future__ import annotations

from .spec import shape_ok

STOREY = 6          # rows per storey (3 m): floor plate + 5
WIN_W, WIN_H = 3, 3
GARAGE_W, GARAGE_H = 5, 4
SPACING = 2         # wall between windows
HALF_BELOW = 4      # rows a split-level wing sinks below the front ground (a garage under it)
STEP = 4            # blocks a wing stands forward or back

# (material, colour word) -> wall block; colour words are matched by substring, first hit wins
WALLS = {
    "wood_siding": [("white", "smooth_quartz"), ("cream", "smooth_sandstone"), ("beige", "smooth_sandstone"),
                    ("yellow", "birch_planks"), ("grey", "polished_andesite"), ("gray", "polished_andesite"),
                    ("blue", "light_blue_terracotta"), ("green", "green_terracotta"), ("red", "red_terracotta"),
                    ("dark brown", "dark_oak_planks"), ("black", "dark_oak_planks"), ("dark", "dark_oak_planks"),
                    ("brown", "spruce_planks"), ("pink", "cherry_planks"), ("", "birch_planks")],
    # brown brick is ordinary brick (mud bricks are tan); dark brick, the darkest terracotta
    "brick": [("white", "smooth_quartz"), ("cream", "smooth_sandstone"), ("yellow", "smooth_sandstone"),
              ("dark", "brown_terracotta"), ("black", "deepslate_bricks"), ("tan", "mud_bricks"),
              ("beige", "mud_bricks"), ("grey", "stone_bricks"), ("gray", "stone_bricks"), ("", "bricks")],
    "stone": [("white", "polished_diorite"), ("cream", "smooth_sandstone"), ("beige", "smooth_sandstone"),
              ("yellow", "sandstone"), ("dark", "deepslate_bricks"), ("brown", "mud_bricks"), ("pink", "granite"),
              ("red", "smooth_red_sandstone"), ("tan", "smooth_sandstone"), ("", "stone_bricks")],
    "stucco": [("white", "smooth_quartz"), ("cream", "smooth_sandstone"), ("beige", "smooth_sandstone"),
               ("yellow", "smooth_sandstone"), ("pink", "white_terracotta"), ("grey", "light_gray_concrete"),
               ("gray", "light_gray_concrete"), ("", "smooth_quartz")],
    "concrete": [("white", "white_concrete"), ("dark", "gray_concrete"), ("", "light_gray_concrete")],
    "glass": [("", "light_blue_stained_glass")],
}
BASES = {"brick": "bricks", "stone": "stone_bricks", "concrete": "smooth_stone"}
ROOFS = [("black", "blackstone"), ("dark", "deepslate_tiles"), ("red", "granite"), ("orange", "cut_copper"),
         ("brown", "dark_oak_planks"), ("green", "oxidized_cut_copper"), ("blue", "deepslate_tiles"),
         ("white", "smooth_quartz"), ("silver", "polished_andesite"), ("tan", "smooth_sandstone"),
         ("beige", "smooth_sandstone"), ("grey", "stone_bricks"), ("gray", "stone_bricks"), ("", "stone_bricks")]
# (blue: warped wood is the one blue-green block with the stairs, slabs and trapdoors trim needs)
TRIMS = [("white", "quartz_block"), ("black", "polished_blackstone"), ("green", "dark_prismarine"),
         ("red", "red_nether_bricks"), ("blue", "warped_planks"), ("teal", "warped_planks"),
         ("yellow", "birch_planks"), ("gold", "yellow_terracotta"), ("dark", "dark_oak_planks"), ("brown", "spruce_planks"),
         ("cream", "smooth_sandstone"), ("tan", "smooth_sandstone"), ("beige", "smooth_sandstone"),
         ("grey", "stone_bricks"), ("gray", "stone_bricks"), ("", "dark_oak_planks")]
DOORS = [("light", "birch_planks"), ("white", "birch_planks"), ("dark", "dark_oak_planks"), ("black", "dark_oak_planks"),
         ("red", "mangrove_planks"), ("", "spruce_planks")]


def pick(table, word) -> str:
    word = str(word or "").lower()
    return next(block for key, block in table if key in word)


def _int(v, default=0, lo=0, hi=99) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _spread(n: int, w: int, width: int, start: int = 0, stop: int | None = None) -> list:
    """Left ends of `n` openings `w` wide spread evenly over [start, stop) of a wall."""
    stop = width if stop is None else stop
    room = stop - start
    if n <= 0 or room < w:
        return []
    n = min(n, max(1, (room + SPACING) // (w + SPACING)))
    gap = (room - n * w) / (n + 1)
    return [start + round(gap * (i + 1) + w * i) for i in range(n)]


def assemble(form: dict) -> dict:
    """A blueprint (dict, `blueprint-0`) from a filled-in form; tolerant of missing fields."""
    wings = [w for w in form.get("wings") or [] if isinstance(w, dict)] or [{}]
    wall = form.get("wall") or {}
    wall_block = pick(WALLS.get(str(wall.get("material", "")).lower(), WALLS["wood_siding"]), wall.get("colour"))
    trim = pick(TRIMS, form.get("trim_colour"))
    base = BASES.get(str((form.get("base") or {}).get("material", "")).lower())
    materials = {"wall": wall_block, "wall_base": base or wall_block, "trim": trim,
                 "roof": pick(ROOFS, form.get("roof_colour")),
                 "roof_edge": trim if shape_ok("stairs_north", trim) else "dark_oak_planks",
                 "sill": trim if shape_ok("stairs_north", trim) else "stone_bricks",
                 "shutter": trim if shape_ok("trapdoor_open_north", trim) else "dark_oak_planks",
                 "window": "light_gray_stained_glass", "door": pick(DOORS, form.get("door_colour")),
                 "garage": "smooth_quartz", "skylight": "glass", "chimney": "bricks"}
    tallest = max(_int(w.get("storeys"), 1, 1, 60) for w in wings)
    rows, x = [], 0
    for i, w in enumerate(wings):
        storeys = _int(w.get("storeys"), 1, 1, 60)
        front = w.get("front") or {}
        across = _int(w.get("windows_across"), 2, 0, 30)
        need = max(across * (WIN_W + SPACING),
                   _int(front.get("garage_doors")) * (GARAGE_W + SPACING) + SPACING,
                   (WIN_W + SPACING) * 3 if front.get("wide_window") else 0)
        width = max(7, need + 4)
        depth = max(9, min(round(width * 0.9), 24)) if tallest <= 3 else max(9, width)
        below = HALF_BELOW if w.get("level") == "half_below" else 0
        rows.append({"w": w, "storeys": storeys, "width": width, "depth": depth, "below": below, "x": x,
                     "front": front, "across": across})
        x += width
    main = max(rows, key=lambda r: (r["storeys"], r["width"]))
    blueprint = {"format": "blueprint-0", "materials": materials, "masses": [], "openings": [], "features": []}
    for i, r in enumerate(rows):
        w, mid = r["w"], f"w{i}"
        offset = {"forward": STEP, "back": -STEP}.get(w.get("front_offset"), 0) if r is not main else 0
        z = main["depth"] - r["depth"] + offset
        roof_type = w.get("roof") if w.get("roof") in ("gable", "hip", "flat", "shed") else "gable"
        ridge_parallel = w.get("ridge") != "perpendicular"
        mass = {"id": mid, "x": r["x"], "z": z, "w": r["width"], "d": r["depth"], "y": -r["below"],
                "walls": r["storeys"] * STOREY + r["below"],
                "floors": sorted({0, *(r["below"] + k * STOREY for k in range(r["storeys"]))}),
                "plinth": (r["below"] + 2) if base else 0,
                "roof": {"type": roof_type, "ridge": "x" if ridge_parallel else "z",
                         "pitch": 1.0 if w.get("pitch") == "steep" else 0.5, "overhang": 1}}
        blueprint["masses"].append(mass)
        rel = r["below"]                                   # the front ground's row in this mass
        width, front = r["width"], r["front"]
        outer = "west" if i == 0 and len(rows) > 1 else "east"
        for k in range(1, r["storeys"]):                   # upper floors: the windows across
            for u in _spread(r["across"], WIN_W, width):
                blueprint["openings"].append({"mass": mid, "side": "south", "kind": "window", "u": u,
                                              "y": rel + k * STOREY + 2, "w": WIN_W, "h": WIN_H})
        start, stop = 0, width
        gd = _int(front.get("garage_doors"))
        for u in _spread(gd, GARAGE_W, width):
            blueprint["openings"].append({"mass": mid, "side": "south", "kind": "garage", "u": u, "y": 0,
                                          "w": GARAGE_W, "h": GARAGE_H})
        if gd:
            continue
        if front.get("door"):                              # the door towards the building's middle
            u = width - 3 if i < len(rows) / 2 - 0.25 else 2
            blueprint["openings"].append({"mass": mid, "side": "south", "kind": "door", "u": u, "y": rel + 1})
            start, stop = (0, u - 1) if u > width / 2 else (u + 3, width)
        if front.get("wide_window"):
            ww = max(WIN_W, min(stop - start - 4, 13))
            u = start + (stop - start - ww) // 2
            blueprint["openings"].append({"mass": mid, "side": "south", "kind": "window", "u": u, "y": rel + 2,
                                          "w": ww, "h": WIN_H, "mullions": list(range(4, ww - 1, 4))})
        else:
            n = _int(front.get("ground_windows"), r["across"])
            for u in _spread(n, WIN_W, width, start, stop):
                blueprint["openings"].append({"mass": mid, "side": "south", "kind": "window", "u": u,
                                              "y": rel + 2, "w": WIN_W, "h": WIN_H})
        side = _int(w.get("side_garage_doors"))
        for u in _spread(side, GARAGE_W, r["depth"]):
            blueprint["openings"].append({"mass": mid, "side": outer, "kind": "garage", "u": u, "y": 0,
                                          "w": GARAGE_W, "h": GARAGE_H})
        if side and r["storeys"] > 0:
            blueprint["openings"].append({"mass": mid, "side": outer, "kind": "window", "u": r["depth"] // 2 - 1,
                                          "y": rel + 2 + (STOREY if rel == 0 and r["storeys"] > 1 else 0),
                                          "w": WIN_W, "h": WIN_H, "shutters": True})
    for i, (r, mass) in enumerate(zip(rows, blueprint["masses"])):
        w = r["w"]
        top = mass["y"] + mass["walls"]
        rise = (mass["d"] + 2) // 2 if mass["roof"]["ridge"] == "x" else (mass["w"] + 2) // 2
        ridge = top + int(rise * mass["roof"]["pitch"])
        if w.get("chimney"):
            cx = mass["x"] + 1 if i == 0 else mass["x"] + mass["w"] - 4
            blueprint["features"].append({"kind": "chimney", "x": cx, "z": mass["z"] + mass["d"] // 2 - 3,
                                          "w": 2, "d": 2, "top": ridge + 2})
        n = _int(w.get("skylights"), 0, 0, 4)
        if n and mass["roof"]["type"] in ("gable", "hip") and mass["roof"]["ridge"] == "x":
            for u in _spread(n, 3, mass["w"] + 2, 2, mass["w"]):
                blueprint["features"].append({"kind": "skylight", "mass": mass["id"], "side": "south", "u": u,
                                              "v": 3, "w": 3, "h": 3})
    return blueprint
