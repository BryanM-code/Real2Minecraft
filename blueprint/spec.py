"""The blueprint format (`blueprint-0`): a building described by its parts, the description
a photo model fills in and `blueprint.build` turns into blocks.

Frame: x east (to the right seen from the front), y up, z south (towards the viewer); the
front is the south side. Units are blocks. A blueprint is plain JSON:

    {"format": "blueprint-0",
     "materials": {"wall": "birch_planks", "roof": "stone_bricks", ...},   # role -> block
     "masses": [{"id": "main", "x": 0, "z": 0, "w": 17, "d": 20, "y": 0,
                 "storeys": 2, "storey_h": 5, "plinth": 1,
                 "roof": {"type": "gable", "ridge": "x", "pitch": 0.5, "overhang": 1}}],
     "openings": [{"mass": "main", "side": "south", "kind": "window",
                   "u": 2, "y": 2, "w": 3, "h": 3, "mullions": [1]}],
     "features": [{"kind": "chimney", "x": 1, "z": 11, "w": 3, "d": 3, "top": 19}],
     "site": {"ground": [...], "boxes": [...], "bushes": [...], "trees": [...]}}

A mass is a box of walls: footprint `x`, `z` (its north-west corner), `w` (along x) by `d`
(along z), base row `y` (its ground floor), walls `storeys` x `storey_h` rows high (or
`walls` rows), floor plates at `floors` (row offsets; default every storey). `plinth` rows
at the bottom are the base material (a number, or one per side). Its roof starts on the row
above the walls.

An opening sits on one wall of a mass: `u` counts blocks from the wall's left end as seen
from outside, `y` rows above the mass's base row, `w` x `h` the clear opening (a door is
always 2 high). Kinds: window, door, garage (recessed one block), panel (a patch of another
material, e.g. brick round an entrance).

Block names drop the "minecraft:" prefix; every role must be a block the vocabulary
(mcvox-v3, 1.21.11) has in the shapes the role needs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from mcvoxel import vocab as V

FORMAT = "blueprint-1"
FORMATS = ("blueprint-0", "blueprint-1")          # blueprint-1 adds shapes, roofs, parts, decor
SIDES = ("south", "north", "east", "west")
ROOF_TYPES = ("gable", "hip", "shed", "flat", "pyramid", "cross_gable", "mansard", "gambrel", "sawtooth",
              "spire", "cone", "dome", "pagoda")
PITCHES = (0.25, 0.5, 0.75, 1.0, 2.0, 3.0)  # rise per block of run: 14, 27, 37, 45, 63, 72 degrees
SHAPES = ("box", "round", "octagon")
DECOR = ("half_timber", "quoins", "bands", "pilasters", "buttresses", "curtain")
WINDOW_STYLES = ("rect", "arched", "lancet")
OPENING_KINDS = ("window", "door", "garage", "panel", "arch")
FEATURE_KINDS = ("chimney", "skylight", "portico", "porch", "balcony", "dormer", "bay", "dome", "stack", "canopy",
                 "awning", "sails", "gallery", "lantern", "finial")
GABLE_ENDS = ("plain", "stepped")                  # stepped: crow-stepped gable walls rising above the roof
# features attached to one wall of a mass: (mass, side, u along it, w wide)
WALL_FEATURES = ("portico", "porch", "balcony", "bay", "awning")

# role -> (default block, shapes the builder may draw it in)
ROLES = {
    "wall": ("birch_planks", ("full",)),
    "wall_base": ("stone_bricks", ("full",)),
    "trim": ("spruce_planks", ("full",)),
    "roof": ("deepslate_tiles", ("full", "slab_bottom", "slab_top")),           # stairs where it has them
    "roof_edge": (None, ("full", "slab_bottom", "slab_top")),                   # None: the roof's
    "window": ("glass", ("full",)),                    # panes where it has them
    "frame": (None, ("full",)),                                                  # None: trim
    "sill": (None, ()),                                                         # None: trim; stairs, else full
    "shutter": (None, ()),                                                      # None: trim; no trapdoor, no shutter
    "door": ("spruce_planks", ("door_south",)),
    "garage": ("white_concrete", ("full",)),
    "floor": ("spruce_planks", ("full",)),
    "chimney": ("bricks", ("full",)),
    "cap": ("stone_bricks", ("slab_bottom",)),
    "skylight": ("glass", ("full",)),
    "column": ("smooth_quartz", ("full",)),
    "railing": ("dark_oak_planks", ()),                 # a fence, else a wall, else full blocks
    "dome": ("oxidized_cut_copper", ("full",)),
    "awning": ("red_nether_bricks", ("slab_bottom",)),
}
FALLBACK = {"roof_edge": "roof", "frame": "trim", "sill": "trim", "shutter": "trim"}

MASS_DEFAULTS = {"y": 0, "storeys": 1, "storey_h": 5, "plinth": 0, "corner_trim": True, "shape": "box"}
ROOF_DEFAULTS = {"type": "gable", "pitch": 1.0, "overhang": 1, "parapet": 1, "high": "north",
                 "battlements": False, "squash": 1.0,
                 "skirt": 0,               # > 0: only this many rows of slope from the eave (a pagoda tier's)
                 "gable_end": "plain"}
OPENING_DEFAULTS = {
    "window": {"w": 2, "h": 3, "mullions": [], "frame": True, "sill": True, "shutters": False, "style": "rect"},
    "arch": {"w": 3, "h": 4, "frame": True},
    "door": {"w": 1, "frame": True},
    "garage": {"w": 5, "h": 4, "frame": True},
    "panel": {"material": None},
}
FEATURE_DEFAULTS = {"chimney": {"w": 2, "d": 2, "cap": True}, "skylight": {"w": 3, "h": 3, "frame": True},
                    "portico": {"depth": 3, "columns": 0, "pediment": True, "height": 0},
                    "porch": {"depth": 3, "height": 4, "railing": True, "wrap": False, "wrap_side": "right"},
                    "balcony": {"depth": 1, "y": 0},
                    "dormer": {"w": 3, "h": 2},
                    "bay": {"depth": 1, "y": 1, "h": 3},
                    "dome": {"squash": 1.0, "drum": 0, "lantern": False},
                    "stack": {"r": 1},
                    "canopy": {"posts": 6},
                    "awning": {"y": 4, "depth": 2},
                    "sails": {"length": 0},
                    "gallery": {},
                    "lantern": {"h": 3, "open": False, "cap": "", "shrink": -1},
                    "finial": {"h": 0}}


class BlueprintError(ValueError):
    """A blueprint that cannot be built; the message lists every problem found."""


def material_index(name: str) -> int | None:
    """Vocabulary index of a block name ("bricks" or "minecraft:bricks"), or None."""
    return V.MATERIAL_INDEX.get(name if name.startswith("minecraft:") else "minecraft:" + name)


def shape_ok(shape: str, name: str) -> bool:
    m = material_index(name)
    return m is not None and bool(V.VALID[V.SHAPE_INDEX[shape], m])


def resolve_materials(*layers: dict) -> dict:
    """role -> block name, later layers over earlier, over the defaults; the roles that
    default to another role's block follow it."""
    out = {role: block for role, (block, _) in ROLES.items()}
    for layer in layers:
        out.update({k: v for k, v in (layer or {}).items() if v is not None})
    for role, base in FALLBACK.items():
        if out[role] is None:
            out[role] = out[base]
    return out


def mass_bounds(m: dict) -> tuple:
    """(x0, x1, z0, z1), inclusive."""
    return m["x"], m["x"] + m["w"] - 1, m["z"], m["z"] + m["d"] - 1


def wall_rows(m: dict) -> int:
    return int(m["walls"]) if "walls" in m else int(m["storeys"]) * int(m["storey_h"])


def wall_length(m: dict, side: str) -> int:
    return m["w"] if side in ("south", "north") else m["d"]


def load(src) -> dict:
    """A blueprint (path, JSON text or dict) with every default filled in; raises
    BlueprintError listing every problem."""
    if isinstance(src, dict):
        bp = copy.deepcopy(src)
    else:
        text = Path(src).read_text(encoding="utf-8") if not str(src).lstrip().startswith("{") else str(src)
        bp = json.loads(text)
    errors = []
    if bp.get("format", FORMAT) not in FORMATS:
        errors.append(f"format: one of {FORMATS}, got {bp.get('format')!r}")
    bp["format"] = FORMAT
    bp.setdefault("materials", {})
    masses = bp.setdefault("masses", [])
    if not masses:
        errors.append("masses: at least one mass is needed")
    ids = set()
    for i, m in enumerate(masses):
        where = f"masses[{i}]"
        for k in ("x", "z", "w", "d"):
            if not isinstance(m.get(k), int):
                errors.append(f"{where}.{k}: an integer is needed")
        for k, v in MASS_DEFAULTS.items():
            m.setdefault(k, v)
        m.setdefault("id", f"m{i}")
        if m["id"] in ids:
            errors.append(f"{where}.id: {m['id']!r} is used twice")
        ids.add(m["id"])
        if any(not isinstance(m.get(k), int) for k in ("w", "d")):
            continue
        if m["w"] < 3 or m["d"] < 3:
            errors.append(f"{where}: walls need w and d of at least 3")
        if m["shape"] not in SHAPES:
            errors.append(f"{where}.shape: one of {SHAPES}, got {m['shape']!r}")
        m["decor"] = list(m.get("decor") or [])
        for dec in m["decor"]:
            if dec not in DECOR:
                errors.append(f"{where}.decor: one of {DECOR}, got {dec!r}")
        rows = wall_rows(m)
        if rows < 2:
            errors.append(f"{where}: walls need at least 2 rows")
        m.setdefault("floors", [k * int(m["storey_h"]) for k in range(max(1, rows // int(m["storey_h"])))])
        if isinstance(m["plinth"], int):
            m["plinth"] = {s: m["plinth"] for s in SIDES}
        else:
            m["plinth"] = {s: int(m["plinth"].get(s, 0)) for s in SIDES}
        roof = m.setdefault("roof", {})
        for k, v in ROOF_DEFAULTS.items():
            roof.setdefault(k, v)
        roof.setdefault("ridge", "x" if m["w"] >= m["d"] else "z")
        if roof["type"] not in ROOF_TYPES:
            errors.append(f"{where}.roof.type: one of {ROOF_TYPES}, got {roof['type']!r}")
        if float(roof["pitch"]) not in PITCHES:
            errors.append(f"{where}.roof.pitch: one of {PITCHES}, got {roof['pitch']!r}")
        roof["pitch"] = float(roof["pitch"])
        if roof["ridge"] not in ("x", "z"):
            errors.append(f"{where}.roof.ridge: 'x' or 'z', got {roof['ridge']!r}")
        if roof["high"] not in SIDES:
            errors.append(f"{where}.roof.high: one of {SIDES}, got {roof['high']!r}")
        if not 0 <= int(roof["overhang"]) <= 3:
            errors.append(f"{where}.roof.overhang: 0 to 3, got {roof['overhang']!r}")
        if roof["gable_end"] not in GABLE_ENDS:
            errors.append(f"{where}.roof.gable_end: one of {GABLE_ENDS}, got {roof['gable_end']!r}")
        if int(roof["skirt"]) < 0:
            errors.append(f"{where}.roof.skirt: 0 or more rows, got {roof['skirt']!r}")
    by_id = {m["id"]: m for m in masses if isinstance(m.get("w"), int) and isinstance(m.get("d"), int)}
    for i, o in enumerate(bp.setdefault("openings", [])):
        where = f"openings[{i}]"
        kind = o.get("kind")
        if kind not in OPENING_KINDS:
            errors.append(f"{where}.kind: one of {OPENING_KINDS}, got {kind!r}")
            continue
        for k, v in OPENING_DEFAULTS[kind].items():
            o.setdefault(k, copy.deepcopy(v))
        if kind == "door":
            o["h"] = 2
        m = by_id.get(o.get("mass"))
        if m is None:
            errors.append(f"{where}.mass: no mass {o.get('mass')!r}")
            continue
        if o.get("side") not in SIDES:
            errors.append(f"{where}.side: one of {SIDES}, got {o.get('side')!r}")
            continue
        for k in ("u", "y", "w", "h"):
            if not isinstance(o.get(k), int):
                errors.append(f"{where}.{k}: an integer is needed")
        if any(not isinstance(o.get(k), int) for k in ("u", "y", "w", "h")):
            continue
        if o["u"] < 0 or o["u"] + o["w"] > wall_length(m, o["side"]) or o["w"] < 1:
            errors.append(f"{where}: u {o['u']} + w {o['w']} is off the {wall_length(m, o['side'])}-block wall")
        if o["y"] < 0 or o["y"] + o["h"] > wall_rows(m) or o["h"] < 1:
            errors.append(f"{where}: y {o['y']} + h {o['h']} is off the {wall_rows(m)}-row wall")
        if kind == "window":
            if o["style"] not in WINDOW_STYLES:
                errors.append(f"{where}.style: one of {WINDOW_STYLES}, got {o['style']!r}")
            o["mullions"] = [int(t) for t in o["mullions"]]
            if any(not 0 <= t < o["w"] for t in o["mullions"]):
                errors.append(f"{where}.mullions: offsets inside the opening (0 to {o['w'] - 1})")
        if kind == "panel" and o.get("material") is not None and material_index(o["material"]) is None:
            errors.append(f"{where}.material: unknown block {o['material']!r}")
    for i, f in enumerate(bp.setdefault("features", [])):
        where = f"features[{i}]"
        kind = f.get("kind")
        if kind not in FEATURE_KINDS:
            errors.append(f"{where}.kind: one of {FEATURE_KINDS}, got {kind!r}")
            continue
        for k, v in FEATURE_DEFAULTS[kind].items():
            f.setdefault(k, v)
        need = {"chimney": ("x", "z", "top"), "stack": ("x", "z", "top"), "canopy": ("x0", "z0", "x1", "z1", "top"),
                "skylight": ("u", "v"), "dormer": ("u",), "sails": (), "gallery": ("y",), "lantern": (), "dome": (),
                "finial": ()}
        need.update({k: ("u", "w") for k in WALL_FEATURES})
        for k in need[kind]:
            if not isinstance(f.get(k), int):
                errors.append(f"{where}.{k}: an integer is needed")
        if kind in ("chimney", "stack", "canopy") or (kind == "dome" and "mass" not in f):
            if kind == "dome" and not all(isinstance(f.get(k), int) for k in ("x", "z", "r", "y")):
                errors.append(f"{where}: a dome needs a mass, or x, z, r and y")
            continue
        m = by_id.get(f.get("mass"))
        if m is None:
            errors.append(f"{where}.mass: no mass {f.get('mass')!r}")
        elif kind in ("skylight", "dormer", "sails", *WALL_FEATURES) and f.get("side") not in SIDES:
            errors.append(f"{where}.side: one of {SIDES}, got {f.get('side')!r}")
    site = bp.setdefault("site", {})
    for k in ("ground", "boxes", "bushes", "trees"):
        site.setdefault(k, [])
    site.setdefault("seed", 0)
    for i, g in enumerate(site["ground"]):
        g.setdefault("material", "grass_block")
        g.setdefault("under", "dirt")
        g.setdefault("depth", 3)
        for k in ("x0", "z0", "x1", "z1", "top"):
            if not isinstance(g.get(k), int):
                errors.append(f"site.ground[{i}].{k}: an integer is needed")
    for i, b in enumerate(site["boxes"]):
        b.setdefault("shape", "full")
        if b["shape"] not in V.SHAPE_INDEX:
            errors.append(f"site.boxes[{i}].shape: unknown shape {b['shape']!r}")
        elif b.get("material") != "air" and not shape_ok(b["shape"], str(b.get("material"))):
            errors.append(f"site.boxes[{i}]: {b.get('material')!r} has no {b['shape']!r}")
    for i, b in enumerate(site["bushes"]):
        b.setdefault("r", 2)
        b.setdefault("material", "azalea_leaves")
    for i, t in enumerate(site["trees"]):
        t.setdefault("h", 9)
        t.setdefault("kind", "oak")
        if material_index(f"{t['kind']}_log") is None or material_index(f"{t['kind']}_leaves") is None:
            errors.append(f"site.trees[{i}].kind: no {t['kind']}_log / {t['kind']}_leaves block")
    for where, layers in [("materials", [bp["materials"]])] + [
            (f"masses[{i}].materials", [bp["materials"], m.get("materials")]) for i, m in enumerate(masses)
            if m.get("materials")]:
        for role, block in resolve_materials(*layers).items():
            if role not in ROLES:
                errors.append(f"{where}.{role}: unknown role (roles: {', '.join(ROLES)})")
                continue
            for shape in ROLES[role][1]:
                if not shape_ok(shape, block):
                    errors.append(f"{where}.{role}: {block!r} has no {shape!r} block")
    for g in site["ground"]:
        for k in ("material", "under"):
            if material_index(g[k]) is None:
                errors.append(f"site.ground: unknown block {g[k]!r}")
    if errors:
        raise BlueprintError("; ".join(errors))
    return bp
