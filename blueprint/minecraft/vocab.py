"""The block vocabulary: every block is a SHAPE and a MATERIAL.

A block state like `minecraft:dark_oak_trapdoor[facing=south,half=bottom,open=true]` is a
material (dark oak) and a shape (a trapdoor opened against the north side of its block). The
builder fills grids of shape and material IDs; `decode` turns an ID pair back into a block
state. `VALID[shape, material]` says which pairs exist in the game: oak has stairs, glass has
a pane, stone has no fence.

Conventions:

* Axes: +X = east, +Z = south, Y up (the `.schem` convention).
* Stairs are named by their HIGH side (`stairs_east` rises to the east); `stairs_top_*` are
  upside-down stairs.
* Doors and trapdoors are named by the side of the block their plate sits on: `door_north`
  sits against the north face, which Minecraft calls `facing=south`.
* A material is named by its full block (`minecraft:oak_planks`); oak stairs, fences, gates,
  doors and trapdoors are shapes of it.

Shape and material IDs never change: new ones are only ever added at the end.

What the game works out from neighbours (stair corners, fence and wall arms, a door's upper
half) is not part of the vocabulary; `states.finish_states` adds it to a finished grid.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .blocks import AUX_PALETTE, BLOCK_PALETTE, BLOCK_VARIANTS, WALL_VARIANTS

AIR = "minecraft:air"
TARGET_VERSION = "1.21.11"            # the Minecraft version schematics are written for
TARGET_DATA_VERSION = 4671

# ── Shapes ────────────────────────────────────────────────────────────────────
SIDES = ("east", "west", "north", "south")
OPPOSITE = {"east": "west", "west": "east", "north": "south", "south": "north"}

SHAPES = ["air", "full", "slab_bottom", "slab_top",
          "stairs_east", "stairs_west", "stairs_north", "stairs_south",
          "fence", "wall",
          "stairs_top_east", "stairs_top_west", "stairs_top_north", "stairs_top_south",
          "pane",
          "trapdoor_bottom", "trapdoor_top",
          "trapdoor_open_north", "trapdoor_open_south", "trapdoor_open_east", "trapdoor_open_west",
          "door_north", "door_south", "door_east", "door_west",
          "full_axis_x", "full_axis_z",
          "fence_gate_x", "fence_gate_z",
          "carpet"]
SHAPE_INDEX = {s: i for i, s in enumerate(SHAPES)}
SHAPE_AIR, SHAPE_FULL, SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP = 0, 1, 2, 3
SHAPE_FENCE, SHAPE_WALL, SHAPE_PANE, SHAPE_CARPET = 8, 9, 14, 29
SHAPE_TRAPDOOR_BOTTOM, SHAPE_TRAPDOOR_TOP = 15, 16
SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z = 25, 26
SHAPE_GATE_X, SHAPE_GATE_Z = 27, 28
IGNORE_SHAPE = 255            # a block the vocabulary leaves out (a door's upper half)
IGNORE_MATERIAL = 65535

STAIR_SHAPES = {("bottom", f): SHAPE_INDEX["stairs_" + f] for f in SIDES}
STAIR_SHAPES.update({("top", f): SHAPE_INDEX["stairs_top_" + f] for f in SIDES})
TRAPDOOR_OPEN_SHAPES = {s: SHAPE_INDEX["trapdoor_open_" + s] for s in SIDES}   # plate side
DOOR_SHAPES = {s: SHAPE_INDEX["door_" + s] for s in SIDES}                     # plate side
# Shapes with a horizontal direction, and shapes with a horizontal axis.
DIRECTIONAL = {i: ("stairs_" + half, f) for (half, f), i in STAIR_SHAPES.items()}
DIRECTIONAL.update({i: ("trapdoor_open", s) for s, i in TRAPDOOR_OPEN_SHAPES.items()})
DIRECTIONAL.update({i: ("door", s) for s, i in DOOR_SHAPES.items()})
AXIAL = {SHAPE_FULL_AXIS_X: ("full", "x"), SHAPE_FULL_AXIS_Z: ("full", "z"),
         SHAPE_GATE_X: ("fence_gate", "x"), SHAPE_GATE_Z: ("fence_gate", "z")}


# ── Materials ─────────────────────────────────────────────────────────────────
# The first 120 materials are air and the two colour palettes, in sorted order.
_PALETTE_MATERIALS = [AIR] + sorted(BLOCK_PALETTE) + sorted(AUX_PALETTE)
assert len(_PALETTE_MATERIALS) == 120, "the first 120 material IDs must not change"

_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry",
          "pale_oak", "bamboo", "crimson", "warped")
_COLORS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
           "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black")
_COPPER_STAGES = ("", "exposed_", "weathered_", "oxidized_")


def _more_materials() -> list[str]:
    """The materials after the first 120, in their fixed order (only ever append)."""
    names = ["pale_oak_planks", "bamboo_planks"]
    # Every wood: logs, stripped logs, wood and stripped wood (stems and hyphae for nether woods).
    for w in _WOODS:
        if w in ("crimson", "warped"):
            names += [f"{w}_stem", f"stripped_{w}_stem", f"{w}_hyphae", f"stripped_{w}_hyphae"]
        elif w == "bamboo":
            names += ["bamboo_block", "stripped_bamboo_block"]
        else:
            names += [f"{w}_log", f"stripped_{w}_log", f"{w}_wood", f"stripped_{w}_wood"]
    names += [f"{w}_leaves" for w in _WOODS if w not in ("bamboo", "crimson", "warped")]
    names += ["azalea_leaves", "flowering_azalea_leaves"]
    # The stained glass colours the palette lacks.
    names += [f"{c}_stained_glass" for c in _COLORS
              if "minecraft:" + c + "_stained_glass" not in AUX_PALETTE]
    # Stone families.
    names += ["mossy_cobblestone", "mossy_stone_bricks", "cracked_stone_bricks",
              "chiseled_stone_bricks",
              "cobbled_deepslate", "deepslate_bricks", "deepslate_tiles", "chiseled_deepslate",
              "polished_tuff", "tuff_bricks", "chiseled_tuff", "chiseled_tuff_bricks",
              "cut_sandstone", "chiseled_sandstone",
              "smooth_red_sandstone", "cut_red_sandstone", "chiseled_red_sandstone",
              "smooth_quartz", "quartz_pillar", "chiseled_quartz_block", "quartz_bricks",
              "red_nether_bricks", "polished_blackstone_bricks", "end_stone_bricks",
              "prismarine_bricks", "polished_basalt", "purpur_pillar"]
    # Copper: every oxidation stage of cut, chiseled, grate and bulb (waxed is an alias).
    for kind in ("cut_copper", "chiseled_copper", "copper_grate", "copper_bulb"):
        names += [st + kind for st in _COPPER_STAGES]
    # The rest of the dyed colour families.
    names += [f"{c}_concrete_powder" for c in _COLORS]
    names += [f"{c}_glazed_terracotta" for c in _COLORS]
    # Other full blocks builders use.
    names += ["mushroom_stem", "brown_mushroom_block", "red_mushroom_block", "hay_block",
              "ochre_froglight", "verdant_froglight", "pearlescent_froglight",
              "glowstone", "sea_lantern", "redstone_lamp", "bookshelf", "beehive", "lodestone",
              "netherrack", "soul_sand", "bedrock", "sand", "suspicious_sand", "podzol",
              "pale_moss_block", "ice", "sponge", "dried_kelp_block", "mangrove_roots",
              "dead_tube_coral_block", "dead_brain_coral_block", "dead_bubble_coral_block",
              "dead_fire_coral_block", "dead_horn_coral_block",
              "coal_block", "redstone_block", "diamond_block", "netherite_block",
              "coal_ore", "iron_ore", "lapis_ore", "diamond_ore", "nether_quartz_ore"]
    names += ["gold_ore", "emerald_ore", "redstone_ore", "emerald_block",
              "chiseled_polished_blackstone", "suspicious_gravel", "rooted_dirt", "red_sand",
              "shroomlight", "honey_block", "bee_nest", "bamboo_mosaic"]
    return ["minecraft:" + n for n in names]


MATERIALS = _PALETTE_MATERIALS + _more_materials()
MATERIAL_INDEX = {m: i for i, m in enumerate(MATERIALS)}
assert len(MATERIAL_INDEX) == len(MATERIALS), "duplicate material"


# ── Families: material -> {kind: block id} ────────────────────────────────────
def _families():
    """Every block each material can take, keyed by kind, and the materials with an axis.

    Kinds: slab, stairs, fence, wall, pane, door, trapdoor, gate, carpet (the full block is
    the material itself). Deepslate and the four copper blocks have no stairs, slab or wall of
    their own here: those belong to cobbled deepslate and cut copper.
    """
    fam = {m: {} for m in MATERIALS[1:]}

    def put(material, **kinds):
        material = "minecraft:" + material
        for kind, block in kinds.items():
            if block:
                fam[material][kind] = "minecraft:" + block

    borrowed = {"minecraft:deepslate", "minecraft:copper_block", "minecraft:exposed_copper",
                "minecraft:weathered_copper", "minecraft:oxidized_copper"}
    for base, (slab, stairs) in BLOCK_VARIANTS.items():
        if base not in borrowed:
            fam[base].update({k: v for k, v in (("slab", slab), ("stairs", stairs)) if v})
    for base, wall in WALL_VARIANTS.items():
        if base not in borrowed:
            fam[base]["wall"] = wall
    for w in _WOODS:
        put(f"{w}_planks", slab=f"{w}_slab", stairs=f"{w}_stairs", fence=f"{w}_fence",
            gate=f"{w}_fence_gate", door=f"{w}_door", trapdoor=f"{w}_trapdoor")
    for c in _COLORS:
        put(f"{c}_wool", carpet=f"{c}_carpet")
        put(f"{c}_stained_glass", pane=f"{c}_stained_glass_pane")
    put("glass", pane="glass_pane")
    put("moss_block", carpet="moss_carpet")
    put("iron_block", pane="iron_bars", door="iron_door", trapdoor="iron_trapdoor")
    for st in _COPPER_STAGES:
        put(st + "copper" if st else "copper_block", pane=st + "copper_bars",
            door=st + "copper_door", trapdoor=st + "copper_trapdoor")
        put(st + "cut_copper", slab=st + "cut_copper_slab", stairs=st + "cut_copper_stairs")
    put("nether_bricks", fence="nether_brick_fence")
    for base, slab, stairs, wall in (
            ("tuff", "tuff_slab", "tuff_stairs", "tuff_wall"),
            ("mossy_cobblestone", "mossy_cobblestone_slab", "mossy_cobblestone_stairs",
             "mossy_cobblestone_wall"),
            ("mossy_stone_bricks", "mossy_stone_brick_slab", "mossy_stone_brick_stairs",
             "mossy_stone_brick_wall"),
            ("cobbled_deepslate", "cobbled_deepslate_slab", "cobbled_deepslate_stairs",
             "cobbled_deepslate_wall"),
            ("deepslate_bricks", "deepslate_brick_slab", "deepslate_brick_stairs",
             "deepslate_brick_wall"),
            ("deepslate_tiles", "deepslate_tile_slab", "deepslate_tile_stairs",
             "deepslate_tile_wall"),
            ("polished_tuff", "polished_tuff_slab", "polished_tuff_stairs", "polished_tuff_wall"),
            ("tuff_bricks", "tuff_brick_slab", "tuff_brick_stairs", "tuff_brick_wall"),
            ("cut_sandstone", "cut_sandstone_slab", None, None),
            ("smooth_red_sandstone", "smooth_red_sandstone_slab", "smooth_red_sandstone_stairs",
             None),
            ("cut_red_sandstone", "cut_red_sandstone_slab", None, None),
            ("smooth_quartz", "smooth_quartz_slab", "smooth_quartz_stairs", None),
            ("red_nether_bricks", "red_nether_brick_slab", "red_nether_brick_stairs",
             "red_nether_brick_wall"),
            ("polished_blackstone_bricks", "polished_blackstone_brick_slab",
             "polished_blackstone_brick_stairs", "polished_blackstone_brick_wall"),
            ("end_stone_bricks", "end_stone_brick_slab", "end_stone_brick_stairs",
             "end_stone_brick_wall"),
            ("prismarine_bricks", "prismarine_brick_slab", "prismarine_brick_stairs", None),
            ("bamboo_mosaic", "bamboo_mosaic_slab", "bamboo_mosaic_stairs", None)):
        put(base, slab=slab, stairs=stairs, wall=wall)
    axis = {"minecraft:" + n for n in (
        "basalt", "polished_basalt", "bone_block", "deepslate", "quartz_pillar", "hay_block",
        "purpur_pillar", "ochre_froglight", "verdant_froglight", "pearlescent_froglight",
        "bamboo_block", "stripped_bamboo_block")}
    axis |= {m for m in MATERIALS if m.endswith(("_log", "_wood", "_stem", "_hyphae"))
             and m != "minecraft:mushroom_stem"}
    return fam, axis


FAMILIES, AXIS_MATERIALS = _families()
_KIND_OF_SHAPE = {SHAPE_SLAB_BOTTOM: "slab", SHAPE_SLAB_TOP: "slab", SHAPE_FENCE: "fence",
                  SHAPE_WALL: "wall", SHAPE_PANE: "pane", SHAPE_CARPET: "carpet",
                  SHAPE_TRAPDOOR_BOTTOM: "trapdoor", SHAPE_TRAPDOOR_TOP: "trapdoor",
                  SHAPE_GATE_X: "gate", SHAPE_GATE_Z: "gate"}
_KIND_OF_SHAPE.update({i: "stairs" for i in STAIR_SHAPES.values()})
_KIND_OF_SHAPE.update({i: "trapdoor" for i in TRAPDOOR_OPEN_SHAPES.values()})
_KIND_OF_SHAPE.update({i: "door" for i in DOOR_SHAPES.values()})

# block id -> (material, kind): the inverse of FAMILIES (each block has one owner)
_OWNER = {m: (m, "full") for m in MATERIALS[1:]}
for _m, _kinds in FAMILIES.items():
    for _k, _block in _kinds.items():
        assert _block not in _OWNER, "block with two owners: " + _block
        _OWNER[_block] = (_m, _k)


def _valid_table() -> np.ndarray:
    valid = np.zeros((len(SHAPES), len(MATERIALS)), dtype=bool)
    valid[SHAPE_AIR, 0] = True
    for m, i in MATERIAL_INDEX.items():
        if i == 0:
            continue
        valid[SHAPE_FULL, i] = True
        if m in AXIS_MATERIALS:
            valid[SHAPE_FULL_AXIS_X, i] = valid[SHAPE_FULL_AXIS_Z, i] = True
        kinds = FAMILIES[m]
        for shape, kind in _KIND_OF_SHAPE.items():
            if kind in kinds:
                valid[shape, i] = True
    return valid


VALID = _valid_table()
VALID.flags.writeable = False

# ── Aliases: blocks that look exactly like another one ─────────────────────────
ALIAS_PREFIXES = {"waxed_": "", "infested_": ""}
ALIAS_NAMES = {"minecraft:petrified_oak_slab": "minecraft:oak_slab"}


def _alias(name: str) -> str | None:
    if name in ALIAS_NAMES:
        return ALIAS_NAMES[name]
    bare = name.split(":", 1)[1] if ":" in name else name
    for prefix, repl in ALIAS_PREFIXES.items():
        if bare.startswith(prefix):
            twin = "minecraft:" + repl + bare[len(prefix):]
            if twin in _OWNER:
                return twin
    return None


def _props(state: str) -> dict:
    if "[" not in state:
        return {}
    body = state[state.index("[") + 1:state.rindex("]")]
    out = {}
    for part in body.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _join(name: str, props: dict) -> str:
    if not props:
        return name
    return "{}[{}]".format(name, ",".join(f"{k}={props[k]}" for k in sorted(props)))


# ── Encoding ──────────────────────────────────────────────────────────────────
def classify_state(state: str) -> dict:
    """How the vocabulary represents one block state.

    Returns {"status", "shape", "material", "reasons", "kind"} with status one of
      "air"      -- air (shape 0, material 0)
      "exact"    -- every property that matters is kept
      "lossy"    -- representable, but a visible property is dropped (`reasons` say which)
      "derived"  -- worked out from the neighbours (a door's upper half): IGNORE
      "ignore"   -- not in the vocabulary (decoration, fluids, unsupported blocks)
    """
    if not state:
        return {"status": "air", "shape": SHAPE_AIR, "material": 0, "reasons": [], "kind": "air"}
    name = state.split("[", 1)[0]
    if name in (AIR, "minecraft:cave_air", "minecraft:void_air"):
        return {"status": "air", "shape": SHAPE_AIR, "material": 0, "reasons": [], "kind": "air"}
    props = _props(state)
    reasons = []
    owner = _OWNER.get(name)
    if owner is None:
        twin = _alias(name)
        if twin is None:
            return {"status": "ignore", "shape": None, "material": None, "reasons": [],
                    "kind": None}
        owner = _OWNER[twin]
        reasons.append("alias")
    material, kind = owner
    mat = MATERIAL_INDEX[material]
    shape = None
    if kind == "full":
        axis = props.get("axis", "y") if material in AXIS_MATERIALS else "y"
        shape = {"x": SHAPE_FULL_AXIS_X, "z": SHAPE_FULL_AXIS_Z}.get(axis, SHAPE_FULL)
        if "facing" in props and props["facing"] != "north":
            reasons.append("facing")                 # glazed terracotta, beehive pattern
        if props.get("lit") == "true":
            reasons.append("lit")                    # redstone lamp, copper bulb glow
    elif kind == "slab":
        t = props.get("type", "bottom")
        shape = SHAPE_FULL if t == "double" else (SHAPE_SLAB_TOP if t == "top" else SHAPE_SLAB_BOTTOM)
        if t == "double":
            reasons.append("double_slab")            # becomes the full block (looks the same)
    elif kind == "stairs":
        half = "top" if props.get("half") == "top" else "bottom"
        shape = STAIR_SHAPES[(half, props.get("facing", "east"))]
    elif kind in ("fence", "wall", "pane", "carpet"):
        shape = {"fence": SHAPE_FENCE, "wall": SHAPE_WALL, "pane": SHAPE_PANE,
                 "carpet": SHAPE_CARPET}[kind]
    elif kind == "trapdoor":
        facing = props.get("facing", "north")
        if props.get("open") == "true":
            shape = TRAPDOOR_OPEN_SHAPES[OPPOSITE[facing]]
            if props.get("half") == "top":
                reasons.append("trapdoor_half")      # an open top trapdoor's texture flips
        else:
            shape = SHAPE_TRAPDOOR_TOP if props.get("half") == "top" else SHAPE_TRAPDOOR_BOTTOM
            if facing != "north":
                reasons.append("trapdoor_facing")    # closed: hinge side only, negligible
    elif kind == "door":
        if props.get("half") == "upper":
            return {"status": "derived", "shape": IGNORE_SHAPE, "material": IGNORE_MATERIAL,
                    "reasons": ["door_upper_half"], "kind": "door"}
        facing = props.get("facing", "north")
        if props.get("open") == "true":
            # An open door's plate swings to the hinge side: right -> clockwise of facing.
            cw = {"north": "east", "east": "south", "south": "west", "west": "north"}
            side = cw[facing] if props.get("hinge") == "right" else OPPOSITE[cw[facing]]
            shape = DOOR_SHAPES[side]
            reasons.append("door_open")
        else:
            shape = DOOR_SHAPES[OPPOSITE[facing]]
    elif kind == "gate":
        shape = SHAPE_GATE_X if props.get("facing", "north") in ("north", "south") else SHAPE_GATE_Z
        if props.get("open") == "true":
            reasons.append("gate_open")
    if not VALID[shape, mat]:                       # e.g. an axis on a material without one
        return {"status": "ignore", "shape": None, "material": None, "reasons": [], "kind": kind}
    status = "lossy" if reasons else "exact"
    return {"status": status, "shape": shape, "material": mat, "reasons": reasons, "kind": kind}


def encode_state(state: str):
    """Block state string -> (shape index, material index).

    A block outside the vocabulary gives (SHAPE_FULL, None); a door's upper half gives
    (IGNORE_SHAPE, IGNORE_MATERIAL).
    """
    c = classify_state(state)
    if c["status"] == "ignore":
        return SHAPE_FULL, None
    return c["shape"], c["material"]


# ── Decoding ──────────────────────────────────────────────────────────────────
_CLOSED_TRAPDOOR = {SHAPE_TRAPDOOR_BOTTOM: "bottom", SHAPE_TRAPDOOR_TOP: "top"}


@lru_cache(maxsize=None)
def _canonical(shape: int, material: int) -> str:
    """The one block state a valid pair stands for (properties sorted)."""
    m = MATERIALS[material]
    kinds = FAMILIES[m]
    if shape == SHAPE_FULL:
        return _join(m, {"persistent": "true"} if m.endswith("_leaves") else {})
    if shape in (SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z):
        return _join(m, {"axis": AXIAL[shape][1]})
    if shape in (SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP):
        return _join(kinds["slab"], {"type": "bottom" if shape == SHAPE_SLAB_BOTTOM else "top"})
    if shape in DIRECTIONAL and DIRECTIONAL[shape][0].startswith("stairs"):
        kind, facing = DIRECTIONAL[shape]
        return _join(kinds["stairs"], {"facing": facing, "half": kind.split("_")[1]})
    if shape == SHAPE_FENCE:
        return _join(kinds["fence"], {**{f: "false" for f in SIDES}, "waterlogged": "false"})
    if shape == SHAPE_WALL:
        return _join(kinds["wall"], {**{f: "none" for f in SIDES}, "up": "true",
                                     "waterlogged": "false"})
    if shape == SHAPE_PANE:
        return _join(kinds["pane"], {**{f: "false" for f in SIDES}, "waterlogged": "false"})
    if shape == SHAPE_CARPET:
        return kinds["carpet"]
    if shape in _CLOSED_TRAPDOOR:
        return _join(kinds["trapdoor"], {"facing": "north", "half": _CLOSED_TRAPDOOR[shape],
                                         "open": "false", "powered": "false",
                                         "waterlogged": "false"})
    if shape in TRAPDOOR_OPEN_SHAPES.values():
        side = DIRECTIONAL[shape][1]
        return _join(kinds["trapdoor"], {"facing": OPPOSITE[side], "half": "bottom",
                                         "open": "true", "powered": "false",
                                         "waterlogged": "false"})
    if shape in DOOR_SHAPES.values():
        side = DIRECTIONAL[shape][1]
        return _join(kinds["door"], {"facing": OPPOSITE[side], "half": "lower", "hinge": "left",
                                     "open": "false", "powered": "false"})
    if shape in (SHAPE_GATE_X, SHAPE_GATE_Z):
        return _join(kinds["gate"], {"facing": "north" if shape == SHAPE_GATE_X else "east",
                                     "in_wall": "false", "open": "false", "powered": "false"})
    raise AssertionError(f"no canonical state for {SHAPES[shape]}")


def decode(shape: int, material: int, strict: bool = False) -> str:
    """(shape index, material index) -> block state string; air for shape 0 or material 0.

    A valid pair decodes to its block state. An invalid pair raises with `strict=True`,
    otherwise it becomes the material's full block. (The builder only ever makes valid pairs.)
    """
    shape, material = int(shape), int(material)
    if shape == SHAPE_AIR or material == 0:
        return AIR
    if shape not in range(len(SHAPES)) or not 0 <= material < len(MATERIALS):
        raise ValueError("shape/material outside vocabulary")
    if VALID[shape, material]:
        return _canonical(shape, material)
    if strict:
        raise ValueError(f"invalid pair: {SHAPES[shape]} x {MATERIALS[material]}")
    return _canonical(SHAPE_FULL, material)
