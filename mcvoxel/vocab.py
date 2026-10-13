"""The factored block vocabulary a model predicts: SHAPE x MATERIAL (`mcvox-v3`).

A blockstate string like `minecraft:dark_oak_trapdoor[facing=south,half=bottom,open=true]` is
two things: a **material** -- the texture family the block belongs to (dark oak) -- and a
**shape** -- geometry plus orientation (a trapdoor opened against the north side of its
voxel). The model predicts the two as separate categorical maps; a material takes every shape
its family has in the game, recorded in `VALID[shape, material]`. `decode` turns every valid
pair into a real blockstate; training targets only ever hold valid pairs, and at inference the
invalid logits are masked.

Conventions (verified, not assumed):

* Axes: +X = east, +Z = south, Y up -- the Sponge `.schem` convention, and the one
  `block_shapes` classifies in.
* Stair `facing` names the HIGH side, for both halves (measured on 42,233 real stairs; see
  the README). `stairs_top_*` are the upside-down stairs.
* Doors and trapdoors are named by the side of the voxel their plate occupies:
  `door_north` / `trapdoor_open_north` sit against the voxel's north face. Minecraft's
  `facing` is then the opposite side (`facing=south`) -- true for doors and trapdoors alike.
* A material is named by its full block (`minecraft:oak_planks`); oak stairs, fences, fence
  gates, doors and trapdoors are shapes of it. Iron (`iron_block`) has bars, a door and a
  trapdoor; each copper oxidation stage (`copper_block`, `exposed_copper`, ...) likewise.

Append-only history: v1 had shapes 0-7, v2 added fence and wall (8, 9); v3 appends shapes
10-29 and materials 120+. Every older ID is unchanged and every v2 pair still decodes
(`decode(..., strict=False)` falls back exactly as v2 did for pairs v3 does not make
valid). The app's colour palettes (`BLOCK_PALETTE`, `BUILDER_PALETTE`, `AUX_PALETTE`) are
untouched: v3 materials live in `V3_MATERIALS` only.

Predicted versus derived. The model predicts what cannot be derived; the rest is restored on
the assembled grid by `blockstates.finish_states`: stair corners (both halves), fence / wall /
pane connections, a door's upper half and hinge (the game's placement rule), a fence gate's
`in_wall`. Properties with no visible effect (waterlogged, powered, leaf distance, mushroom
faces, grass snowiness) are dropped as derived. Properties dropped *with* a visible effect
make the encoding lossy, and `classify_state` names the reason.

Supervision policy (`encode_target`): interior voxels, blocks outside the vocabulary (decor,
fluids, snow layers, unsupported blocks) and a door's upper half are IGNORE, never air.

Validation targets Minecraft `TARGET_VERSION` (DataVersion `TARGET_DATA_VERSION`), the
newest 1.21.x release; its block-property table, extracted from PrismarineJS
minecraft-data (MIT), ships in `mcvoxel/blocktables/`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import block_shapes as bs
from .block_palette import AUX_PALETTE, BLOCK_PALETTE

VOCAB_VERSION = "mcvox-v3"
AIR = "minecraft:air"
TARGET_VERSION = "1.21.11"
TARGET_DATA_VERSION = 4671
V2_DATA_VERSION = 3465                      # Minecraft 1.20.1, what mcvox-v2 validated for
SCHEMA_PATH = Path(__file__).with_name("blocktables") / "blocks_1_21_11.json"

# ── Shapes (append-only) ──────────────────────────────────────────────────────
SIDES = ("east", "west", "north", "south")
OPPOSITE = {"east": "west", "west": "east", "north": "south", "south": "north"}

SHAPES = ["air", "full", "slab_bottom", "slab_top",
          "stairs_east", "stairs_west", "stairs_north", "stairs_south",        # v1: 0-7
          "fence", "wall",                                                      # v2: 8-9
          "stairs_top_east", "stairs_top_west", "stairs_top_north", "stairs_top_south",
          "pane",
          "trapdoor_bottom", "trapdoor_top",
          "trapdoor_open_north", "trapdoor_open_south", "trapdoor_open_east", "trapdoor_open_west",
          "door_north", "door_south", "door_east", "door_west",
          "full_axis_x", "full_axis_z",
          "fence_gate_x", "fence_gate_z",
          "carpet"]                                                             # v3: 10-29
SHAPE_INDEX = {s: i for i, s in enumerate(SHAPES)}
V2_SHAPE_COUNT = 10
SHAPE_AIR, SHAPE_FULL, SHAPE_SLAB_BOTTOM, SHAPE_SLAB_TOP = 0, 1, 2, 3
SHAPE_FENCE, SHAPE_WALL, SHAPE_PANE, SHAPE_CARPET = 8, 9, 14, 29
SHAPE_TRAPDOOR_BOTTOM, SHAPE_TRAPDOOR_TOP = 15, 16
SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z = 25, 26
SHAPE_GATE_X, SHAPE_GATE_Z = 27, 28
IGNORE_SHAPE = 255            # interior / outside the supervised region / not representable
IGNORE_MATERIAL = 65535

STAIR_SHAPES = {("bottom", f): SHAPE_INDEX["stairs_" + f] for f in SIDES}
STAIR_SHAPES.update({("top", f): SHAPE_INDEX["stairs_top_" + f] for f in SIDES})
TRAPDOOR_OPEN_SHAPES = {s: SHAPE_INDEX["trapdoor_open_" + s] for s in SIDES}   # plate side
DOOR_SHAPES = {s: SHAPE_INDEX["door_" + s] for s in SIDES}                     # plate side
# Shapes with a horizontal direction (rotate / mirror remap it) or a horizontal axis.
DIRECTIONAL = {i: ("stairs_" + half, f) for (half, f), i in STAIR_SHAPES.items()}
DIRECTIONAL.update({i: ("trapdoor_open", s) for s, i in TRAPDOOR_OPEN_SHAPES.items()})
DIRECTIONAL.update({i: ("door", s) for s, i in DOOR_SHAPES.items()})
AXIAL = {SHAPE_FULL_AXIS_X: ("full", "x"), SHAPE_FULL_AXIS_Z: ("full", "z"),
         SHAPE_GATE_X: ("fence_gate", "x"), SHAPE_GATE_Z: ("fence_gate", "z")}

# block_shapes code -> vocab shape index
FROM_BLOCK_SHAPES = {bs.SHAPE_FULL: 1, bs.SHAPE_SLAB_BOTTOM: 2, bs.SHAPE_SLAB_TOP: 3,
                     bs.SHAPE_STAIR_E: 4, bs.SHAPE_STAIR_W: 5, bs.SHAPE_STAIR_N: 6,
                     bs.SHAPE_STAIR_S: 7}
_FACING = {4: "east", 5: "west", 6: "north", 7: "south"}           # v2 stair shapes
_FACING_TO_SHAPE = {v: k for k, v in _FACING.items()}

# ── Materials (append-only) ───────────────────────────────────────────────────
V2_MATERIALS = [AIR] + sorted(BLOCK_PALETTE) + sorted(AUX_PALETTE)
assert len(V2_MATERIALS) == 120, "the v2 material list must not change"

_WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry",
          "pale_oak", "bamboo", "crimson", "warped")
_COLORS = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
           "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black")
_COPPER_STAGES = ("", "exposed_", "weathered_", "oxidized_")


def _v3_material_names() -> list[str]:
    """Materials v3 appends, in their fixed order. Never reorder; only append."""
    names = ["pale_oak_planks", "bamboo_planks"]
    # Every 1.21 wood: logs, stripped logs, wood and stripped wood (stems, hyphae for nether).
    for w in _WOODS:
        if w in ("crimson", "warped"):
            names += [f"{w}_stem", f"stripped_{w}_stem", f"{w}_hyphae", f"stripped_{w}_hyphae"]
        elif w == "bamboo":
            names += ["bamboo_block", "stripped_bamboo_block"]
        else:
            names += [f"{w}_log", f"stripped_{w}_log", f"{w}_wood", f"stripped_{w}_wood"]
    names += [f"{w}_leaves" for w in _WOODS if w not in ("bamboo", "crimson", "warped")]
    names += ["azalea_leaves", "flowering_azalea_leaves"]
    # The stained glass colours v2 lacked (v2 has plain, tinted and 7 colours).
    names += [f"{c}_stained_glass" for c in _COLORS
              if "minecraft:" + c + "_stained_glass" not in AUX_PALETTE]
    # Stone families: the deepslate, tuff, sandstone and quartz families complete; the
    # others where the dataset uses them (>= 2 builds or >= 200 supervised blocks).
    names += ["mossy_cobblestone", "mossy_stone_bricks", "cracked_stone_bricks",
              "chiseled_stone_bricks",
              "cobbled_deepslate", "deepslate_bricks", "deepslate_tiles", "chiseled_deepslate",
              "polished_tuff", "tuff_bricks", "chiseled_tuff", "chiseled_tuff_bricks",
              "cut_sandstone", "chiseled_sandstone",
              "smooth_red_sandstone", "cut_red_sandstone", "chiseled_red_sandstone",
              "smooth_quartz", "quartz_pillar", "chiseled_quartz_block", "quartz_bricks",
              "red_nether_bricks", "polished_blackstone_bricks", "end_stone_bricks",
              "prismarine_bricks", "polished_basalt", "purpur_pillar"]
    # Copper: every oxidation stage of cut, chiseled, grate and bulb (waxed = alias).
    for kind in ("cut_copper", "chiseled_copper", "copper_grate", "copper_bulb"):
        names += [st + kind for st in _COPPER_STAGES]
    # Dyed colour families completed, as v2 already had wool, concrete and terracotta.
    names += [f"{c}_concrete_powder" for c in _COLORS]
    names += [f"{c}_glazed_terracotta" for c in _COLORS]
    # Full blocks the dataset builds with (>= 2 builds or >= 200 supervised blocks), plus
    # the red mushroom block and the other froglights to complete those families.
    names += ["mushroom_stem", "brown_mushroom_block", "red_mushroom_block", "hay_block",
              "ochre_froglight", "verdant_froglight", "pearlescent_froglight",
              "glowstone", "sea_lantern", "redstone_lamp", "bookshelf", "beehive", "lodestone",
              "netherrack", "soul_sand", "bedrock", "sand", "suspicious_sand", "podzol",
              "pale_moss_block", "ice", "sponge", "dried_kelp_block", "mangrove_roots",
              "dead_tube_coral_block", "dead_brain_coral_block", "dead_bubble_coral_block",
              "dead_fire_coral_block", "dead_horn_coral_block",
              "coal_block", "redstone_block", "diamond_block", "netherite_block",
              "coal_ore", "iron_ore", "lapis_ore", "diamond_ore", "nether_quartz_ore"]
    return ["minecraft:" + n for n in names]


V3_MATERIALS = _v3_material_names()
MATERIALS = V2_MATERIALS + V3_MATERIALS
MATERIAL_INDEX = {m: i for i, m in enumerate(MATERIALS)}
assert len(MATERIAL_INDEX) == len(MATERIALS), "duplicate material"
V2_MATERIAL_COUNT = len(V2_MATERIALS)


# ── Families: material -> {kind: block id} ────────────────────────────────────
def _families():
    """Every block each material can take, keyed by kind; plus the axis materials.

    Kinds: slab, stairs, fence, wall, pane, door, trapdoor, gate, carpet (the full block is
    the material itself). Built from v2's tables, then 1.21's: v2's two shortcuts that v3
    replaces with real materials -- deepslate's stairs, slab and wall (which are cobbled
    deepslate's) and copper's stairs and slab (cut copper's) -- are left out here and stay
    decodable for v2 targets through the legacy path.
    """
    fam = {m: {} for m in MATERIALS[1:]}

    def put(material, **kinds):
        material = "minecraft:" + material
        for kind, block in kinds.items():
            if block:
                fam[material][kind] = "minecraft:" + block

    legacy = {"minecraft:deepslate", "minecraft:copper_block", "minecraft:exposed_copper",
              "minecraft:weathered_copper", "minecraft:oxidized_copper"}
    for base, (slab, stairs) in bs.BLOCK_VARIANTS.items():
        if base not in legacy:
            fam[base].update({k: v for k, v in (("slab", slab), ("stairs", stairs)) if v})
    for base, wall in bs.WALL_VARIANTS.items():
        if base not in legacy:
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
            ("prismarine_bricks", "prismarine_brick_slab", "prismarine_brick_stairs", None)):
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

# ── Read-time aliases (not vocabulary): blocks that look exactly like another ──
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


# ── v2 tables, frozen: the legacy decode path and DataVersion 3465 validation ──
_V2_MATERIAL_INDEX = {m: i for i, m in enumerate(V2_MATERIALS)}
_VARIANT_BASE = {}            # v2: variant block -> (material, kind)
for _base, (_slab, _stairs) in bs.BLOCK_VARIANTS.items():
    if _slab:
        _VARIANT_BASE[_slab] = (_base, "slab")
    if _stairs:
        _VARIANT_BASE[_stairs] = (_base, "stairs")
for _family, _kind in ((bs.FENCE_VARIANTS, "fence"), (bs.WALL_VARIANTS, "wall")):
    _VARIANT_BASE.update({v: (k, _kind) for k, v in _family.items()})


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
    """How v3 represents one blockstate.

    Returns {"status", "shape", "material", "reasons", "kind"} with status one of
      "air"      -- air (shape 0, material 0)
      "exact"    -- every predicted property kept; dropped ones are derived or invisible
      "lossy"    -- representable, but a visible property is dropped (`reasons` say which)
      "derived"  -- rebuilt at assembly (a door's upper half): IGNORE in targets
      "ignore"   -- outside the vocabulary (decor, fluids, unsupported): IGNORE in targets
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
    if not VALID[shape, mat]:                       # e.g. an axis on a non-axis material
        return {"status": "ignore", "shape": None, "material": None, "reasons": [], "kind": kind}
    status = "lossy" if reasons else "exact"
    return {"status": status, "shape": shape, "material": mat, "reasons": reasons, "kind": kind}


def encode_state(state: str):
    """Blockstate string -> (shape index, material index).

    Blocks outside the vocabulary return (SHAPE_FULL, None), as in v2; a door's upper half
    (derived at assembly) returns (IGNORE_SHAPE, IGNORE_MATERIAL).
    """
    c = classify_state(state)
    if c["status"] == "ignore":
        return SHAPE_FULL, None
    return c["shape"], c["material"]


def encode_target(blocks, palette, shell, exterior, interior=None):
    """A palette grid + its masks -> (shape uint8, material uint16) training targets.

    `blocks` indexes `palette` (blockstate strings); `shell`, `exterior` (and `interior`)
    are the supervision masks of `schemset.interior`, any axis order as long as all agree.
    Exterior air -> air. Shell blocks -> their (shape, material) when the vocabulary holds
    them; decor, unsupported blocks and door upper halves -> IGNORE. Interior and anything
    else -> IGNORE. Never turns an unrepresentable block into air.
    """
    blocks = np.asarray(blocks)
    lut_shape = np.full(len(palette), IGNORE_SHAPE, dtype=np.uint8)
    lut_mat = np.full(len(palette), IGNORE_MATERIAL, dtype=np.uint16)
    for i, state in enumerate(palette):
        c = classify_state(str(state))
        if c["status"] in ("exact", "lossy", "air"):
            lut_shape[i], lut_mat[i] = c["shape"], c["material"]
    shape = np.full(blocks.shape, IGNORE_SHAPE, dtype=np.uint8)
    material = np.full(blocks.shape, IGNORE_MATERIAL, dtype=np.uint16)
    ext = np.asarray(exterior, dtype=bool)
    shape[ext] = SHAPE_AIR
    material[ext] = 0
    sup = np.asarray(shell, dtype=bool) & ~ext
    if interior is not None:
        sup &= ~np.asarray(interior, dtype=bool)
    shape[sup] = lut_shape[blocks[sup]]
    material[sup] = lut_mat[blocks[sup]]
    air_in_shell = sup & (shape == SHAPE_AIR)       # an air palette entry inside the shell
    shape[air_in_shell], material[air_in_shell] = IGNORE_SHAPE, IGNORE_MATERIAL
    return shape, material


# ── Decoding ──────────────────────────────────────────────────────────────────
_CLOSED_TRAPDOOR = {SHAPE_TRAPDOOR_BOTTOM: "bottom", SHAPE_TRAPDOOR_TOP: "top"}


@lru_cache(maxsize=None)
def _canonical(shape: int, material: int) -> str:
    """The one blockstate a valid pair stands for (properties sorted)."""
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
    """(shape index, material index) -> blockstate string. Air for shape 0 or material 0.

    A VALID pair decodes to its canonical state (round-trips through `encode_state`).
    An invalid pair raises with `strict=True`. Otherwise it falls back: a v2 pair exactly as
    v2 decoded it (the nearest variant-capable material by colour, or v2's deepslate and
    copper shortcuts), any other pair to the material's full block.
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
    if shape < V2_SHAPE_COUNT and material < V2_MATERIAL_COUNT:
        return _decode_v2(shape, material)
    return _canonical(SHAPE_FULL, material)


def _decode_v2(shape: int, material: int) -> str:
    """mcvox-v2's decode, unchanged (the legacy path for pairs v3 does not make valid)."""
    if shape == SHAPE_AIR or material == 0:
        return AIR
    base = MATERIALS[material]
    if shape in (SHAPE_FENCE, SHAPE_WALL):
        from .block_palette import match_blocks_restricted
        family = bs.FENCE_VARIANTS if shape == SHAPE_FENCE else bs.WALL_VARIANTS
        if base not in family:
            rgb = BLOCK_PALETTE.get(base) or AUX_PALETTE.get(base)
            base = match_blocks_restricted([rgb], list(family))[0][0]
        name = family[base]
        if shape == SHAPE_FENCE:
            return name + "[east=false,north=false,south=false,waterlogged=false,west=false]"
        return name + "[east=none,north=none,south=none,up=true,waterlogged=false,west=none]"
    if shape == SHAPE_FULL or base in AUX_PALETTE:
        return base
    want_stairs = shape in _FACING
    if base not in bs.BLOCK_VARIANTS or bs.BLOCK_VARIANTS[base][1 if want_stairs else 0] is None:
        from .block_palette import match_blocks_restricted
        pool = bs._STAIRS_CAPABLE if want_stairs else bs._SLAB_CAPABLE
        rgb = BLOCK_PALETTE.get(base) or AUX_PALETTE.get(base)
        base = match_blocks_restricted([rgb], pool)[0][0]
    slab, stairs = bs.BLOCK_VARIANTS[base]
    if shape == 2:
        return "{}[type=bottom]".format(slab)
    if shape == 3:
        return "{}[type=top]".format(slab)
    return "{}[facing={},half=bottom]".format(stairs, _FACING[shape])


def valid_pairs() -> dict:
    """{shape name: [material ids]} for every shape (the inference mask, compactly)."""
    return {SHAPES[s]: [int(m) for m in np.flatnonzero(VALID[s])] for s in range(len(SHAPES))}


def vocab_json() -> str:
    return json.dumps({
        "version": VOCAB_VERSION, "shapes": SHAPES, "materials": MATERIALS,
        "ignore_shape": IGNORE_SHAPE, "ignore_material": IGNORE_MATERIAL,
        "v2_shape_count": V2_SHAPE_COUNT, "v2_material_count": V2_MATERIAL_COUNT,
        "valid": valid_pairs(), "valid_pairs": int(VALID.sum()),
        "target_version": TARGET_VERSION, "target_data_version": TARGET_DATA_VERSION,
        "aliases": {"prefixes": ALIAS_PREFIXES, "names": ALIAS_NAMES},
        "derived": ["stairs shape (corners)", "fence/wall/pane connections", "wall up",
                    "door upper half and hinge", "fence gate in_wall", "waterlogged",
                    "powered", "leaf distance/persistent", "mushroom faces", "snowy"],
    }, indent=1)


# ── Validation ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def block_schema() -> dict:
    """{block name (no namespace): {property: [values]}} for `TARGET_VERSION`."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["blocks"]


def validate_state(state: str, data_version: int = V2_DATA_VERSION):
    """Reject a state that is not a real block of the given version.

    `TARGET_DATA_VERSION` (1.21.11) checks the block and every property value against
    the bundled minecraft-data table. `V2_DATA_VERSION` (3465, 1.20.1) keeps mcvox-v2's
    contract unchanged: only v2's materials and variants, with the properties v2 emits.
    """
    if data_version == TARGET_DATA_VERSION:
        schema = block_schema()
        name = state.split("[", 1)[0]
        bare = name.split(":", 1)[1] if name.startswith("minecraft:") else None
        if bare is None or bare not in schema:
            raise ValueError("unknown block in {}: {}".format(TARGET_VERSION, name))
        for key, value in _props(state).items():
            if value not in schema[bare].get(key, ()):
                raise ValueError("invalid property for {}: {}".format(TARGET_VERSION, state))
        return
    if data_version != V2_DATA_VERSION:
        raise ValueError("no block table for DataVersion {} (have {} and {})".format(
            data_version, V2_DATA_VERSION, TARGET_DATA_VERSION))
    name = state.split("[", 1)[0]
    if name in _V2_MATERIAL_INDEX:
        allowed = {}
    elif name in _VARIANT_BASE:
        _, kind = _VARIANT_BASE[name]
        allowed = {
            "slab": {"type": {"top", "bottom", "double"}, "waterlogged": {"true", "false"}},
            "stairs": {"facing": set(_FACING_TO_SHAPE), "half": {"top", "bottom"},
                       "shape": {"straight", "inner_left", "inner_right", "outer_left", "outer_right"},
                       "waterlogged": {"true", "false"}},
            "fence": {**{f: {"true", "false"} for f in _FACING_TO_SHAPE}, "waterlogged": {"true", "false"}},
            "wall": {**{f: {"none", "low", "tall"} for f in _FACING_TO_SHAPE},
                     "up": {"true", "false"}, "waterlogged": {"true", "false"}},
        }[kind]
    else:
        raise ValueError("unknown target block: " + name)
    for key, value in _props(state).items():
        if key not in allowed or value not in allowed[key]:
            raise ValueError("invalid target property: " + state)
