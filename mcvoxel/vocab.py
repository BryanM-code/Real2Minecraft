"""The factored block vocabulary a model predicts: SHAPE x MATERIAL.

A blockstate string like `minecraft:oak_stairs[facing=east,half=bottom]` is two things: a
material (oak planks) and a shape (stairs rising to the east). Predicting them as two small
categorical maps instead of one flat list of ~4,000 states keeps every class populated and
lets any material take any shape its family supports.

Conventions (verified, not assumed):

* Axes: +X = east, +Z = south, Y up -- the Sponge `.schem` convention, and the one
  `block_shapes` classifies in.
* Stair `facing` names the HIGH side. Measured on the 106 real builds in BuildDataCollector:
  for 42,233 bottom-half stairs, the block one up and one toward `facing` is solid 84.0% of
  the time and one up and away 5.0%. `block_shapes.SHAPE_STAIR_E` (full bottom, top half
  on +X) therefore writes `facing=east`, as it does.

Version 2 appends fences and walls. Version 1 shape and material indices are unchanged.
Connections are derived on the assembled grid, not predicted per training window.
Upside-down stairs, panes, doors and trapdoors remain outside this vocabulary.
"""
from __future__ import annotations

import json

from . import block_shapes as bs
from .block_palette import AUX_PALETTE, BLOCK_PALETTE

VOCAB_VERSION = "mcvox-v2"
AIR = "minecraft:air"

SHAPES = ["air", "full", "slab_bottom", "slab_top",
          "stairs_east", "stairs_west", "stairs_north", "stairs_south", "fence", "wall"]
SHAPE_FENCE, SHAPE_WALL = 8, 9
SHAPE_AIR, SHAPE_FULL = 0, 1
IGNORE_SHAPE = 255            # interior / outside the supervised region
IGNORE_MATERIAL = 65535

MATERIALS = [AIR] + sorted(BLOCK_PALETTE) + sorted(AUX_PALETTE)
MATERIAL_INDEX = {m: i for i, m in enumerate(MATERIALS)}

# block_shapes code -> vocab shape index
FROM_BLOCK_SHAPES = {bs.SHAPE_FULL: 1, bs.SHAPE_SLAB_BOTTOM: 2, bs.SHAPE_SLAB_TOP: 3,
                     bs.SHAPE_STAIR_E: 4, bs.SHAPE_STAIR_W: 5, bs.SHAPE_STAIR_N: 6,
                     bs.SHAPE_STAIR_S: 7}
_FACING = {4: "east", 5: "west", 6: "north", 7: "south"}
_FACING_TO_SHAPE = {v: k for k, v in _FACING.items()}

_VARIANT_BASE = {}
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


def encode_state(state: str):
    """Blockstate string -> (shape index, material index). Unknown materials -> None."""
    if not state or state.split("[", 1)[0] in (AIR, "minecraft:cave_air", "minecraft:void_air"):
        return SHAPE_AIR, 0
    name = state.split("[", 1)[0]
    props = _props(state)
    base, kind = _VARIANT_BASE.get(name, (name, "full"))
    mat = MATERIAL_INDEX.get(base)
    if kind == "slab":
        t = props.get("type", "bottom")
        shape = SHAPE_FULL if t == "double" else (2 if t == "bottom" else 3)
    elif kind == "stairs":
        # v1 has no upside-down stairs: a top-half stair reads as a top slab.
        shape = 3 if props.get("half") == "top" else _FACING_TO_SHAPE.get(props.get("facing", "east"), 4)
    elif kind in ("fence", "wall"):
        shape = SHAPE_FENCE if kind == "fence" else SHAPE_WALL
    else:
        shape = SHAPE_FULL
    return shape, mat


def decode(shape: int, material: int) -> str:
    """(shape index, material index) -> blockstate string. Air for shape 0.

    A material with no variant for the requested shape takes the nearest variant-capable
    material by colour (the rule `block_shapes.apply_shapes` uses), so every decoded state
    is a block that exists.
    """
    if shape == SHAPE_AIR or material == 0:
        return AIR
    if shape not in range(len(SHAPES)) or not 0 <= material < len(MATERIALS):
        raise ValueError("shape/material outside vocabulary")
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


def vocab_json() -> str:
    return json.dumps({"version": VOCAB_VERSION, "shapes": SHAPES, "materials": MATERIALS,
                       "ignore_shape": IGNORE_SHAPE, "ignore_material": IGNORE_MATERIAL},
                      indent=1)


def validate_state(state: str, data_version: int = 3465):
    """Reject unsupported target states; this contract targets Minecraft 1.20.1."""
    if data_version != 3465:
        raise ValueError("target vocabulary is validated for DataVersion 3465")
    name = state.split("[", 1)[0]
    if name in MATERIAL_INDEX:
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
