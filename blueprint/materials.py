"""Colour and material words -> blocks.

The vision model reports colours and materials in words ("cream wood siding", "dark grey
roof"); these tables turn them into blocks. Within a table, colour words are matched by
substring and the first hit wins, so a more specific word comes before a general one
("dark brown" before "brown"), and the last entry ("") is the default.
"""
from __future__ import annotations

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
