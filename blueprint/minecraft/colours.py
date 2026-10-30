"""The colour the renderer gives a block state.

`game_colour` is what renders use: a full block's texture average from `blocks.BLOCK_PALETTE`
when it has one, otherwise `colour_rules`. The rules cover every other block (stairs,
slabs, fences, logs, panes, dyed blocks) in three tiers, cheapest first:

1. `BASE_COLORS`: an explicit entry, for blocks whose colour is their own.
2. Family rules: a dye prefix (`blue_`) crossed with a material suffix (`_wool`,
   `_concrete`, `_terracotta`), plus wood and stone families.
3. A neutral grey, so an unknown block still draws as solid mass.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .blocks import BLOCK_PALETTE

FALLBACK = (132, 132, 138)

# ── The 16 dye colours ────────────────────────────────────────────────────────
# Used for wool/concrete/terracotta/glass/carpet/bed/banner/shulker families.
DYES = {
    "white": (233, 236, 236), "orange": (240, 118, 19), "magenta": (189, 68, 179),
    "light_blue": (58, 175, 217), "yellow": (248, 198, 39), "lime": (112, 185, 25),
    "pink": (237, 141, 172), "gray": (62, 68, 71), "light_gray": (142, 142, 134),
    "cyan": (21, 137, 145), "purple": (121, 42, 172), "blue": (53, 57, 157),
    "brown": (114, 71, 40), "green": (84, 109, 27), "red": (161, 39, 34),
    "black": (25, 25, 29),
}

# Materials that take a dye prefix, with a multiplier applied to the dye colour.
# Terracotta is muted and earthy; glass is lightened because it reads as translucent.
DYED_MATERIALS = {
    "_wool": 1.0, "_concrete": 1.0, "_concrete_powder": 1.08, "_carpet": 1.0,
    "_terracotta": 0.72, "_glazed_terracotta": 0.85, "_stained_glass": 1.15,
    "_stained_glass_pane": 1.15, "_bed": 1.0, "_banner": 1.0, "_wall_banner": 1.0,
    "_shulker_box": 0.9, "_candle": 1.0,
}

# ── Wood types ────────────────────────────────────────────────────────────────
# One colour per wood; planks/stairs/slabs/fences/doors all inherit it, and logs get a
# darker bark tone.
WOODS = {
    "oak": (162, 130, 78), "spruce": (114, 84, 48), "birch": (192, 175, 121),
    "jungle": (160, 115, 80), "acacia": (168, 90, 50), "dark_oak": (66, 43, 20),
    "mangrove": (117, 54, 48), "cherry": (226, 177, 172), "bamboo": (193, 158, 63),
    "crimson": (104, 52, 76), "warped": (43, 104, 99),
}
BARK = {
    "oak": (109, 85, 51), "spruce": (58, 39, 20), "birch": (215, 213, 207),
    "jungle": (85, 67, 27), "acacia": (103, 96, 86), "dark_oak": (60, 46, 26),
    "mangrove": (84, 60, 55), "cherry": (60, 45, 51), "bamboo": (146, 122, 44),
    "crimson": (92, 25, 29), "warped": (43, 104, 99),
}
WOOD_SUFFIXES = ("_planks", "_stairs", "_slab", "_fence", "_fence_gate", "_door",
                 "_trapdoor", "_sign", "_wall_sign", "_pressure_plate", "_button",
                 "_hanging_sign", "_wall_hanging_sign", "_wall_fan")
BARK_SUFFIXES = ("_log", "_wood", "_stem", "_hyphae", "_roots")

# ── Explicit blocks ───────────────────────────────────────────────────────────
BASE_COLORS = {
    # Stone family
    "stone": (125, 125, 125), "cobblestone": (122, 121, 122),
    "mossy_cobblestone": (110, 118, 96), "smooth_stone": (158, 158, 158),
    "stone_bricks": (122, 121, 122), "mossy_stone_bricks": (110, 118, 96),
    "cracked_stone_bricks": (118, 117, 118), "chiseled_stone_bricks": (119, 118, 119),
    "granite": (149, 103, 85), "polished_granite": (154, 107, 89),
    "diorite": (188, 188, 190), "polished_diorite": (192, 193, 195),
    "andesite": (136, 136, 137), "polished_andesite": (132, 135, 135),
    "deepslate": (77, 77, 81), "cobbled_deepslate": (77, 77, 80),
    "deepslate_bricks": (71, 71, 74), "deepslate_tiles": (54, 54, 57),
    "tuff": (108, 109, 102), "calcite": (223, 224, 220), "dripstone_block": (140, 108, 90),
    "blackstone": (42, 36, 42), "basalt": (80, 80, 86), "obsidian": (21, 18, 30),
    "bedrock": (85, 85, 85), "netherrack": (97, 38, 38), "end_stone": (221, 223, 165),
    "end_stone_bricks": (218, 224, 162), "purpur_block": (169, 125, 169),
    "purpur_pillar": (171, 129, 171), "quartz_block": (235, 229, 222),
    "smooth_quartz": (235, 229, 222), "chiseled_quartz_block": (232, 226, 218),
    "quartz_pillar": (235, 231, 224), "quartz_bricks": (234, 228, 219),
    # Dirt / ground
    "dirt": (134, 96, 67), "coarse_dirt": (119, 85, 59), "rooted_dirt": (144, 103, 76),
    "grass_block": (95, 159, 53), "grass_path": (148, 121, 65), "dirt_path": (148, 121, 65),
    "podzol": (91, 64, 27), "mycelium": (111, 99, 95), "farmland": (94, 62, 34),
    "sand": (219, 207, 163), "red_sand": (190, 102, 33), "gravel": (131, 127, 126),
    "clay": (160, 166, 179), "soul_sand": (81, 62, 50), "soul_soil": (75, 57, 46),
    "snow": (249, 254, 254), "snow_block": (249, 254, 254), "powder_snow": (248, 253, 253),
    "ice": (145, 183, 253), "packed_ice": (141, 180, 250), "blue_ice": (116, 167, 253),
    "magma_block": (142, 63, 31),
    # Ores & metal blocks
    "coal_ore": (105, 105, 105), "iron_ore": (136, 130, 127), "gold_ore": (145, 129, 96),
    "diamond_ore": (121, 143, 143), "emerald_ore": (110, 137, 116),
    "lapis_ore": (107, 117, 141), "redstone_ore": (133, 107, 107),
    "copper_ore": (127, 128, 121), "coal_block": (16, 15, 15),
    "iron_block": (220, 220, 220), "gold_block": (246, 208, 61),
    "diamond_block": (98, 219, 213), "emerald_block": (42, 203, 88),
    "lapis_block": (30, 67, 140), "redstone_block": (175, 24, 5),
    "netherite_block": (66, 61, 63), "copper_block": (192, 107, 79),
    "oxidized_copper": (82, 162, 132), "weathered_copper": (108, 153, 117),
    "exposed_copper": (161, 125, 102),
    "pale_oak_planks": (227, 217, 214), "bone_block": (229, 225, 207), "mushroom_stem": (203, 196, 185),
    # Building blocks
    "bricks": (150, 97, 83), "nether_bricks": (44, 22, 26),
    "red_nether_bricks": (69, 7, 9), "prismarine": (99, 156, 151),
    "prismarine_bricks": (99, 171, 158), "dark_prismarine": (51, 91, 75),
    "sea_lantern": (172, 199, 190), "glowstone": (171, 131, 84),
    "shroomlight": (240, 146, 70), "sandstone": (216, 203, 155),
    "smooth_sandstone": (219, 207, 163), "chiseled_sandstone": (215, 202, 157),
    "cut_sandstone": (216, 203, 155), "red_sandstone": (186, 99, 29),
    "smooth_red_sandstone": (190, 102, 33), "cut_red_sandstone": (186, 99, 29),
    "chiseled_red_sandstone": (182, 98, 32), "bookshelf": (110, 86, 53),
    "crafting_table": (124, 87, 51), "furnace": (109, 108, 108),
    "glass": (191, 216, 226), "glass_pane": (191, 216, 226), "tinted_glass": (44, 40, 47),
    "hay_block": (166, 138, 25), "melon": (111, 145, 30), "pumpkin": (198, 118, 24),
    "carved_pumpkin": (198, 118, 24), "jack_o_lantern": (213, 145, 42),
    "note_block": (98, 66, 45), "jukebox": (99, 66, 47), "tnt": (183, 61, 47),
    "sponge": (195, 192, 75), "slime_block": (111, 192, 91), "honey_block": (251, 179, 48),
    "target": (216, 199, 189), "beacon": (117, 216, 208), "conduit": (159, 133, 106),
    "chest": (162, 130, 78), "trapped_chest": (162, 130, 78), "barrel": (118, 90, 51),
    "ladder": (137, 111, 66), "scaffolding": (183, 149, 90), "cauldron": (73, 73, 73),
    "anvil": (73, 73, 74), "enchanting_table": (128, 63, 60),
    "brewing_stand": (124, 103, 81), "lectern": (150, 118, 66),
    "smoker": (86, 78, 71), "blast_furnace": (86, 86, 86), "loom": (131, 106, 68),
    "cartography_table": (95, 76, 54), "fletching_table": (198, 179, 125),
    "smithing_table": (55, 52, 61), "grindstone": (142, 142, 142),
    "stonecutter": (124, 122, 118), "composter": (121, 87, 47),
    "bell": (247, 202, 84), "lantern": (116, 89, 67), "soul_lantern": (86, 116, 121),
    "torch": (250, 212, 108), "wall_torch": (250, 212, 108),
    "soul_torch": (108, 202, 214), "campfire": (206, 129, 46),
    "redstone_lamp": (95, 60, 33), "iron_bars": (98, 100, 102),
    "cobweb": (229, 235, 235), "mud": (60, 56, 61), "mud_bricks": (137, 105, 78),
    "packed_mud": (142, 106, 79), "amethyst_block": (133, 97, 191),
    "moss_block": (89, 109, 45), "sculk": (12, 26, 33),
    "white_bed": (233, 236, 236), "flower_pot": (126, 66, 50),
    # Redstone / utility. Small blocks, but they show up on rooflines and walls.
    "dropper": (109, 108, 108), "dispenser": (109, 108, 108),
    "hopper": (75, 75, 78), "observer": (98, 98, 98), "piston": (150, 137, 106),
    "sticky_piston": (150, 137, 106), "piston_head": (150, 137, 106),
    "end_rod": (223, 219, 208), "chain": (55, 59, 70),
    "iron_trapdoor": (202, 202, 202), "iron_door": (202, 202, 202),
    "tripwire": (128, 128, 128), "tripwire_hook": (139, 122, 89),
    "sign": (162, 130, 78), "wall_sign": (162, 130, 78),
    "lever": (124, 106, 82), "repeater": (158, 152, 150), "comparator": (158, 152, 150),
    "redstone_wire": (175, 24, 5), "rail": (124, 118, 108),
    "powered_rail": (140, 116, 71), "detector_rail": (134, 116, 100),
    "activator_rail": (127, 108, 96), "skull": (194, 194, 194), "head": (194, 194, 194),
    "banner": (162, 130, 78), "armor_stand": (185, 176, 155),
    "daylight_detector": (137, 116, 87), "structure_block": (85, 73, 87),
    "spawner": (26, 39, 51), "end_portal_frame": (91, 118, 100),
    # Plants
    "oak_leaves": (60, 143, 40), "spruce_leaves": (49, 88, 49),
    "birch_leaves": (128, 167, 85), "jungle_leaves": (52, 145, 22),
    "acacia_leaves": (86, 145, 31), "dark_oak_leaves": (52, 130, 27),
    "mangrove_leaves": (60, 143, 40), "cherry_leaves": (232, 172, 199),
    "azalea_leaves": (91, 132, 41), "flowering_azalea_leaves": (110, 133, 62),
    "grass": (95, 159, 53), "tall_grass": (95, 159, 53), "fern": (86, 140, 51),
    "large_fern": (86, 140, 51), "vine": (63, 129, 33), "lily_pad": (32, 128, 48),
    "sugar_cane": (148, 192, 101), "cactus": (85, 127, 44), "bamboo": (128, 168, 41),
    "dandelion": (232, 224, 62), "poppy": (194, 54, 46), "wheat": (176, 165, 65),
    "sunflower": (238, 213, 66), "rose_bush": (168, 47, 46), "peony": (206, 166, 209),
    "lilac": (191, 158, 199), "dead_bush": (109, 78, 35),
    # Flowers and foliage a modern world is full of, found missing while importing a
    # Bedrock export -- without these a garden renders as grey gravel.
    "azure_bluet": (221, 227, 189), "cornflower": (70, 106, 199),
    "oxeye_daisy": (226, 231, 205), "lily_of_the_valley": (231, 236, 222),
    "allium": (169, 125, 200), "blue_orchid": (47, 174, 200),
    "red_tulip": (190, 47, 41), "orange_tulip": (196, 118, 33),
    "white_tulip": (222, 229, 222), "pink_tulip": (222, 178, 216),
    "short_grass": (95, 159, 53), "spore_blossom": (222, 116, 168),
    "torchflower": (233, 156, 62), "pitcher_plant": (155, 106, 189),
    "glow_lichen": (114, 133, 111), "flowering_azalea": (110, 133, 62),
    "azalea": (91, 132, 41), "spruce_sapling": (49, 88, 49),
    "oak_sapling": (60, 143, 40), "birch_sapling": (128, 167, 85),
    "hanging_roots": (150, 111, 88), "big_dripleaf": (109, 148, 55),
    "small_dripleaf": (97, 138, 47), "moss_carpet": (89, 109, 45),
    # Fixtures and utility blocks.
    "ender_chest": (42, 62, 62), "lightning_rod": (155, 111, 92),
    "heavy_weighted_pressure_plate": (220, 220, 220),
    "light_weighted_pressure_plate": (246, 208, 61),
    "wither_skeleton_skull": (57, 57, 57), "skeleton_skull": (194, 194, 194),
    "player_head": (140, 105, 78), "zombie_head": (82, 118, 71),
    "creeper_head": (114, 189, 96), "dragon_head": (36, 30, 40),
    "barrier": (196, 60, 60), "structure_void": (140, 90, 140),
    "item_frame": (129, 103, 68), "glow_item_frame": (150, 132, 84),
    # Liquids
    "water": (63, 118, 228), "lava": (223, 118, 24),
}

# Stone-ish families whose colour comes from a base material embedded in the block name.
# Ordered longest-hint-first so "polished_blackstone_brick_stairs" matches blackstone
# rather than brick, and each hint names the BASE_COLORS key it borrows from -- a bare
# "quartz" is not a block, but "quartz_stairs" should still look like quartz.
STONE_FAMILY_HINTS = (
    ("deepslate", "deepslate"), ("blackstone", "blackstone"),
    ("red_sandstone", "red_sandstone"), ("sandstone", "sandstone"),
    ("prismarine", "prismarine"), ("quartz", "quartz_block"), ("purpur", "purpur_block"),
    ("nether_brick", "nether_bricks"), ("stone_brick", "stone_bricks"),
    ("cobblestone", "cobblestone"), ("andesite", "andesite"), ("diorite", "diorite"),
    ("granite", "granite"), ("basalt", "basalt"), ("tuff", "tuff"),
    ("calcite", "calcite"), ("oxidized_", "oxidized_copper"), ("weathered_", "weathered_copper"),
    ("exposed_", "exposed_copper"), ("copper", "copper_block"), ("amethyst", "amethyst_block"),
    ("mud_brick", "mud_bricks"), ("end_stone", "end_stone"), ("brick", "bricks"),
    ("stone", "stone"),
)


def _scale(rgb, factor: float):
    return tuple(min(255, max(0, int(round(c * factor)))) for c in rgb)


def _dyed(name: str):
    """Match `<dye>_<material>`, e.g. "light_blue_stained_glass_pane"."""
    for suffix, factor in DYED_MATERIALS.items():
        if name.endswith(suffix):
            dye = DYES.get(name[: -len(suffix)])
            if dye is not None:
                return _scale(dye, factor)
    return None


def _wooden(name: str):
    """Match a wood type against the plank-ish and bark-ish suffix families."""
    for wood, rgb in WOODS.items():
        if not name.startswith(wood + "_"):
            continue
        if name.endswith(BARK_SUFFIXES) or "stripped" in name:
            return BARK.get(wood, rgb)
        if name.endswith(WOOD_SUFFIXES) or name == wood:
            return rgb
    if name.startswith("stripped_"):
        for wood, rgb in BARK.items():
            if name.startswith("stripped_" + wood):
                return _scale(WOODS.get(wood, rgb), 1.05)
    return None


def _stone_family(name: str):
    """Fall back to the base material named inside a compound stone block name."""
    for hint, base in STONE_FAMILY_HINTS:
        if hint not in name:
            continue
        rgb = BASE_COLORS.get(base)
        if rgb is None:
            continue
        # Polished/smooth variants read lighter, cracked/chiselled slightly darker.
        if "polished" in name or "smooth" in name:
            return _scale(rgb, 1.1)
        if "cracked" in name or "chiseled" in name:
            return _scale(rgb, 0.94)
        return rgb
    return None


@lru_cache(maxsize=4096)
def colour_rules(blockstate: str) -> tuple[int, int, int]:
    """Approximate RGB for a block state string, namespaced or not, with or without props."""
    name = str(blockstate)
    cut = name.find("[")
    if cut >= 0:
        name = name[:cut]
    if ":" in name:
        name = name.split(":", 1)[1]
    name = name.strip().lower()

    return (BASE_COLORS.get(name)
            or _dyed(name)
            or _wooden(name)
            or _stone_family(name)
            or FALLBACK)


def game_colour(blockstate: str) -> tuple[int, int, int]:
    """A full block's texture average when the palette has it, else `colour_rules`."""
    base = str(blockstate).split("[")[0]
    return BLOCK_PALETTE.get(base if ":" in base else "minecraft:" + base) or colour_rules(blockstate)


def colour_table(names, colour_of=game_colour) -> np.ndarray:
    """An (N, 3) uint8 table of colours, one row per palette name (an empty name is grey)."""
    table = np.empty((len(names), 3), dtype=np.uint8)
    for index, state in enumerate(names):
        table[index] = colour_of(state) if state else FALLBACK
    return table

