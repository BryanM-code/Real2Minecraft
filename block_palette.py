#!/usr/bin/env python3
"""Minecraft block palette + perceptual (CIELAB) nearest-color matching.

Maps a sampled voxel RGB color to the closest real Minecraft block so exports use
multiple materials (stone body, sandstone walls, wool accents, ...) instead of
monochrome stone.

`BLOCK_PALETTE` holds id -> representative sRGB color (approximate average of the
block's 1.20 texture).  Only blocks with a visually UNIFORM texture are included —
noisy/patterned blocks (glistering melon, bookshelves, logs' bark rings) read as
clutter in a wall.  Plain glass is deliberately absent: its average color is nearly
white, so it used to steal white-wall voxels while real windows (dark in photos)
never matched it.

`match_blocks()` converts palette and query colors to CIELAB and nearest-neighbours
in that space, with the L (lightness) axis scaled by `L_WEIGHT` so HUE outranks
lightness: a dark green should become a dark-green block, not blackstone, and a dark
blue-gray window should stay bluish.  Add or tune entries freely; the KD-tree is
rebuilt lazily from the dict.
"""

import numpy as np

# Lightness weight for matching: < 1 makes hue/saturation dominate lightness, keeping
# dark-but-hued colors (foliage shadow, window glass) in the right hue family.
L_WEIGHT = 0.65

# id -> (r, g, b).  Curated ~113-block set spanning the common build materials.
BLOCK_PALETTE: dict[str, tuple[int, int, int]] = {
    # greys / stones
    "minecraft:stone":            (126, 126, 126),
    "minecraft:cobblestone":      (122, 122, 122),
    "minecraft:stone_bricks":     (122, 121, 122),
    "minecraft:andesite":         (136, 136, 137),
    "minecraft:polished_andesite": (132, 134, 133),
    "minecraft:smooth_stone":     (158, 158, 158),
    "minecraft:gravel":           (131, 127, 126),
    "minecraft:tuff":             (108, 109, 102),
    "minecraft:basalt":           (80, 81, 86),
    "minecraft:smooth_basalt":    (72, 72, 78),
    "minecraft:deepslate":        (77, 77, 80),
    "minecraft:polished_deepslate": (72, 72, 73),
    "minecraft:polished_blackstone": (53, 48, 56),
    # whites / light
    "minecraft:diorite":          (188, 188, 189),
    "minecraft:polished_diorite": (192, 193, 194),
    "minecraft:calcite":          (224, 225, 221),
    "minecraft:quartz_block":     (235, 229, 222),
    "minecraft:bone_block":       (229, 224, 196),
    "minecraft:snow_block":       (249, 254, 254),
    "minecraft:iron_block":       (220, 220, 220),
    "minecraft:clay":             (160, 166, 179),
    # tans / sandstones / browns (castle stone lives here)
    "minecraft:sandstone":        (216, 203, 155),
    "minecraft:smooth_sandstone": (224, 213, 160),
    "minecraft:end_stone":        (219, 222, 158),
    "minecraft:red_sandstone":    (186, 99, 29),
    "minecraft:packed_mud":       (143, 116, 90),
    "minecraft:mud_bricks":       (137, 103, 78),
    "minecraft:mud":              (60, 57, 60),
    "minecraft:dirt":             (134, 96, 67),
    "minecraft:coarse_dirt":      (119, 85, 59),
    "minecraft:dripstone_block":  (134, 107, 92),
    "minecraft:granite":          (149, 103, 85),
    "minecraft:polished_granite": (154, 106, 89),
    "minecraft:raw_iron_block":   (166, 135, 107),
    # wood ramp (planks: light -> dark, plus the exotic hues)
    "minecraft:birch_planks":     (192, 175, 121),
    "minecraft:oak_planks":       (162, 131, 79),
    "minecraft:jungle_planks":    (160, 115, 80),
    "minecraft:acacia_planks":    (168, 90, 50),
    "minecraft:spruce_planks":    (114, 84, 48),
    "minecraft:dark_oak_planks":  (66, 43, 20),
    "minecraft:mangrove_planks":  (117, 54, 48),
    "minecraft:cherry_planks":    (226, 178, 172),
    "minecraft:crimson_planks":   (101, 48, 70),
    "minecraft:warped_planks":    (43, 104, 99),
    # copper oxidation ramp
    "minecraft:copper_block":     (192, 107, 79),
    "minecraft:exposed_copper":   (161, 125, 103),
    "minecraft:weathered_copper": (108, 153, 110),
    "minecraft:oxidized_copper":  (82, 162, 132),
    # reds
    "minecraft:bricks":           (150, 97, 83),
    "minecraft:nether_bricks":    (44, 21, 26),
    # greens (grass / foliage)
    "minecraft:grass_block":      (95, 159, 53),
    "minecraft:moss_block":       (89, 109, 45),
    # blues / teals
    "minecraft:prismarine":       (99, 156, 151),
    "minecraft:dark_prismarine":  (51, 91, 75),
    "minecraft:packed_ice":       (141, 180, 222),
    "minecraft:lapis_block":      (30, 67, 140),
    # saturated accents
    "minecraft:gold_block":       (246, 208, 62),
    "minecraft:amethyst_block":   (133, 97, 191),
    "minecraft:purpur_block":     (169, 125, 169),
    # darks — deepslate/blackstone above cover "dark stone"; these are near-black.
    "minecraft:blackstone":       (42, 38, 44),
    "minecraft:obsidian":         (15, 10, 24),
    # concrete (all 16) — flat modern colors
    "minecraft:white_concrete":   (207, 213, 214),
    "minecraft:orange_concrete":  (224, 97, 1),
    "minecraft:magenta_concrete": (169, 48, 159),
    "minecraft:light_blue_concrete": (36, 137, 199),
    "minecraft:yellow_concrete":  (240, 175, 21),
    "minecraft:lime_concrete":    (94, 169, 24),
    "minecraft:pink_concrete":    (214, 101, 143),
    "minecraft:gray_concrete":    (54, 57, 61),
    "minecraft:light_gray_concrete": (125, 125, 115),
    "minecraft:cyan_concrete":    (21, 119, 136),
    "minecraft:purple_concrete":  (100, 32, 156),
    "minecraft:blue_concrete":    (45, 47, 143),
    "minecraft:brown_concrete":   (96, 60, 32),
    "minecraft:green_concrete":   (73, 91, 36),
    "minecraft:red_concrete":     (142, 33, 33),
    "minecraft:black_concrete":   (8, 10, 15),
    # terracotta (all 16 + plain) — muted earth versions of every hue
    "minecraft:terracotta":       (152, 94, 67),
    "minecraft:white_terracotta": (209, 178, 161),
    "minecraft:orange_terracotta": (161, 83, 37),
    "minecraft:magenta_terracotta": (149, 88, 108),
    "minecraft:light_blue_terracotta": (113, 108, 137),
    "minecraft:yellow_terracotta": (186, 133, 35),
    "minecraft:lime_terracotta":  (103, 117, 52),
    "minecraft:pink_terracotta":  (161, 78, 78),
    "minecraft:gray_terracotta":  (57, 42, 35),
    "minecraft:light_gray_terracotta": (135, 106, 97),
    "minecraft:cyan_terracotta":  (86, 91, 91),
    "minecraft:purple_terracotta": (118, 69, 86),
    "minecraft:blue_terracotta":  (74, 59, 91),
    "minecraft:brown_terracotta": (77, 51, 35),
    "minecraft:green_terracotta": (76, 83, 42),
    "minecraft:red_terracotta":   (143, 61, 46),
    "minecraft:black_terracotta": (37, 22, 16),
    # wool (all 16) — saturated soft colors
    "minecraft:white_wool":       (233, 236, 236),
    "minecraft:orange_wool":      (240, 118, 19),
    "minecraft:magenta_wool":     (189, 68, 179),
    "minecraft:light_blue_wool":  (58, 175, 217),
    "minecraft:yellow_wool":      (248, 197, 39),
    "minecraft:lime_wool":        (112, 185, 25),
    "minecraft:pink_wool":        (237, 141, 172),
    "minecraft:gray_wool":        (62, 68, 71),
    "minecraft:light_gray_wool":  (142, 142, 134),
    "minecraft:cyan_wool":        (21, 137, 145),
    "minecraft:purple_wool":      (121, 42, 172),
    "minecraft:blue_wool":        (53, 57, 157),
    "minecraft:brown_wool":       (114, 71, 40),
    "minecraft:green_wool":       (84, 109, 27),
    "minecraft:red_wool":         (160, 39, 34),
    "minecraft:black_wool":       (20, 21, 25),
}


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (0-255, shape (...,3)) to CIELAB under the D65 white point."""
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])  # normalize to D65 white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of `_srgb_to_lab`: CIELAB (D65) -> sRGB 0-255, out-of-gamut clipped."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    eps, kappa = 216 / 24389, 24389 / 27
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def _finv(f):
        f3 = f ** 3
        return np.where(f3 > eps, f3, (116 * f - 16) / kappa)

    xyz = np.stack([_finv(fx),
                    np.where(L > kappa * eps, fy ** 3, L / kappa),
                    _finv(fz)], axis=-1)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    m_inv = np.array([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [0.0557, -0.2040, 1.0570]])
    lin = np.clip(xyz @ m_inv.T, 0.0, None)
    srgb = np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin)
    return np.clip(srgb * 255.0, 0, 255)


def _match_space(rgb: np.ndarray) -> np.ndarray:
    """LAB with L scaled by L_WEIGHT — the space nearest-neighbour matching runs in."""
    lab = _srgb_to_lab(rgb)
    lab[..., 0] *= L_WEIGHT
    return lab


# Weighted-LAB KD-tree over the full palette, built once and cached.
_TREE: tuple | None = None


def _palette_tree():
    """Return (ids, cKDTree in match space) for the full palette, cached on first use."""
    global _TREE
    if _TREE is None:
        from scipy.spatial import cKDTree
        ids = list(BLOCK_PALETTE)
        lab = _match_space(np.array([BLOCK_PALETTE[k] for k in ids], dtype=np.float64))
        _TREE = (ids, cKDTree(lab))
    return _TREE


def match_blocks(colors_rgb) -> list[str]:
    """Map (r,g,b) colors to the nearest Minecraft block id (weighted CIELAB).

    Each color is matched independently against the whole BLOCK_PALETTE — no per-object
    restriction — so block choice is driven purely by the sampled color.
    """
    ids, tree = _palette_tree()
    q = _match_space(np.asarray(colors_rgb, dtype=np.float64).reshape(-1, 3))
    _, idx = tree.query(q)
    return [ids[i] for i in np.atleast_1d(idx)]


# Weighted-LAB KD-trees over palette SUBSETS (e.g. slab/stairs-capable blocks), cached
# by the frozenset of allowed ids.
_RESTRICTED_TREES: dict = {}


def match_blocks_restricted(colors_rgb, allowed_ids):
    """Nearest palette match among `allowed_ids` only; returns (ids, distances).

    Same weighted-LAB space as match_blocks.  Used by block_shapes to swap a voxel whose
    matched material has no slab/stairs variant to the closest one that does, gated on the
    returned distance.  The (ids, cKDTree) is cached per allowed-id set.
    """
    key = frozenset(allowed_ids)
    cached = _RESTRICTED_TREES.get(key)
    if cached is None:
        from scipy.spatial import cKDTree
        ids = [k for k in allowed_ids if k in BLOCK_PALETTE]
        lab = _match_space(np.array([BLOCK_PALETTE[k] for k in ids], dtype=np.float64))
        cached = (ids, cKDTree(lab))
        _RESTRICTED_TREES[key] = cached
    ids, tree = cached
    q = _match_space(np.asarray(colors_rgb, dtype=np.float64).reshape(-1, 3))
    dist, idx = tree.query(q)
    idx = np.atleast_1d(idx)
    return [ids[i] for i in idx], np.atleast_1d(dist)
