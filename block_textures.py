#!/usr/bin/env python3
"""16x16 block textures for the voxel preview + a packed atlas for fast rendering.

Each Minecraft block id maps to a 16x16 RGB tile.  A tile is loaded from
`textures/blocks/<name>.png` (name = id without the "minecraft:" prefix) when present,
so any CC0 / resource pack can be dropped in to override; otherwise a pixel-art tile is
generated procedurally from the block's palette color + a pattern keyed to its material
(stone speckle, plank grain, brick rows, glass pane, ...).  `build_atlas()` packs all the
tiles used into one image with per-block UV rects so the whole model renders with a single
texture.

Textures use nearest-neighbour sampling at render time to keep the crisp pixelated look.
"""

from pathlib import Path
import numpy as np

from block_palette import BLOCK_PALETTE

TILE = 16
TEX_DIR = Path(__file__).parent / "textures" / "blocks"

_rng = np.random.default_rng(7)   # deterministic textures across runs


def _shade(base, amt):
    """base RGB nudged by amt (per-channel additive), clipped to bytes."""
    return np.clip(np.asarray(base, dtype=int) + amt, 0, 255).astype(np.uint8)


def _noisy(base, spread):
    """16x16 tile of base color with per-pixel brightness noise."""
    n = _rng.integers(-spread, spread + 1, size=(TILE, TILE, 1))
    return _shade(np.broadcast_to(np.asarray(base, int), (TILE, TILE, 3)), n)


def _planks(base):
    """Horizontal planks with grain + vertical seams (oak/spruce)."""
    t = _noisy(base, 12)
    for row in (0, 5, 10, 15):                     # plank seams
        t[row, :, :] = _shade(base, -45)
    for x in (3, 8, 13):                            # vertical seams, staggered
        t[0:5, x, :] = _shade(base, -35)
        t[5:10, (x + 4) % TILE, :] = _shade(base, -35)
        t[10:16, x, :] = _shade(base, -35)
    return t


def _bricks(base):
    """Offset brick rows with mortar lines."""
    t = _noisy(base, 8)
    mortar = _shade(base, -55)
    for row in range(0, TILE, 4):
        t[row, :, :] = mortar
    for row in range(0, TILE, 8):                   # vertical mortar, offset per band
        t[row:row + 4, 0::8, :] = mortar
        t[row + 4:row + 8, 4::8, :] = mortar
    return t


def _glass(base):
    """Light translucent-looking pane: pale fill, brighter border."""
    t = _noisy(_shade(base, 10), 4)
    border = _shade(base, 35)
    t[0, :, :] = t[-1, :, :] = border
    t[:, 0, :] = t[:, -1, :] = border
    return t


def _cobble(base):
    """Chunky stone speckle for cobble/stone-bricks/andesite."""
    t = _noisy(base, 22)
    # a few darker grout lines
    for (r, c) in [(5, 0), (5, 8), (10, 4), (10, 12), (0, 6), (11, 0)]:
        t[r, c:c + 4, :] = _shade(base, -40)
    return t


def _grass(base):
    """Green with speckle, a touch brighter on top."""
    t = _noisy(base, 16)
    t[0:3, :, :] = _shade(base, 18)
    return t


def _generate_tile(block_id: str):
    """Procedural 16x16 tile for a block id, keyed to material by name."""
    base = BLOCK_PALETTE.get(block_id, (128, 128, 128))
    name = block_id.split(":", 1)[-1]
    if "glass" in name:
        return _glass(base)
    if "planks" in name or name.endswith("_log"):
        return _planks(base)
    if "brick" in name:
        return _bricks(base)
    if "grass" in name or "moss" in name:
        return _grass(base)
    if any(k in name for k in ("cobble", "stone_brick", "andesite", "diorite",
                               "granite", "gravel", "deepslate", "blackstone",
                               "tuff", "basalt")):
        return _cobble(base)
    if any(k in name for k in ("concrete", "wool", "terracotta", "quartz",
                               "calcite", "bone", "gold", "obsidian", "coal",
                               "copper", "iron", "lapis", "amethyst", "purpur",
                               "snow", "ice", "clay", "prismarine")):
        return _noisy(base, 6)          # smooth materials: subtle grain
    return _noisy(base, 14)             # stone / sandstone / dirt / mud: medium grain


def get_tile(block_id: str) -> np.ndarray:
    """16x16x3 uint8 tile: from textures/blocks/<name>.png if present, else generated.

    A partial-block id (`oak_stairs[facing=east,half=bottom]`) is normalized to its base
    palette id first, so slabs/stairs render as their parent material's tile.
    """
    from block_shapes import base_block
    block_id = base_block(block_id)
    png = TEX_DIR / (block_id.split(":", 1)[-1] + ".png")
    if png.exists():
        try:
            from PIL import Image
            img = Image.open(png).convert("RGB").resize((TILE, TILE), Image.NEAREST)
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            pass
    return _generate_tile(block_id)


def build_atlas(block_ids):
    """Pack the unique block ids' tiles into one atlas image.

    Returns (atlas_rgb (uint8 HxWx3), uv_map {id: (u0, v0, u1, v1)}).  UVs are inset by
    half a texel so nearest-neighbour sampling never bleeds across tile borders.  v is
    given in VTK convention (origin bottom-left).
    """
    ids = list(dict.fromkeys(block_ids))           # unique, order-preserving
    cols = int(np.ceil(np.sqrt(len(ids)))) or 1
    rows = int(np.ceil(len(ids) / cols))
    atlas = np.zeros((rows * TILE, cols * TILE, 3), dtype=np.uint8)
    uv_map = {}
    half = 0.5 / TILE
    for i, bid in enumerate(ids):
        r, c = divmod(i, cols)
        atlas[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = get_tile(bid)
        u0 = c / cols + half / cols
        u1 = (c + 1) / cols - half / cols
        # Flip vertically for VTK's bottom-left texture origin.
        v1 = 1.0 - (r / rows + half / rows)
        v0 = 1.0 - ((r + 1) / rows - half / rows)
        uv_map[bid] = (u0, v0, u1, v1)
    return atlas, uv_map
