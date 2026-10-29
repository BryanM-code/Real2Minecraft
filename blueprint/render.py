"""Preview renders of a built blueprint with BuildDataCollector's orbit renderer (the same
one the review gallery and the looks score use). Runs in BuildDataCollector's environment
(..\\BuildDataCollector\\.venv), with this repository's root as the working directory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BDC = Path(__file__).resolve().parents[2] / "BuildDataCollector"
# (yaw, pitch): yaw 0 looks at the front (south) side, negative yaw from the front right
VIEWS = ((-35, 28), (35, 28), (145, 28), (215, 28))
BACKGROUND = (246, 246, 244)


def render_views(shape: np.ndarray, material: np.ndarray, out_dir, views=VIEWS, size: int = 900,
                 prefix: str = "view") -> list:
    """Save one PNG per (yaw, pitch) in `out_dir`; returns their paths. Stair corners, door
    halves and fence / pane arms are derived first, as the schematic writer does."""
    from photofirst.schem import palette_grid
    grid, names = palette_grid(shape, material)                      # (Y, Z, X)
    blocks = np.ascontiguousarray(np.transpose(grid, (2, 0, 1)))     # (X, Y, Z)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for yaw, pitch in views:
        im = render_blocks(blocks, names, yaw, pitch, size)
        if im is None:
            continue
        path = out_dir / f"{prefix}_{yaw:+d}_{pitch}.png"
        im.save(path)
        paths.append(path)
    return paths


def render_rgba(shape: np.ndarray, material: np.ndarray, yaw: float, pitch: float, size: int = 512) -> np.ndarray | None:
    """The renderer's own RGBA frame (transparent background, not cropped) of (shape,
    material) grids."""
    from photofirst.schem import palette_grid
    grid, names = palette_grid(shape, material)
    blocks = np.ascontiguousarray(np.transpose(grid, (2, 0, 1)))
    im = _render(blocks, names, yaw=yaw, pitch=pitch, size=size)
    return None if im is None else np.array(im.convert("RGBA"))


def _render(blocks: np.ndarray, names: list, **kw):
    """BuildDataCollector's render of an (X, Y, Z) palette grid, in the game's colours: its own
    table tints the dyed families from the dye (white terracotta comes out grey, plain terracotta
    its fallback grey, light blue terracotta bright blue where the game's is a greyish violet),
    so the blocks `mcvoxel`'s palette has (its textures' averages) take that palette's colour."""
    if str(BDC) not in sys.path:
        sys.path.insert(0, str(BDC))
    import block_colors
    from schemset.review import render

    from mcvoxel.block_palette import BLOCK_PALETTE
    original = block_colors.color_for

    def game(state: str):
        base = str(state).split("[")[0]
        return BLOCK_PALETTE.get(base if ":" in base else "minecraft:" + base) or original(state)
    block_colors.color_for = game
    try:
        return render(blocks, names, **kw)
    finally:
        block_colors.color_for = original


# colours of a code view: hues 45 degrees or more apart, so the renderer's face shading and edge
# blending (which keep the hue) still tell them apart; code 0 is drawn white
CODE_HUES = (0, 45, 90, 135, 180, 225, 270, 315)


def _hue_rgb(h: float) -> tuple:
    import colorsys
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 360.0, 1.0, 1.0))


def render_codes(shape: np.ndarray, material: np.ndarray, codes: np.ndarray, colour_of: dict, yaw: float,
                 pitch: float, size: int = 512) -> np.ndarray | None:
    """(H, W) ints: for each pixel of the render (same camera and frame as `render_rgba`),
    the code (`codes`, an (X, Y, Z) grid of small ints) of the block drawn there, -1 for
    none. `colour_of` gives each code a slot: None for white, 0-7 for a hue of CODE_HUES.
    Blocks keep their shapes; only their colours change."""
    from photofirst.schem import palette_grid
    if str(BDC) not in sys.path:
        sys.path.insert(0, str(BDC))
    import block_colors
    from schemset.review import render
    grid, names = palette_grid(shape, material)                      # (Y, Z, X)
    c = np.transpose(codes, (1, 2, 0)).astype(np.int64)
    combined = grid.astype(np.int64) * 256 + c
    uniq, inv = np.unique(combined, return_inverse=True)
    new_names, rgb = [], {}
    for u in uniq:
        name, code = names[int(u // 256)], int(u % 256)
        if name in ("minecraft:air", "") or u // 256 == 0:
            new_names.append("minecraft:air")
            continue
        base = name.split(":", 1)[-1]
        label = f"minecraft:r2mcode{code}_{base}"
        slot = colour_of.get(code)
        rgb[label.split("[")[0].split(":", 1)[1]] = (255, 255, 255) if slot is None else _hue_rgb(CODE_HUES[slot])
        new_names.append(label)
    new_grid = inv.reshape(grid.shape).astype(np.int32)
    new_grid[grid == 0] = 0
    if new_names[0] != "minecraft:air":                              # palette index 0 must be air
        new_names = ["minecraft:air"] + new_names
        new_grid = np.where(grid == 0, 0, new_grid + 1)
    original = block_colors.color_for

    def coloured(state: str):
        base = str(state).split("[")[0].split(":", 1)[-1]
        return rgb.get(base) or original(state)
    block_colors.color_for = coloured
    try:
        im = render(np.ascontiguousarray(np.transpose(new_grid, (2, 0, 1))), new_names, yaw=yaw, pitch=pitch, size=size)
    finally:
        block_colors.color_for = original
    if im is None:
        return None
    a = np.array(im.convert("RGBA")).astype(float)
    out = np.full(a.shape[:2], -1, np.int32)
    v = a[..., :3].max(-1)
    lit = (a[..., 3] > 127) & (v > 25)
    r, g, b = (a[..., k] / np.maximum(v, 1) for k in range(3))     # brightness divided out: hue and saturation
    mn = np.minimum(np.minimum(r, g), b)
    d = np.maximum(1.0 - mn, 1e-6)
    hue = np.where(r >= 1.0, ((g - b) / d) % 6, np.where(g >= 1.0, (b - r) / d + 2, (r - g) / d + 4)) * 60
    chroma = lit & (1.0 - mn > 0.35)
    out[lit & ~chroma] = next((code for code, s in colour_of.items() if s is None), -1)
    slot = np.round(hue / 45.0).astype(int) % 8
    for code, s in colour_of.items():
        if s is not None:
            out[chroma & (slot == s)] = code
    return out


def render_blocks(blocks: np.ndarray, names: list, yaw: float, pitch: float, size: int = 900):
    """RGB image of an (X, Y, Z) palette grid, cropped to the build and scaled to `size`."""
    im = _render(blocks, names, yaw=yaw, pitch=pitch, size=2 * size)
    if im is None:
        return None
    return finish(im, size)


def finish(im, size: int):
    """On the background, cropped to what is drawn (plus a margin), scaled to fit `size`."""
    from PIL import Image
    im = im.convert("RGBA")
    box = im.getchannel("A").getbbox()
    if box:
        pad = int(0.04 * max(box[2] - box[0], box[3] - box[1]))
        im = im.crop((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
    bg = Image.new("RGBA", im.size, BACKGROUND + (255,))
    bg.alpha_composite(im)
    scale = size / max(bg.size)
    return bg.convert("RGB").resize((max(1, round(bg.width * scale)), max(1, round(bg.height * scale))),
                                    Image.LANCZOS)
