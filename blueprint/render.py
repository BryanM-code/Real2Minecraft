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


def render_rgba(shape: np.ndarray, material: np.ndarray, yaw: float, pitch: float, size: int = 512,
                labels: bool = False) -> np.ndarray | None:
    """The renderer's own RGBA frame (transparent background, not cropped) of (shape,
    material) grids; with `labels` windows are drawn black, doors red and the rest white
    (`label_name`), from the same camera, for the evaluator's window and door measures."""
    from photofirst.schem import palette_grid
    grid, names = palette_grid(shape, material)
    if labels:
        names = [label_name(n) for n in names]
    blocks = np.ascontiguousarray(np.transpose(grid, (2, 0, 1)))
    if str(BDC) not in sys.path:
        sys.path.insert(0, str(BDC))
    from schemset.review import render
    im = render(blocks, names, yaw=yaw, pitch=pitch, size=size)
    return None if im is None else np.array(im.convert("RGBA"))


def label_name(name: str) -> str:
    """A block name as the label view draws it: glass black, doors red, the rest white in
    the same shape (stairs, slabs, fences and walls keep their states)."""
    base, _, states = name.partition("[")
    states = "[" + states if states else ""
    if base == "minecraft:air":
        return name
    if base.endswith("_pane"):
        return "minecraft:black_stained_glass_pane" + states
    if "glass" in base:
        return "minecraft:black_concrete"
    if base.endswith("_door"):
        return "minecraft:red_concrete"
    for part, white in (("_stairs", "minecraft:quartz_stairs"), ("_slab", "minecraft:quartz_slab"),
                        ("_fence_gate", "minecraft:birch_fence_gate"), ("_fence", "minecraft:birch_fence"),
                        ("_wall", "minecraft:diorite_wall"), ("_trapdoor", "minecraft:birch_trapdoor")):
        if base.endswith(part):
            return white + states
    return "minecraft:white_concrete"


def render_blocks(blocks: np.ndarray, names: list, yaw: float, pitch: float, size: int = 900):
    """RGB image of an (X, Y, Z) palette grid, cropped to the build and scaled to `size`."""
    from PIL import Image
    if str(BDC) not in sys.path:
        sys.path.insert(0, str(BDC))
    from schemset.review import render
    im = render(blocks, names, yaw=yaw, pitch=pitch, size=2 * size)
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
