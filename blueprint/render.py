"""Pictures of a built blueprint: preview renders, and the views the checker compares with
the photo.

Blocks are drawn as shaded cubes (stairs, slabs, fences, panes and doors in their real
shapes) by an orthographic camera at any yaw and pitch, in the game's colours
(`minecraft.colours.game_colour`). Yaw 0 looks at the front (south) side; a negative yaw
looks from the front right.

    render_views   preview PNGs from four sides
    render_rgba    one view as an RGBA array (transparent background)
    render_codes   which part or label drew each pixel of a view (same camera as render_rgba)
    finish         a view on the background, cropped to the build
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from .minecraft.colours import colour_rules, colour_table, game_colour
from .minecraft.schem import palette_grid

# (yaw, pitch): yaw 0 looks at the front (south) side, negative yaw from the front right
VIEWS = ((-35, 28), (35, 28), (145, 28), (215, 28))
BACKGROUND = (246, 246, 244)
MAX_SURFACE = 60_000              # visible blocks drawn at most; bigger builds are drawn coarser


def render_views(shape: np.ndarray, material: np.ndarray, out_dir, views=VIEWS, size: int = 900,
                 prefix: str = "view") -> list:
    """Save one PNG per (yaw, pitch) in `out_dir`; returns their paths. Stair corners, door
    halves and fence / pane arms are worked out first, as the schematic writer does."""
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
    """One view of (shape, material) grids as an RGBA array (transparent background, not
    cropped)."""
    grid, names = palette_grid(shape, material)
    blocks = np.ascontiguousarray(np.transpose(grid, (2, 0, 1)))
    im = draw(blocks, names, game_colour, yaw=yaw, pitch=pitch, size=size)
    return None if im is None else np.array(im.convert("RGBA"))


# colours of a code view: hues 45 degrees or more apart, so the face shading and edge
# blending (which keep the hue) still tell them apart; code 0 is drawn white
CODE_HUES = (0, 45, 90, 135, 180, 225, 270, 315)


def _hue_rgb(h: float) -> tuple:
    import colorsys
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 360.0, 1.0, 1.0))


def render_codes(shape: np.ndarray, material: np.ndarray, codes: np.ndarray, colour_of: dict, yaw: float,
                 pitch: float, size: int = 512) -> np.ndarray | None:
    """(H, W) ints: for each pixel of the view (same camera and frame as `render_rgba`),
    the code (`codes`, an (X, Y, Z) grid of small ints) of the block drawn there, -1 for
    none. `colour_of` gives each code a slot: None for white, 0-7 for a hue of CODE_HUES.
    Blocks keep their shapes; only their colours change."""
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

    def coloured(state: str):
        base = str(state).split("[")[0].split(":", 1)[-1]
        return rgb.get(base) or colour_rules(state)
    im = draw(np.ascontiguousarray(np.transpose(new_grid, (2, 0, 1))), new_names, coloured,
              yaw=yaw, pitch=pitch, size=size)
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
    im = draw(blocks, names, game_colour, yaw=yaw, pitch=pitch, size=2 * size)
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


def draw(blocks: np.ndarray, names: list, colour_of, yaw: float = 45, pitch: float = 30,
         cut_above: int | None = None, size: int = 320):
    """PIL RGBA image of an (X, Y, Z) palette grid (`names[i]` is palette entry i's block
    state, 0 is air), each block coloured by `colour_of(state)`; None if there is nothing to
    draw."""
    if not np.any(blocks):
        return None
    grid = np.ascontiguousarray(np.transpose(blocks, (1, 2, 0)))     # (Y, Z, X)
    try:
        return _render_orbit(grid, list(names), colour_of, yaw, pitch, cut_above, size, MAX_SURFACE)
    except Exception:                                                 # noqa: BLE001
        return None


# ── The orthographic block renderer ───────────────────────────────────────────
# Under an orthographic camera every block projects to the same shape wherever it is, so one
# stencil is drawn per block shape per frame and then copied once per visible block, farthest
# first. Pictures are drawn at SUPERSAMPLE x and scaled down, which is the antialiasing.

SUPERSAMPLE = 2
MIN_CELL_PX = 2                 # smallest cube edge, in supersampled pixels
VIEW_FRAME_MARGIN = 1.03        # room round the build at its widest angle
# Face shading: tops lit fully; sides by a fixed sun in the build's own coordinates, so a
# face keeps its brightness however the camera turns.
SHADE_TOP = 1.0
LIGHT_DIR = (-0.92, 0.0, -0.39)
SHADE_SIDE_BASE, SHADE_SIDE_RANGE = 0.635, 0.092
AIR_NAMES = frozenset({"minecraft:air", "minecraft:cave_air", "minecraft:void_air"})


def _base_name(blockstate: str) -> str:
    """`"minecraft:oak_stairs[facing=east]"` -> `"oak_stairs"`."""
    name = str(blockstate or "")
    cut = name.find("[")
    if cut >= 0:
        name = name[:cut]
    cut = name.find(":")
    if cut >= 0:
        name = name[cut + 1:]
    return name.strip()


def _solid(grid, names):
    """Boolean grid: True wherever the block is not air."""
    air = np.array([str(n).split("[", 1)[0] in AIR_NAMES for n in names], dtype=bool)
    return ~air[grid]


def _downsample(grid, solid, factor: int):
    """Shrink the voxel grid by `factor`, keeping the first solid block in each cell.

    Averaging block *ids* would be meaningless, so this picks a representative instead:
    walk the factor^3 sub-positions and take the first solid one found. factor is 2-4 in
    practice, so this is a few dozen vectorised passes, not a Python loop over voxels.
    """
    h, l, w = grid.shape
    nh, nl, nw = h // factor, l // factor, w // factor
    if nh < 1 or nl < 1 or nw < 1:
        return grid, solid

    out = np.zeros((nh, nl, nw), dtype=grid.dtype)
    out_solid = np.zeros((nh, nl, nw), dtype=bool)
    for dy in range(factor):
        for dz in range(factor):
            for dx in range(factor):
                sub = grid[dy:dy + nh * factor:factor,
                           dz:dz + nl * factor:factor,
                           dx:dx + nw * factor:factor]
                sub_solid = solid[dy:dy + nh * factor:factor,
                                  dz:dz + nl * factor:factor,
                                  dx:dx + nw * factor:factor]
                take = sub_solid & ~out_solid
                out[take] = sub[take]
                out_solid |= sub_solid
    return out, out_solid


def _surface(solid):
    """Voxels that are solid and touch empty space -- the only ones that can be seen.

    A hollowed-out build renders identically to a filled one, so for a typical house this
    throws away most of the work before any drawing happens.
    """
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    enclosed = (padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1] &
                padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1] &
                padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:])
    return solid & ~enclosed


def _fit_to_budget(base_grid, base_solid, max_surface: int, fit_cell):
    """Downsample until the cubes are big enough to see and few enough to draw.

    The budget is measured against the *surface* count, since that is what the draw loop
    actually iterates -- a solidly-filled build has orders of magnitude more voxels than
    visible ones, and gating on the total would render it needlessly blocky. Each attempt
    downsamples the original rather than the previous result, so the output depends only on
    the final factor and rounding never compounds.

    Returns (grid, solid, surface, factor), or None when there is nothing left to draw.
    """
    if not base_solid.any():
        return None
    grid, solid, factor = base_grid, base_solid, 1
    while True:
        h, l, w = grid.shape
        visible = _surface(solid)
        if factor >= 16 or (fit_cell(w, h, l) >= MIN_CELL_PX
                            and int(visible.sum()) <= max_surface):
            return grid, solid, visible, factor
        factor += 1
        grid, solid = _downsample(base_grid, base_solid, factor)
        if not solid.any():
            return None


def _view_axes(yaw_deg: float, pitch_deg: float):
    """Camera basis: build coords -> (screen right, screen down, towards the viewer).

    Yaw spins the build about its vertical axis; pitch raises the camera above the horizon.
    Each returned triple dots with a build coordinate to give one component, so the whole
    projection is three dot products and no matrices.
    """
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return ((cy, 0.0, sy),                      # screen x
            (-sy * sp, -cp, cy * sp),           # screen y, growing downwards
            (-sy * cp, sp, cy * cp))            # depth, growing towards the viewer


def _fit_orbit_cell(w: int, h: int, l: int, pitch_deg: float, canvas: int) -> int:
    """Cube edge at which the build fits from *any* yaw, so spinning never resizes it.

    Fitted against the diagonal of the footprint -- the widest it ever projects, at 45
    degrees to it -- rather than against the current angle, so views of the same build from
    different sides are drawn at the same scale.

    The margin is not decoration: at the diagonal the build fills the frame to the pixel, and
    per-voxel rounding plus the stencil's grown edge would put that last pixel outside it.
    """
    pitch = math.radians(pitch_deg)
    spread = math.hypot(w, l) * VIEW_FRAME_MARGIN
    down = (h * abs(math.cos(pitch)) + spread * abs(math.sin(pitch))) * VIEW_FRAME_MARGIN
    return int(min(canvas / max(spread, 1), canvas / max(down, 1)))


def _face_shade(normal) -> float:
    """How lit one face of a cube is, under a sun fixed in the build's own coordinates."""
    if normal == (0, 1, 0):
        return SHADE_TOP
    if normal == (0, -1, 0):
        return SHADE_SIDE_BASE - 2 * SHADE_SIDE_RANGE      # the underside, seen from below
    lit = sum(a * b for a, b in zip(normal, LIGHT_DIR))
    return SHADE_SIDE_BASE + SHADE_SIDE_RANGE * lit


_CUBE_BOX = ((0.0, 0.0, 0.0, 1.0, 1.0, 1.0),)


_TILING_SUFFIXES = ("_slab", "_stairs", "_door", "_trapdoor", "_carpet")


def _tiles(base: str) -> bool:
    return base.endswith(_TILING_SUFFIXES) or _shape_boxes(base, {}) == _CUBE_BOX


def _shape_boxes(base: str, state: dict):
    """The boxes one blockstate fills, in unit-cube coordinates. () means "draw nothing"."""
    if base.endswith("_slab"):
        kind = state.get("type", "bottom")
        if kind == "top":
            return ((0.0, 0.5, 0.0, 1.0, 1.0, 1.0),)
        if kind == "double":
            return _CUBE_BOX
        return ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0),)

    if base.endswith("_stairs"):
        # `facing` names the full-height half: on real roofs most stairs face uphill, the
        # side the tall part has to be on for the steps to descend outwards.
        facing = state.get("facing", "north")
        half = state.get("half", "bottom")
        step = ((0.0, 0.0, 0.0, 1.0, 0.5, 1.0) if half == "bottom"
                else (0.0, 0.5, 0.0, 1.0, 1.0, 1.0))
        y0, y1 = (0.5, 1.0) if half == "bottom" else (0.0, 0.5)
        tall = {"north": (0.0, y0, 0.0, 1.0, y1, 0.5),
                "south": (0.0, y0, 0.5, 1.0, y1, 1.0),
                "west": (0.0, y0, 0.0, 0.5, y1, 1.0),
                "east": (0.5, y0, 0.0, 1.0, y1, 1.0)}.get(facing)
        corner = state.get("shape", "straight")
        if corner != "straight" and facing in ("east", "west", "north", "south"):
            ccw = {"north": "west", "west": "south", "south": "east", "east": "north"}
            side = ccw[facing] if corner.endswith("left") else ccw[ccw[ccw[facing]]]
            boxes = [step]
            for ix in (0, 1):
                for iz in (0, 1):
                    high = {"east": ix == 1, "west": ix == 0, "north": iz == 0, "south": iz == 1}
                    occupied = (high[facing] and high[side]) if corner.startswith("outer") else (high[facing] or high[side])
                    if occupied:
                        boxes.append((ix*.5,y0,iz*.5,(ix+1)*.5,y1,(iz+1)*.5))
            return tuple(boxes)
        return (step, tall) if tall else (step,)

    if base.endswith("_trapdoor"):
        facing, half = state.get("facing", "north"), state.get("half", "bottom")
        if state.get("open") == "true":
            t = 0.19
            return {"north": (0.0, 0.0, 0.0, 1.0, 1.0, t),
                    "south": (0.0, 0.0, 1.0 - t, 1.0, 1.0, 1.0),
                    "west": (0.0, 0.0, 0.0, t, 1.0, 1.0),
                    "east": (1.0 - t, 0.0, 0.0, 1.0, 1.0, 1.0)}.get(
                        facing, (0.0, 0.0, 0.0, 1.0, 1.0, t)),
        return (((0.0, 0.0, 0.0, 1.0, 0.19, 1.0),) if half == "bottom"
                else ((0.0, 0.81, 0.0, 1.0, 1.0, 1.0),))

    if base.endswith("_pane") or base.endswith("_bars"):
        return ((0.44, 0.0, 0.0, 0.56, 1.0, 1.0), (0.0, 0.0, 0.44, 1.0, 1.0, 0.56))
    if base.endswith("_fence"):
        boxes = [(0.375, 0.0, 0.375, 0.625, 1.0, 0.625)]
        for direction in ("east", "west", "north", "south"):
            if state.get(direction) != "true":
                continue
            for y0, y1 in ((.375, .5625), (.75, .9375)):
                boxes.append({"east": (.5, y0, .4375, 1., y1, .5625),
                              "west": (0., y0, .4375, .5, y1, .5625),
                              "north": (.4375, y0, 0., .5625, y1, .5),
                              "south": (.4375, y0, .5, .5625, y1, 1.)}[direction])
        return tuple(boxes)
    if base.endswith("_wall"):
        boxes = [(0.25, 0., 0.25, .75, 1., .75)] if state.get("up", "true") == "true" else []
        for direction in ("east", "west", "north", "south"):
            kind = state.get(direction, "none")
            if kind == "none":
                continue
            height = 1. if kind == "tall" else .875
            boxes.append({"east": (.5, 0., .3125, 1., height, .6875),
                          "west": (0., 0., .3125, .5, height, .6875),
                          "north": (.3125, 0., 0., .6875, height, .5),
                          "south": (.3125, 0., .5, .6875, height, 1.)}[direction])
        return tuple(boxes)
    if base.endswith("_fence_gate"):
        return ((0.0, 0.3, 0.44, 1.0, 1.0, 0.56),)
    if base.endswith("_door"):
        return ((0.0, 0.0, 0.0, 1.0, 1.0, 0.19),)
    if base.endswith("_carpet"):
        return ((0.0, 0.0, 0.0, 1.0, 0.0625, 1.0),)
    if base.endswith(("_pressure_plate",)):
        return ((0.06, 0.0, 0.06, 0.94, 0.0625, 0.94),)
    if base.endswith(("_button", "_torch", "_lever")):
        return ((0.35, 0.0, 0.35, 0.65, 0.6, 0.65),)
    return _CUBE_BOX


def _box_faces(box):
    """(normal, 4 corners) for each face of an axis-aligned box."""
    x0, y0, z0, x1, y1, z1 = box
    return (
        ((0, 1, 0), ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))),
        ((0, -1, 0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
        ((1, 0, 0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1, 0, 0), ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))),
        ((0, 0, 1), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0, 0, -1), ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))),
    )


def _orbit_sprite(cell: int, right, down, near, boxes=_CUBE_BOX, grow=True):
    """One block as this camera sees it: (shade per pixel, mask, x offset, y offset).

    The offsets say where the block's (0,0,0) corner sits inside the stencil, so a voxel is
    drawn by projecting that one corner and blitting. Only the faces turned towards the
    camera are drawn -- the other three are behind them, and the painter's pass over the
    voxels themselves takes care of what is behind what.

    Every shape is drawn on the *cube's* stencil, at the cube's offsets, so a slab and a full
    block can be blitted by the same loop with no per-shape bookkeeping -- a half-height slab
    is simply a stencil with its lower half empty.
    """
    from PIL import Image, ImageDraw

    def flatten(corner):
        return (sum(c * a for c, a in zip(corner, right)) * cell,
                sum(c * a for c, a in zip(corner, down)) * cell)

    cube = [flatten(c) for _, corners in _box_faces(_CUBE_BOX[0]) for c in corners]
    xs = [p[0] for p in cube]
    ys = [p[1] for p in cube]
    # A pixel of margin all round, for the dilation below to grow into.
    off_x, off_y = 1 - math.floor(min(xs)), 1 - math.floor(min(ys))
    width = int(math.ceil(max(xs) - min(xs))) + 3
    height = int(math.ceil(max(ys) - min(ys))) + 3

    stencil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(stencil)
    shades = [0.0]
    # Farthest box first, so the near half of a stair is drawn over the far half and not the
    # other way round. `near` grows towards the viewer, so ascending order is back to front --
    # negating it here painted every stair's tread over its own riser, which is what made a
    # roof of stairs look like it had bites taken out of it.
    ordered = sorted(boxes, key=lambda b: sum(
        ((b[i] + b[i + 3]) / 2.0) * near[i] for i in range(3)))
    for box in ordered:
        for normal, corners in _box_faces(box):
            if sum(n * a for n, a in zip(normal, near)) <= 0:
                continue                                   # facing away from the camera
            shades.append(_face_shade(normal))
            draw.polygon([(flatten(c)[0] + off_x, flatten(c)[1] + off_y) for c in corners],
                         fill=len(shades) - 1)

    labels = np.asarray(stencil)
    # Shapes that tile edge to edge are grown by a pixel. Voxels are blitted at whole pixels
    # while the lattice they sit on is not, so two neighbours can round apart and leave a
    # hairline of background between them -- visible as pinholes along every joint in a roof.
    # Shapes that stand alone in their block (a pane, a fence post) are left as they are:
    # they have space around them, and widening one would only make it the wrong size.
    if grow:
        labels = _grow_labels(labels)
    shade = np.asarray(shades, dtype=np.float32)[labels]
    return shade, labels > 0, off_x, off_y


def _grow_labels(labels):
    """Widen a stencil by one pixel, so neighbouring cubes meet instead of nearly meeting.

    Voxels are blitted at whole pixels while the lattice they sit on is not, so adjacent
    cubes can round apart and leave a hairline of background between them. Growing each
    stencil into its own margin closes that: cubes now overlap by a pixel instead, and the
    painter's pass already knows which of two overlapping cubes wins.
    """
    grown = labels.copy()
    rows, cols = labels.shape
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        target = grown[max(dy, 0):rows + min(dy, 0), max(dx, 0):cols + min(dx, 0)]
        source = labels[max(-dy, 0):rows + min(-dy, 0), max(-dx, 0):cols + min(-dx, 0)]
        take = (target == 0) & (source != 0)
        target[take] = source[take]
    return grown


def _render_orbit(grid, names: list, colour_of, yaw: float, pitch: float, cut_above: Optional[int],
                  size: int, max_surface: int) -> Optional["object"]:
    """Draw a decoded grid from an arbitrary camera onto a fixed, centred square frame.

    The frame is worked out from the build's bounding box rather than from what is actually
    drawn, and its scale from the footprint's diagonal rather than from the current yaw, so
    turning, tilting or cutting the top off leaves the build the same size in the same place.
    """
    from PIL import Image

    canvas_px = size * SUPERSAMPLE
    right, down, near = _view_axes(yaw, pitch)
    fitted = _fit_to_budget(grid, _solid(grid, names), max_surface,
                            lambda w, h, l: _fit_orbit_cell(w, h, l, pitch, canvas_px))
    if fitted is None:
        return None
    grid, solid, visible, factor = fitted

    # Framed against the blocks rather than the grid: schematics routinely carry empty air
    # around the build, and fitting to that would shrink the build to leave room for nothing.
    (y0, y1), (z0, z1), (x0, x1) = _content_box(solid)
    cell = max(MIN_CELL_PX, _fit_orbit_cell(x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1,
                                            pitch, canvas_px))
    sprites, off_x, off_y = _palette_sprites(names, cell, right, down, near)
    sprite_h, sprite_w = sprites[0][1].shape

    # Centre the build on the canvas. Centred on the *volume* the cubes fill, which runs half
    # a cube past the last voxel's own coordinate in each direction -- centring on the
    # coordinates instead leaves the build sitting half a block off in the frame.
    # The padding is a sprite's worth on each side, so a cube at the very edge is drawn whole
    # and then cropped, rather than clipped mid-blit.
    pad = max(sprite_h, sprite_w) + 2
    centre = ((x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0, (z0 + z1 + 1) / 2.0)
    origin_x = sum(c * a for c, a in zip(centre, right)) * cell - canvas_px / 2.0 - pad + off_x
    origin_y = sum(c * a for c, a in zip(centre, down)) * cell - canvas_px / 2.0 - pad + off_y
    framed = canvas_px + 2 * pad
    canvas = np.zeros((framed, framed, 4), dtype=np.uint8)

    if cut_above is not None:
        # Cut the solid mask, then re-find the surface: the newly exposed floors and room
        # interiors are surface now, and drawing the pre-cut surface would lid them over.
        solid = solid.copy()
        solid[max(0, min(int(cut_above) // factor, grid.shape[0] - 1)) + 1:] = False
        visible = _surface(solid)

    ys, zs, xs = np.nonzero(visible)
    if len(ys) == 0:
        return _blank(size)                          # cut below the whole build

    fx, fy, fz = xs.astype(np.float64), ys.astype(np.float64), zs.astype(np.float64)
    project_on = lambda axis: fx * axis[0] + fy * axis[1] + fz * axis[2]   # noqa: E731
    left = np.rint(project_on(right) * cell - origin_x).astype(np.int64)
    top = np.rint(project_on(down) * cell - origin_y).astype(np.int64)
    depth = project_on(near)

    # An oversized build can be clamped to MIN_CELL_PX and overrun the frame; drop whatever
    # falls outside it rather than letting a blit run off the end of the canvas.
    inside = ((left >= 0) & (top >= 0)
              & (left + sprite_w <= framed) & (top + sprite_h <= framed))
    if not inside.all():
        keep = np.nonzero(inside)[0]
        left, top, depth = left[keep], top[keep], depth[keep]
        ys, zs, xs = ys[keep], zs[keep], xs[keep]
        if len(ys) == 0:
            return _blank(size)

    palette = grid[ys, zs, xs]
    colors = colour_table(names, colour_of)[palette]

    # Painter's algorithm: farthest first, so nearer blocks overwrite what they hide. Each
    # voxel picks the stencil its own blockstate calls for, which is what lets a roof of
    # stairs read as a roof of stairs instead of a staircase of cubes.
    for i in np.argsort(depth, kind="stable"):
        shaded_rgb, mask = sprites[palette[i]]
        if shaded_rgb is None:
            continue
        y0, x0 = top[i], left[i]
        tile = canvas[y0:y0 + sprite_h, x0:x0 + sprite_w]
        tile[mask, :3] = shaded_rgb * colors[i]
        tile[mask, 3] = 255

    cropped = canvas[pad:pad + canvas_px, pad:pad + canvas_px]
    return Image.fromarray(cropped, "RGBA").resize((size, size), Image.LANCZOS)


def _palette_sprites(names, cell, right, down, near):
    """One stencil per palette entry, sharing the cube's frame. (sprites, off_x, off_y).

    Built per shape and shared between palette entries that have the same one, so a build
    whose palette holds forty blockstates still only draws a handful of stencils -- the cost
    is a function of how many *shapes* are on screen, not of how many blocks.
    """
    by_shape = {}
    sprites = []
    off_x = off_y = 0
    for name in names:
        base = _base_name(name)
        state = _blockstate(name)
        boxes = _shape_boxes(base, state)
        key = boxes
        entry = by_shape.get(key)
        if entry is None:
            shade, mask, off_x, off_y = _orbit_sprite(
                cell, right, down, near, boxes, grow=_tiles(base))
            entry = (shade[mask][:, None] if mask.any() else None, mask)
            by_shape[key] = entry
        sprites.append(entry)
    return sprites, off_x, off_y


def _blockstate(name: str) -> dict:
    """`"minecraft:oak_stairs[facing=east,half=top]"` -> {"facing": "east", "half": "top"}."""
    cut = name.find("[")
    if cut < 0 or not name.endswith("]"):
        return {}
    out = {}
    for part in name[cut + 1:-1].split(","):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _content_box(solid):
    """(y, z, x) first/last indices holding a block -- the build inside its grid."""
    spans = []
    for axis in (0, 1, 2):
        along = np.any(solid, axis=tuple(a for a in (0, 1, 2) if a != axis))
        hit = np.nonzero(along)[0]
        spans.append((int(hit[0]), int(hit[-1])))
    return spans


def _blank(size: int):
    """An empty frame -- what a cutaway below the whole build honestly looks like."""
    from PIL import Image

    return Image.new("RGBA", (size, size), (0, 0, 0, 0))
