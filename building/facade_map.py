"""B4 — facade application: put the photo's facade onto the massing's walls.

Three jobs:
  extract_walls      - walk each massing level's footprint boundary and emit maximal
                       straight wall segments {dir, plane, u_range, v_range}, merged
                       vertically across levels.
  pick_photo_facing  - decide which outward direction the photo shows, by comparing an
                       exposed-face color projection of the raw voxels (pure numpy, no
                       rendering) against the rectified facade in LAB.
  map_facades        - paint per-wall (v,u) class/RGB grids: the photo-facing wall gets
                       the facade cell grid cell-by-cell; every other wall gets the same
                       floor lines and a mirror-tiled repeat of the pattern at the same
                       physical bay width (what a human builder does with unseen sides).

Orientation convention (load-bearing): a wall's grid is stored v-up (v=0 at the wall's
base), u along +axis order.  A viewer OUTSIDE the building sees u running left-to-right
for +Z and -X walls, and right-to-left for -Z and +X walls — `_U_FLIP` encodes this so
photos never paint mirrored.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from mcvoxel.block_palette import srgb_to_lab, match_blocks, match_blocks_restricted
from color_repair import _normalize_tone
from building.ir import CELL_WALL, CELL_WINDOW, CELL_ACCENT, CELL_DOOR

DIRS = ("-Z", "+Z", "-X", "+X")
_U_FLIP = {"-Z": True, "+Z": False, "-X": False, "+X": True}

# Window sub-palette: matched ONLY here (never in the global palette — glass's
# near-white average steals white walls, see block_palette docstring).
WINDOW_PALETTE_IDS = (
    "minecraft:glass", "minecraft:white_stained_glass",
    "minecraft:light_gray_stained_glass", "minecraft:gray_stained_glass",
    "minecraft:black_stained_glass", "minecraft:light_blue_stained_glass",
    "minecraft:blue_stained_glass", "minecraft:cyan_stained_glass",
    "minecraft:tinted_glass",
)


# ── wall extraction ──────────────────────────────────────────────────────────

def _level_footprint(level: dict, W: int, L: int) -> np.ndarray:
    fp = np.zeros((W, L), dtype=bool)
    for x0, z0, x1, z1 in level["rects"]:
        fp[max(0, x0):min(W, x1), max(0, z0):min(L, z1)] = True
    return fp


def _runs(bools: np.ndarray):
    """Maximal runs of True -> [(start, stop_exclusive)]."""
    idx = np.flatnonzero(bools)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks] + 1, idx[-1] + 1]
    return list(zip(starts.tolist(), stops.tolist()))


def _segments_for_level(fp: np.ndarray, y0: int, y1: int):
    """Wall segments of one footprint: exposed boundary cells grouped into runs."""
    W, L = fp.shape
    segs = []
    exp = {
        "-Z": fp & ~np.pad(fp, ((0, 0), (1, 0)))[:, :-1],   # nothing at z-1
        "+Z": fp & ~np.pad(fp, ((0, 0), (0, 1)))[:, 1:],
        "-X": fp & ~np.pad(fp, ((1, 0), (0, 0)))[:-1, :],
        "+X": fp & ~np.pad(fp, ((0, 1), (0, 0)))[1:, :],
    }
    for d in DIRS:
        e = exp[d]
        if d in ("-Z", "+Z"):
            for plane in range(L):
                for u0, u1 in _runs(e[:, plane]):
                    segs.append({"dir": d, "plane": int(plane),
                                 "u_range": [int(u0), int(u1)],
                                 "v_range": [int(y0), int(y1)]})
        else:
            for plane in range(W):
                for u0, u1 in _runs(e[plane, :]):
                    segs.append({"dir": d, "plane": int(plane),
                                 "u_range": [int(u0), int(u1)],
                                 "v_range": [int(y0), int(y1)]})
    return segs


def extract_walls(massing_ir: dict, reg_solid: np.ndarray) -> list:
    """Massing -> wall list, merged vertically across levels, deterministic ids."""
    W, H, L = reg_solid.shape
    segs = []
    for lv in massing_ir["levels"]:
        fp = _level_footprint(lv, W, L)
        segs.extend(_segments_for_level(fp, lv["y0"], lv["y1"]))

    # Merge segments identical in (dir, plane, u_range) with touching v ranges.
    segs.sort(key=lambda s: (s["dir"], s["plane"], s["u_range"][0],
                             s["u_range"][1], s["v_range"][0]))
    merged = []
    for s in segs:
        if (merged and merged[-1]["dir"] == s["dir"]
                and merged[-1]["plane"] == s["plane"]
                and merged[-1]["u_range"] == s["u_range"]
                and merged[-1]["v_range"][1] == s["v_range"][0]):
            merged[-1]["v_range"][1] = s["v_range"][1]
        else:
            merged.append(dict(s))
    for i, s in enumerate(merged):
        s["id"] = i
    return merged


# ── photo-facing detection ───────────────────────────────────────────────────

def _face_image(solid: np.ndarray, vol: np.ndarray, d: str) -> np.ndarray:
    """Exposed-face color projection for direction d, oriented as seen from OUTSIDE
    (row 0 = top).  vol is (W,H,L,3) uint8; empty columns get the mean color."""
    if d in ("-Z", "+Z"):
        s = solid[:, :, ::-1] if d == "+Z" else solid
        v = vol[:, :, ::-1, :] if d == "+Z" else vol
        first = np.argmax(s, axis=2)                   # (W,H): first solid from outside
        valid = s.any(axis=2)
        img = np.take_along_axis(v, first[:, :, None, None], axis=2)[:, :, 0]  # (W,H,3)
        img = np.transpose(img, (1, 0, 2))             # (H, W, 3): rows=y, cols=x=u
        val = valid.T
    else:
        s = solid[::-1] if d == "+X" else solid
        v = vol[::-1] if d == "+X" else vol
        first = np.argmax(s, axis=0)                   # (H,L)
        valid = s.any(axis=0)
        img = np.take_along_axis(v, first[None, :, :, None], axis=0)[0]  # (H,L,3): u=z
        val = valid
    img = img[::-1]                                     # row 0 = top
    val = val[::-1]
    if _U_FLIP[d]:
        img = img[:, ::-1]
        val = val[:, ::-1]
    out = img.astype(np.float64)
    if val.any() and (~val).any():
        out[~val] = out[val].mean(axis=0)
    return out


def pick_photo_facing(raw_solid: np.ndarray, colors_raw, facade_bundle,
                      params: dict) -> dict:
    """Which outward direction does the photo show?  ΔE between each face projection
    and the rectified facade, with the calibrated front convention as a tiebreaker.

    Convention calibration (dev_building.py --facing on 30 cached GLB+cutout pairs,
    2026-07-14): winner tally {'+Z': 18, '-Z': 4, '+X': 4, '-X': 4}.  TRELLIS leans
    toward facing the input photo at +Z but not reliably enough to trust outright
    (60%), and the misses were all low-margin scatter — hence convention-within-slack
    only: correlation decides whenever it has a clear margin.
    """
    convention = params.get("front_convention")
    if colors_raw is None or facade_bundle is None:
        return {"dir": convention or "-Z", "method": "default", "score": None,
                "scores_de": {}}

    toned = _normalize_tone(list(colors_raw))
    occ = np.argwhere(raw_solid)
    vol = np.zeros(raw_solid.shape + (3,), dtype=np.uint8)
    vol[occ[:, 0], occ[:, 1], occ[:, 2]] = np.asarray(toned, dtype=np.uint8)

    rect = facade_bundle["rect_images"][0]
    n = params["facing_thumb"]
    ref = np.asarray(rect.resize((n, n), Image.BOX), dtype=np.float64)
    ref_lab = srgb_to_lab(ref.reshape(-1, 3))

    scores = {}
    for d in DIRS:
        face = _face_image(raw_solid, vol, d)
        thumb = np.asarray(Image.fromarray(face.astype(np.uint8)).resize(
            (n, n), Image.BOX), dtype=np.float64)
        lab = srgb_to_lab(thumb.reshape(-1, 3))
        scores[d] = float(np.linalg.norm(lab - ref_lab, axis=1).mean())

    best = min(DIRS, key=lambda d: scores[d])
    method = "correlation"
    chosen = best
    if convention in scores and scores[convention] <= (
            params["front_convention_slack"] * scores[best]):
        chosen = convention
        method = "convention"
    return {"dir": chosen, "method": method, "score": scores[chosen],
            "scores_de": scores}


# ── facade -> wall painting ──────────────────────────────────────────────────

def _mirror_frac(t: np.ndarray) -> np.ndarray:
    """Triangle wave: mirror-tiled fraction in [0, 1] (seam-free pattern repeat)."""
    t = np.mod(t, 2.0)
    return np.where(t > 1.0, 2.0 - t, t)


def _paint_wall(wall: dict, facade: dict, cls_grid: np.ndarray, rgb_grid: np.ndarray,
                photo_wall: dict, extended: bool):
    """(v,u) class + RGB grids for one wall from a facade's cell grids."""
    u0, u1 = wall["u_range"]
    v0, v1 = wall["v_range"]
    w, h = u1 - u0, v1 - v0
    row_edges = np.asarray(facade["grid"]["row_edges_frac"], dtype=np.float64)
    col_edges = np.asarray(facade["grid"]["col_edges_frac"], dtype=np.float64)

    pv0, pv1 = photo_wall["v_range"]
    pu0, pu1 = photo_wall["u_range"]
    ph = max(1, pv1 - pv0)
    pw = max(1, pu1 - pu0)

    vv = np.arange(h) + 0.5
    uu = np.arange(w) + 0.5
    # v: absolute Y through the photo wall's frame (floors wrap the building).
    y_abs = v0 + vv
    v_frac_top = 1.0 - np.clip((y_abs - pv0) / ph, 0.0, 1.0)
    rows = np.clip(np.searchsorted(row_edges, v_frac_top, side="right") - 1,
                   0, cls_grid.shape[0] - 1)
    # u: direct fraction on the photo wall; mirror-tiled at the photo wall's physical
    # width on extended walls so the bay rhythm continues around the corner.  Every
    # wall reads left-to-right as seen from outside, like the photo.
    if extended:
        u_frac = _mirror_frac(uu / pw)
    else:
        u_frac = uu / w
    if _U_FLIP[wall["dir"]]:
        u_frac = 1.0 - u_frac
    cols = np.clip(np.searchsorted(col_edges, u_frac, side="right") - 1,
                   0, cls_grid.shape[1] - 1)

    cls = cls_grid[np.ix_(rows, cols)].copy()          # (v, u); v index 0 = ground
    rgb = rgb_grid[np.ix_(rows, cols)].copy()

    # De-speckle: a 1-block window is classification noise (photo shadows), not a
    # window — real windows survive as 2+ block components.
    win_labels, n_win = ndimage.label(cls == CELL_WINDOW)
    if n_win:
        sizes = np.bincount(win_labels.ravel())
        tiny = np.isin(win_labels, np.flatnonzero(sizes < 2)) & (win_labels > 0)
        cls[tiny] = CELL_WALL

    wall_cells = cls == CELL_WALL
    wall_fill = (rgb[wall_cells].reshape(-1, 3).mean(axis=0).astype(np.uint8)
                 if wall_cells.any() else np.array([160, 160, 160], np.uint8))
    if extended:
        # No doors on unseen sides; a tiled door would be nonsense.
        rgb[cls == CELL_DOOR] = wall_fill
        cls[cls == CELL_DOOR] = CELL_WALL
    # Corner rule: cells within 1 block of a segment end read as wall.
    if w > 2:
        for edge in (0, w - 1):
            rgb[:, edge] = wall_fill
            cls[:, edge] = CELL_WALL
    return cls, rgb


def map_facades(walls: list, facade_bundle, facing: dict, colors_raw,
                raw_solid: np.ndarray, params: dict):
    """-> (walls_ir, wall_grids {id: (class (v,u) uint8, rgb (v,u,3) uint8)})."""
    warnings = []
    materials, materials_rgb = _pick_materials(facade_bundle, colors_raw,
                                               raw_solid, params, warnings)

    wall_grids = {}
    floor_lines_y = []
    photo_dirs = {}
    if facade_bundle is not None:
        facades = facade_bundle["ir"]["facades"]
        # Facade 0 goes to the detected facing; a split second facade goes to the
        # better-scoring perpendicular direction.
        photo_dirs[facing["dir"]] = 0
        if len(facades) > 1:
            perp = ("-X", "+X") if facing["dir"] in ("-Z", "+Z") else ("-Z", "+Z")
            scores = facing.get("scores_de") or {}
            side = min(perp, key=lambda d: scores.get(d, 1e9))
            photo_dirs[side] = 1

        # The photo wall per photo dir = the largest wall segment on that dir.
        photo_walls = {}
        for d, fid in photo_dirs.items():
            cands = [wl for wl in walls if wl["dir"] == d]
            if cands:
                photo_walls[d] = max(
                    cands, key=lambda wl: (wl["u_range"][1] - wl["u_range"][0])
                    * (wl["v_range"][1] - wl["v_range"][0]))
        ref_wall = photo_walls.get(facing["dir"])
        if ref_wall is None and walls:
            ref_wall = max(walls, key=lambda wl: (wl["u_range"][1] - wl["u_range"][0])
                           * (wl["v_range"][1] - wl["v_range"][0]))

        min_pitch = params["min_floor_pitch_blocks"]
        for wl in walls:
            fid = photo_dirs.get(wl["dir"], 0)
            facade = facades[fid]
            cls_grid, rgb_grid = facade_bundle["cells"][fid]
            is_photo = (wl is photo_walls.get(wl["dir"])
                        and wl["dir"] in photo_dirs)
            pw = photo_walls.get(wl["dir"]) if is_photo else ref_wall
            cls, rgb = _paint_wall(wl, facade, cls_grid, rgb_grid, pw or wl,
                                   extended=not is_photo)
            # In the periodic modes a row is a floor: windows need >= 2 blocks per
            # floor to read, else the whole wall goes plain.  Freeform rows are just
            # map resolution, not floors — small windows simply vanish in the
            # downsample, so no gate applies.
            wall_h = wl["v_range"][1] - wl["v_range"][0]
            n_rows = max(1, len(facade["grid"]["row_edges_frac"]) - 1)
            if (facade["grid"]["mode"] != "freeform"
                    and wall_h / n_rows < min_pitch):
                if (cls == CELL_WINDOW).any():
                    warnings.append(
                        "wall {}: floor pitch < {} blocks — windows disabled".format(
                            wl["id"], min_pitch))
                cls[cls == CELL_WINDOW] = CELL_WALL
            wl["source"] = ("photo:{}".format(fid) if is_photo
                            else "extended:{}".format(fid))
            wall_grids[wl["id"]] = (cls, rgb)

        # Global floor lines from the reference wall's row edges (v-up, absolute Y) —
        # but only when floors resolve to a sane pitch; a mis-detected fine grid would
        # otherwise fill the whole interior with plates.
        if ref_wall is not None:
            f0 = facades[photo_dirs.get(facing["dir"], 0)]
            pv0, pv1 = ref_wall["v_range"]
            ph = pv1 - pv0
            n_rows = max(1, len(f0["grid"]["row_edges_frac"]) - 1)
            if ph / n_rows >= 3.0:
                ys = sorted({int(round(pv0 + (1.0 - e) * ph))
                             for e in f0["grid"]["row_edges_frac"]})
                floor_lines_y = [y for y in ys if pv0 < y < pv1]
    else:
        for wl in walls:
            wl["source"] = "default"
            u0, u1 = wl["u_range"]
            v0, v1 = wl["v_range"]
            cls = np.full((v1 - v0, u1 - u0), CELL_WALL, dtype=np.uint8)
            rgb = np.tile(np.asarray(materials_rgb["wall"], np.uint8),
                          (v1 - v0, u1 - u0, 1))
            wall_grids[wl["id"]] = (cls, rgb)

    walls_ir = {
        "photo_facing": facing,
        "floor_lines_y": [int(y) for y in floor_lines_y],
        "materials": materials,
        "materials_rgb": {k: [int(c) for c in v] for k, v in materials_rgb.items()},
        "walls": [{k: v for k, v in wl.items()} for wl in walls],
        "warnings": warnings,
    }
    return walls_ir, wall_grids


def _pick_materials(facade_bundle, colors_raw, raw_solid, params, warnings):
    """Class -> (block id, display RGB).  Photo cells are ground truth when present;
    the roof falls back to the tone-lifted top voxels (the photo rarely shows roofs).

    Class medians go through the sculpture pipeline's `_normalize_tone` before palette
    matching: photos carry shading/exposure, but Minecraft re-lights the build, so
    materials should be matched at lit-albedo brightness (a shaded cream wall must
    become a cream block, not a brown one).  The IR cell colors stay photo-true.
    """
    def med_rgb(cls_code, default):
        if facade_bundle is None:
            return default
        vals = []
        for fid, (cls, rgb) in facade_bundle["cells"].items():
            m = cls == cls_code
            if m.any():
                vals.append(rgb[m].reshape(-1, 3))
        if not vals:
            return default
        return tuple(int(v) for v in np.median(np.concatenate(vals), axis=0))

    wall_rgb = med_rgb(CELL_WALL, (170, 165, 158))
    accent_rgb = med_rgb(CELL_ACCENT, wall_rgb)
    window_rgb = med_rgb(CELL_WINDOW, (60, 70, 84))
    if facade_bundle is not None:
        # One shared gain (computed over ALL cells so class medians stay consistent),
        # windows excluded from lifting: glass should stay dark.
        all_cells = np.concatenate(
            [rgb.reshape(-1, 3) for _, rgb in facade_bundle["cells"].values()])
        toned = _normalize_tone([tuple(c) for c in all_cells]
                                + [wall_rgb, accent_rgb])
        wall_rgb = tuple(int(v) for v in toned[-2])
        accent_rgb = tuple(int(v) for v in toned[-1])

    roof_rgb = wall_rgb
    if colors_raw is not None:
        toned = np.asarray(_normalize_tone(list(colors_raw)), dtype=np.float64)
        occ = np.argwhere(raw_solid)
        H = raw_solid.shape[1]
        top = occ[:, 1] >= H - max(1, int(round(0.15 * H)))
        if top.any():
            roof_rgb = tuple(int(v) for v in np.median(toned[top], axis=0))
    elif facade_bundle is None:
        warnings.append("no texture colors and no facade — default materials")

    materials = {
        "wall": match_blocks([wall_rgb])[0],
        "accent": match_blocks([accent_rgb])[0],
        "roof": match_blocks([roof_rgb])[0],
        "window": match_blocks_restricted([window_rgb], WINDOW_PALETTE_IDS)[0][0],
        "door": params["door_block"],
    }
    materials["floor"] = materials["wall"]
    materials_rgb = {"wall": wall_rgb, "accent": accent_rgb, "roof": roof_rgb,
                     "window": window_rgb, "door": (96, 66, 40), "floor": wall_rgb}
    return materials, materials_rgb
