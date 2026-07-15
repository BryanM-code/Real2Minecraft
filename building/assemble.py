"""B5 — assembly: massing + wall grids -> Minecraft-native blocks.

The only building-mode stage that names Minecraft block ids.  Produces:
  - a hollow shell (1-block walls; Minecraft builds are hollow, and it makes the
    windows real glass you can see through) with optional floor plates,
  - per-voxel semantic labels (the training target for the future learned model),
  - a voxel_data dict with EXACTLY the sculpture pipeline's contract, so the viewer,
    schematic export, and dev-render harness consume it unchanged.
"""

import numpy as np
from scipy import ndimage

from building import ir
from building.ir import (SEM_AIR, SEM_WALL, SEM_WINDOW, SEM_DOOR, SEM_ROOF,
                         SEM_FLOOR, SEM_ACCENT, SEM_FOUNDATION,
                         CELL_WALL, CELL_WINDOW, CELL_ACCENT, CELL_DOOR)

_CELL_TO_SEM = {CELL_WALL: SEM_WALL, CELL_WINDOW: SEM_WINDOW,
                CELL_ACCENT: SEM_ACCENT, CELL_DOOR: SEM_DOOR}
_SEM_TO_MATERIAL = {SEM_WALL: "wall", SEM_WINDOW: "window", SEM_DOOR: "door",
                    SEM_ROOF: "roof", SEM_FLOOR: "floor", SEM_ACCENT: "accent",
                    SEM_FOUNDATION: "wall"}


def assemble(massing_ir: dict, reg_solid: np.ndarray, walls_ir: dict,
             wall_grids: dict, scale_factor: float, watertight: bool,
             params: dict):
    """-> (voxel_data, semantic_labels uint8 (W,H,L))."""
    W, H, L = reg_solid.shape
    struct = ndimage.generate_binary_structure(3, 1)

    interior = ndimage.binary_erosion(reg_solid, structure=struct)
    shell = reg_solid & ~interior

    labels = np.zeros((W, H, L), dtype=np.uint8)
    labels[shell] = SEM_WALL

    # Roof: top-exposed cells that are NOT side-exposed (the parapet ring stays wall).
    above = np.zeros_like(reg_solid)
    above[:, :-1, :] = reg_solid[:, 1:, :]
    exposed_top = reg_solid & ~above
    side = np.zeros_like(reg_solid)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = np.zeros_like(reg_solid)
        sx, sz = shift
        src_x = slice(max(0, -sx), W - max(0, sx))
        dst_x = slice(max(0, sx), W - max(0, -sx))
        src_z = slice(max(0, -sz), L - max(0, sz))
        dst_z = slice(max(0, sz), L - max(0, -sz))
        n[dst_x, :, dst_z] = reg_solid[src_x, :, src_z]
        side |= reg_solid & ~n
    labels[exposed_top & ~side & shell] = SEM_ROOF

    # Paint the wall grids onto their planes.
    for wl in walls_ir["walls"]:
        grids = wall_grids.get(wl["id"])
        if grids is None:
            continue
        cls, _rgb = grids
        u0, u1 = wl["u_range"]
        v0, v1 = wl["v_range"]
        plane = wl["plane"]
        sem = np.zeros_like(cls)
        for cell, s in _CELL_TO_SEM.items():
            sem[cls == cell] = s
        # cls is (v, u), v index 0 = ground.
        if wl["dir"] in ("-Z", "+Z"):
            target = labels[u0:u1, v0:v1, plane]           # (u, v)
            on = shell[u0:u1, v0:v1, plane]
            target[on] = sem.T[on]
            labels[u0:u1, v0:v1, plane] = target
        else:
            target = labels[plane, v0:v1, u0:u1]           # (v, u)
            on = shell[plane, v0:v1, u0:u1]
            target[on] = sem[on]
            labels[plane, v0:v1, u0:u1] = target

    # Roof-taper levels ARE the roof: their stepped rings must read as roof material
    # (stepped shingles), never as wall pattern — this intentionally OVERRIDES any
    # facade cells the wall painting put there (windows on a roof step are nonsense).
    facade_classes = (SEM_WALL, SEM_WINDOW, SEM_DOOR, SEM_ACCENT)
    for lv in massing_ir["levels"]:
        if lv["role"] == "roof_taper":
            band = labels[:, lv["y0"]:lv["y1"], :]
            band[np.isin(band, facade_classes)] = SEM_ROOF
            labels[:, lv["y0"]:lv["y1"], :] = band

    # Floor plates + foundation re-solidify interior cells.
    occupancy = shell.copy()
    if params["floor_plates"]:
        for y in walls_ir["floor_lines_y"]:
            if 0 < y < H:
                plate = interior[:, y, :]
                occupancy[:, y, :] |= plate
                lab_y = labels[:, y, :]
                lab_y[plate & (lab_y == SEM_AIR)] = SEM_FLOOR
                labels[:, y, :] = lab_y
    ground = interior[:, 0, :]
    occupancy[:, 0, :] |= ground
    lab_0 = labels[:, 0, :]
    lab_0[ground & (lab_0 == SEM_AIR)] = SEM_FOUNDATION
    labels[:, 0, :] = lab_0

    # Fill 1-cell roof pits (raw-footprint noise between stacked levels): a column
    # whose four lateral neighbours all top out higher gets one block added, so decks
    # read flat.  Two passes handle 2-deep dimples; real courtyards are wider than one
    # cell and untouched.
    for _ in range(2):
        has = occupancy.any(axis=1)
        top = np.where(has, H - 1 - np.argmax(occupancy[:, ::-1, :], axis=1), -10)
        nb_min = np.full((W, L), 10 * H)
        for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.full((W, L), -10)
            src_x = slice(max(0, -sx), W - max(0, sx))
            dst_x = slice(max(0, sx), W - max(0, -sx))
            src_z = slice(max(0, -sz), L - max(0, sz))
            dst_z = slice(max(0, sz), L - max(0, -sz))
            shifted[dst_x, dst_z] = top[src_x, src_z]
            nb_min = np.minimum(nb_min, shifted)
        pit = has & (nb_min >= top + 1) & (top + 1 < H)
        if not pit.any():
            break
        px, pz = np.nonzero(pit)
        py = top[pit] + 1
        occupancy[px, py, pz] = True
        labels[px, py, pz] = SEM_ROOF

    # Emit blocks (argwhere is C-order sorted -> deterministic).
    occ = np.argwhere(occupancy)
    lo = occ.min(axis=0)
    blocks = [(int(x - lo[0]), int(y - lo[1]), int(z - lo[2])) for x, y, z in occ]
    dims = tuple(int(v) for v in (occ.max(axis=0) - lo + 1))

    materials = walls_ir["materials"]
    materials_rgb = walls_ir["materials_rgb"]
    vox_labels = labels[occ[:, 0], occ[:, 1], occ[:, 2]]
    block_ids, colors = [], []
    for s in vox_labels:
        key = _SEM_TO_MATERIAL[int(s)]
        block_ids.append(materials[key])
        colors.append(tuple(int(c) for c in materials_rgb[key]))

    voxel_data = {
        "blocks": blocks,
        "colors": colors,
        "block_ids": block_ids,
        "shapes": None,
        "block_type": "stone",                # legacy field, kept for compatibility
        "dims": dims,
        "block_count": len(blocks),
        "scale_factor": float(scale_factor),
        "watertight": bool(watertight),
        "color_source": "facade",
    }
    return voxel_data, labels
