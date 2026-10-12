"""Voxel model -> Minecraft blocks, deterministically: the target side of a training pair.

Reuses Real2Minecraft's colour-to-block chain -- `quantize_colors` (k-means in the
palette's weighted-LAB space, spatial coherence), `match_blocks`, `consolidate_block_ids`,
`apply_shapes` -- on the SHELL of a clean source model, and adds the rules only a clean
model can support:

* **Glass.** Voxels whose surface samples mostly came from glass materials (BLEND alpha,
  translucent base colour, or a glass/window name) take the nearest glass from
  `AUX_PALETTE`. Colour alone cannot find windows; the material says so outright.
* **Doors.** Door-named materials take the nearest wood planks (v1 has no door blocks).
* Glass and door voxels stay full cubes: a glass slab is not a window.

TRELLIS-specific colour repairs (tone lift, shadow lift, shadow repaint, photo-guided
repaint) are NOT applied: the source here is a clean model whose colours are albedo already.
"""
from __future__ import annotations

import numpy as np

from . import block_shapes as bs
from .block_palette import AUX_PALETTE, match_blocks, match_blocks_restricted
from .color_quantize import consolidate_block_ids, quantize_colors
from .scene import KIND_DOOR, KIND_GLASS
from .vocab import FROM_BLOCK_SHAPES, IGNORE_MATERIAL, IGNORE_SHAPE, MATERIAL_INDEX, encode_state

__all__ = ["assign_blocks", "BlockTarget"]

_PLANKS = sorted(k for k in bs.BLOCK_VARIANTS if k.endswith("_planks"))


class BlockTarget:
    """Per-voxel blocks for a voxel model's shell, plus the model-ready encodings."""

    def __init__(self, occ, states, shape_grid, material_grid, info):
        self.occ = occ                      # (N,3) shell voxel indices (x,y,z)
        self.states = states                # list of N blockstate strings
        self.shape = shape_grid             # (X,Y,Z) uint8 vocab shape; 255 = ignore
        self.material = material_grid       # (X,Y,Z) uint16 vocab material; 65535 = ignore
        self.info = info

    def sponge_grid(self, solid_shape):
        """(grid (Y,Z,X) uint16 palette indices, names) for `schem_write.write_schem`."""
        names = ["minecraft:air"]
        index = {"minecraft:air": 0}
        grid = np.zeros((solid_shape[1], solid_shape[2], solid_shape[0]), dtype=np.uint16)
        for (x, y, z), s in zip(self.occ, self.states):
            i = index.get(s)
            if i is None:
                i = index[s] = len(names)
                names.append(s)
            grid[y, z, x] = i
        return grid, names


def assign_blocks(vm, glass_min: float = 0.5, door_min: float = 0.5,
                  shadow_repaint: bool = False, palette=None, quantize: dict | None = None,
                  coherent_materials: bool = False) -> BlockTarget:
    """Blocks for every shell voxel of `vm` (a grid.VoxelModel with shapes and colours).

    `palette` restricts matching (and the slab / stair material swap) to a subset of
    BLOCK_PALETTE, e.g. `block_palette.BUILDER_PALETTE`; None = the full palette.
    `quantize` overrides `quantize_colors` settings (k_init, merge_de, absorb_de, ...): the
    defaults were tuned to flatten TRELLIS noise, which a clean model does not have.

    `shadow_repaint` is passed to `quantize_colors`. Its shadow-to-albedo stage is meant for
    TRELLIS colours with baked lighting; on clean source models it only recolours genuinely
    dark neutral materials (measured on 40 pilot targets: 4 changed, e.g. 13% of one
    building's grey trim repainted as its sandstone walls), so it is off by default here.
    """
    occ = np.argwhere(vm.shell)
    info = {"shell": int(len(occ)), "shadow_repaint": bool(shadow_repaint)}
    if len(occ) == 0:
        raise ValueError("voxel model has an empty shell")
    rgb = vm.rgb[occ[:, 0], occ[:, 1], occ[:, 2]].astype(np.int64)
    colors = [tuple(int(c) for c in row) for row in rgb]

    q = quantize_colors(occ, colors, vm.solid.shape, shadow_repaint=shadow_repaint, **(quantize or {}))
    allowed = None if palette is None else list(palette)
    ids = match_blocks(q) if allowed is None else match_blocks_restricted(q, allowed)[0]
    ids = consolidate_block_ids(q, ids)
    info["materials_after_quantize"] = len(set(ids))

    kind = vm.kind_frac[occ[:, 0], occ[:, 1], occ[:, 2]].astype(np.float32)
    glass = kind[:, KIND_GLASS] >= glass_min
    door = (kind[:, KIND_DOOR] >= door_min) & ~glass
    if glass.any():
        g_ids, _ = match_blocks_restricted([q[i] for i in np.flatnonzero(glass)], list(AUX_PALETTE))
        for i, g in zip(np.flatnonzero(glass), g_ids):
            ids[i] = g
    if door.any():
        planks = _PLANKS if allowed is None else [b for b in _PLANKS if b in allowed] or _PLANKS
        d_ids, _ = match_blocks_restricted([q[i] for i in np.flatnonzero(door)], planks)
        for i, d in zip(np.flatnonzero(door), d_ids):
            ids[i] = d
    info["glass_voxels"] = int(glass.sum())
    info["door_voxels"] = int(door.sum())

    shapes = vm.shapes[occ[:, 0], occ[:, 1], occ[:, 2]].astype(np.int8)
    shapes[glass | door] = bs.SHAPE_FULL
    if coherent_materials:
        from .architecture import material_families
        ids, info['material_families'] = material_families(
            occ, ids, q, shapes, vm.solid.shape, glass | door, allowed)
    states, shapes = bs.apply_shapes(ids, q, shapes, allowed=allowed)
    detail_counts = {"fence": 0, "wall": 0, "material_rejected": 0}
    if vm.post_mask is not None and vm.post_mask.any():
        from scipy import ndimage
        from .vocab import decode, SHAPE_FENCE, SHAPE_WALL
        # One material decision for a whole vertical run, using source colours rather
        # than a quantiser that may absorb a three-voxel post into the facade colour.
        labels, nr = ndimage.label(vm.post_mask & vm.shell)
        occ_labels = labels[tuple(occ.T)]
        for r in range(1, nr + 1):
            rows = np.flatnonzero((occ_labels == r) & ~glass & ~door)
            if len(rows) < 2:
                continue
            families = vm.post_family[tuple(occ[rows].T)] if vm.post_family is not None else np.zeros(len(rows), int)
            if (families == 3).any():
                detail_counts["material_rejected"] += len(rows)
                continue
            rgb_post = np.median(rgb[rows], axis=0)
            pools = [[b for b in family if allowed is None or b in allowed]
                     for family in (bs.FENCE_VARIANTS, bs.WALL_VARIANTS)]
            matches = [match_blocks_restricted([rgb_post], pool) if pool else ([], [float('inf')]) for pool in pools]
            if (families == 1).any():
                family = 0
            elif (families == 2).any():
                family = 1
            else:
                family = int(matches[1][1][0] < matches[0][1][0])
                if matches[family][1][0] > 12:
                    detail_counts["material_rejected"] += len(rows)
                    continue
            if not matches[family][0]:
                continue
            base = matches[family][0][0]
            state = decode((SHAPE_FENCE, SHAPE_WALL)[family], MATERIAL_INDEX[base])
            for row in rows:
                states[row] = state
            detail_counts[("fence", "wall")[family]] += len(rows)
    info["context_details"] = detail_counts

    shape_grid = np.full(vm.solid.shape, IGNORE_SHAPE, dtype=np.uint8)
    material_grid = np.full(vm.solid.shape, IGNORE_MATERIAL, dtype=np.uint16)
    air = ~vm.solid
    shape_grid[air] = 0
    material_grid[air] = 0
    unknown = 0
    for (x, y, z), s in zip(occ, states):
        sh, mat = encode_state(s)
        if mat is None:
            unknown += 1
            continue
        shape_grid[x, y, z] = sh
        material_grid[x, y, z] = mat
    info["unencodable"] = unknown
    info["shape_counts"] = {int(k): int(v) for k, v in zip(*np.unique(shape_grid[vm.shell], return_counts=True))}
    return BlockTarget(occ, states, shape_grid, material_grid, info)
