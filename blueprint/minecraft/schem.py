"""Shape and material grids -> a Sponge v2 `.schem` file (Minecraft 1.21.11).

Each (shape, material) pair is decoded to its block state (`vocab.decode`); then the states
that depend on neighbours (stair corners, door halves, fence, wall and pane arms) are worked
out over the whole grid (`states.finish_states`), as the game would.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .states import finish_states
from .vocab import AIR, TARGET_DATA_VERSION, decode


def palette_grid(shape: np.ndarray, material: np.ndarray) -> tuple:
    """(grid (Y, Z, X) int32 of palette indices, names) with every neighbour-dependent state
    finished; air is index 0. `shape` and `material` are (X, Y, Z)."""
    shape = np.asarray(shape)
    material = np.asarray(material)
    solid = (shape != 0) & (material != 0) & (shape != 255)
    keys = shape.astype(np.int64) * 65536 + material.astype(np.int64)
    uniq, inv = np.unique(keys[solid], return_inverse=True)
    names = [AIR] + [decode(int(k // 65536), int(k % 65536)) for k in uniq]
    grid = np.zeros(shape.shape, np.int32)
    grid[solid] = inv.ravel() + 1
    grid, names = finish_states(np.ascontiguousarray(np.transpose(grid, (1, 2, 0))), names)
    return np.asarray(grid, np.int32), list(names)


def write_schem(shape: np.ndarray, material: np.ndarray, path) -> dict:
    """Write (shape, material) (X, Y, Z) to `path`; returns {"blocks", "palette", "dims"}."""
    import nbtlib
    from nbtlib import ByteArray, Compound, Int, IntArray, Short
    grid, names = palette_grid(shape, material)
    H, L, W = grid.shape
    used = np.unique(grid)
    remap = np.zeros(len(names), np.int64)
    remap[used] = np.arange(len(used))                         # drop states finish_states replaced
    grid = remap[grid]
    names = [names[i] for i in used]
    data = bytearray()
    for v in grid.ravel():                                     # Sponge order: x fastest, then z, then y
        v = int(v)
        while True:                                            # each index as a varint
            if v & ~0x7F == 0:
                data.append(v)
                break
            data.append((v & 0x7F) | 0x80)
            v >>= 7
    nbt = Compound({
        "Version": Int(2),
        "DataVersion": Int(TARGET_DATA_VERSION),
        "Width": Short(W), "Height": Short(H), "Length": Short(L),
        "PaletteMax": Int(len(names)),
        "Palette": Compound({n: Int(i) for i, n in enumerate(names)}),
        "BlockData": ByteArray(data),
        "Offset": IntArray([0, 0, 0]),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbtlib.File(nbt, root_name="Schematic").save(path, gzipped=True)   # the root is the schematic
    return {"blocks": int((grid != 0).sum()), "palette": len(names), "dims": [W, H, L]}
