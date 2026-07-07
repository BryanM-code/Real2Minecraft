"""Sponge Schematic v2 (.schem) export — the buildable output artifact."""

from pathlib import Path

import numpy as np


def export_schematic(voxel_data: dict, out_path: Path):
    """Write voxel_data to a Sponge Schematic v2 (.schem) gzipped NBT file."""
    try:
        import nbtlib
        from nbtlib import Compound, Int, Short, ByteArray, IntArray
    except ImportError:
        raise RuntimeError(
            "nbtlib is required for .schem export.\n"
            "Run:  r2m_v2\\Scripts\\pip install nbtlib"
        )

    W, H, L = voxel_data["dims"]
    blocks    = voxel_data["blocks"]
    block_ids = voxel_data.get("block_ids") or ["minecraft:stone"] * len(blocks)

    # Palette: air is always id 0; every distinct block type used gets the next id.
    palette: dict[str, int] = {"minecraft:air": 0}
    for bid in block_ids:
        if bid not in palette:
            palette[bid] = len(palette)

    # Build 3D block-id grid in YZX order (Sponge Schematic v2 convention); 0 = air.
    grid = np.zeros((H, L, W), dtype=np.int32)
    for (x, y, z), bid in zip(blocks, block_ids):
        if 0 <= y < H and 0 <= z < L and 0 <= x < W:
            grid[y, z, x] = palette[bid]

    def _varint(v: int) -> bytes:
        buf = bytearray()
        while True:
            if v & ~0x7F == 0:
                buf.append(v)
                break
            buf.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(buf)

    block_data = bytearray()
    for val in grid.flatten():
        block_data.extend(_varint(int(val)))

    schematic = Compound({
        "Version":     Int(2),
        "DataVersion": Int(3465),       # Minecraft 1.20.1
        "Width":       Short(W),
        "Height":      Short(H),
        "Length":      Short(L),
        "PaletteMax":  Int(len(palette)),
        "Palette":     Compound({bid: Int(idx) for bid, idx in palette.items()}),
        "BlockData":   ByteArray(block_data),
        "Offset":      IntArray([0, 0, 0]),
    })

    nbtlib.File({"": schematic}).save(str(out_path), gzipped=True)
