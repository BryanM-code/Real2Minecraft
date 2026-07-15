"""Intermediate-representation (IR) I/O for building mode.

Every stage of the building pipeline communicates through the IRs saved here — they are
simultaneously the runtime data contract AND the training targets of the future learned
image->build model, so writing must be canonical (sorted keys, fixed float precision)
so identical runs produce byte-identical files that can be hashed for regression.

This module holds NO algorithms: only schema constants, canonical save/load, and small
validators.
"""

import json
from pathlib import Path

import numpy as np

from building.params import SCHEMA_VERSION

# ── Semantic voxel labels (irs/semantic_voxels.npz) ──────────────────────────
SEM_AIR = 0
SEM_WALL = 1
SEM_WINDOW = 2
SEM_DOOR = 3
SEM_ROOF = 4
SEM_FLOOR = 5
SEM_ACCENT = 6
SEM_FOUNDATION = 7

SEM_NAMES = {
    SEM_AIR: "air", SEM_WALL: "wall", SEM_WINDOW: "window", SEM_DOOR: "door",
    SEM_ROOF: "roof", SEM_FLOOR: "floor", SEM_ACCENT: "accent",
    SEM_FOUNDATION: "foundation",
}

# Facade cell classes (irs/facade_cells.npz + wall_grids.npz share this coding).
CELL_WALL = 0
CELL_WINDOW = 1
CELL_ACCENT = 2
CELL_DOOR = 3


def _canonical(obj):
    """Recursively convert to plain JSON types with floats rounded to 6 dp.

    numpy scalars/arrays sneak into IRs easily; normalizing here keeps the JSON
    canonical (same bytes for the same values) so file hashes are meaningful.
    """
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path: Path, obj: dict):
    """Canonical JSON write: version-stamped, sorted keys, 6-dp floats, LF newline."""
    out = dict(obj)
    out.setdefault("version", SCHEMA_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_canonical(out), sort_keys=True, indent=1)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(path: Path, **arrays):
    """Uncompressed-deterministic npz (compression levels can vary across zlib builds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)


def load_npz(path: Path) -> dict:
    with np.load(str(path)) as data:
        return {k: data[k] for k in data.files}


def validate_massing(ir: dict):
    """Cheap structural sanity for the massing IR (raises AssertionError)."""
    assert ir["levels"], "massing has no levels"
    prev_y1 = 0
    for lv in ir["levels"]:
        assert lv["y0"] == prev_y1, "levels must tile Y contiguously"
        assert lv["y1"] > lv["y0"]
        assert lv["rects"], "level without rects"
        for x0, z0, x1, z1 in lv["rects"]:
            assert x1 > x0 and z1 > z0, "degenerate rect"
        prev_y1 = lv["y1"]
    W, H, L = ir["grid_dims"]
    assert prev_y1 <= H
