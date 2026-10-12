"""Blockstate rules that depend on orientation or on neighbours.

Two jobs, both deterministic:

* **Augmentation symmetry.** Rotating or mirroring a training grid about the vertical axis
  must also rotate or mirror every stair's `facing`, or the augmented label says "rises to
  the east" over a slope that now rises to the south. `rotate_y` / `mirror_x` do the grid
  and the label together, derived from the index transform itself so the two cannot
  disagree.
* **Stair corners.** A stair's `shape` (straight / inner_* / outer_*) is not something to
  predict: Minecraft derives it from the neighbouring stairs, but only when a neighbour
  updates -- a pasted schematic keeps whatever it stored. `stair_shapes` runs the game's own
  rule over a finished grid, after chunks are reassembled, so hips and valleys paste right.
"""
from __future__ import annotations

import re

import numpy as np

from .vocab import SHAPES

__all__ = ["rotate_y", "mirror_x", "stair_shapes", "structural_connections", "finish_states", "FACINGS"]

FACINGS = {"east": (1, 0), "west": (-1, 0), "south": (0, 1), "north": (0, -1)}   # (dx, dz)
_SHAPE_OF_FACING = {f: SHAPES.index("stairs_" + f) for f in FACINGS}
_FACING_OF_SHAPE = {v: k for k, v in _SHAPE_OF_FACING.items()}


def _facing_map(transform):
    """How a (x, z) index transform maps each horizontal facing."""
    out = {}
    for name, (dx, dz) in FACINGS.items():
        p0 = np.array(transform(0, 0))
        p1 = np.array(transform(dx, dz))
        d = tuple(int(v) for v in p1 - p0)
        out[name] = next(k for k, v in FACINGS.items() if v == d)
    return out


def rotate_y(shape_grid, k: int = 1, *others):
    """Rotate (X,Y,Z) grids by k*90 degrees about +Y, remapping stair facings.

    Implemented with np.rot90 on axes (0, 2); the facing permutation is derived from
    where np.rot90 sends a unit step, so the grid and the labels always agree.
    """
    k %= 4
    if k == 0:
        return (shape_grid,) + tuple(others)

    X, Z = shape_grid.shape[0], shape_grid.shape[2]

    def transform(x, z, n=k, X=X, Z=Z):
        for _ in range(n):             # np.rot90(axes=(0,2)): (x, z) -> (Z-1-z, x)
            x, z = Z - 1 - z, x
            X, Z = Z, X
        return x, z

    fmap = _facing_map(transform)
    out = np.rot90(shape_grid, k, axes=(0, 2)).copy()
    remapped = out.copy()
    for src, dst in fmap.items():
        remapped[out == _SHAPE_OF_FACING[src]] = _SHAPE_OF_FACING[dst]
    return (remapped,) + tuple(np.rot90(o, k, axes=(0, 2)).copy() for o in others)


def mirror_x(shape_grid, *others):
    """Mirror (X,Y,Z) grids across the YZ plane (east <-> west), remapping facings."""
    X = shape_grid.shape[0]
    fmap = _facing_map(lambda x, z: (X - 1 - x, z))
    out = shape_grid[::-1].copy()
    remapped = out.copy()
    for src, dst in fmap.items():
        remapped[out == _SHAPE_OF_FACING[src]] = _SHAPE_OF_FACING[dst]
    return (remapped,) + tuple(o[::-1].copy() for o in others)


_CCW = {"north": "west", "west": "south", "south": "east", "east": "north"}
_STAIR_RE = re.compile(r"_stairs\[")


def _prop(state, key):
    m = re.search(key + r"=(\w+)", state)
    return m.group(1) if m else None


def stair_shapes(grid, names):
    """Set `shape=` on every stair in a Sponge (Y,Z,X) grid the way Minecraft computes it.

    Returns (grid, names) with new palette entries appended as needed. The rule is the
    game's StairBlock#getStairsShape: look at the stair in front (toward `facing`); if it
    has the same half and faces across our axis, this is an OUTER corner (left when it
    faces our counter-clockwise side). Otherwise look behind; the same test gives an INNER
    corner. Otherwise straight.
    """
    names = list(names)
    stair_idx = {i for i, n in enumerate(names) if _STAIR_RE.search(n)}
    if not stair_idx:
        return grid, names
    info = {i: (_prop(names[i], "facing"), _prop(names[i], "half")) for i in stair_idx}
    H, L, W = grid.shape
    out = grid.copy()
    index = {n: i for i, n in enumerate(names)}

    def at(y, z, x):
        if 0 <= y < H and 0 <= z < L and 0 <= x < W:
            i = int(grid[y, z, x])
            return info.get(i)
        return None

    ys, zs, xs = np.nonzero(np.isin(grid, list(stair_idx)))
    for y, z, x in zip(ys.tolist(), zs.tolist(), xs.tolist()):
        i = int(grid[y, z, x])
        facing, half = info[i]
        if facing is None:
            continue
        dx, dz = FACINGS[facing]
        axis = "x" if dx else "z"
        shape = "straight"
        front = at(y, z + dz, x + dx)
        if front and front[1] == half and ("x" if FACINGS[front[0]][0] else "z") != axis:
            side = FACINGS[_opposite(front[0])]
            if _can_take(at, y, z + side[1], x + side[0], facing, half):
                shape = "outer_left" if front[0] == _CCW[facing] else "outer_right"
        if shape == "straight":
            back = at(y, z - dz, x - dx)
            if back and back[1] == half and ("x" if FACINGS[back[0]][0] else "z") != axis:
                side = FACINGS[back[0]]
                if _can_take(at, y, z + side[1], x + side[0], facing, half):
                    shape = "inner_left" if back[0] == _CCW[facing] else "inner_right"
        base = re.sub(r",?shape=\w+", "", names[i])
        new = base[:-1] + ",shape={}]".format(shape) if base.endswith("]") else base
        j = index.get(new)
        if j is None:
            j = index[new] = len(names)
            names.append(new)
        out[y, z, x] = j
    return out, names


def _opposite(facing):
    return {"north": "south", "south": "north", "east": "west", "west": "east"}[facing]


def _can_take(at, y, z, x, facing, half):
    other = at(y, z, x)
    return other is None or other[0] != facing or other[1] != half


def structural_connections(grid, names):
    """Fence/wall states on a complete (Y,Z,X) target, after tile assembly.

    Full side faces connect, slab edges do not, stairs connect on their high side.
    Walls use low arms unless a covering block above requires tall arms. The centre
    post disappears on an uncovered straight run. Waterlogging stays false.
    """
    from .vocab import encode_state, SHAPE_FENCE, SHAPE_WALL, _props
    names = list(names)
    parsed = [(n.split("[", 1)[0], _props(n), encode_state(n)[0]) for n in names]
    ids = [i for i, (_, _, sh) in enumerate(parsed) if sh in (SHAPE_FENCE, SHAPE_WALL)]
    if not ids:
        return grid, names
    H, L, W = grid.shape
    out, index = grid.copy(), {n: i for i, n in enumerate(names)}

    def at(y, z, x):
        return parsed[int(out[y, z, x])] if 0 <= y < H and 0 <= z < L and 0 <= x < W else ("minecraft:air", {}, 0)

    def full_face(block, side):
        name, props, sh = block
        if sh == 1:
            return "glass" not in name
        if 4 <= sh <= 7:
            return props.get("facing") == side
        return False

    # Resolve upper walls first: a lower wall depends on the arms/post above it.
    for y, z, x in np.argwhere(np.isin(grid, ids))[::-1].tolist():
        name, old, shape = at(y, z, x)
        links = {}
        above = at(y + 1, z, x)
        covers = above[2] in (1, 2, 4, 5, 6, 7)
        for direction, (dx, dz) in FACINGS.items():
            neighbour = at(y, z + dz, x + dx)
            connected = neighbour[2] == shape or full_face(neighbour, _opposite(direction))
            if shape == SHAPE_FENCE:
                links[direction] = str(connected).lower()
            else:
                tall = covers or (above[2] == SHAPE_WALL and above[1].get(direction, "none") != "none")
                links[direction] = ("tall" if tall else "low") if connected else "none"
        if shape == SHAPE_WALL:
            active = {f for f, value in links.items() if value != "none"}
            straight = active in ({"east", "west"}, {"north", "south"})
            above_post = above[2] == SHAPE_FENCE or (above[2] == SHAPE_WALL and above[1].get("up") == "true")
            links["up"] = str(not straight or covers or above_post).lower()
        links["waterlogged"] = "false"
        state = name + "[" + ",".join(k + "=" + links[k] for k in sorted(links)) + "]"
        if state not in index:
            index[state] = len(names)
            names.append(state)
            parsed.append((name,links,shape))
        out[y, z, x] = index[state]
    return out, names


def finish_states(grid, names):
    """Derive all neighbour-dependent states, once the full grid is assembled."""
    return structural_connections(*stair_shapes(grid, names))
