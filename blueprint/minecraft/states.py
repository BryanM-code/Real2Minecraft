"""Block states the game works out from neighbours, added to a finished grid.

A pasted schematic keeps exactly the states it stores, so these must be right in the file:
stair corners (`stair_shapes`), a door's hinge and upper half (`door_halves`), and fence,
wall and pane arms plus a fence gate's `in_wall` (`structural_connections`).
`finish_states` runs all three, in that order. Grids here are in `.schem` order (Y, Z, X).
"""
from __future__ import annotations

import re

import numpy as np

FACINGS = {"east": (1, 0), "west": (-1, 0), "south": (0, 1), "north": (0, -1)}   # (dx, dz)


def _parse(state):
    if "[" not in state:
        return state, {}
    name, body = state.split("[", 1)
    props = dict(part.split("=", 1) for part in body.rstrip("]").split(",") if "=" in part)
    return name, props


def _join(name, props):
    return name if not props else name + "[" + ",".join(
        k + "=" + props[k] for k in sorted(props)) + "]"


# ── Stair corners ─────────────────────────────────────────────────────────────
_CCW = {"north": "west", "west": "south", "south": "east", "east": "north"}
_STAIR_RE = re.compile(r"_stairs\[")


def _prop(state, key):
    m = re.search(key + r"=(\w+)", state)
    return m.group(1) if m else None


def stair_shapes(grid, names):
    """Set `shape=` on every stair in a Sponge (Y,Z,X) grid the way Minecraft computes it.

    Returns (grid, names) with new palette entries appended as needed. The rule is the
    game's StairBlock#getStairsShape, the same for both halves: look at the stair in front
    (toward `facing`); if it has the same half and faces across our axis, this is an OUTER
    corner (left when it faces our counter-clockwise side). Otherwise look behind; the same
    test gives an INNER corner. Otherwise straight.
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


# ── Doors ─────────────────────────────────────────────────────────────────────
_CW = {"north": "east", "east": "south", "south": "west", "west": "north"}


def _full_block_shapes():
    from .vocab import SHAPE_FULL, SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z
    return (SHAPE_FULL, SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z)


def door_halves(grid, names):
    """Give every lower door half its hinge and its upper half, by the game's rules.

    DoorBlock#getHinge: count full-collision blocks beside the door at both heights (left
    -1 each, right +1 each, left/right relative to the door's facing); a lower door of the
    same kind beside it forces the hinge away from it. Positive -> right, negative -> left,
    tied -> left (the game then uses the click position, which a schematic does not have).
    The upper half takes the lower half's facing, hinge, open and powered, one block up.
    """
    from .vocab import encode_state
    names = list(names)
    lower = {i for i, n in enumerate(names)
             if n.split("[", 1)[0].endswith("_door") and _prop(n, "half") == "lower"}
    if not lower:
        return grid, names
    H, L, W = grid.shape
    out = grid.copy()
    index = {n: i for i, n in enumerate(names)}
    full = _full_block_shapes()
    solid = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        sh, mat = encode_state(n)
        solid[i] = mat is not None and sh in full

    def palette(state):
        j = index.get(state)
        if j is None:
            j = index[state] = len(names)
            names.append(state)
        return j

    def at(y, z, x):
        return int(grid[y, z, x]) if 0 <= y < H and 0 <= z < L and 0 <= x < W else None

    for y, z, x in np.argwhere(np.isin(grid, list(lower))).tolist():
        i = int(grid[y, z, x])
        name, props = _parse(names[i])
        facing = props.get("facing", "north")
        left, right = FACINGS[_CCW[facing]], FACINGS[_CW[facing]]
        score = 0
        for (dx, dz), sign in ((left, -1), (right, 1)):
            for dy in (0, 1):
                j = at(y + dy, z + dz, x + dx)
                if j is not None and j < len(solid) and solid[j]:
                    score += sign
        left_door = at(y, z + left[1], x + left[0])
        right_door = at(y, z + right[1], x + right[0])
        left_door = left_door in lower and names[left_door].split("[", 1)[0] == name
        right_door = right_door in lower and names[right_door].split("[", 1)[0] == name
        if (left_door and not right_door) or score > 0:
            hinge = "right"
        else:
            hinge = "left"
        base = {k: v for k, v in props.items() if k not in ("hinge", "half")}
        out[y, z, x] = palette(_join(name, {**base, "half": "lower", "hinge": hinge}))
        if y + 1 < H:
            out[y + 1, z, x] = palette(_join(name, {**base, "half": "upper", "hinge": hinge}))
    return out, names


# ── Fences, walls, panes, fence gates ─────────────────────────────────────────
def structural_connections(grid, names):
    """Fence / wall / pane arms and fence-gate `in_wall` on a complete (Y,Z,X) target.

    Full side faces connect, slab edges do not, stairs connect on their high side (both
    halves). Fences join fences of their own kind (wooden or nether brick) and fence gates
    that open along the arm's axis; walls join walls, panes and such gates; panes join panes,
    walls and full faces, glass included (fences and walls keep v2's rule of not joining
    glass). Leaves join nothing. Walls use low arms unless a covering block above requires
    tall arms; the centre post disappears on an uncovered straight run. A fence gate is
    `in_wall` when a wall stands beside it along its span. Waterlogging stays false.
    """
    from .vocab import (_props, encode_state, SHAPE_FENCE, SHAPE_WALL, SHAPE_PANE,
                        SHAPE_GATE_X, SHAPE_GATE_Z, SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z)
    names = list(names)
    parsed = [(n.split("[", 1)[0], _props(n), encode_state(n)[0]) for n in names]
    linked = (SHAPE_FENCE, SHAPE_WALL, SHAPE_PANE)
    gates = (SHAPE_GATE_X, SHAPE_GATE_Z)
    ids = [i for i, (_, _, sh) in enumerate(parsed) if sh in linked]
    gate_ids = [i for i, (_, _, sh) in enumerate(parsed) if sh in gates]
    if not ids and not gate_ids:
        return grid, names
    H, L, W = grid.shape
    out, index = grid.copy(), {n: i for i, n in enumerate(names)}

    def at(y, z, x):
        return parsed[int(out[y, z, x])] if 0 <= y < H and 0 <= z < L and 0 <= x < W else ("minecraft:air", {}, 0)

    def full_face(block, side, glass_ok=False):
        name, props, sh = block
        if name.endswith("_leaves"):
            return False
        if sh in (1, SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z):
            return glass_ok or "glass" not in name
        if 4 <= sh <= 7 or 10 <= sh <= 13:
            return props.get("facing") == side
        return False

    def gate_opens_along(block, direction):
        """A gate spanning X (facing north/south) joins arms to its east and west."""
        sh = block[2]
        return (sh == SHAPE_GATE_X and direction in ("east", "west")) or \
               (sh == SHAPE_GATE_Z and direction in ("north", "south"))

    def fence_kind(name):
        return "nether" if name == "minecraft:nether_brick_fence" else "wood"

    def connects(me, neighbour, direction):
        name, _, shape = me
        n_name, _, n_shape = neighbour
        back = _opposite(direction)
        if shape == SHAPE_FENCE:
            return ((n_shape == SHAPE_FENCE and fence_kind(n_name) == fence_kind(name))
                    or (n_shape in gates and gate_opens_along(neighbour, direction))
                    or full_face(neighbour, back))
        if shape == SHAPE_WALL:
            return (n_shape in (SHAPE_WALL, SHAPE_PANE)
                    or (n_shape in gates and gate_opens_along(neighbour, direction))
                    or full_face(neighbour, back))
        return n_shape in (SHAPE_PANE, SHAPE_WALL) or full_face(neighbour, back, glass_ok=True)

    # Resolve upper walls first: a lower wall depends on the arms/post above it.
    for y, z, x in np.argwhere(np.isin(grid, ids))[::-1].tolist():
        name, old, shape = at(y, z, x)
        links = {}
        above = at(y + 1, z, x)
        covers = above[2] in (1, 2, 4, 5, 6, 7, SHAPE_FULL_AXIS_X, SHAPE_FULL_AXIS_Z)
        for direction, (dx, dz) in FACINGS.items():
            connected = connects((name, old, shape), at(y, z + dz, x + dx), direction)
            if shape in (SHAPE_FENCE, SHAPE_PANE):
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
            parsed.append((name, links, shape))
        out[y, z, x] = index[state]

    for y, z, x in np.argwhere(np.isin(grid, gate_ids)).tolist():
        name, props, shape = at(y, z, x)
        sides = ("east", "west") if shape == SHAPE_GATE_X else ("north", "south")
        in_wall = any(at(y, z + FACINGS[s][1], x + FACINGS[s][0])[2] == SHAPE_WALL for s in sides)
        state = _join(name, {**props, "in_wall": str(in_wall).lower()})
        if state not in index:
            index[state] = len(names)
            names.append(state)
            parsed.append((name, _parse(state)[1], shape))
        out[y, z, x] = index[state]
    return out, names


def finish_states(grid, names):
    """Derive all neighbour-dependent states, once the full grid is assembled."""
    return structural_connections(*door_halves(*stair_shapes(grid, names)))
