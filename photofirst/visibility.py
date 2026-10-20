"""What one photo can see of a build: orthographic ray casting through a block grid.

The camera is the evaluation photos' (BuildDataCollector `thumbs._view_axes`, rendered by
`synth.trellis_compare.render_photo`): `yaw` spins the build about +Y, `pitch` raises the
camera above the horizon, and the projection is orthographic. `cast` marches parallel rays
from the viewer through the grid. On each ray the first block that is not air is seen; an
IGNORE block (interior the target does not supervise) stops the ray but is not reported.
The air in front of it is seen air. A block also counts as seen when one of its faces turned
towards the viewer has a clear line to it (`_faces_seen`): rays half a block apart miss
surfaces seen at a grazing angle (a roof from 12 degrees up), which a photo, with many pixels
per block, does show. Everything else is hidden from that photo: what a pipeline that
follows the photo has to complete.
"""
from __future__ import annotations

import math

import numpy as np

IGNORE = 255
EVAL_VIEW = (35.0, 12.0)          # yaw, pitch of the real-TRELLIS evaluation photos


def view_axes(yaw_deg: float, pitch_deg: float):
    """(screen right, screen down, towards the viewer) in grid coordinates."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return (np.array([cy, 0.0, sy]), np.array([-sy * sp, -cp, cy * sp]),
            np.array([-sy * cp, sp, cy * cp]))


def cast(ts: np.ndarray, yaw: float, pitch: float, spacing: float = 0.5, step: float = 0.25,
         chunk: int = 1024):
    """(seen solid, seen air) boolean grids for target shapes `ts` (0 air, IGNORE interior)
    seen from (`yaw`, `pitch`): rays every `spacing` blocks across the screen, sampled
    every `step` blocks along the view."""
    opaque = ts != 0
    dims = np.array(ts.shape)
    right, down, near = view_axes(yaw, pitch)
    corners = np.array([[x, y, z] for x in (0, dims[0]) for y in (0, dims[1]) for z in (0, dims[2])],
                       dtype=np.float64)
    su, sv, sd = corners @ right, corners @ down, corners @ near
    uu, vv = (a.ravel() for a in np.meshgrid(np.arange(su.min(), su.max() + spacing, spacing),
                                             np.arange(sv.min(), sv.max() + spacing, spacing),
                                             indexing="ij"))
    depth = sd.max() + 1.0 - np.arange(0.0, np.ptp(sd) + 3.0, step)      # nearest first
    seen_solid = np.zeros(ts.shape, bool)
    seen_air = np.zeros(ts.shape, bool)
    steps = np.arange(len(depth))[None, :]
    for s in range(0, len(uu), chunk):
        p = (uu[s:s + chunk, None, None] * right + vv[s:s + chunk, None, None] * down
             + depth[None, :, None] * near)                                  # (rays, steps, 3)
        idx = np.floor(p).astype(np.int64)
        inside = ((idx >= 0) & (idx < dims)).all(-1)
        hit = np.zeros(inside.shape, bool)
        ii = idx[inside]
        hit[inside] = opaque[ii[:, 0], ii[:, 1], ii[:, 2]]
        first = np.where(hit.any(1), hit.argmax(1), hit.shape[1])
        a = idx[inside & (steps < first[:, None])]
        seen_air[a[:, 0], a[:, 1], a[:, 2]] = True
        r = np.nonzero(first < hit.shape[1])[0]
        h = idx[r, first[r]]
        seen_solid[h[:, 0], h[:, 1], h[:, 2]] = True
    seen_solid |= _faces_seen(opaque, near, step)
    return seen_solid & (ts != IGNORE), seen_air


def cast_from(ts: np.ndarray, centre, right, down, near, focal: float, size: int, every: int = 2,
              step: float = 0.25, chunk: int = 1024):
    """`cast` for a perspective camera: at `centre` (grid coordinates), axes `right`, `down`
    and `near` (towards the camera, which looks along -near), `focal` in pixels, a `size` x
    `size` image; one ray through every `every`-th pixel, plus the faces with a clear line to
    the centre."""
    opaque = ts != 0
    dims = np.array(ts.shape)
    centre, right, down, near = (np.asarray(a, np.float64) for a in (centre, right, down, near))
    px = (np.arange(0, size, every) + 0.5 - size / 2.0) / float(focal)
    uu, vv = (a.ravel() for a in np.meshgrid(px, px, indexing="ij"))
    dirs = uu[:, None] * right + vv[:, None] * down - near
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    corners = np.array([[x, y, z] for x in (0, dims[0]) for y in (0, dims[1]) for z in (0, dims[2])], np.float64)
    dist = np.linalg.norm(corners - centre, axis=1)
    inside_box = ((centre >= 0) & (centre <= dims)).all()
    t = np.arange(0.0 if inside_box else max(float(dist.min()) - float(np.linalg.norm(dims)), 0.0),
                  float(dist.max()) + 1.0, step)
    seen_solid = np.zeros(ts.shape, bool)
    seen_air = np.zeros(ts.shape, bool)
    steps = np.arange(len(t))[None, :]
    for s in range(0, len(dirs), chunk):
        idx = np.floor(centre + t[None, :, None] * dirs[s:s + chunk, None, :]).astype(np.int64)
        inside = ((idx >= 0) & (idx < dims)).all(-1)
        hit = np.zeros(inside.shape, bool)
        ii = idx[inside]
        hit[inside] = opaque[ii[:, 0], ii[:, 1], ii[:, 2]]
        first = np.where(hit.any(1), hit.argmax(1), hit.shape[1])
        a = idx[inside & (steps < first[:, None])]
        seen_air[a[:, 0], a[:, 1], a[:, 2]] = True
        r = np.nonzero(first < hit.shape[1])[0]
        h = idx[r, first[r]]
        seen_solid[h[:, 0], h[:, 1], h[:, 2]] = True
    seen_solid |= _faces_seen(opaque, None, step, centre=centre)
    return seen_solid & (ts != IGNORE), seen_air


def _faces_seen(opaque: np.ndarray, near, step: float = 0.25, chunk: int = 2048, centre=None) -> np.ndarray:
    """Blocks with a face the viewer sees: the face points towards the viewer, is open to a
    block that is not opaque (or the grid's edge), and nothing opaque lies on the ray from its
    centre towards the viewer (along `near`, or to the camera `centre` for a perspective view)."""
    dims = np.array(opaque.shape)
    seen = np.zeros(opaque.shape, bool)
    t = np.arange(step, float(np.linalg.norm(dims)) + 2.0, step)
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3)
            n[axis] = sign
            if centre is None and n @ near <= 1e-6:
                continue
            neighbour = np.zeros(opaque.shape, bool)          # the block beyond the face is opaque
            src, dst = [slice(None)] * 3, [slice(None)] * 3
            src[axis], dst[axis] = (slice(1, None), slice(0, -1)) if sign > 0 else (slice(0, -1), slice(1, None))
            neighbour[tuple(dst)] = opaque[tuple(src)]
            faces = np.argwhere(opaque & ~neighbour & ~seen)
            for s in range(0, len(faces), chunk):
                f = faces[s:s + chunk]
                c = f + 0.5 + 0.5 * n
                if centre is None:
                    d = np.broadcast_to(near, c.shape)
                else:
                    d = centre - c
                    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                    facing = d @ n > 1e-6
                    f, c, d = f[facing], c[facing], d[facing]
                c = c + 1e-3 * d
                idx = np.floor(c[:, None, :] + t[None, :, None] * d[:, None, :]).astype(np.int64)
                inside = ((idx >= 0) & (idx < dims)).all(-1)
                hit = np.zeros(inside.shape, bool)
                ii = idx[inside]
                hit[inside] = opaque[ii[:, 0], ii[:, 1], ii[:, 2]]
                clear = f[~hit.any(1)]
                seen[clear[:, 0], clear[:, 1], clear[:, 2]] = True
    return seen
