"""Chunked processing for buildings larger than the model's window.

The model sees a fixed cube (64^3 to start). A 300-block tower is processed as overlapping
windows and reassembled into one grid:

* **Coverage.** `window_starts` places windows every `window - overlap` voxels and snaps
  the last one to the grid's end, so every voxel is covered and every window is full-size
  wherever the grid allows. An axis shorter than the window gets one padded window.
* **Shape: soft blending.** Each window's shape probabilities are weighted by a ramp that
  is 1 in the window's core and falls off across the overlap, except on sides that touch
  the grid boundary (where no neighbour exists to take over). Probabilities are summed and
  normalised, then argmax'd. Predictions near a window's edge -- where the model has least
  context -- are outvoted by the window that sees the same voxel centrally.
* **Material: most-central window wins.** Blending ~130 material probability maps over a
  large grid would cost gigabytes; instead each voxel keeps the material from the window in
  which its blending weight is highest.

Neighbour-dependent block states (stair corners) are derived after assembly, over the whole
grid, by `blockstates.stair_shapes` -- never predicted per window.

Memory: the shape accumulator is (n_shapes, X, Y, Z) float16 -- 8 x 2 bytes a voxel, so a
256 x 384 x 256 bounding box needs about 400 MB; everything else is one byte or two a voxel.
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["window_starts", "iter_windows", "ramp", "assemble"]


def window_starts(n: int, window: int, overlap: int):
    """Start offsets along one axis. Windows overlap by at least `overlap`."""
    if n <= window:
        return [0]
    stride = window - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than the window")
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    return starts


def iter_windows(shape, window: int, overlap: int):
    """(start (3,), slices into the grid, slices into the window) for every window."""
    per_axis = [window_starts(n, window, overlap) for n in shape]
    for s in itertools.product(*per_axis):
        grid_sl = tuple(slice(a, min(a + window, n)) for a, n in zip(s, shape))
        win_sl = tuple(slice(0, g.stop - g.start) for g in grid_sl)
        yield np.array(s), grid_sl, win_sl


def ramp(length: int, overlap: int, low_edge: bool, high_edge: bool):
    """1-D blending weight: 1 in the core, linear falloff across `overlap` at each side
    that has a neighbour; flat to the boundary on sides that touch the grid edge."""
    w = np.ones(length, dtype=np.float32)
    if overlap <= 0:
        return w
    r = (np.arange(overlap, dtype=np.float32) + 1.0) / (overlap + 1.0)
    k = min(overlap, length)
    if not low_edge:
        w[:k] = np.minimum(w[:k], r[:k])
    if not high_edge:
        w[length - k:] = np.minimum(w[length - k:], r[:k][::-1])
    return w


def assemble(shape, window: int, overlap: int, predict, n_shapes: int):
    """Run `predict` on every window and reassemble one (shape, material) grid.

    `predict(start, grid_slices)` returns (shape_probs (n_shapes, w, w, w) float,
    material (w, w, w) int) for the full-size window starting at `start` -- the caller pads
    input windows that run past the grid. Returns (shape_idx uint8 grid, material int32
    grid, weight float32 grid).
    """
    shape = tuple(int(v) for v in shape)
    acc = np.zeros((n_shapes,) + shape, dtype=np.float16)
    wsum = np.zeros(shape, dtype=np.float32)
    best = np.zeros(shape, dtype=np.float32)
    material = np.zeros(shape, dtype=np.int32)
    for start, gsl, wsl in iter_windows(shape, window, overlap):
        probs, mat = predict(start, gsl)
        ws = [ramp(g.stop - g.start, overlap, g.start == 0, g.stop == n)
              for g, n in zip(gsl, shape)]
        w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
        p = probs[(slice(None),) + wsl].astype(np.float32)
        acc[(slice(None),) + gsl] += (p * w[None]).astype(np.float16)
        wsum[gsl] += w
        m = mat[wsl]
        view_best = best[gsl]
        view_mat = material[gsl]
        better = w > view_best
        view_mat[better] = m[better]
        view_best[better] = w[better]
        best[gsl] = view_best
        material[gsl] = view_mat
    shape_idx = np.argmax(acc.astype(np.float32) / np.maximum(wsum, 1e-6)[None], axis=0).astype(np.uint8)
    return shape_idx, material, wsum
