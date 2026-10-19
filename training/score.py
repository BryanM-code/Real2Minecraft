"""Score a predicted block grid against a human build's target (Phase 5).

Both grids are (shape uint8, material uint16) in the target's frame, mcvox-v3 ids, the
target with IGNORE (255 / 65535) wherever nothing is supervised: the interior, decor, blocks
outside the vocabulary, beyond the capture. Only supervised voxels count: the build's
visible shell and the exterior air around it. What the prediction puts in an ignored voxel
is never scored.

    geometry   solid = any shape but air. Over the supervised voxels: IoU, precision,
               recall and F1 of the predicted solid against the target's, and a tolerant F1
               where a solid voxel within one block (26-neighbourhood) of the other side's
               solid counts as matched (a target voxel is matched by any predicted solid,
               ignored voxels included: a wall one block inward is a near miss)
    shape      share of the voxels solid in both whose shape label is exact (stairs and
               their facing, slabs, panes, doors...)
    material   share of the voxels solid in both whose material id is exact, and the CIELAB
               colour difference between the two materials (`material_colors.json`), so a
               near-miss (dark oak for spruce) costs less than a wrong colour
    block      share of the target's solid voxels predicted exactly (shape and material)

`align` puts a prediction made in its own frame (the rule-based baseline normalises, turns
and scales the mesh itself) into the target's frame: the quarter turn about +Y and the
integer translation that best overlap the two solids. The learned model predicts in the
target's frame already; running `align` on it should return the identity.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

IGNORE_SHAPE = 255
IGNORE_MATERIAL = 65535
AIR = 0
COLORS_PATH = Path(__file__).with_name("material_colors.json")


@lru_cache(maxsize=1)
def material_lab() -> np.ndarray:
    """(n_materials, 3) CIELAB per material id; NaN for air."""
    from mcvoxel.block_palette import srgb_to_lab
    from mcvoxel.vocab import MATERIALS
    rows = json.loads(COLORS_PATH.read_text(encoding="utf-8"))["materials"]
    if [r[0] for r in rows] != list(MATERIALS):
        raise ValueError("material_colors.json does not match the vocabulary; regenerate it")
    lab = np.full((len(rows), 3), np.nan)
    for i, (_, rgb) in enumerate(rows):
        if rgb is not None:
            lab[i] = srgb_to_lab(np.asarray(rgb, dtype=np.float64))
    return lab


def _ratio(a, b):
    return float(a) / float(b) if b else float("nan")


def score(pred_shape, pred_material, tgt_shape, tgt_material) -> dict:
    """The metrics above for one build; grids of equal shape in the target's frame."""
    from scipy import ndimage
    pred_shape, tgt_shape = np.asarray(pred_shape), np.asarray(tgt_shape)
    pred_material, tgt_material = np.asarray(pred_material), np.asarray(tgt_material)
    if pred_shape.shape != tgt_shape.shape:
        raise ValueError(f"grid shapes differ: {pred_shape.shape} vs {tgt_shape.shape}")
    sup = tgt_shape != IGNORE_SHAPE
    t = sup & (tgt_shape != AIR)
    p = sup & (pred_shape != AIR) & (pred_shape != IGNORE_SHAPE)
    tp, fp, fn = int((t & p).sum()), int((p & ~t).sum()), int((t & ~p).sum())
    near = np.ones((3, 3, 3), dtype=bool)
    p_hit = int((p & ndimage.binary_dilation(t, near)).sum())
    # A wall predicted one block inward lands in the ignored interior: still a near miss.
    p_any = (pred_shape != AIR) & (pred_shape != IGNORE_SHAPE)
    t_hit = int((t & ndimage.binary_dilation(p_any, near)).sum())
    prec_tol, rec_tol = _ratio(p_hit, p.sum()), _ratio(t_hit, t.sum())
    both = t & p
    out = {"supervised": int(sup.sum()), "target_solid": int(t.sum()), "pred_solid": int(p.sum()),
           "iou": _ratio(tp, tp + fp + fn), "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
           "f1": _ratio(2 * tp, 2 * tp + fp + fn),
           "f1_tol1": (2 * prec_tol * rec_tol / (prec_tol + rec_tol)) if prec_tol + rec_tol > 0 else float("nan"),
           "shape_acc": float((pred_shape[both] == tgt_shape[both]).mean()) if both.any() else float("nan"),
           "material_acc": float((pred_material[both] == tgt_material[both]).mean()) if both.any() else float("nan"),
           "block_acc": float(((pred_shape == tgt_shape) & (pred_material == tgt_material))[t].mean())
           if t.any() else float("nan")}
    lab = material_lab()
    pm, tm = pred_material[both].astype(np.int64), tgt_material[both].astype(np.int64)
    ok = (pm < len(lab)) & (tm < len(lab))
    de = np.linalg.norm(lab[pm[ok]] - lab[tm[ok]], axis=1) if ok.any() else np.empty(0)
    de = de[np.isfinite(de)]
    out["material_de_mean"] = float(de.mean()) if len(de) else float("nan")
    out["material_de_p50"] = float(np.median(de)) if len(de) else float("nan")
    return out


def summarize(rows: list, keys=("iou", "f1", "f1_tol1", "precision", "recall", "shape_acc",
                                "material_acc", "material_de_p50", "block_acc")) -> dict:
    """Mean over builds and p10 / p50 / p90 of each metric (NaN rows left out)."""
    out = {"builds": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = ({"mean": round(float(v.mean()), 4),
                   "p10_p50_p90": [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])]}
                  if len(v) else None)
    return out


def _place(grid, shift, shape, fill):
    """`grid` translated by integer `shift` into a new array of `shape` (cropped / padded)."""
    out = np.full(shape, fill, dtype=grid.dtype)
    src, dst = [], []
    for g, s, n in zip(grid.shape, shift, shape):
        lo, hi = max(0, s), min(n, g + s)
        if hi <= lo:
            return out
        dst.append(slice(lo, hi))
        src.append(slice(lo - s, hi - s))
    out[tuple(dst)] = grid[tuple(src)]
    return out


def align(pred_shape, pred_material, tgt_shape, tgt_material, turns=(0, 1, 2, 3)) -> tuple:
    """(shape, material, info): the prediction turned by k quarter turns about +Y and shifted
    by the integer translation that maximises the overlap of its solid with the target's
    supervised solid (FFT cross-correlation), cropped / padded to the target's frame with
    air. A tie in overlap (a footprint symmetric under a quarter turn) goes to the turn with
    more exactly matching blocks, then to the smaller turn. `info` holds the turn, the
    shift, the overlap and how many predicted solid voxels fell outside the frame."""
    from mcvoxel.blockstates import rotate_y
    from scipy.signal import fftconvolve
    tgt_shape, tgt_material = np.asarray(tgt_shape), np.asarray(tgt_material)
    t_mask = (tgt_shape != IGNORE_SHAPE) & (tgt_shape != AIR)
    t = t_mask.astype(np.float32)
    best = None
    for k in turns:
        ps, pm = rotate_y(np.asarray(pred_shape), k, np.asarray(pred_material))
        p = ((ps != AIR) & (ps != IGNORE_SHAPE)).astype(np.float32)
        if not p.any() or not t.any():
            continue
        # corr[n] = sum_x t[x] p[x - d] with d = n - (len(p) - 1): pred voxel x lands at x + d.
        corr = fftconvolve(t, p[::-1, ::-1, ::-1], mode="full")
        idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
        shift = tuple(int(i) - (n - 1) for i, n in zip(idx, p.shape))
        out_s = _place(ps, shift, tgt_shape.shape, AIR)
        out_m = _place(pm, shift, tgt_shape.shape, 0)
        overlap = int(round(float(corr[idx])))
        exact = int((t_mask & (out_s == tgt_shape) & (out_m == tgt_material)).sum())
        key = (overlap, exact)
        if best is None or key > best[0]:
            solid_all = int(p.sum())
            solid_in = int(((out_s != AIR) & (out_s != IGNORE_SHAPE)).sum())
            best = (key, {"turns": int(k), "shift": list(shift), "overlap": overlap,
                          "outside_frame": solid_all - solid_in}, out_s, out_m)
    if best is None:
        return (np.zeros(tgt_shape.shape, np.uint8), np.zeros(tgt_shape.shape, np.uint16),
                {"turns": 0, "shift": [0, 0, 0], "overlap": 0, "outside_frame": 0})
    _, info, out_s, out_m = best
    return out_s, out_m, info
