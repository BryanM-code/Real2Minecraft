"""B3 — facade analysis: the photo side of building mode.

From the cutout (whose alpha is the exact object mask) this stage estimates the visible
facade quadrilateral, rectifies it with a true projective warp (PIL PERSPECTIVE), and
turns it into a grid of classified cells (wall / window / accent / door) plus detected
floor/bay rhythm.  It knows nothing about blocks or voxels — everything geometric is
expressed in FRACTIONS of the rectified facade so the IR is scale-independent (stable
training targets no matter what block size the user picked).

Cell grids are ALWAYS emitted as cells_class_<i>/cells_rgb_<i> with explicit
row_edges_frac/col_edges_frac: regular mode uses the detected floor/bay grid, freeform
mode (low periodicity confidence — e.g. woven-band facades) uses a fixed 64-row map.
Downstream mapping (B4) is identical for both; `grid.mode` records which path fired.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

from block_palette import _srgb_to_lab
from building.ir import CELL_WALL, CELL_WINDOW, CELL_ACCENT, CELL_DOOR


# ── geometry helpers ─────────────────────────────────────────────────────────

def _perspective_coeffs(src_quad, w, h):
    """PIL PERSPECTIVE coefficients mapping output (w x h) pixels -> source quad."""
    dst = [(0, 0), (w, 0), (w, h), (0, h)]
    A, B = [], []
    for (X, Y), (x, y) in zip(src_quad, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); B.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); B.append(Y)
    return np.linalg.solve(np.array(A, dtype=np.float64),
                           np.array(B, dtype=np.float64))


def _fit_line(ys, xs, trim_frac):
    """x = a*y + b by least squares, with one trimmed re-fit."""
    A = np.stack([ys, np.ones_like(ys)], axis=1)
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    resid = np.abs(A @ coef - xs)
    keep = resid <= np.percentile(resid, 100.0 * (1.0 - trim_frac))
    if keep.sum() >= 4:
        coef, *_ = np.linalg.lstsq(A[keep], xs[keep], rcond=None)
    return coef                                       # (a, b)


def _fit_quad(mask: np.ndarray, params: dict):
    """Facade quad [TL, TR, BR, BL] in image px from the object mask silhouette."""
    ys_all, xs_all = np.nonzero(mask)
    if len(ys_all) < 64:
        return None
    # Trimmed vertical extent (mass-based percentiles resist speckle).
    y_top = float(np.percentile(ys_all, params["facade_trim_y_pct"]))
    y_bot = float(np.percentile(ys_all, 100.0 - params["facade_trim_y_pct"]))

    # Side lines from per-row extremes over the middle band of rows.
    pad = (100.0 - params["facade_side_rows_pct"]) / 2.0
    r0 = float(np.percentile(ys_all, pad))
    r1 = float(np.percentile(ys_all, 100.0 - pad))
    rows = np.arange(int(np.ceil(r0)), int(np.floor(r1)) + 1)
    rows = rows[mask[rows].any(axis=1)]
    if len(rows) < 8:
        x0, x1 = float(xs_all.min()), float(xs_all.max())
        return [[x0, y_top], [x1, y_top], [x1, y_bot], [x0, y_bot]]
    first = np.argmax(mask[rows], axis=1).astype(np.float64)
    last = (mask.shape[1] - 1
            - np.argmax(mask[rows][:, ::-1], axis=1)).astype(np.float64)
    ry = rows.astype(np.float64)
    al, bl = _fit_line(ry, first, params["facade_fit_trim_frac"])
    ar, br = _fit_line(ry, last, params["facade_fit_trim_frac"])
    quad = [[al * y_top + bl, y_top], [ar * y_top + br, y_top],
            [ar * y_bot + br, y_bot], [al * y_bot + bl, y_bot]]
    tw = quad[1][0] - quad[0][0]
    bw = quad[2][0] - quad[3][0]
    if tw <= 4 or bw <= 4:                            # crossed/degenerate fit
        x0, x1 = float(xs_all.min()), float(xs_all.max())
        return [[x0, y_top], [x1, y_top], [x1, y_bot], [x0, y_bot]]
    return quad


def _detect_corner_split(mask: np.ndarray, gray: np.ndarray, params: dict):
    """Gated two-facade corner detection: a strong vertical-edge energy peak that
    coincides with a kink in the silhouette top.  Returns IR dict (+ split x or None)."""
    ir = {"attempted": True, "detected": False, "split_x_px": None, "confidence": 0.0}
    xs = np.nonzero(mask.any(axis=0))[0]
    if len(xs) < 40:
        return ir
    x_lo, x_hi = int(xs[0]), int(xs[-1])
    width = x_hi - x_lo + 1

    grad = np.abs(np.gradient(gray, axis=1))
    energy = (grad * mask).sum(axis=0)
    energy = ndimage.gaussian_filter1d(energy, 3.0)
    m0 = x_lo + int(0.20 * width)
    m1 = x_lo + int(0.80 * width)
    if m1 - m0 < 8:
        return ir
    band = energy[m0:m1]
    peak_x = m0 + int(np.argmax(band))
    base = float(np.median(band))
    peak = float(band.max())
    conf = (peak - base) / peak if peak > 1e-9 else 0.0
    ir["confidence"] = conf

    # Silhouette-top kink: strongest jump in the top profile near the peak.
    top_y = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 0).astype(np.float64)
    jump = np.abs(np.diff(ndimage.gaussian_filter1d(top_y, 2.0)))
    if jump[m0:m1].size == 0:
        return ir
    kink_x = m0 + int(np.argmax(jump[m0:m1]))
    if (conf >= params["corner_split_conf"]
            and abs(kink_x - peak_x) <= params["corner_split_kink_tol"] * width):
        ir["detected"] = True
        ir["split_x_px"] = int(peak_x)
    return ir


# ── rhythm detection ─────────────────────────────────────────────────────────

def _autocorr_peak(profile: np.ndarray, lag_min: int, lag_max: int,
                   fundamental_ratio: float = 0.80):
    """(best_lag, normalized peak) of the autocorrelation of a 1D profile.

    Length-corrected so long lags aren't penalized (a 2-story facade's floor pitch is
    ~half the profile), and harmonic-corrected: when lag/2 or lag/3 is nearly as
    strong, the shorter lag is the fundamental pitch and wins.
    """
    a = profile - profile.mean()
    denom = float((a * a).sum())
    n = len(a)
    lag_max = min(lag_max, n // 2)
    if denom < 1e-9 or lag_max <= lag_min:
        return 0, 0.0

    def corr(k):
        return float((a[:-k] * a[k:]).sum() / denom * n / (n - k))

    lags = np.arange(lag_min, lag_max + 1)
    ac = np.array([corr(int(k)) for k in lags])
    best = int(np.argmax(ac))
    lag, conf = int(lags[best]), float(ac[best])
    for div in (3, 2):                       # prefer the fundamental over harmonics
        sub = lag // div
        if sub >= lag_min:
            c = float(ac[sub - lag_min])
            if c >= fundamental_ratio * conf:
                lag, conf = sub, c
    return lag, conf


def _comb_phase(detrended: np.ndarray, pitch: int) -> int:
    """Offset whose comb positions are darkest (floor lines / mullions read dark)."""
    scores = [float(detrended[o::pitch].mean()) for o in range(pitch)]
    return int(np.argmin(scores))


def _edges_from_pitch(size: int, pitch: int, phase: int) -> list:
    """Cell boundary positions: 0, the comb lines strictly inside, and size."""
    edges = [0]
    p = phase
    while p < size:
        if 3 <= p <= size - 3:
            edges.append(int(p))
        p += pitch
    edges.append(size)
    return edges


# ── cell classification ──────────────────────────────────────────────────────

def _kmeans_lab(labs: np.ndarray, k: int, iters: int, merge_de: float):
    """Deterministic Lloyd k-means in LAB: luminance-quantile init, fixed iterations,
    then a centroid merge pass.  Returns (assignments, ordered unique cluster ids)."""
    n = len(labs)
    k = min(k, n)
    order = np.argsort(labs[:, 0], kind="stable")
    seeds = [order[int(q * (n - 1))] for q in np.linspace(0.1, 0.9, k)]
    cent = labs[seeds].copy()
    for _ in range(iters):
        d = np.linalg.norm(labs[:, None, :] - cent[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        for i in range(k):
            m = assign == i
            if m.any():
                cent[i] = labs[m].mean(axis=0)
    # Merge near-identical centroids (lowest index wins).
    remap = np.arange(k)
    for i in range(k):
        for j in range(i + 1, k):
            if remap[j] == j and np.linalg.norm(cent[i] - cent[j]) < merge_de:
                remap[j] = remap[i]
    assign = remap[assign]
    return assign, [int(c) for c in np.unique(assign)]


def _classify_cells(cell_rgb: np.ndarray, params: dict):
    """cells (R,C,3 uint8) -> (class grid (R,C) uint8, classes IR list)."""
    R, C = cell_rgb.shape[:2]
    flat = cell_rgb.reshape(-1, 3).astype(np.float64)
    labs = _srgb_to_lab(flat)
    assign, clusters = _kmeans_lab(labs, params["facade_kmeans_k"],
                                   params["facade_kmeans_iters"],
                                   params["facade_merge_de"])
    n = len(flat)
    stats = []
    for cid in clusters:
        m = assign == cid
        stats.append({
            "cid": cid,
            "frac": float(m.sum() / n),
            "med_l": float(np.median(labs[m, 0])),
            "med_b": float(np.median(labs[m, 2])),
            "rgb": [int(v) for v in np.median(flat[m], axis=0)],
            "lab": [float(v) for v in np.median(labs[m], axis=0)],
        })

    # Window: among mid-fraction clusters, the darkest (tie: bluest).
    window_cid = None
    cands = [s for s in stats
             if params["facade_window_frac_min"] <= s["frac"]
             <= params["facade_window_frac_max"]]
    if cands and len(stats) > 1:
        window_cid = min(cands, key=lambda s: (s["med_l"], s["med_b"]))["cid"]

    # Wall: the cluster dominating the BOTTOM 2/3 of rows.  A visible roof (house
    # silhouettes include it) or sky fringe concentrates in the top rows and must not
    # out-vote the true wall material.
    lower = assign.reshape(R, C)[R // 3:, :].ravel()
    remaining = [s for s in stats if s["cid"] != window_cid]
    wall_cid = (max(remaining, key=lambda s: int((lower == s["cid"]).sum()))["cid"]
                if remaining else None)

    cls = np.full(n, CELL_ACCENT, dtype=np.uint8)
    if wall_cid is not None:
        cls[assign == wall_cid] = CELL_WALL
    if window_cid is not None:
        cls[assign == window_cid] = CELL_WINDOW

    classes_ir = []
    for s in stats:
        name = ("window" if s["cid"] == window_cid
                else "wall" if s["cid"] == wall_cid else "accent")
        classes_ir.append({"name": name, "lab": s["lab"], "rgb": s["rgb"],
                           "cell_frac": s["frac"]})
    return cls.reshape(R, C), classes_ir


def _cell_medians(arr: np.ndarray, row_edges: list, col_edges: list) -> np.ndarray:
    """Per-cell median RGB over the interior 60% of each cell."""
    R, C = len(row_edges) - 1, len(col_edges) - 1
    out = np.zeros((R, C, 3), dtype=np.uint8)
    for r in range(R):
        y0, y1 = row_edges[r], row_edges[r + 1]
        iy = max(1, int(0.2 * (y1 - y0)))
        y0i, y1i = min(y0 + iy, y1 - 1), max(y1 - iy, y0 + 1)
        for c in range(C):
            x0, x1 = col_edges[c], col_edges[c + 1]
            ix = max(1, int(0.2 * (x1 - x0)))
            x0i, x1i = min(x0 + ix, x1 - 1), max(x1 - ix, x0 + 1)
            region = arr[y0i:y1i, x0i:x1i].reshape(-1, 3)
            if len(region) == 0:
                region = arr[y0:y1, x0:x1].reshape(-1, 3)
            out[r, c] = np.median(region, axis=0)
    return out


# ── main entry ───────────────────────────────────────────────────────────────

def analyze_facade(photo_path, cutout_path, params: dict,
                   quad_override=None, grid_override=None):
    """Photo/cutout -> facade-layout IR bundle, or None when no usable cutout mask.

    Returns {"ir": <facade_layout dict>, "cells": {fid: (class_grid, rgb_grid)},
             "rect_images": {fid: PIL.Image}}.
    `quad_override` = [[x,y] TL,TR,BR,BL] image px; `grid_override` = {n_rows, n_cols}
    (uniform grid, forces regular mode) — both come from the refine UI.
    """
    src = cutout_path if cutout_path is not None else photo_path
    if src is None:
        return None
    img = Image.open(src).convert("RGBA")
    rgba = np.asarray(img, dtype=np.uint8)
    alpha = rgba[..., 3]
    has_alpha = (alpha < 250).mean() >= 0.02
    if not has_alpha and quad_override is None:
        return None                       # no mask and no user quad -> nothing to fit
    if has_alpha:
        mask = alpha > params["facade_alpha_thresh"]
        labels, nlab = ndimage.label(mask)
        if nlab == 0:
            if quad_override is None:
                return None
            mask = np.ones_like(alpha, dtype=bool)
        else:
            sizes = np.bincount(labels.ravel()); sizes[0] = 0
            mask = labels == int(sizes.argmax())
    else:
        mask = np.ones_like(alpha, dtype=bool)        # user quad drives everything

    # Composite over mid-gray: background pixels can never contribute real colors.
    rgb = rgba[..., :3].astype(np.float64)
    a = (alpha.astype(np.float64) / 255.0)[..., None]
    comp = (rgb * a + 128.0 * (1.0 - a)).astype(np.uint8)
    gray = comp.astype(np.float64) @ np.array([0.299, 0.587, 0.114])

    # Quads: user override > gated corner split > single auto quad.
    corner_ir = {"attempted": False, "detected": False,
                 "split_x_px": None, "confidence": 0.0}
    if quad_override is not None:
        quads = [[list(map(float, p)) for p in quad_override]]
        quad_sources = ["user"]
    else:
        corner_ir = _detect_corner_split(mask, gray, params)
        if corner_ir["detected"]:
            sx = corner_ir["split_x_px"]
            left = mask.copy(); left[:, sx:] = False
            right = mask.copy(); right[:, :sx] = False
            quads, quad_sources = [], []
            for sub in (left, right):
                q = _fit_quad(sub, params)
                if q is not None:
                    quads.append(q); quad_sources.append("auto")
            if len(quads) != 2:
                corner_ir["detected"] = False
        if not corner_ir["detected"]:
            q = _fit_quad(mask, params)
            if q is None:
                return None
            quads, quad_sources = [q], ["auto"]

    out_h = params["facade_rect_height"]
    facades, cells, rect_images = [], {}, {}
    comp_img = Image.fromarray(comp)
    for fid, (quad, qsource) in enumerate(zip(quads, quad_sources)):
        tl, tr, br, bl = quad
        top_w = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        bot_w = np.hypot(br[0] - bl[0], br[1] - bl[1])
        left_h = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
        right_h = np.hypot(br[0] - tr[0], br[1] - tr[1])
        aspect = ((top_w + bot_w) / 2.0) / max(1.0, (left_h + right_h) / 2.0)
        out_w = int(np.clip(round(out_h * aspect),
                            params["facade_rect_min_w"], params["facade_rect_max_w"]))
        coeffs = _perspective_coeffs(quad, out_w, out_h)
        rect = comp_img.transform((out_w, out_h), Image.PERSPECTIVE,
                                  tuple(coeffs), resample=Image.BILINEAR)
        rect_arr = np.asarray(rect, dtype=np.uint8)
        rect_images[fid] = rect

        # Rhythm: detrended L-profile autocorrelation, rows then columns.
        lab_img = _srgb_to_lab(rect_arr.reshape(-1, 3).astype(np.float64))
        L = lab_img[:, 0].reshape(out_h, out_w)
        row_prof = L.mean(axis=1)
        row_det = row_prof - ndimage.gaussian_filter1d(
            row_prof, params["facade_detrend_sigma"])
        row_pitch, row_conf = _autocorr_peak(
            row_det, params["facade_lag_min"], params["facade_lag_max"],
            params["facade_fundamental_ratio"])
        mid = slice(int(0.2 * out_h), max(int(0.2 * out_h) + 1, int(0.8 * out_h)))
        col_prof = L[mid].mean(axis=0)
        col_det = col_prof - ndimage.gaussian_filter1d(
            col_prof, params["facade_detrend_sigma"])
        col_pitch, col_conf = _autocorr_peak(
            col_det, params["facade_lag_min"], params["facade_lag_max"],
            params["facade_fundamental_ratio"])

        conf_gate = params["facade_conf_regular"]
        if grid_override:
            mode = "regular"
        elif row_conf >= conf_gate and col_conf >= conf_gate:
            mode = "regular"
        elif row_conf >= conf_gate:
            mode = "rows_only"
        else:
            mode = "freeform"

        # Cell edges. Regular modes follow the detected comb; freeform uses a fixed
        # fine map (64 rows) that reproduces aperiodic patterns cell-by-cell.
        if grid_override:
            n_rows = max(1, int(grid_override["n_rows"]))
            n_cols = max(1, int(grid_override["n_cols"]))
            row_edges = [int(round(v)) for v in np.linspace(0, out_h, n_rows + 1)]
            col_edges = [int(round(v)) for v in np.linspace(0, out_w, n_cols + 1)]
        elif mode in ("regular", "rows_only"):
            row_phase = _comb_phase(row_det, row_pitch)
            row_edges = _edges_from_pitch(out_h, row_pitch, row_phase)
            if mode == "regular":
                col_phase = _comb_phase(col_det, col_pitch)
                col_edges = _edges_from_pitch(out_w, col_pitch, col_phase)
            else:
                approx = max(1, int(round(out_w / max(row_pitch, 8))))
                col_edges = [int(round(v))
                             for v in np.linspace(0, out_w, approx + 1)]
        else:
            n_rows = params["facade_freeform_rows"]
            n_cols = int(np.clip(round(n_rows * out_w / out_h), 8,
                                 params["facade_freeform_max_cols"]))
            row_edges = [int(round(v)) for v in np.linspace(0, out_h, n_rows + 1)]
            col_edges = [int(round(v)) for v in np.linspace(0, out_w, n_cols + 1)]

        cell_rgb = _cell_medians(rect_arr, row_edges, col_edges)
        cls, classes_ir = _classify_cells(cell_rgb, params)

        # Doors: non-wall cells on the bottom row, middle third of columns.
        door_cells = []
        R, C = cls.shape
        for c in range(C // 3, (2 * C) // 3 + 1):
            if c < C and cls[R - 1, c] != CELL_WALL:
                cls[R - 1, c] = CELL_DOOR
                door_cells.append([R - 1, c])

        facades.append({
            "id": fid,
            "quad_px": [[float(p[0]), float(p[1])] for p in quad],
            "quad_order": "TL,TR,BR,BL",
            "quad_source": qsource,
            "rectified": "irs/facade_rect_{}.png".format(fid),
            "rectified_size": [int(out_w), int(out_h)],
            "grid": {
                "mode": mode,
                "row_pitch_frac": float(row_pitch) / out_h if row_pitch else 0.0,
                "row_conf": row_conf, "n_rows": len(row_edges) - 1,
                "col_pitch_frac": float(col_pitch) / out_w if col_pitch else 0.0,
                "col_conf": col_conf, "n_cols": len(col_edges) - 1,
                "row_edges_frac": [e / out_h for e in row_edges],
                "col_edges_frac": [e / out_w for e in col_edges],
            },
            "classes": classes_ir,
            "door_cells": door_cells,
            "confidence": float((row_conf + col_conf) / 2.0),
        })
        cells[fid] = (cls, cell_rgb)

    facade_ir = {
        "image_size": [int(rgba.shape[1]), int(rgba.shape[0])],
        "corner_split": corner_ir,
        "facades": facades,
    }
    return {"ir": facade_ir, "cells": cells, "rect_images": rect_images}
