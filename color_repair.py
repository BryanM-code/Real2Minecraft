"""Photo-guided color repair for TRELLIS bakes (the "Enhanced" color mode).

TRELLIS textures carry hallucinated colors on surfaces the photo never saw, baked
shadows, and a globally dark tone.  These passes repair COLORS ONLY — geometry is
never touched — using the input photo (cutout) as the ground truth for which colors
belong on the build.  Calibration notes live in the project memory
(color-repair-lessons): the thresholds here are measured, not guessed.
"""

import os

import numpy as np
from PIL import Image


def _photo_reference_lab(image_path, size: int = 96, border: int = 3,
                         bg_de: float = 10.0, min_bin_frac: float = 0.02,
                         max_colors: int = 4096):
    """LAB colour population of the input photo's object (foreground) pixels.

    The photo is the ground truth for which colours belong on the build (CLAUDE.md:
    the photo is the *reference*).  When the image carries a real alpha channel (a
    _prepare_cutout output), that alpha IS the object mask.  Otherwise background is
    estimated colour-wise from the image border ring (sky, lawn, backdrop); pixels far
    from every border colour — plus the centre patch, where the object sits — count as
    foreground.

    Only the DOMINANT colour families are kept (ab-plane bins holding >= `min_bin_frac`
    of the foreground): rare pixels — shadow slivers, anti-aliased silhouette blends —
    would otherwise legitimise hallucinations of nearly their colour (a house's teal
    bake finding support in a few cool-shadow pixels).  Returns (K,3) float64 LAB
    (unweighted), or None on any failure.
    """
    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
        from block_palette import _srgb_to_lab

        rgba = Image.open(image_path).convert("RGBA").resize((size, size))
        arr = np.asarray(rgba, dtype=np.float64)
        lab = _srgb_to_lab(arr[..., :3]).reshape(-1, 3)
        alpha = arr[..., 3]
        if (alpha < 250).any():
            # A real cutout (our _prepare_cutout output): the alpha IS the object mask
            # — exact, no heuristics.  Erode to drop anti-aliased silhouette blends.
            fg2d = ndimage.binary_erosion(alpha > 128, iterations=2)
            if int(fg2d.sum()) < 50:
                return None
        else:
            # Plain photo: estimate background colour-wise from the border ring, keep
            # pixels far from it plus the centre patch where the object sits.
            interior = np.zeros((size, size), dtype=bool)
            interior[border:-border, border:-border] = True
            d, _ = cKDTree(lab[~interior.ravel()]).query(lab)
            fg2d = interior & (d > bg_de).reshape(size, size)
            # Erode: strips the anti-aliased silhouette (object-sky blends read as
            # colours that exist in neither object nor background).
            fg2d = ndimage.binary_erosion(fg2d, iterations=2)
            c0, c1 = int(size * 0.3), int(size * 0.7)
            fg2d[c0:c1, c0:c1] = True
        cols = lab[fg2d.ravel()]
        # Keep only dominant colour families (>= min_bin_frac of foreground per ab bin).
        bins = np.floor(cols[:, 1:] / 8.0).astype(np.int64)
        key = bins[:, 0] * 4096 + bins[:, 1]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        dominant = cols[cnt[inv] >= min_bin_frac * len(cols)]
        if len(dominant) >= 50:
            cols = dominant
        if len(cols) > max_colors:
            keep = np.random.default_rng(0).choice(len(cols), max_colors, replace=False)
            cols = cols[keep]
        return cols
    except Exception:
        return None


# Lightness weight in the photo-support distance: the TRELLIS bake is globally darker
# than the photo, so support leans on hue/saturation with only a mild L penalty.
# The repaint threshold is deliberately CONSERVATIVE: it flags only egregious
# inventions (a lawn baked onto a house's underside, ~dE 48); measured real-but-
# drifted colors reach ~dE 13.6 (hard tower's cream facade), so tightening past ~16
# starts repainting real surfaces.
PHOTO_L_WEIGHT = 0.25


def _repair_unsupported_colors(occ, colors, photo_lab, thresh: float = 16.0,
                               max_frac: float = 0.5):
    """Repaint voxels whose colour does not exist anywhere in the input photo.

    TRELLIS hallucinates colours on surfaces the photo never saw (teal walls on a
    white house, gold specks in green foliage).  Any voxel colour with no close match
    in the photo's foreground population (hue-weighted LAB, see PHOTO_L_WEIGHT) is
    replaced: preferably by its mirror across X/Z when the mirror is supported (keeps
    the build's structure), else by the nearest supported voxel.

    Safety gate: if more than `max_frac` of voxels would change, the photo mask is
    probably wrong (object fills the frame) — leave everything alone.
    """
    if not colors or photo_lab is None or not len(photo_lab):
        return colors
    from scipy.spatial import cKDTree
    from block_palette import _srgb_to_lab

    occ = np.asarray(occ)
    col = np.asarray(colors, dtype=float)
    w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
    d, _ = cKDTree(np.asarray(photo_lab, dtype=np.float64) * w).query(
        _srgb_to_lab(col) * w)
    bad = d > thresh
    if not bad.any() or float(bad.mean()) > max_frac:
        return colors

    out = col.copy()
    good = ~bad
    dims = occ.max(axis=0) + 1
    gidx = np.full(tuple(dims), -1, dtype=np.int64)
    gidx[occ[:, 0], occ[:, 1], occ[:, 2]] = np.arange(len(occ))
    fixed = np.zeros(len(occ), dtype=bool)
    for ax in (0, 2):
        lo_c, hi_c = int(occ[:, ax].min()), int(occ[:, ax].max())
        mcoord = occ.copy()
        mcoord[:, ax] = (lo_c + hi_c) - occ[:, ax]
        mj = gidx[mcoord[:, 0], mcoord[:, 1], mcoord[:, 2]]     # mirror index, or -1
        use = bad & ~fixed & (mj >= 0) & good[np.where(mj >= 0, mj, 0)]
        out[use] = col[mj[use]]
        fixed |= use
    rem = bad & ~fixed
    if rem.any() and good.any():
        _, idx = cKDTree(occ[good]).query(occ[rem])
        out[rem] = col[good][idx]
    return [tuple(int(c) for c in rgb) for rgb in out]


def _snap_suspect_hues(colors, photo_lab, lo: float = 8.0, hi: float = 16.0):
    """Snap the HUE of suspicious colours onto the photo's palette, keeping lightness.

    Colours whose photo-support distance falls in the [lo, hi) band are too far from the
    photo to be trusted but not egregious enough for `_repair_unsupported_colors` to
    repaint (that threshold must stay >= ~16, see PHOTO_L_WEIGHT note).  For those, the
    voxel keeps its own L (shading/detail survives) but takes the LAB ab of its nearest
    photo colour — a wrong-hued clump becomes a shaded patch of a colour that actually
    exists in the photo, while real-but-drifted colours snap to their own colour family
    (a near no-op).  Colours below `lo` are already photo-true and stay untouched.
    """
    if not colors or photo_lab is None or not len(photo_lab):
        return colors
    from scipy.spatial import cKDTree
    from block_palette import _srgb_to_lab, _lab_to_srgb

    col = np.asarray(colors, dtype=np.float64)
    lab = _srgb_to_lab(col)
    w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
    ref = np.asarray(photo_lab, dtype=np.float64)
    d, idx = cKDTree(ref * w).query(lab * w)
    band = (d >= lo) & (d < hi)
    if not band.any():
        return colors
    lab[band, 1:] = ref[idx[band], 1:]
    col[band] = _lab_to_srgb(lab[band])
    return [tuple(int(c) for c in np.rint(rgb)) for rgb in col]


def _fill_unobserved_colors(occ, colors, photo_lab=None, pair_de: float = 20.0,
                            min_pairs: int = 30):
    """Repair TRELLIS's hallucinated far side by mirror-pair disagreement.

    TRELLIS bakes the side facing the input photo faithfully; surfaces the photo never
    saw are diffusion-generated, which shows up as regions that are DARK (grey tower
    back) or WRONG-HUED (teal house walls).  Whole-half statistics miss these (a big
    shared roof dilutes any half-mean), so the repair works pair-by-pair.  Runs on
    TONE-LIFTED colours so photo comparisons are at photo brightness.  Per horizontal
    axis (X then Z):

      1. Pair every voxel with its mirror across the model's centre plane and keep the
         pairs that STRONGLY disagree (LAB dE >= `pair_de`).  Agreeing voxels — shared
         roofs, symmetric bodies — drop out untouched.  Fewer than `min_pairs`
         disagreements means real asymmetry/noise, not a baked artifact: skip.
      2. Decide which side of the disagreeing set to trust, by cascade:
           a. a LARGE median-lightness gap -> the brighter side (a dark grey back vs a
              lit facade; hallucinated fills trend dim).  This outranks the photo test
              because a real-but-drifted bright facade can look photo-unsupported;
           b. PHOTO SUPPORT -> when one side's colours clearly exist in the input
              photo and the other's clearly don't (teal walls on a cream house),
              trust the supported side;
           c. small lightness gap, then colour diversity, else leave the axis alone.
      3. Suspect members of disagreeing pairs take their mirror's colour
         position-for-position, so the hallucinated patch inherits the real material.

    Always runs (not gated on any symmetry guess); colour-only, never changes geometry.
    `occ` is the (N,3) voxel-index array aligned 1:1 with `colors`.
    """
    if not colors or len(colors) < 4:
        return colors
    from block_palette import _srgb_to_lab

    occ = np.asarray(occ)
    out = np.asarray(colors, dtype=float)

    ptree = None
    if photo_lab is not None and len(photo_lab):
        try:
            from scipy.spatial import cKDTree
            w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
            ptree = cKDTree(np.asarray(photo_lab, dtype=np.float64) * w)
        except Exception:
            ptree = None

    dims = occ.max(axis=0) + 1
    gidx = np.full(tuple(dims), -1, dtype=np.int64)
    gidx[occ[:, 0], occ[:, 1], occ[:, 2]] = np.arange(len(occ))

    # One sweep per horizontal axis.  (A second sweep was tried and rejected: after the
    # first repaint the pair statistics describe repaired colours, and the follow-up
    # decisions repaint real surfaces — e.g. half a roof with wall colours.)
    for ax in (0, 2):
        lo_c, hi_c = int(occ[:, ax].min()), int(occ[:, ax].max())
        if hi_c - lo_c < 3:
            continue
        lab = _srgb_to_lab(out)

        mcoord = occ.copy()
        mcoord[:, ax] = (lo_c + hi_c) - occ[:, ax]
        mj = gidx[mcoord[:, 0], mcoord[:, 1], mcoord[:, 2]]     # mirror index, or -1
        center = (lo_c + hi_c) / 2.0
        is_lo = occ[:, ax] < center
        # One index per pair: lo-side voxels that have a hi-side mirror partner.
        paired = is_lo & (mj >= 0) & ~is_lo[np.where(mj >= 0, mj, 0)]
        if not paired.any():
            continue
        pi = np.where(paired)[0]                                # lo member
        pj = mj[pi]                                             # hi member
        de = np.linalg.norm(lab[pi] - lab[pj], axis=1)
        bad = de >= pair_de
        if int(bad.sum()) < min_pairs:
            continue
        bi, bj = pi[bad], pj[bad]

        # 2. Trust cascade (see docstring).
        lum_lo = float(np.median(lab[bi][:, 0]))
        lum_hi = float(np.median(lab[bj][:, 0]))
        sup_lo = sup_hi = None
        if ptree is not None:
            w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
            sup_lo = float(np.median(ptree.query(lab[bi] * w)[0]))
            sup_hi = float(np.median(ptree.query(lab[bj] * w)[0]))

        div_lo = float(lab[bi].std(axis=0).mean())
        div_hi = float(lab[bj].std(axis=0).mean())
        if abs(lum_lo - lum_hi) >= 25.0:
            # An unambiguous lit-vs-dark split.  Only this outranks the photo test: a
            # real-but-hue-drifted bright facade can look photo-unsupported (hard
            # tower, dL~35), while a hallucination is rarely THIS much brighter.
            trust_lo, rule = lum_lo > lum_hi, "bright-strong"
        elif sup_lo is not None and (sup_lo - sup_hi) >= 3.0:
            trust_lo, rule = False, "photo"  # lo's colours aren't in the photo -> fake
        elif sup_lo is not None and (sup_hi - sup_lo) >= 3.0:
            trust_lo, rule = True, "photo"
        elif abs(lum_lo - lum_hi) >= 6.0:
            trust_lo, rule = lum_lo > lum_hi, "bright-weak"
        elif div_lo >= 1.3 * div_hi:
            trust_lo, rule = True, "diversity"
        elif div_hi >= 1.3 * div_lo:
            trust_lo, rule = False, "diversity"
        else:
            continue                       # can't tell which side is real -> hands off

        # 3. Repaint the suspect member of each disagreeing pair from its mirror.
        suspect = bj if trust_lo else bi
        donor = bi if trust_lo else bj
        suspect_med = np.median(lab[suspect], axis=0)
        out[suspect] = out[donor]

        # 4. Unpaired voxels on the suspect side wearing the same (fake) colouring —
        # the hallucinated patch usually extends past where mirror partners exist —
        # follow their nearest trusted/repainted neighbour.  When the photo test made
        # the call, membership is per-voxel photo support (the patch varies more than
        # a fixed radius around its median covers); otherwise colour similarity.
        repainted = np.zeros(len(occ), dtype=bool)
        repainted[suspect] = True
        side = ~is_lo if trust_lo else is_lo
        if rule == "photo" and ptree is not None:
            w = np.array([PHOTO_L_WEIGHT, 1.0, 1.0])
            cand = side & ~repainted & (ptree.query(lab * w)[0] >= 7.5)
        else:
            cand = side & ~repainted & \
                (np.linalg.norm(lab - suspect_med, axis=1) <= 12.0)
        donors = ~side | repainted
        n_extra = int(cand.sum())
        if n_extra and donors.any():
            try:
                from scipy.spatial import cKDTree
                _, idx = cKDTree(occ[donors]).query(occ[cand])
                out[cand] = out[donors][idx]
            except Exception:
                n_extra = 0
        if os.environ.get("R2M_DEBUG"):
            print("  [fill_unobs] ax={} bad={} extra={} lumL {:.1f}/{:.1f} sup {}/{} "
                  "div {:.1f}/{:.1f} -> {} trust {}".format(
                      ax, len(bi), n_extra, lum_lo, lum_hi,
                      "-" if sup_lo is None else "{:.1f}".format(sup_lo),
                      "-" if sup_hi is None else "{:.1f}".format(sup_hi),
                      div_lo, div_hi, rule, "lo" if trust_lo else "hi"))

    return [tuple(int(c) for c in rgb) for rgb in out]


def _inpaint_dark_voxels(occ, colors, floor_frac: float = 0.35,
                         abs_floor: float = 30.0, max_frac: float = 0.40,
                         min_blob_frac: float = 0.01, min_blob_floor: int = 8):
    """Repair baked-shadow / unobserved dark BLOBS from the nearest well-lit voxel.

    TRELLIS bakes parts the photo never saw (a roof top shot from ground level,
    undersides) near-black, leaving dark blobs that _fill_unobserved_colors's
    front/back mirror can't reach — they aren't a half-space split on X/Z.  Any voxel
    whose luminance falls below max(abs_floor, floor_frac * the model's median
    luminance) takes the colour of the nearest voxel above that floor.

    Only LARGE dark components are repainted (26-connected size >=
    max(min_blob_floor, min_blob_frac * N)): bake-shadow blobs span hundreds of
    voxels, while genuine small dark detail — window frames, doors, trim — forms
    small components that must survive to the palette matcher.

    Safety gate: skips entirely when more than `max_frac` of voxels qualify — then the
    object is genuinely dark (a black car), not shadow-damaged.  Colour-only.
    """
    if not colors or len(colors) < 4:
        return colors
    occ = np.asarray(occ)
    col = np.asarray(colors, dtype=float)
    lum = col @ np.array([0.299, 0.587, 0.114])
    floor = max(abs_floor, floor_frac * float(np.median(lum)))
    dark_raw = lum < floor                 # inpaint SOURCE excludes all of these
    dark = dark_raw
    n_dark_raw = int(dark_raw.sum())
    if dark.any() and float(dark.mean()) <= max_frac:
        # Keep small dark components: genuine detail, not bake shadow.
        try:
            from scipy import ndimage
            min_blob = max(min_blob_floor, int(round(min_blob_frac * len(occ))))
            shape = tuple(int(m) + 1 for m in occ.max(axis=0))
            grid = np.zeros(shape, dtype=bool)
            didx = occ[dark]
            grid[didx[:, 0], didx[:, 1], didx[:, 2]] = True
            lab, n_comp = ndimage.label(
                grid, structure=ndimage.generate_binary_structure(3, 3))
            sizes = np.bincount(lab.ravel())
            keep_small = sizes[lab[didx[:, 0], didx[:, 1], didx[:, 2]]] < min_blob
            d2 = dark.copy()
            d2[np.flatnonzero(dark)[keep_small]] = False
            dark = d2
        except Exception:
            pass
    if os.environ.get("R2M_DEBUG"):
        print("[inpaint] floor={:.1f} dark={}/{} ({:.0%}), {} in large blobs{}".format(
            floor, n_dark_raw, len(lum), n_dark_raw / max(len(lum), 1),
            int(dark.sum()),
            "  SKIPPED (over max_frac)" if n_dark_raw / max(len(lum), 1) > max_frac
            else ""))
    if not dark.any() or n_dark_raw / max(len(lum), 1) > max_frac:
        return colors
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(occ[~dark_raw]).query(occ[dark])
        col[dark] = col[~dark_raw][idx]
    except Exception:
        return colors
    return [tuple(int(c) for c in rgb) for rgb in col]


def _normalize_tone(colors, hi_pct: float = 98.0, target_hi: int = 235,
                    max_gain: float = 2.0):
    """Lift globally-dark TRELLIS textures so colors map to sensible materials.

    TRELLIS bakes low-value textures, so a whole model's colors can sit in a dark band
    that the palette matches to near-black stone (a green tree coming out mostly
    blackstone).  Compute a single brightness gain that brings the model's bright end (the
    `hi_pct` luminance percentile) up to `target_hi`, clamped to [1, max_gain] (a model that
    is already bright is left alone, never darkened).

    Hue-preserving: instead of `col*gain` then clipping each channel at 255 (which turns a
    bright green into yellow once R and G clip), the per-voxel gain is capped so the voxel's
    BRIGHTEST channel lands at exactly 255 — the colour saturates in brightness without ever
    clipping a channel, so hue/saturation are kept (green stays green).
    """
    if not colors:
        return colors
    col = np.asarray(colors, dtype=float)
    lum = col @ np.array([0.299, 0.587, 0.114])
    p_hi = float(np.percentile(lum, hi_pct))
    if p_hi < 1e-3:
        return colors                      # essentially all black -> leave as-is
    gain = min(max_gain, max(1.0, target_hi / p_hi))
    if gain <= 1.0 + 1e-3:
        return colors                      # already bright enough
    chan_max = col.max(axis=1)             # brightest channel per voxel
    safe_gain = np.divide(255.0, chan_max, out=np.full_like(chan_max, gain),
                          where=chan_max > 0)                    # cap so no channel clips
    g = np.minimum(gain, safe_gain)[:, None]
    out = np.clip(col * g, 0, 255)         # clip is now a no-op safety net; hue preserved
    return [tuple(int(c) for c in rgb) for rgb in out]


def _lift_shadows(colors, gamma=None):
    """Brighten dark regions before block matching so they match mid-tone blocks.

    TRELLIS bakes dark, so shadowed/unlit voxels land on near-black blocks even when the
    material is a dark brown or grey.  A luminance gamma lift (gamma < 1) raises the darks
    far more than the highlights, pulling them into the palette's richer mid-tone range.
    Hue-preserving via the same brightest-channel cap as _normalize_tone (no channel
    clips, so a dark brown stays brown, just lighter).

    Runs LAST, right before matching — after the Enhanced-mode photo repairs — so it only
    changes block choice, never the calibrated photo comparisons.  `gamma` defaults to
    R2M_SHADOW_GAMMA (1.0 = off, lower = stronger lift).
    """
    if not colors:
        return colors
    if gamma is None:
        gamma = float(os.environ.get("R2M_SHADOW_GAMMA", "0.85"))
    if gamma >= 0.999:
        return colors                      # off
    col = np.asarray(colors, dtype=float)
    lum = col @ np.array([0.299, 0.587, 0.114])
    tgt = 255.0 * np.power(np.clip(lum / 255.0, 0.0, 1.0), gamma)
    lift = np.ones_like(lum)
    nz = lum > 1.0
    lift[nz] = tgt[nz] / lum[nz]
    chan_max = col.max(axis=1)             # cap so the brightest channel never clips (hue kept)
    cap = np.divide(255.0, chan_max, out=np.full_like(chan_max, np.inf), where=chan_max > 0)
    lift = np.minimum(lift, cap)[:, None]
    out = np.clip(col * lift, 0, 255)
    return [tuple(int(c) for c in rgb) for rgb in out]
