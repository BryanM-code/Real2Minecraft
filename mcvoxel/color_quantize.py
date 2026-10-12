"""Region quantization: collapse noisy per-voxel colors into flat block regions.

The last color stage before palette matching (Enhanced mode only).  Calibration
notes live in the project memory (color-repair-lessons) — the thresholds here are
measured on the cached test GLBs; don't retune blind.
"""

import os

import numpy as np


def quantize_colors(occ, colors, grid_shape,
                     k_init: int = 24,
                     merge_de: float = 12.0,
                     merge_spread_factor: float = 1.2,
                     region_divisor: int = 2000,
                     min_region_floor: int = 3,
                     absorb_de: float = 25.0,
                     absorb_always_size: int = 1,
                     shadow_chroma_max: float = 6.0,
                     shadow_l_min: float = 28.0,
                     shadow_dl_min: float = 12.0,
                     donor_chroma_min: float = 8.0,
                     shadow_contact_frac: float = 0.25,
                     shadow_max_frac: float = 0.35,
                     shadow_repaint: bool = True):
    """Collapse noisy per-voxel colors into a few flat color regions before matching.

    TRELLIS textures carry baked lighting/AO/noise, so a surface a human reads as ONE
    material is hundreds of slightly different shades; matched voxel-by-voxel they
    straddle palette boundaries and come out as random near-miss block patches.

    Stage A — global quantization: k-means in the palette's weighted-LAB match space
    (L x L_WEIGHT, so cluster boundaries align with block boundaries), then close
    centroids are agglomeratively merged.  The merge radius is spread-aware: two
    clusters merge when their distance is under max(merge_de,
    merge_spread_factor * (rms_spread_i + rms_spread_j)) capped at 2*merge_de —
    k-means loves carving ONE noisy material into sub-clusters, and those betray
    themselves by internal spread comparable to their separation, while genuinely
    distinct materials are tight relative to their distance.  Global on purpose:
    disconnected same-material surfaces (front and back wall) land in the same
    cluster and therefore the same block.  Down-weighted L + the merge radius also
    absorb baked shading gradients into the surface's one color.

    Stage A2 — shadow → albedo repaint: a cast shadow baked into the texture (roof
    overhang darkening a wall) forms its own near-neutral dark cluster; Minecraft
    re-lights builds, so that's lighting, not material.  A cluster D qualifies when
    it is near-neutral (C < shadow_chroma_max — genuine materials, even warm beige,
    carry more chroma) and not true-black (L >= shadow_l_min, protecting window
    frames/posts).  Shadows are LOCAL, so each connected COMPONENT of D picks its
    own donor: the brighter chromatic cluster (L_B - L_D >= shadow_dl_min,
    C_B >= donor_chroma_min) holding the largest share of the component's
    brighter-neighbour boundary (>= shadow_contact_frac; contacts with darker
    clusters say nothing about D's material, shadows grade into deeper shadow).
    The component's voxels take the donor's centroid — the LIT albedo — so a wall
    shadow rejoins the wall while a roof-edge shadow rejoins the roof.  If accepted
    repaints sum past shadow_max_frac of the build, everything is skipped (misfire
    guard: a legitimately dark/two-tone model, not a shadow).  Stage A2 exists for
    TRELLIS bakes, whose textures carry lighting; `shadow_repaint=False` skips it for
    sources whose colours are albedo already, where a near-neutral dark cluster (a slate
    roof, asphalt, dark trim) is a real material.

    Stage B — spatial coherence: connected components (26-conn) smaller than
    max(min_region_floor, N/region_divisor) are absorbed into their most-adjacent
    neighbouring cluster — unconditionally at size <= `absorb_always_size` (lone
    specks), otherwise only when the two centroids sit within `absorb_de`, so small
    genuine detail (a dark window on a light wall) survives.

    Runs AFTER the photo-guided repairs (their calibration needs the raw per-voxel
    variation) and only simplifies — it never re-judges hallucinations.  Fully
    deterministic (fixed k-means rng, index-ordered tie-breaks): same GLB -> same
    build.  Color-only; returns [(r,g,b)] parallel to `occ`, geometry untouched.
    """
    if not colors or len(colors) < 8:
        return colors
    import warnings
    from scipy import ndimage
    from .block_palette import L_WEIGHT, srgb_to_lab, lab_to_srgb

    col = np.clip(np.asarray(colors, dtype=np.float64), 0, 255)
    labw = srgb_to_lab(col)
    labw[:, 0] *= L_WEIGHT

    # ── Stage A: global quantization in weighted LAB ──────────────────────────
    uniq, inv = np.unique(col, axis=0, return_inverse=True)
    if len(uniq) <= k_init:                # tiny/flat builds: uniques ARE the clusters
        cent = srgb_to_lab(uniq)
        cent[:, 0] *= L_WEIGHT
        labels = inv.reshape(-1).astype(np.int64)
    else:
        from scipy.cluster.vq import kmeans2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # empty-cluster warnings are handled below
            cent, labels = kmeans2(labw, k_init, iter=20, minit="++", rng=0)
        labels = labels.astype(np.int64)
        keep = np.bincount(labels, minlength=len(cent)) > 0
        cent = cent[keep]
        labels = (np.cumsum(keep) - 1)[labels]
    sizes = np.bincount(labels, minlength=len(cent)).astype(np.float64)
    k_start = len(cent)

    # Per-cluster scatter about the centroid, kept as sum-of-squares so merges can
    # combine it exactly (parallel-axis theorem); rms spread = sqrt(ssq/size).
    diffs = labw - cent[labels]
    ssq = np.bincount(labels, weights=np.einsum("ij,ij->i", diffs, diffs),
                      minlength=len(cent))

    # Merge closest pair (relative to its threshold) until none qualifies.
    cap = 2.0 * merge_de
    while len(cent) > 1:
        d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
        spread = np.sqrt(ssq / np.maximum(sizes, 1.0))
        thr = np.clip(merge_spread_factor * (spread[:, None] + spread[None, :]),
                      merge_de, cap)
        ratio = d / thr
        ratio[np.diag_indices(len(cent))] = np.inf
        i, j = np.unravel_index(int(np.argmin(ratio)), ratio.shape)
        if ratio[i, j] >= 1.0:
            break
        if i > j:
            i, j = j, i                    # keep the lower index (determinism)
        merged = (cent[i] * sizes[i] + cent[j] * sizes[j]) / (sizes[i] + sizes[j])
        ssq[i] = (ssq[i] + sizes[i] * float(np.sum((cent[i] - merged) ** 2))
                  + ssq[j] + sizes[j] * float(np.sum((cent[j] - merged) ** 2)))
        cent[i] = merged
        sizes[i] += sizes[j]
        cent = np.delete(cent, j, axis=0)
        sizes = np.delete(sizes, j)
        ssq = np.delete(ssq, j)
        labels = np.where(labels == j, i, labels)
        labels = np.where(labels > j, labels - 1, labels)

    occ_a = np.asarray(occ)
    n_vox = len(occ_a)
    min_region = max(min_region_floor, int(round(n_vox / region_divisor)))
    struct = ndimage.generate_binary_structure(3, 3)   # 26-connectivity
    X, Y, Z = grid_shape

    def _sl(d, n):                         # (source, neighbour) slices for offset d
        return (slice(0, n - d), slice(d, n)) if d >= 0 else (slice(-d, n), slice(0, n + d))

    dbg = bool(os.environ.get("R2M_DEBUG"))

    # ── Stage A2: shadow → albedo repaint (per connected shadow patch) ────────
    K = len(cent)
    if shadow_repaint and K >= 2:
        Lu = cent[:, 0] / L_WEIGHT                   # unweighted lightness
        Cc = np.linalg.norm(cent[:, 1:], axis=1)     # chroma
        grid = np.full(grid_shape, -1, dtype=np.int64)
        grid[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]] = labels
        shadowy = np.where((Cc < shadow_chroma_max) & (Lu >= shadow_l_min))[0]
        new_lab = labels.copy()
        n_repaint = 0
        for D in shadowy:
            D = int(D)
            donor_ok = (Lu - Lu[D] >= shadow_dl_min) & (Cc >= donor_chroma_min)
            donor_ok[shadowy] = False                # shadow clusters can't donate
            if not donor_ok.any():
                continue
            comp, n_comp = ndimage.label(grid == D, structure=struct)
            if n_comp == 0:
                continue
            # Per component: contact counts with each neighbouring cluster.
            pair_keys = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if (dx, dy, dz) == (0, 0, 0):
                            continue
                        (s0, t0), (s1, t1), (s2, t2) = (_sl(dx, X), _sl(dy, Y),
                                                        _sl(dz, Z))
                        a = comp[s0, s1, s2].ravel()     # component id (1..n_comp)
                        b = grid[t0, t1, t2].ravel()     # neighbour cluster id
                        m = (a > 0) & (b >= 0) & (b != D)
                        if m.any():
                            pair_keys.append((a[m] - 1) * K + b[m])
            if not pair_keys:
                continue
            keys, counts = np.unique(np.concatenate(pair_keys), return_counts=True)
            cids, ncl = keys // K, keys % K
            brighter = Lu[ncl] > Lu[D]               # denominator: brighter contacts
            bright_tot = np.bincount(cids[brighter], weights=counts[brighter],
                                     minlength=n_comp)
            ok = donor_ok[ncl]
            # Best qualifying donor per component (max contact; ties -> low cluster).
            order = np.lexsort((ncl[ok], -counts[ok], cids[ok]))
            c_ok, n_ok, cnt_ok = cids[ok][order], ncl[ok][order], counts[ok][order]
            first = np.ones(len(order), dtype=bool)
            first[1:] = c_ok[1:] != c_ok[:-1]
            tgt_comp, tgt_donor, tgt_cnt = c_ok[first], n_ok[first], cnt_ok[first]
            frac = tgt_cnt / np.maximum(bright_tot[tgt_comp], 1)
            take = frac >= shadow_contact_frac
            if not take.any():
                if dbg:
                    print("[shadow] D#{} L={:.1f} C={:.1f}: no component passed "
                          "contact {:.2f}".format(D, Lu[D], Cc[D],
                                                  shadow_contact_frac))
                continue
            comp_to_donor = np.full(n_comp + 1, -1, dtype=np.int64)
            comp_to_donor[tgt_comp[take] + 1] = tgt_donor[take]
            vox_comp = comp[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]]
            donor_lab = comp_to_donor[vox_comp]
            upd = donor_lab >= 0
            new_lab = np.where(upd, donor_lab, new_lab)
            n_repaint += int(upd.sum())
            if dbg:
                for c, dn, fr in zip(tgt_comp[take], tgt_donor[take], frac[take]):
                    print("[shadow] D#{} L={:.1f} C={:.1f} comp {} ({} vox) -> "
                          "B#{} L={:.1f} C={:.1f} (contact {:.2f})".format(
                              D, Lu[D], Cc[D], int(c) + 1,
                              int((vox_comp == c + 1).sum()), int(dn),
                              Lu[dn], Cc[dn], fr))
        if n_repaint:
            if n_repaint > shadow_max_frac * n_vox:
                if dbg:
                    print("[shadow] skipped: {:.0%} of voxels would repaint "
                          "(> {:.0%} guard) — treating as real dark material".format(
                              n_repaint / max(n_vox, 1), shadow_max_frac))
            else:
                labels = new_lab

    # ── Stage B: absorb small spatial fragments ───────────────────────────────
    absorbed = []
    for _pass in range(2):
        K = len(cent)
        grid = np.full(grid_shape, -1, dtype=np.int64)
        grid[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]] = labels

        # Connected components per cluster, packed into one global component grid.
        comp = np.full(grid_shape, -1, dtype=np.int64)
        comp_cluster, comp_sizes, n_comp = [], [], 0
        for k in range(K):
            lab_k, n_k = ndimage.label(grid == k, structure=struct)
            if n_k == 0:
                continue
            mask = lab_k > 0
            comp[mask] = lab_k[mask] - 1 + n_comp
            comp_sizes.extend(np.bincount(lab_k[mask])[1:].tolist())
            comp_cluster.extend([k] * n_k)
            n_comp += n_k
        comp_sizes = np.asarray(comp_sizes, dtype=np.int64)
        comp_cluster = np.asarray(comp_cluster, dtype=np.int64)
        small = comp_sizes < min_region
        if not small.any():
            break

        # (small component, adjacent other-cluster) contact counts over all 26 offsets.
        pair_keys = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    (s0, t0), (s1, t1), (s2, t2) = _sl(dx, X), _sl(dy, Y), _sl(dz, Z)
                    a = comp[s0, s1, s2].ravel()
                    b = grid[t0, t1, t2].ravel()
                    m = (a >= 0) & (b >= 0)
                    a, b = a[m], b[m]
                    m = small[a] & (comp_cluster[a] != b)
                    if m.any():
                        pair_keys.append(a[m] * K + b[m])
        if not pair_keys:
            break
        keys, counts = np.unique(np.concatenate(pair_keys), return_counts=True)
        comp_ids, neigh = keys // K, keys % K
        # Modal neighbour per component; ties -> lower cluster index (determinism).
        order = np.lexsort((neigh, -counts, comp_ids))
        first = np.ones(len(order), dtype=bool)
        first[1:] = comp_ids[order][1:] != comp_ids[order][:-1]
        tgt_comp, tgt_cluster = comp_ids[order][first], neigh[order][first]

        # Absorb: lone specks always; bigger fragments only across a small color gap.
        d_cent = np.linalg.norm(cent[comp_cluster[tgt_comp]] - cent[tgt_cluster], axis=1)
        allow = (comp_sizes[tgt_comp] <= absorb_always_size) | (d_cent < absorb_de)
        tgt_comp, tgt_cluster = tgt_comp[allow], tgt_cluster[allow]
        if len(tgt_comp) == 0:
            break
        comp_to_new = np.full(n_comp, -1, dtype=np.int64)
        comp_to_new[tgt_comp] = tgt_cluster
        new_lab = comp_to_new[comp[occ_a[:, 0], occ_a[:, 1], occ_a[:, 2]]]
        upd = new_lab >= 0
        if not upd.any():
            break
        labels = np.where(upd, new_lab, labels)
        absorbed.append(int(upd.sum()))

    if os.environ.get("R2M_DEBUG"):
        top = ", ".join("{}".format(int(s))
                        for s in sorted(np.bincount(labels, minlength=len(cent)),
                                        reverse=True)[:8])
        print("[quantize] {} voxels: {} clusters -> {} after merge_de={} "
              "(top sizes {}); min_region={}, absorbed/pass={}".format(
                  n_vox, k_start, len(cent), merge_de, top, min_region,
                  absorbed or [0]))

    cent_lab = cent.copy()
    cent_lab[:, 0] /= L_WEIGHT
    cent_rgb = np.clip(np.rint(lab_to_srgb(cent_lab)), 0, 255).astype(np.int64)
    return [tuple(int(c) for c in cent_rgb[l]) for l in labels]


def consolidate_block_ids(colors, block_ids, max_de: float = 10.0):
    """Post-match cleanup: merge block ids of clusters that landed on near-identical
    MATERIALS (distance in the weighted-LAB match space).

    Quantization leaves only a handful of distinct colors, but two surviving clusters
    can still match palette blocks that read as the same material (stone vs
    cobblestone, ~4 apart) — a fake material boundary.  Working per unique color
    (cheap: quantization guarantees few), the smaller cluster adopts the bigger
    cluster's id whenever their MATCHED palette colors sit within `max_de`.  Global
    and deterministic — same fix on every surface, unlike a per-voxel majority vote.
    """
    if not colors or not block_ids:
        return block_ids
    from .block_palette import BLOCK_PALETTE, match_space

    col = np.clip(np.asarray(colors, dtype=np.float64), 0, 255)
    uniq, first, inv, counts = np.unique(
        col, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    if len(uniq) <= 1:
        return block_ids
    final = [block_ids[i] for i in first]      # one id per unique color (same-color
                                               # voxels always match the same block)
    pal = {bid: match_space(np.asarray(BLOCK_PALETTE[bid], dtype=np.float64))
           for bid in set(final) if bid in BLOCK_PALETTE}
    order = np.lexsort((np.arange(len(uniq)), -counts))   # biggest cluster first
    for pos in range(1, len(order)):
        i = int(order[pos])
        for pos2 in range(pos):
            j = int(order[pos2])
            if final[i] == final[j]:
                break
            if final[i] in pal and final[j] in pal and \
                    float(np.linalg.norm(pal[final[i]] - pal[final[j]])) <= max_de:
                final[i] = final[j]
                break
    return [final[u] for u in inv]
