"""Per-voxel color sampling from the mesh's texture (or baked vertex colors).

The texture path is the primary one: mode-seeking, side-aware surface splatting that
keeps small high-contrast detail (windows, trim) while suppressing baked-lighting
artifacts.  The vertex-color path is the fallback for texture-less sources.
"""

import os

import numpy as np
import trimesh


_MAX_BATCH_SAMPLES = 2_000_000


def _bilinear_sample(image, uv):
    """Bilinear-sample an (H,W,3) uint8 image at (N,2) glTF UVs.

    glTF convention: u wraps, v has its origin at the bottom-left (so image row 0 is
    v=1).  Returns (N,3) float64 colors.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float64)
    u = np.mod(uv[:, 0], 1.0) * (w - 1)
    v = (1.0 - np.mod(uv[:, 1], 1.0)) * (h - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
            img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def _sample_voxel_colors_texture(occ, origin, grid_shape, verts, faces, uv, image,
                                 vert_thresh: float = 0.35,
                                 conflict_frac: float = 0.20,
                                 conflict_min: int = 2,
                                 mode_radius: float = 40.0):
    """RGB per voxel: mode-seeking mean of the texture over the voxel's surface patch.

    Surface splatting with supersampling: every triangle is covered with barycentric
    sample points at ~1/3-voxel spacing, each sample looks up the texture at its
    interpolated UV (bilinear).  The voxel's color is the mean of ONLY the samples
    within `mode_radius` (RGB) of its median-luminance sample — i.e. the majority
    material's samples.  A plain mean would average a bimodal voxel (black window
    frame + cream wall, dark ridge trim + brown shingles) into a mid-grey that
    exists nowhere in the texture and reads as a different material downstream; a
    plain median keeps raw texture noise that inflates cluster spreads.  The
    mode-seeking mean does both jobs: boundary voxels snap to the majority material
    (windows/doors/trim survive) while within-material noise still averages out.

    Side-aware: samples are bucketed by face orientation (UP / DOWN / SIDE at
    |n_y| > `vert_thresh`).  A voxel that collects meaningful samples from BOTH the
    top and bottom of a thin slab (a roof ~1 voxel thick: bright shingles above,
    baked-shadow underside below) keeps only the brighter vertical side — the lit
    albedo — instead of averaging them into a muddy grey that then reads as a
    different material.  Minecraft re-lights the build, so the baked-shadow side is
    lighting, not material.  Walls (all SIDE) and thick surfaces are unaffected.

    Voxels no surface sample lands in (flood-filled interiors) fall back to a
    closest-point texture lookup.  `verts` are in the scaled voxel frame (axis 1 =
    up); a point p lies in voxel floor(p) - origin.  Returns [(r,g,b)] aligned with
    `occ`, or None if sampling fails (caller then uses the vertex-color path).

    A single-texture front end to `sample_voxel_surface`, which does the work for any
    colour source (several textures, base colours, vertex colours).
    """
    try:
        tri_uv = uv[faces]                                          # (F,3,2)

        def group_colors(sel, bary):
            uvs = np.einsum("mk,tkd->tmd", bary, tri_uv[sel]).reshape(-1, 2)
            return _bilinear_sample(image, uvs)

        def point_colors(tri_id, bary):
            uv_i = (bary[:, :, None] * uv[faces[tri_id]]).sum(axis=1)
            return _bilinear_sample(image, uv_i)

        res = sample_voxel_surface(occ, origin, grid_shape, verts, faces,
                                   group_colors, point_colors,
                                   vert_thresh=vert_thresh, conflict_frac=conflict_frac,
                                   conflict_min=conflict_min, mode_radius=mode_radius)
        return [tuple(int(c) for c in rgb) for rgb in res["rgb"]]
    except Exception:
        return None


def sample_voxel_surface(occ, origin, grid_shape, verts, faces, group_colors, point_colors,
                         vert_thresh: float = 0.35,
                         conflict_frac: float = 0.20,
                         conflict_min: int = 2,
                         mode_radius: float = 40.0,
                         face_labels=None, n_labels: int = 0):
    """Mode-seeking, side-aware per-voxel colour from ANY colour source.

    `group_colors(sel, bary)` returns (T*M, 3) float colours for the triangles selected by
    boolean mask `sel`, sampled at the M shared barycentric points `bary` (tri-major, the
    same layout as `np.einsum("mk,tkd->tmd", bary, tri[sel]).reshape(-1, d)`).
    `point_colors(tri_id, bary)` returns (K,3) colours at one barycentric point per
    triangle -- used only for voxels no sample landed in. Pass None to skip that
    closest-point search and leave them black (the result's "sampled" mask says which).

    `face_labels` (F,) ints in [0, n_labels) are tallied per voxel over every sample
    that lands in it; the result's "label_frac" is (N, n_labels). Used to carry material
    properties (glass, door, ...) from faces to voxels.

    Accumulates per OCCUPIED voxel only. Luminance sums stay float32 and are added in
    the original sample order, so results are bit-identical to the pre-2026-09 sampler,
    which kept dense per-cell arrays (~48 bytes a cell: a 26M-cell tower's bounding box
    needed over a gigabyte for voxels that were never read).
    """
    occ = np.asarray(occ)
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    tri_v = verts[faces]                                            # (F,3,3)

    # Face up-components for the orientation buckets.  Guard against globally inverted
    # winding via the signed volume (sum dot(v0, e1 x e2) = 6V; negative -> flipped).
    fn = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    nrm = np.linalg.norm(fn, axis=1)
    ny = np.divide(fn[:, 1], nrm, out=np.zeros_like(nrm), where=nrm > 1e-12)
    if float(np.einsum("ij,ij->i", fn, tri_v[:, 0]).sum()) < 0.0:
        ny = -ny
    tri_bucket = np.where(ny > vert_thresh, 0,                      # 0=UP
                          np.where(ny < -vert_thresh, 1, 2))        # 1=DOWN 2=SIDE

    edges = np.linalg.norm(tri_v - np.roll(tri_v, 1, axis=1), axis=2)
    n_div = np.clip(np.ceil(edges.max(axis=1) * 3).astype(np.int64), 1, 96)

    n_occ = len(occ)
    ncells = int(np.prod(grid_shape))
    hi = np.asarray(grid_shape, dtype=np.int64) - 1
    sums = np.zeros((3 * n_occ, 3), dtype=np.float32)               # per (voxel, bucket)
    counts = np.zeros(3 * n_occ, dtype=np.int32)
    labels = np.zeros((n_occ, max(1, n_labels)), dtype=np.int32) if face_labels is not None else None

    # Row lookup: linear cell -> occ row (-1 = unoccupied); samples outside occupied
    # voxels are never read back, so they are dropped at the door.
    lin_occ = np.ravel_multi_index((occ[:, 0], occ[:, 1], occ[:, 2]), grid_shape)
    voxrow = np.full(ncells, -1, dtype=np.int32)
    voxrow[lin_occ] = np.arange(n_occ, dtype=np.int32)
    parts_row, parts_sb, parts_col = [], [], []

    # Batch triangles by subdivision count so each group splats in one shot -- in chunks
    # of at most ~_MAX_BATCH_SAMPLES samples, so a few thousand large triangles cannot ask
    # for gigabytes at once. Chunks run in triangle order, so the sample sequence (and the
    # float32 sums built from it) is exactly what one unchunked pass would produce.
    batches = []
    for n in np.unique(n_div):
        idx = np.flatnonzero(n_div == n)
        per = max(1, _MAX_BATCH_SAMPLES // ((n + 1) * (n + 2) // 2))
        for c in range(0, len(idx), per):
            batches.append((n, idx[c:c + per]))
    for n, tri_idx in batches:
        sel = np.zeros(len(faces), dtype=bool)
        sel[tri_idx] = True
        gi, gj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (gi + gj) <= n
        b0 = gi[keep] / n
        b1 = gj[keep] / n
        bary = np.stack([b0, b1, 1.0 - b0 - b1], axis=1)            # (M,3)
        pos = np.einsum("mk,tkd->tmd", bary, tri_v[sel]).reshape(-1, 3)
        cols = group_colors(sel, bary)
        vidx = np.clip(np.floor(pos).astype(np.int64) - origin, 0, hi)
        lin = np.ravel_multi_index((vidx[:, 0], vidx[:, 1], vidx[:, 2]), grid_shape)
        sb = np.repeat(tri_bucket[sel], len(b0))        # sample bucket (tri-major)
        row = voxrow[lin]
        m = row >= 0
        np.add.at(sums, row[m] * 3 + sb[m], cols[m])
        np.add.at(counts, row[m] * 3 + sb[m], 1)
        if labels is not None:
            lab = np.repeat(np.asarray(face_labels)[sel], len(b0))
            np.add.at(labels, (row[m], lab[m]), 1)
        parts_row.append(row[m])
        parts_sb.append(sb[m].astype(np.uint8))
        parts_col.append(np.clip(np.rint(cols[m]), 0, 255).astype(np.uint8))

    s_up = sums[0::3].astype(np.float64)
    s_dn = sums[1::3].astype(np.float64)
    c_up = counts[0::3].astype(np.float64)
    c_dn = counts[1::3].astype(np.float64)
    c_sd = counts[2::3].astype(np.float64)
    total = c_up + c_dn + c_sd

    wl = np.array([0.299, 0.587, 0.114])
    lum_up = np.divide(s_up @ wl, c_up, out=np.zeros_like(c_up), where=c_up > 0)
    lum_dn = np.divide(s_dn @ wl, c_dn, out=np.zeros_like(c_dn), where=c_dn > 0)
    # Top/bottom conflict -> keep the brighter vertical side (ties keep UP).
    conflict = np.minimum(c_up, c_dn) >= np.maximum(conflict_min, conflict_frac * total)
    drop_up = conflict & (lum_up < lum_dn)
    drop_dn = conflict & ~drop_up
    if os.environ.get("R2M_DEBUG"):
        print("[sample] {} voxels, {} top/bottom conflicts "
              "(dropped underside {}, top {})".format(
                  n_occ, int(conflict.sum()), int(drop_dn.sum()), int(drop_up.sum())))

    # Mode-seeking mean per voxel: find the median-by-luminance sample (the majority
    # material — a real texture color, never a blend), then average only the samples
    # within mode_radius of it.  Boundary voxels snap to the majority material;
    # within-material noise still averages out.
    out = np.zeros((n_occ, 3), dtype=np.float64)
    have = np.zeros(n_occ, dtype=bool)
    if parts_row:
        s_row = np.concatenate(parts_row)
        s_sb = np.concatenate(parts_sb)
        s_col = np.concatenate(parts_col)
        kept = ~((drop_up[s_row] & (s_sb == 0)) | (drop_dn[s_row] & (s_sb == 1)))
        s_row, s_col = s_row[kept], s_col[kept]
        if len(s_row):
            s_lum = s_col.astype(np.float32) @ wl.astype(np.float32)
            order = np.lexsort((s_lum, s_row))   # stable: same input, same pick
            r_sorted = s_row[order]
            starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
            seg_len = np.diff(np.r_[starts, len(r_sorted)])
            med = order[starts + (seg_len - 1) // 2]   # lower median: real sample
            rows = r_sorted[starts]
            med_by_row = np.zeros((n_occ, 3), dtype=np.float32)
            med_by_row[rows] = s_col[med].astype(np.float32)
            near = (np.linalg.norm(s_col.astype(np.float32) - med_by_row[s_row], axis=1)
                    < mode_radius)
            acc = np.zeros((n_occ, 3), dtype=np.float64)
            cnt = np.zeros(n_occ, dtype=np.int64)
            np.add.at(acc, s_row[near], s_col[near].astype(np.float64))
            np.add.at(cnt, s_row[near], 1)
            have = cnt > 0                 # every sampled voxel keeps >= its median
            out[have] = acc[have] / cnt[have, None]

    missing = ~have
    if missing.any() and point_colors is not None:
        cmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        centers = occ[missing] + 0.5 + origin
        closest, _, tri_id = trimesh.proximity.closest_point(cmesh, centers)
        bary = trimesh.triangles.points_to_barycentric(cmesh.triangles[tri_id], closest)
        bary = np.nan_to_num(bary, nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
        out[missing] = point_colors(tri_id, bary)

    rgb = np.clip(np.rint(np.nan_to_num(out, nan=0.0)), 0, 255).astype(np.int16)
    result = {"rgb": rgb, "sampled": have, "label_frac": None}
    if labels is not None:
        tot = labels.sum(axis=1, keepdims=True)
        result["label_frac"] = np.divide(labels, np.maximum(tot, 1), dtype=np.float32)
    return result


def _sample_voxel_colors(pts, verts, faces, vcolors):
    """RGB color per voxel center, sampled from the colored mesh surface.

    For each voxel center, find the closest point on the mesh surface and
    barycentric-blend the three vertex colors of that triangle.  This is more
    accurate than nearest-vertex (which can grab a color from the wrong side of
    a thin feature).  Falls back to nearest-vertex, then to None.

    `verts`/`faces`/`vcolors` describe the colored mesh already scaled into the
    voxel frame so it lines up with `pts`.  Returns a list of (r,g,b) ints in the
    same order as `pts`, or None.
    """
    if vcolors is None:
        return None
    try:
        cmesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        closest, _, tri_id = trimesh.proximity.closest_point(cmesh, pts)
        tris = cmesh.triangles[tri_id]                                   # (K,3,3)
        bary = trimesh.triangles.points_to_barycentric(tris, closest)    # (K,3)
        tri_cols = vcolors[faces[tri_id]].astype(np.float64)             # (K,3,3)
        blended = (bary[:, :, None] * tri_cols).sum(axis=1)              # (K,3)
        # Degenerate triangles give NaN/inf barycentric coords; sanitize before the int
        # cast, otherwise NaN->int yields a garbage huge value (breaks downstream int casts).
        blended = np.nan_to_num(blended, nan=0.0, posinf=255.0, neginf=0.0)
        blended = np.clip(np.rint(blended), 0, 255).astype(np.int16)
        return [tuple(int(c) for c in rgb) for rgb in blended]
    except Exception:
        pass
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(verts).query(pts)
        return [tuple(int(c) for c in rgb) for rgb in vcolors[idx]]
    except Exception:
        return None
