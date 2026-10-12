"""Source-evidenced architecture decisions shared by dataset and application callers.

No asset names, category templates or metre assumptions enter these functions.
Geometry is never normalized to a common building size. Low confidence preserves
the mesh; all distances here are in the caller's block frame unless documented.
"""
from __future__ import annotations

import numpy as np

REPRESENTATION_VERSION = 'architecture-v1'


def axis_planes(verts, faces, tolerance=0.025):
    """Area-weighted nearly axis-aligned source planes, in input coordinates.

Used for phase/resolution evidence, not to reconstruct rectangular footprints.
Normals must lie within ~4 degrees of an axis. Merging tolerance is positional.
"""
    tri = np.asarray(verts, float)[np.asarray(faces)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = np.abs(cross) / np.maximum(2 * area[:, None], 1e-20)
    centre = tri.mean(axis=1)
    answer = []
    for ax in range(3):
        ids = np.flatnonzero((normal[:, ax] > .997) & (area > 0))
        ids = ids[np.argsort(centre[ids, ax], kind='stable')]
        groups = []
        for i in ids:
            pos, weight = float(centre[i, ax]), float(area[i])
            if groups and abs(pos - groups[-1][0] / groups[-1][1]) <= tolerance:
                groups[-1][0] += pos * weight; groups[-1][1] += weight; groups[-1][2] += 1
            else:
                groups.append([pos * weight, weight, 1])
        answer.append([(s / w, w, n) for s, w, n in groups])
    return answer


def align_grid(verts, faces, blocks_per_unit):
    """Choose one deterministic X/Z phase from broad source wall planes.

Weighted distances to block boundaries are minimized; no per-plane movement,
axis rescaling or grid rotations occur. Ground remains exactly at Y=0.
"""
    v = np.asarray(verts, float) * blocks_per_unit
    lo = v.min(axis=0)
    planes = axis_planes(v - lo, faces)
    phase = np.zeros(3)
    evidence = []
    candidates = np.arange(128, dtype=float) / 128
    for ax in (0, 2):
        rows = np.asarray(planes[ax], float)
        if not len(rows):
            evidence.append({'axis': ax, 'planes': 0, 'confidence': 'none'})
            continue
        rows = rows[rows[:, 1] >= max(1., rows[:, 1].max() * .015)]
        if not len(rows):
            evidence.append({'axis': ax, 'planes': 0, 'confidence': 'none'})
            continue
        pos, weight = rows[:, 0], rows[:, 1]
        delta = pos[:, None] + candidates
        error = np.abs(delta - np.round(delta))
        loss = np.average(error ** 2, axis=0, weights=weight)
        phase[ax] = candidates[int(np.argmin(loss))]
        evidence.append({'axis': ax, 'planes': len(rows), 'confidence': 'geometric',
                         'weighted_boundary_rms': float(np.sqrt(loss.min())),
                         'zero_phase_rms': float(np.sqrt(loss[0]))})
    return -lo + phase, {'policy': 'area_weighted_wall_phase_v1', 'phase': phase.tolist(), 'evidence': evidence}


def resolution_evidence(verts, faces, physical_bpu, target_feature_blocks=2.):
    """Suggest enlargement using *repeated* source plane spacings.

Measurements are in physical-baseline blocks. Single tiny bevels cannot request
arbitrary enlargement. Geometry cannot resolve features painted into a texture:
these remain explicitly unmeasured, even when the geometric estimate is confident.
"""
    v = np.asarray(verts, float) * physical_bpu
    height = float(np.ptp(v[:, 1]))
    groups = axis_planes(v - v.min(0), faces)
    evidence = []
    for ax, rows in enumerate(groups):
        rows = np.asarray(rows, float)
        if len(rows) < 4:
            continue
        rows = rows[rows[:, 1] >= max(.2, rows[:, 1].max() * .08)]
        if len(rows) < 4:
            continue
        # Nearby parallel surfaces commonly describe the two sides of a beam or
        # trim, not the width of an opening. Cluster those before measuring repeats.
        clusters = []
        merge = max(.12, height * .015)
        for pos, weight, _ in rows:
            if clusters and pos - clusters[-1][2] < merge:
                clusters[-1][0] += pos*weight; clusters[-1][1] += weight; clusters[-1][2] = pos
            else:
                clusters.append([pos*weight, weight, pos])
        gaps = np.diff([s/w for s,w,_ in clusters])
        gaps = gaps[(gaps >= max(.2, height * .02)) & (gaps <= height * .3)]
        for g in gaps:
            repeated = gaps[np.abs(gaps - g) <= max(.04, g * .12)]
            if len(repeated) >= 3:
                evidence.append({'axis': ax, 'spacing': float(np.median(repeated)), 'repeats': len(repeated)})
    # Deduplicate modes before choosing a conservative lower quartile spacing.
    unique = {}
    for e in evidence:
        key = (e['axis'], round(e['spacing'], 2))
        unique[key] = e
    evidence = list(unique.values())
    feature = float(np.quantile([e['spacing'] for e in evidence], .25)) if evidence else None
    enlargement = max(1., target_feature_blocks / feature) if feature else 1.
    return {'version': REPRESENTATION_VERSION, 'feature_spacing_baseline_blocks': feature,
            'requested_enlargement': enlargement, 'feature_target_blocks': target_feature_blocks,
            'confidence': 'advisory' if evidence else 'unmeasured', 'evidence': evidence,
            'automatic_enlargement_approved': False,
            'texture_features': 'unmeasured', 'physical_size_is_resolution': False}


def enclosed_cross_sections(occupied):
    """Fill only roof-covered, closed horizontal cross sections.

    A floor and three walls are insufficient evidence: open fronts, arcades and
    breezeways stay open. Closed surfaces still use the ordinary 3-D flood fill.
    """
    from scipy import ndimage
    above = np.maximum.accumulate(occupied[:, ::-1, :], axis=1)[:, ::-1, :]
    added = np.zeros_like(occupied)
    for y in range(occupied.shape[1]):
        added[:, y, :] = ndimage.binary_fill_holes(occupied[:, y, :]) & ~occupied[:, y, :] & above[:, y, :]
    return added


def coherent_wall_shapes(solid, shapes, verts, faces, origin):
    """Continuous vertical mesh surfaces use full wall courses, including curves.

    Occupancy, openings and profiles are unchanged. A cell must be backed by a
    vertical source face, continue both above and below and have no inclined or
    horizontal mesh face. Tops, window edges, ledges and stair runs stay geometric.
    """
    from .raster import rasterize_surface
    from .block_shapes import SHAPE_FULL
    tri = np.asarray(verts)[np.asarray(faces)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vertical = np.abs(n[:, 1]) <= .04 * np.linalg.norm(n, axis=1)
    walls, _ = rasterize_surface(verts, np.asarray(faces)[vertical], origin=origin, shape=solid.shape)
    other, _ = rasterize_surface(verts, np.asarray(faces)[~vertical], origin=origin, shape=solid.shape)
    supported = np.zeros_like(solid)
    supported[:, 1:-1, :] = solid[:, :-2, :] & solid[:, 2:, :]
    mask = walls & ~other & supported & solid & (shapes > SHAPE_FULL)
    out = shapes.copy(); out[mask] = SHAPE_FULL
    return out, {'vertical_surface_full_cells': int(mask.sum()), 'occupancy_changed': 0,
                 'method': 'source_vertical_faces_and_vertical_continuity'}


def material_families(occ, ids, colours, shapes, solid_shape, protected, allowed=None):
    """One compatible family per connected quantized material region.

    Existing colour boundaries, glass and doors remain boundaries. A few stray
    partial cells cannot recolour a large facade; at least three partial cells and
    12% partial coverage are needed. Roof groups are additionally source conditioned.
    """
    from scipy import ndimage
    from . import block_shapes as bs
    from .block_palette import match_blocks_restricted
    out = list(ids)
    changed = regions = rejected = 0
    index = np.full(solid_shape, -1, np.int32); index[tuple(occ.T)] = np.arange(len(occ))
    for base in sorted(set(ids)):
        rows = np.flatnonzero((np.asarray(ids) == base) & ~protected)
        mask = np.zeros(solid_shape, bool); mask[tuple(occ[rows].T)] = True
        labels, _ = ndimage.label(mask)
        for label, box in enumerate(ndimage.find_objects(labels), 1):
            if box is None:
                continue
            rr = index[box][labels[box] == label]
            partial = shapes[rr] > bs.SHAPE_FULL
            if partial.sum() < 3 or partial.mean() < .12:
                continue
            stair = np.isin(shapes[rr], list(bs._FACING)).any()
            pool = bs._STAIRS_CAPABLE if stair else bs._SLAB_CAPABLE
            if allowed is not None:
                pool = [p for p in pool if p in allowed]
            if not pool:
                continue
            match, distance = match_blocks_restricted([np.median(np.asarray(colours)[rr], axis=0)], pool)
            # Large colour shifts are not justified merely by shape compatibility.
            if base not in pool and distance[0] > 8.:
                rejected += 1
                continue
            family = base if base in pool else match[0]
            regions += 1
            for i in rr:
                changed += int(out[i] != family); out[i] = family
    return out, {'regions': regions, 'cells_changed': changed, 'colour_shift_rejected': rejected,
                 'policy': 'connected_material_family_v1'}
