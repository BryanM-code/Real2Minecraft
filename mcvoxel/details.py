"""Conservative structural detail evidence, before colour-to-block assignment.

Only vertically continuous, narrow, anchored elements separated from wall mass qualify.
Material compatibility is a separate decision in assign; geometry never invents wood.
"""
from __future__ import annotations

import re
import numpy as np


def structural_posts(solid, frac, layers, scale, scene, verts, origin):
    from .grid import _layer_tables, _neighbour

    count, low, high = _layer_tables(scale)
    widths = count[layers]
    narrow = (solid & (widths[..., 0] <= scale // 2) & (widths[..., 2] <= scale // 2)
              & (widths[..., 1] >= scale - 1) & (frac <= 0.36))
    # A thin corner of a facade is attached to broad wall mass at the same height.
    wall = solid & ~narrow
    adjacent = np.zeros_like(solid)
    for axis in (0, 2):
        for step in (-1, 1):
            adjacent |= _neighbour(wall, axis, step, False)
    candidate = narrow & ~adjacent
    # A sub-block post straddling a coarse boundary can occupy 2x2 cells. Consolidate
    # that single fine cross-section before choosing fence/wall shapes, otherwise one
    # slender source post becomes four connected Minecraft posts.
    from scipy import ndimage
    components, ncomp = ndimage.label(candidate)
    duplicates = np.zeros_like(solid)
    for label in range(1, ncomp + 1):
        cells = np.argwhere(components == label)
        if len(cells) < 2 or np.ptp(cells[:, 1]) < 1:
            continue
        if np.ptp(cells[:, 0]) > 1 or np.ptp(cells[:, 2]) > 1:
            continue
        lay = layers[tuple(cells.T)]
        lo = cells + low[lay] / float(scale)
        hi = cells + (high[lay] + 1) / float(scale)
        span = hi.max(axis=0) - lo.min(axis=0)
        if span[0] > .8 or span[2] > .8:
            continue
        centre = (hi.max(axis=0) + lo.min(axis=0)) / 2
        x, z = np.floor(centre[[0, 2]] - 1e-7).astype(int)
        drop = cells[(cells[:, 0] != x) | (cells[:, 2] != z)]
        duplicates[tuple(drop.T)] = True
    candidate &= ~duplicates
    posts = np.zeros_like(solid)
    parapets = np.zeros_like(solid)
    runs = 0
    for x, z in np.argwhere(candidate.any(axis=1)):
        ys = np.flatnonzero(candidate[x, :, z])
        for run in np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1):
            if len(run) < 2:
                continue
            a, b = int(run[0]), int(run[-1])
            def broad(y):
                return (0 <= y < solid.shape[1] and solid[x,y,z] and
                        (widths[x,y,z,0] > scale//2 or widths[x,y,z,2] > scale//2))
            base = a == 0 or (broad(a-1) and high[layers[x,a-1,z,1]] >= scale-2)
            overhead = broad(b+1) and low[layers[x,b+1,z,1]] <= 1
            if base and overhead:
                posts[x, run, z] = True
                runs += 1
    # One/two-block-high raised rails/parapets need explicit architectural names as
    # well as thin sheet evidence. This avoids promoting generic facade/roof edges.
    for normal,tangent in ((0,2),(2,0)):
        thin = (solid & (widths[...,normal]<=scale//2) & (widths[...,tangent]>=scale-1)
                & (widths[...,1]>=scale-1) & (frac<=.5))
        thin &= ~_neighbour(solid,normal,-1,False) & ~_neighbour(solid,normal,1,False)
        for x,z in np.argwhere(thin.any(axis=1)):
            ys=np.flatnonzero(thin[x,:,z])
            for run in np.split(ys,np.flatnonzero(np.diff(ys)>1)+1):
                a,b=int(run[0]),int(run[-1])
                if len(run)<=2 and a>=3 and solid[x,a-1,z] and not solid[x,b+1:min(b+3,solid.shape[1]),z].any():
                    parapets[x,run,z]=True
    family = np.zeros(solid.shape, np.uint8)
    pts = np.argwhere(posts|parapets)
    if len(pts):
        import trimesh
        mesh = trimesh.Trimesh(verts, scene.faces, process=False)
        lay = layers[tuple(pts.T)]
        centres = pts + (low[lay] + high[lay] + 1) / (2.0 * scale) + origin
        _, _, face = trimesh.proximity.closest_point(mesh, centres)
        for cell, f in zip(pts, face):
            name = scene.materials[int(scene.face_mat[f])].name
            if scene.face_part is not None:
                name += " " + scene.part_names[int(scene.face_part[f])]
            if re.search(r"metal|steel|iron|glass|leaf|leaves|plant|cable|wire", name, re.I):
                code = 3
            elif re.search(r"wood|timber|plank|oak|pine|cedar|mahogany", name, re.I):
                code = 1
            elif re.search(r"stone|brick|concrete|masonry|marble|granite|plaster", name, re.I):
                code = 2
            else:
                code = 0
            family[tuple(cell)] = code
            if parapets[tuple(cell)] and not (code in (1,2) and re.search(r"parapet|railing|balustrade",name,re.I)):
                parapets[tuple(cell)] = False
    # Only consolidate components that actually yielded a supported post.
    retained = np.unique(components[posts])
    duplicates &= np.isin(components, retained[retained > 0])
    return posts|parapets, family, duplicates, {"anchored_narrow_runs": runs, "candidate_voxels": int(posts.sum()),
                           "named_parapet_or_railing_voxels": int(parapets.sum()),
                           "duplicate_post_cells": int(duplicates.sum()),
                           "rule": "fine-width<=half; vertical>=2; base+overhead; separated-from-wall"}
