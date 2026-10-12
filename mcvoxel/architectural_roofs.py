"""Confidence-aware roof evidence: fit surfaces before selecting block courses.

Small source tiles and bevels do not independently set pitch. A robust local plane
needs two-dimensional support and sub-block residuals. Unsupported ornament remains
geometric. Oblique/curved surfaces use shared horizontal half-block contour levels.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy import ndimage


def surface_evidence(height, grad, shape_xz, sub):
    X, Z = shape_xz
    H = height.reshape(X, sub, Z, sub).transpose(0, 2, 1, 3).reshape(X, Z, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        h = np.nanmedian(np.where(np.isfinite(H), H, np.nan), axis=-1)
    valid = np.isfinite(h)
    # Simultaneous robust local regression: 25 observations, three coefficients.
    design = np.array([[1., x, z] for x in range(-2, 3) for z in range(-2, 3)])
    padded = np.pad(h, 2, constant_values=np.nan)
    observations = np.stack([padded[2+int(dx):2+int(dx)+X, 2+int(dz):2+int(dz)+Z]
                             for _, dx, dz in design], axis=-1)
    mask = np.isfinite(observations) & valid[..., None]
    values = np.nan_to_num(observations)
    # A level change greater than the maximum representable pitch is a separate roof.
    mask &= np.abs(values - np.nan_to_num(h)[..., None]) <= (1 + 2.5 * np.linalg.norm(design[:, 1:], axis=1))
    coefficients = np.zeros((X, Z, 3))
    for _ in range(3):
        weight = mask.astype(float)
        lhs = np.einsum('...k,ki,kj->...ij', weight, design, design)
        rhs = np.einsum('...k,...k,ki->...i', weight, values, design)
        # Regularize the fit only numerically; singular neighborhoods are rejected below.
        lhs += np.eye(3) * 1e-8
        coefficients = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        residual = np.abs(np.einsum('...i,ki->...k', coefficients, design) - values)
        mask &= residual <= .4
    count = mask.sum(axis=-1)
    error = np.sqrt(np.sum(np.where(mask, residual ** 2, 0), axis=-1) / np.maximum(count, 1))
    has_x = ((mask * design[:, 1]) > 0).any(-1) & ((mask * design[:, 1]) < 0).any(-1)
    has_z = ((mask * design[:, 2]) > 0).any(-1) & ((mask * design[:, 2]) < 0).any(-1)
    # Boundary planes may have support only to one side: rank and extent establish it.
    x_span = np.max(np.where(mask, design[:, 1], -10), -1) - np.min(np.where(mask, design[:, 1], 10), -1)
    z_span = np.max(np.where(mask, design[:, 2], -10), -1) - np.min(np.where(mask, design[:, 2], 10), -1)
    confident = valid & (count >= 9) & (x_span >= 2) & (z_span >= 2) & (error <= .22)
    confident &= np.abs(coefficients[..., 0] - np.nan_to_num(h)) <= .35
    slope = np.linalg.norm(coefficients[..., 1:], axis=-1)
    confident &= slope <= 2.5
    # Use fitted surface samples only when there is actual mesh coverage at that ray.
    dx, dz = np.meshgrid((np.arange(sub)+.5)/sub-.5, (np.arange(sub)+.5)/sub-.5, indexing='ij')
    fitted = coefficients[..., 0, None] + coefficients[..., 1, None]*dx.ravel() + coefficients[..., 2, None]*dz.ravel()
    new_h = np.where(confident[..., None] & np.isfinite(H), fitted, H)
    G = grad.reshape(X, sub, Z, sub, 2).transpose(0, 2, 1, 3, 4).reshape(X, Z, -1, 2)
    new_g = np.where(confident[..., None, None], coefficients[..., None, 1:], G)
    return (new_h.reshape(X,Z,sub,sub).transpose(0,2,1,3).reshape(X*sub,Z*sub),
            new_g.reshape(X,Z,sub,sub,2).transpose(0,2,1,3,4).reshape(X*sub,Z*sub,2),
            {'height': coefficients[..., 0], 'gradient': coefficients[..., 1:],
             'confidence': confident, 'residual': error, 'raw_height': h})


def constrain_plan(plan, evidence, materials=None):
    from . import block_shapes as bs
    confidence = evidence['confidence']
    h, g = evidence['height'], evidence['gradient']
    preserved = 0
    for row in plan.info.get('plane_decisions', []):
        if not row.get('planned', True):
            continue
        mask = plan.region == row['region']
        trusted = float(confidence[mask].mean()) if mask.any() else 0.
        # Do not independently reinterpret tiny roof fragments or poorly fitting surfaces.
        if row['columns'] < 6 or trusted < .6 or row.get('max_surface_error', 0) > .85:
            preserved += int(mask.sum())
            plan.level[mask] = -1; plan.shape[mask] = -1
            plan.group[mask] = 0; plan.cls[mask] = 0; plan.facing[mask] = -1
            row.update(planned=False, fallback='preserve_source_geometry', confidence=trusted)
        else:
            row['confidence'] = trusted
    # Terraced contours for oblique and locally curved roofs. Quantization phase is
    # global in Y, so a cone/apse cannot acquire an independent phase at every azimuth.
    primary = np.max(np.abs(g), axis=-1)
    secondary = np.min(np.abs(g), axis=-1)
    oblique = confidence & (primary >= .25) & (secondary > .2 * primary)
    # Only original mesh-supported, anchored candidates from the existing planner.
    eligible = np.zeros_like(confidence)
    for row in plan.info.get('plane_decisions', []):
        if row.get('source_pitch', 0) >= .25:
            eligible |= plan.region == row['region']
    contour = ndimage.binary_dilation(oblique, iterations=2) & confidence & eligible & (primary >= .25)
    labels, _ = ndimage.label(contour)
    records = plan.info.get('plane_decisions', [])
    region_id = max([r['region'] for r in records], default=0)
    for label, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        local = labels[box] == label
        if local.sum() < 9:
            continue
        mask = labels == label
        region_id += 1
        q = np.floor(2*h[mask] + .5 + 1e-7).astype(int)
        plan.level[mask] = np.maximum(0, (q - 1)//2)
        plan.shape[mask] = np.where(q % 2, bs.SHAPE_SLAB_BOTTOM, bs.SHAPE_FULL)
        plan.cls[mask] = 2; plan.facing[mask] = -1; plan.region[mask] = region_id
        records.append({'region': region_id, 'columns': int(mask.sum()), 'planned': True,
                        'direction': None, 'pattern': 'source-contours', 'source_pitch': float(np.median(np.linalg.norm(g[mask],axis=-1))),
                        'selected_pitch': None, 'phase_height': 0., 'confidence': float(confidence[mask].mean()),
                        'max_surface_error': float(np.max(np.abs(q/2 - h[mask])))})
    # Material regions retain source material boundaries, independently of pitch pieces.
    labels_mat = np.zeros(plan.level.shape, int) if materials is None else materials
    plan.group[:] = 0
    next_group = 0
    for region in np.unique(plan.region[plan.level >= 0]):
        candidate = (plan.region == region) & (plan.level >= 0) & (plan.cls >= 2)
        for material in np.unique(labels_mat[candidate]):
            lab, n = ndimage.label(candidate & (labels_mat == material))
            plan.group[lab > 0] = lab[lab > 0] + next_group; next_group += n
    # Recompute coverage after confidence fallbacks and contour replacement.
    for row in records:
        row['columns'] = int(((plan.region == row['region']) & (plan.level >= 0)).sum()) if row.get('planned', True) else row['columns']
    plan.info.update(architecture='source_surfaces_v1', confidence_fallback_columns=preserved,
                     confident_surface_columns=int(confidence.sum()), planned=int((plan.level >= 0).sum()),
                     roof_groups=next_group, regions=len(records),
                     by_class={str(c): int((plan.cls == c).sum()) for c in range(1,5)})
    return plan


def lower_roofs(vm, verts, faces, params, scene, max_layers=4):
    """Process exposed lower roof runs without deleting any higher solid.

    Candidates require two blocks of air above and a spatial patch of at least
    nine columns. The same confidence-aware source fitting is used at each level.
    Four is an explicit evidence-layer budget; skipped layers are reported.
    """
    from dataclasses import replace
    from .grid import shape_shell
    from .roofs import builder_roofs
    solid = vm.solid
    X, Y, Z = solid.shape
    ys = np.arange(Y)[None, :, None]
    above = np.zeros_like(solid); above[:, :-1, :] = solid[:, 1:, :]
    above2 = np.zeros_like(solid); above2[:, :-2, :] = solid[:, 2:, :]
    has = solid.any(1)
    top = np.where(has, Y-1-np.argmax(solid[:, ::-1, :],1), -1)
    exposed = solid & ~above & ~above2 & (ys < top[:, None, :] - 2)
    plans = []
    for _ in range(max_layers):
        lower_has = exposed.any(1)
        if not lower_has.any():
            break
        low = np.where(lower_has, Y-1-np.argmax(exposed[:, ::-1, :],1), -1)
        labels, _ = ndimage.label(lower_has)
        sizes = np.bincount(labels.ravel())
        valid = lower_has & (sizes[labels] >= 9) & (low > 0)
        if not valid.any():
            break
        ceiling_coarse = np.where(valid, low + 1.4, -np.inf)
        ceiling = np.repeat(np.repeat(ceiling_coarse, params.sub, 0), params.sub, 1)
        truncate = ys <= low[:, None, :]
        local = vm.solid & truncate
        upper = vm.solid & ~truncate
        local_vm = replace(vm, solid=local, shell=vm.shell & local, surface=vm.surface & local,
                           shapes=np.where(local, vm.shapes, -1), info=dict(vm.info))
        rebuilt, plan = builder_roofs(local_vm, verts, faces, params, scene, ceiling=ceiling, _include_lower=False)
        plan.info['layer_kind'] = 'exposed_lower_surface'
        # Enforce original open separation. No lower rewrite may invade the upper roof.
        permitted = ys <= (low + 1)[:, None, :]
        changed_solid = np.where(permitted, rebuilt.solid, local) | upper
        changed_shape = np.where(permitted & ~upper, rebuilt.shapes, vm.shapes)
        rgb = None if vm.rgb is None else np.where((permitted & ~upper)[..., None], rebuilt.rgb, vm.rgb)
        kind = None if vm.kind_frac is None else np.where((permitted & ~upper)[..., None], rebuilt.kind_frac, vm.kind_frac)
        vm = replace(vm, solid=changed_solid, shell=shape_shell(changed_solid, changed_shape),
                     shapes=changed_shape, rgb=rgb, kind_frac=kind,
                     surface=vm.surface & changed_solid)
        if (plan.level >= 0).any():
            plans.append(plan)
        exposed &= ys < low[:, None, :] - 2
    vm.info['lower_surface_budget'] = {'max_layers': max_layers, 'remaining_columns': int(exposed.any(1).sum())}
    return vm, plans
