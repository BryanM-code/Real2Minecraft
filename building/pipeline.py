"""Building-mode orchestrator: B1 yaw -> B2 massing -> B3 facade -> B4 mapping -> B5
assembly, with every stage IR written into a dataset sample as it is produced.

`run_building` is the app/harness entry point; `rerun_from_facade` re-runs only
B3->B4->B5 with user overrides from the refine UI (seconds — no TRELLIS, no SAT) and
captures the correction into the sample as labeled training data.
"""

import time
from pathlib import Path

import numpy as np

from voxel_sampling import _sample_voxel_colors_texture
from building import assemble as _assemble
from building import facade as _facade
from building import facade_map as _facade_map
from building import massing as _massing
from building import yaw_align as _yaw
from building.dataset import Sample
from building.errors import BuildingModeError
from building.params import PARAMS


def _status(cb, msg):
    if cb:
        cb(msg)


def run_building(image_path, cutout_path, glb_path, norm_info: dict,
                 target_blocks: int, status_cb=None, dataset_root=None,
                 dataset_enabled=None, run_id=None):
    """Full building pipeline -> (voxel_data, build_state).

    `norm_info` is normalize_mesh's dict (tex_verts/tex_faces/tex_uv/tex_image).
    `dataset_root` overrides outputs/dataset (the dev harness isolates its runs);
    `run_id` defaults to the GLB stem so samples cross-reference the TRELLIS cache.
    """
    params = dict(PARAMS)
    params["target_blocks"] = int(target_blocks)
    enabled = params["dataset_enabled"] if dataset_enabled is None else dataset_enabled

    run_id = run_id or (Path(glb_path).stem if glb_path else "run")
    sample = Sample(run_id, root=dataset_root, enabled=enabled)
    sample.write_inputs(image_path, cutout_path, glb_path)

    state = {"sample": sample, "params": params,
             "photo_path": image_path, "cutout_path": cutout_path}

    verts = np.asarray(norm_info["tex_verts"], dtype=np.float64)
    faces = np.asarray(norm_info["tex_faces"], dtype=np.int64)

    # ── B1: yaw alignment ─────────────────────────────────────────────────────
    _status(status_cb, "Aligning facades to the grid...")
    t0 = time.time()
    try:
        yaw_ir = _yaw.estimate_yaw(verts, faces, params)
        verts_aligned = (_yaw.apply_yaw(verts, yaw_ir["theta_deg"])
                         if yaw_ir["applied"] else verts.copy())
    except BuildingModeError:
        sample.record_stage("yaw", time.time() - t0, ok=False)
        sample.save_manifest()
        raise
    sample.write_ir_json("yaw.json", yaw_ir)
    sample.record_stage("yaw", time.time() - t0, ok=True, file="yaw.json")

    aligned_info = dict(norm_info)
    aligned_info["tex_verts"] = verts_aligned
    state.update(yaw_ir=yaw_ir, aligned_info=aligned_info)

    # ── B2: massing ───────────────────────────────────────────────────────────
    _status(status_cb, "Regularizing massing...")
    t0 = time.time()
    try:
        raw_solid, scale_factor, origin, watertight = _massing.solid_from_mesh(
            verts_aligned, faces, target_blocks)
        massing_ir, reg_solid = _massing.build_massing(raw_solid, params)
    except BuildingModeError:
        sample.record_stage("massing", time.time() - t0, ok=False)
        sample.save_manifest()
        raise
    massing_ir["scale_factor"] = float(scale_factor)
    sample.write_ir_json("massing.json",
                         {k: v for k, v in massing_ir.items()})
    sample.write_ir_npz("massing_grid.npz",
                        raw_solid=raw_solid, reg_solid=reg_solid)
    sample.record_stage("massing", time.time() - t0, ok=True, file="massing.json")
    state.update(raw_solid=raw_solid, reg_solid=reg_solid, massing_ir=massing_ir,
                 scale_factor=scale_factor, origin=origin, watertight=watertight)

    # ── raw voxel colors (facing detection + material fallbacks) ─────────────
    colors_raw = None
    if norm_info.get("tex_uv") is not None:
        occ_raw = np.argwhere(raw_solid)
        colors_raw = _sample_voxel_colors_texture(
            occ_raw, origin, raw_solid.shape,
            verts_aligned * scale_factor, faces,
            norm_info["tex_uv"], norm_info["tex_image"])
    state["colors_raw"] = colors_raw

    voxel_data = _facade_stages(state, status_cb)
    return voxel_data, state


def _facade_stages(state: dict, status_cb=None,
                   quad_override=None, grid_override=None):
    """B3 -> B4 -> B5 (the part `rerun_from_facade` repeats), writing IRs + final."""
    sample: Sample = state["sample"]
    params = state["params"]
    warnings = []

    # ── B3: facade analysis ──────────────────────────────────────────────────
    _status(status_cb, "Analyzing the facade...")
    t0 = time.time()
    facade_bundle = _facade.analyze_facade(
        state["photo_path"], state["cutout_path"], params,
        quad_override=quad_override, grid_override=grid_override)
    if facade_bundle is not None:
        sample.write_ir_json("facade_layout.json", facade_bundle["ir"])
        cells_arrays = {}
        for fid, (cls, rgb) in facade_bundle["cells"].items():
            cells_arrays["cells_class_{}".format(fid)] = cls
            cells_arrays["cells_rgb_{}".format(fid)] = rgb
        sample.write_ir_npz("facade_cells.npz", **cells_arrays)
        if sample.enabled:
            for fid, img in facade_bundle["rect_images"].items():
                img.save(sample.ir_path("facade_rect_{}.png".format(fid)))
        sample.record_stage("facade", time.time() - t0, ok=True,
                            file="facade_layout.json")
    else:
        warnings.append("no usable cutout mask — massing-only build")
        sample.record_stage("facade", time.time() - t0, ok=False)
    state["facade_bundle"] = facade_bundle

    # ── B4: facade application ───────────────────────────────────────────────
    _status(status_cb, "Mapping the facade onto the walls...")
    t0 = time.time()
    walls = _facade_map.extract_walls(state["massing_ir"], state["reg_solid"])
    facing = _facade_map.pick_photo_facing(
        state["raw_solid"], state["colors_raw"], facade_bundle, params)
    walls_ir, wall_grids = _facade_map.map_facades(
        walls, facade_bundle, facing, state["colors_raw"],
        state["raw_solid"], params)
    warnings.extend(walls_ir.pop("warnings", []))
    sample.write_ir_json("walls.json", walls_ir)
    grid_arrays = {}
    for wid, (cls, rgb) in wall_grids.items():
        grid_arrays["class_{}".format(wid)] = cls
        grid_arrays["rgb_{}".format(wid)] = rgb
    sample.write_ir_npz("wall_grids.npz", **grid_arrays)
    sample.record_stage("facade_map", time.time() - t0, ok=True, file="walls.json")
    state.update(walls_ir=walls_ir, wall_grids=wall_grids)

    # ── B5: assembly ─────────────────────────────────────────────────────────
    _status(status_cb, "Assembling the build...")
    t0 = time.time()
    voxel_data, labels = _assemble.assemble(
        state["massing_ir"], state["reg_solid"], walls_ir, wall_grids,
        state["scale_factor"], state["watertight"], params)
    sample.write_ir_npz("semantic_voxels.npz", labels=labels)
    sample.record_stage("assemble", time.time() - t0, ok=True)
    state.update(voxel_data=voxel_data, semantic_labels=labels)

    sample.write_final(voxel_data, params, warnings)
    _write_previews(sample, voxel_data)
    return voxel_data


def _write_previews(sample: Sample, voxel_data: dict):
    """Front/back preview renders (cosmetic — headless VTK failures are swallowed)."""
    if not sample.enabled:
        return
    try:
        from dev_rebuild import render_screenshot
        render_screenshot(voxel_data, sample.dir / "final" / "preview.png")
    except Exception:
        pass


def rerun_from_facade(state: dict, quad_override=None, grid_override=None,
                      status_cb=None):
    """Refine-UI path: keep TRELLIS/massing, re-run B3->B5 with user overrides.

    Captures (auto prediction, human correction) into the dataset sample before the
    corrected IRs overwrite the originals.  Returns the new voxel_data.
    """
    sample: Sample = state["sample"]
    before = {}
    bundle = state.get("facade_bundle")
    if bundle is not None and bundle["ir"]["facades"]:
        f0 = bundle["ir"]["facades"][0]
        before = {"quad_px": f0["quad_px"], "n_rows": f0["grid"]["n_rows"],
                  "n_cols": f0["grid"]["n_cols"], "quad_source": f0["quad_source"]}
    after = dict(before)
    if quad_override is not None:
        after["quad_px"] = [[float(p[0]), float(p[1])] for p in quad_override]
        after["quad_source"] = "user"
    if grid_override is not None:
        after["n_rows"] = int(grid_override["n_rows"])
        after["n_cols"] = int(grid_override["n_cols"])

    sample.snapshot_auto_ir()
    voxel_data = _facade_stages(state, status_cb,
                                quad_override=quad_override,
                                grid_override=grid_override)
    sample.append_human_edit("facade", 0, before, after)
    return voxel_data
