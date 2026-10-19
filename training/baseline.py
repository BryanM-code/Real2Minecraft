"""The rule-based baseline (D18) on the real-TRELLIS evaluation sets, scored like the model.

Usage (the app's environment, which has the app's modules; BuildDataCollector beside this
repo, or --bdc):

    r2m_app\\Scripts\\python.exe -m training.baseline --dataset <BDC>/data/real-schems-v1
        [--sets smooth blocky photos] [--modes building sculpture] [--only <id> ...]
        [--out <dataset>/runs/baseline]

Per evaluation build, the TRELLIS mesh goes through the app as a user's would:
`normalize_mesh`, then building mode B1-B5 (`run_building`) or the sculpture voxelizer
(`voxelize_mesh`, Enhanced colour, partial shapes). The target size is the longest side of
the mesh as the evaluation placed it (`schemset.real_trellis._fit` plus the shift), so the
baseline builds at the scale the learned model's input has. The photo is the one TRELLIS
got (the render for the two render sets; the cutout or the photo for the real photos).

The blocks are encoded with mcvox-v3 (`vocab.encode_state`; a block outside the vocabulary
is solid with an unknown material), put into the target's frame by `score.align` (the app
normalises, turns and scales the mesh itself, so its output has its own frame) and scored
with `score.score`. `align` also tries the mirror image and records its overlap
(`overlap_mirror`): far above the unmirrored overlap would mean the app's axes and the
targets' disagree in handedness.

Writes <out>/<set>/<mode>/<id>.json (scores, alignment, target size, timings) and
<id>.npz (the aligned shape / material grids), and <out>/summary.json (per set, mode and
split: `score.summarize`). Resumable: a build with a json is skipped (--force).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SETS = {"smooth": "real_trellis_smooth", "blocky": "real_trellis"}
MODES = ("building", "sculpture")


def _bdc(path: Path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def to_grid(voxel_data: dict):
    """App voxel_data (blocks relative to their min corner, blockstate ids) -> (shape uint8,
    material uint16) in mcvox-v3. Outside the vocabulary: solid, material IGNORE."""
    from mcvoxel import vocab
    dims = tuple(int(d) for d in voxel_data["dims"])
    shape = np.zeros(dims, np.uint8)
    mat = np.zeros(dims, np.uint16)
    blocks = np.asarray(voxel_data["blocks"], dtype=np.int64).reshape(-1, 3)
    cache = {}
    for (x, y, z), bid in zip(blocks, voxel_data["block_ids"]):
        if bid not in cache:
            s, m = vocab.encode_state(str(bid))
            cache[bid] = (s, vocab.IGNORE_MATERIAL if m is None else m)
        shape[x, y, z], mat[x, y, z] = cache[bid]
    return shape, mat


def jobs(dataset: Path, sets) -> list:
    """(set, id, split, glb, photo, cutout, placement) for every placed evaluation build."""
    out = []
    for name in sets:
        if name == "photos":
            from schemset.real_photo import PHOTOS, build_dir, photo_path
            idx = json.loads((dataset / "pairs" / "real_photo_index.json").read_text(encoding="utf-8"))
            rows = {r["id"]: r for r in idx["builds"] if r.get("status") == "ok"}
            for p in PHOTOS:
                r = rows.get(p["id"])
                if not r:
                    continue
                d = build_dir(dataset, p)
                used = (r.get("cutout") or {}).get("cutout_used")
                image = d / "cutout.png" if used and (d / "cutout.png").is_file() else photo_path(dataset, p)
                out.append((name, p["id"], r["split"], d / "trellis.glb", image, image, r["placement"]))
            continue
        folder = dataset / "pairs" / SETS[name]
        idx = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        for b in idx["builds"]:
            if b.get("status") != "ok":
                continue
            d = folder / b["id"]
            out.append((name, b["id"], b["split"], d / "trellis.glb", d / "photo.png", d / "photo.png",
                        b["placement"]))
    return out


def placed_extent(dataset: Path, bid: str, glb: Path, placement: dict) -> float:
    """Longest side (blocks) of the TRELLIS mesh as the evaluation placed it."""
    from mcvoxel.scene import load_scene
    from schemset.pairs import target_volume
    from schemset.real_trellis import _fit
    z = np.load(dataset / "normalized" / f"{bid}.npz")
    vol = target_volume(z["target_shape"], z["exterior"], z["ground_height"])
    v, _ = _fit(load_scene(glb).verts, placement["yaw_deg"], vol, placement.get("scale_mult", 1.0),
                placement.get("aspect", 1.0))
    return float((v.max(axis=0) - v.min(axis=0)).max())


def run_one(dataset: Path, job, mode: str, work: Path) -> dict:
    """One build through one app mode -> the record (and the aligned grids). The GLB is
    copied into `work` first: `normalize_mesh` writes its OBJ next to the file it reads,
    and the dataset's folders stay as they are."""
    import os
    import shutil
    from mesh_normalize import normalize_mesh
    from mcvoxel.blockstates import mirror_x
    from training import score as S
    name, bid, split, src_glb, photo, cutout, placement = job
    work.mkdir(parents=True, exist_ok=True)
    glb = work / f"{name}__{bid}.glb"
    shutil.copyfile(src_glb, glb)
    z = np.load(dataset / "normalized" / f"{bid}.npz")
    ts, tm = z["target_shape"], z["target_material"]
    size = placed_extent(dataset, bid, glb, placement)
    target_blocks = max(4, int(round(size)))
    rec = {"set": name, "id": bid, "split": split, "mode": mode, "target_blocks": target_blocks,
           "placed_extent_blocks": round(size, 2)}
    t0 = time.time()
    cwd = os.getcwd()
    os.chdir(REPO)                     # the app's modules resolve paths from the repo root
    try:
        norm_path, norm_info = normalize_mesh(glb)
        if mode == "building":
            from building.errors import BuildingModeError
            from building.pipeline import run_building
            try:
                voxel_data, _ = run_building(image_path=photo, cutout_path=cutout, glb_path=glb,
                                             norm_info=norm_info, target_blocks=target_blocks,
                                             dataset_enabled=False)
            except BuildingModeError as exc:      # the app shows an error: no build
                rec["failed"] = f"BuildingModeError: {exc}"
                voxel_data = None
        else:
            from voxelize import voxelize_mesh
            voxel_data = voxelize_mesh(norm_path, norm_info, target_blocks=target_blocks,
                                       color_repair=True, partial_shapes=True, image_path=photo)
    finally:
        os.chdir(cwd)
        for f in (glb, work / f"{glb.stem}_normalized.obj", work / f"{glb.stem}_normalized.mtl"):
            f.unlink(missing_ok=True)
    rec["seconds_app"] = round(time.time() - t0, 1)
    if voxel_data is None or not voxel_data.get("blocks"):
        ps, pm = np.zeros_like(ts), np.zeros(ts.shape, np.uint16)
        rec["align"] = None
    else:
        own_s, own_m = to_grid(voxel_data)
        rec["app_dims"] = list(own_s.shape)
        rec["app_blocks"] = int(len(voxel_data["blocks"]))
        ps, pm, info = S.align(own_s, own_m, ts, tm)
        ms, mm = mirror_x(own_s, own_m)
        _, _, info_m = S.align(ms, mm, ts, tm)
        info["overlap_mirror"] = info_m["overlap"]
        rec["align"] = info
    rec["scores"] = S.score(ps, pm, ts, tm)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec, ps, pm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--bdc", type=Path, default=REPO.parent / "BuildDataCollector")
    ap.add_argument("--sets", nargs="*", default=["smooth", "blocky", "photos"])
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    _bdc(args.bdc.resolve())
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    out = (args.out or dataset / "runs" / "baseline").resolve()
    from training import score as S
    todo = [j for j in jobs(dataset, args.sets) if args.only is None or j[1] in args.only]
    print(f"{len(todo)} evaluation builds x {len(args.modes)} mode(s) -> {out}", flush=True)
    for job in todo:
        for mode in args.modes:
            d = out / job[0] / mode
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{job[1]}.json"
            if path.is_file() and not args.force:
                continue
            try:
                rec, ps, pm = run_one(dataset, job, mode, out / "_work")
                np.savez_compressed(d / f"{job[1]}.npz", shape=ps, material=pm)
            except Exception as exc:                 # recorded, never silent
                rec = {"set": job[0], "id": job[1], "split": job[2], "mode": mode,
                       "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
            path.write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
            sc = rec.get("scores") or {}
            print(f"  {job[0]:7s} {mode:9s} {job[1][:44]:44s} "
                  + (f"IoU {sc['iou']:.3f} F1~1 {sc['f1_tol1']:.3f} shape {sc['shape_acc']:.3f} "
                     f"material {sc['material_acc']:.3f} dE {sc['material_de_p50']:.1f}"
                     f"{'  FAILED ' + rec['failed'][:60] if rec.get('failed') else ''}"
                     if sc else "ERROR " + rec.get("error", "")[:90]), flush=True)
    summary = {}
    for name in args.sets:
        for mode in args.modes:
            d = out / name / mode
            recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.is_dir() else []
            for split in ("val", "test", "all"):
                rows = [r["scores"] for r in recs if r.get("scores") and (split == "all" or r["split"] == split)]
                if rows:
                    summary[f"{name}/{mode}/{split}"] = {
                        **S.summarize(rows), "failed": sum(1 for r in recs if r.get("failed")
                                                           and (split == "all" or r["split"] == split)),
                        "errors": sum(1 for r in recs if r.get("error")
                                      and (split == "all" or r["split"] == split))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    for key, s in summary.items():
        print(f"{key:28s} n={s['builds']:2d} IoU {s['iou']['mean'] if s['iou'] else None} "
              f"F1~1 {s['f1_tol1']['mean'] if s['f1_tol1'] else None} shape {s['shape_acc']['mean'] if s['shape_acc'] else None} "
              f"material {s['material_acc']['mean'] if s['material_acc'] else None} "
              f"dE {s['material_de_p50']['mean'] if s['material_de_p50'] else None} failed {s['failed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
