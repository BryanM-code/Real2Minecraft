"""The multi-photo front end (`front_mv`, MapAnything) on rendered photo sets, scored.

    r2m_train\\Scripts\\python -m photofirst.evaluate_mv --dataset <BDC>/data/real-schems-v1
        --views 0 1 2 3 [--splits val test] [--run pf-a3 --checkpoint best_front.pt] [--tag mv4]
        [--hints none|intrinsics|rough|coarse|true] [--views-from da3_rough] [--refine]

For each build with renders of all `--views` (`photofirst.render --views`), the photos go
through MapAnything together (with `--hints`, conditioned on camera hints:
`front_mv.render_hints`), or their per-photo arrays come from another front end
(`--views-from`, e.g. `front_da3`'s). The reconstruction is put into the target's frame by the
similarity from its cameras to the renders' (`front_mv.similarity_from_cameras`) and turned
into blocks (`front_mv.to_grid`). Scored against what those photos truly show (the union of
their `cast_from` sets): recall and precision of the seen blocks (exact and within one block),
solid on air the photos show, seen air on solid. With `--run`, the completion model on the
blocks (end to end). Also reports how far the reconstructed cameras were from the true ones.
Writes <dataset>/photofirst/front/<tag>/<id>.npz (the input; .pred.npz with --run) and
front_eval_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from training.score import score, summarize

from . import data as P
from . import front_mv as M
from .evaluate_front import photo_metrics, seen_metrics
from .render import view_key
from .views import OUT_DIR


def rotation_error(Ra, Rb) -> float:
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--run", default=None, help="a photofirst run (completion model)")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--hints", choices=M.HINTS + ("compass",), default="none",
                    help="camera hints for MapAnything (front_mv.render_hints); compass: direction within 12 degrees "
                         "(a phone's heading), tilt estimated from each photo (real.tilt), the focal known (EXIF)")
    ap.add_argument("--placement", choices=("oracle", "free"), default="oracle",
                    help="free: no target (place.free_placement), the prediction aligned to the target afterwards "
                         "(quarter turn + shift, training.score.align) for scoring")
    ap.add_argument("--renders", default="persp",
                    help="the photos: persp (render --views: the evaluation builds' corners) or persp_street "
                         "(render --street: street-level photos, per build)")
    ap.add_argument("--set", choices=sorted(M.STREET_EVAL), default=None,
                    help="with --renders persp_street: a fixed photo set (front_mv.STREET_EVAL) instead of --views")
    ap.add_argument("--ground", choices=("points", "cameras"), default="points",
                    help="with --placement free: the ground at the lowest visible point, or at the street renders' "
                         "eye height below the cameras (place.ground_from_cameras)")
    ap.add_argument("--free-scale", choices=("target", "metric", "cameras"), default="target",
                    help="with --placement free: the target's height in blocks, MapAnything's metric scale, or the "
                         "scale from the true cameras (isolates the placement's turn and position from the scale)")
    ap.add_argument("--views-from", default=None,
                    help="per-photo arrays from another front end: <dataset>/photofirst/mv_views/<tag> (front_da3)")
    ap.add_argument("--res", type=int, default=1, choices=(1, 2), help="front_mv.to_grid cells per block")
    ap.add_argument("--passes", type=int, default=1, help="a masked model's decoding passes (photofirst.masked)")
    ap.add_argument("--carve", action="store_true", help="space carving across the photos (front_mv.to_grid)")
    ap.add_argument("--coarse-scale", action="store_true",
                    help="with --refine, search the scale over 0.75-1.33 first (a front end unsure of its scale)")
    ap.add_argument("--refine", action="store_true",
                    help="refine the placement against the target (front_mv.refine_to_target), as for training pairs")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    pf = dataset / OUT_DIR
    tag = args.tag or ("mv" + "".join(str(v) for v in args.views))
    out_dir = pf / "front" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = pf / "tmp_mv"
    tmp.mkdir(exist_ok=True)
    street = args.renders != "persp"
    cams = {} if street else json.loads((pf / "persp" / "cameras.json").read_text(encoding="utf-8"))
    store = P.Store(dataset)
    model = None
    if args.run:
        from training.predict import predict_grid
        from .masked import decode, load_any
        model, _ = load_any(pf / "runs" / args.run / args.checkpoint)
    ma = None if args.views_from else M.load_mapanything()
    rng = np.random.default_rng(0)        # the same hints as front_da3 for the same builds
    moge = None
    if args.hints == "compass":
        from .front import load_moge
        from .real import steady_pitches, tilt
        moge = load_moge()
    if street:
        builds = sorted(b for b in store.builds if (pf / args.renders / b / "cameras.json").is_file())
    else:
        builds = sorted({c.get("build", k) for k, c in cams.items()})
    rows = []
    for bid in builds:
        if args.ids and bid not in args.ids:
            continue
        if street:
            folder = pf / args.renders / bid
            cams = json.loads((folder / "cameras.json").read_text(encoding="utf-8"))
            keys = list(M.STREET_EVAL[args.set]) if args.set else [str(v) for v in args.views]
        else:
            folder = pf / "persp"
            keys = [view_key(bid, v) for v in args.views]
        if not all(k in cams for k in keys) or store.builds.get(bid, {}).get("split") not in args.splits:
            continue
        ts, tm = store.target(bid)
        ts, tm = np.asarray(ts), np.asarray(tm)
        paths, alphas = [], []
        for k in keys:
            im = np.array(Image.open(folder / f"{k}.png").convert("RGBA"))
            alphas.append(im[..., 3] > 127)
            Image.fromarray(np.where(alphas[-1][..., None], im[..., :3], 255).astype(np.uint8)).save(tmp / f"{k}.png")
            paths.append(tmp / f"{k}.png")
        if args.views_from:
            src = pf / "mv_views" / args.views_from / f"{bid}.npz"
            if not src.exists():
                continue
            views = M.load_views(src)
        elif args.hints == "compass":
            K, _ = M.render_hints(cams, keys, "intrinsics", rng)
            pitches, _ = steady_pitches([tilt(moge, np.array(Image.open(p).convert("RGB")), a)["pitch_deg"]
                                         for p, a in zip(paths, alphas)])
            yaws = [cams[k]["yaw"] + rng.uniform(-12.0, 12.0) for k in keys]
            hint_pitch_err = [pp - cams[k]["pitch"] for pp, k in zip(pitches, keys)]
            views = M.run_mapanything(ma, paths, K, M.rough_poses(yaws, pitches))
        else:
            views = M.run_mapanything(ma, paths, *M.render_hints(cams, keys, args.hints, rng))
        h, w = views[0]["pts"].shape[:2]
        masks = [np.array(Image.fromarray(a).resize((w, h), Image.NEAREST)) for a in alphas]
        true_R = [np.stack([np.array(cams[k]["right"]), np.array(cams[k]["down"]), -np.array(cams[k]["near"])], 1)
                  for k in keys]
        sim = M.similarity_from_cameras([v["cam2world"] for v in views], [cams[k]["centre"] for k in keys], true_R)
        cam_err = [rotation_error(sim["R"] @ v["cam2world"][:3, :3], Rk) for v, Rk in zip(views, true_R)]
        free = None
        if args.placement == "free":
            from training.score import align
            from . import place as PL
            from .front import place as put
            t_solid = (ts != 0) & (ts != 255)
            ys = np.nonzero(t_solid.any((0, 2)))[0]
            eye = None
            if args.ground == "cameras":                           # standing: a raised spot is not known
                from .render import eye_blocks
                eye = eye_blocks(json.loads((dataset / "normalized" / "records" / f"{bid}.json").read_text(encoding="utf-8")))
            sim_f, dims_f, pinfo = PL.free_placement(
                views, masks, height_blocks=float(ys.max() - ys.min() + 1) if args.free_scale == "target" else None,
                metres_per_block=1.0 / sim["s"] if args.free_scale == "cameras" else 1.0,
                eye_units=None if eye is None else eye / sim["s"])
            D = np.asarray(sim_f["R"]) @ np.asarray(sim["R"]).T       # oracle grid -> free grid
            up_err = float(np.degrees(np.arccos(np.clip(D[1, 1], -1, 1))))
            yaw = float(np.degrees(np.arctan2(D[0, 2], D[0, 0])))
            free = {"placement": pinfo, "up_error_deg": up_err, "yaw_off_grid_deg": abs((yaw + 45.0) % 90.0 - 45.0),
                    "dims": list(dims_f)}
            if args.hints == "compass":
                free["hint_pitch_error_deg"] = hint_pitch_err
            inp_f = M.to_grid(views, sim_f, dims_f, masks, res=args.res, carve=args.carve)
            blk_f = M.to_blocks(inp_f, dims_f)
            _, _, ainfo = align(blk_f[..., 0].astype(np.uint8), np.zeros(dims_f, np.uint16), ts, tm)
            free["input_alignment"] = ainfo
            on_target, _ = M.onto_target(inp_f, dims_f, ts, tm)      # the same alignment, at the input's own cells
            np.savez_compressed(out_dir / f"{bid}.npz", input=on_target, free_input=inp_f,   # and as placed, for
                                free_turns=np.int64(ainfo["turns"]), free_shift=np.asarray(ainfo["shift"]))  # evaluate_saved --free
            blocks = put(blk_f, ainfo["turns"], ainfo["shift"], ts.shape)
        elif args.refine:
            sim = M.refine_to_target(views, sim, (ts != 0) & (ts != 255), masks,
                                     coarse_scales=np.exp(np.linspace(np.log(0.75), np.log(1.33), 11))
                                     if args.coarse_scale else None)
        centre_err = [float(np.linalg.norm(sim["s"] * (sim["R"] @ v["cam2world"][:3, 3]) + sim["t"] - np.array(cams[k]["centre"])))
                      for v, k in zip(views, keys)]
        if free is None:
            inp = M.to_grid(views, sim, ts.shape, masks, res=args.res, carve=args.carve)
            np.savez_compressed(out_dir / f"{bid}.npz", input=inp)
            blocks = M.to_blocks(inp, ts.shape)
        t_seen = np.zeros(ts.shape, bool)
        t_air = np.zeros(ts.shape, bool)
        for k in keys:
            packed = np.load(folder / f"{k}.seen.npy")
            t_seen |= np.unpackbits(packed[0])[:ts.size].reshape(ts.shape).astype(bool)
            t_air |= np.unpackbits(packed[1])[:ts.size].reshape(ts.shape).astype(bool)
        t_air &= ~t_seen
        solid, air = blocks[..., 0].astype(bool), blocks[..., 1].astype(bool)
        row = {"id": bid, "split": store.builds[bid]["split"], "scale": sim["s"], "refine": sim.get("refine"), "free": free,
               "camera_rotation_error_deg": cam_err, "camera_centre_error_blocks": centre_err,
               "front": {**seen_metrics(solid, air, ts, t_seen, t_air), **photo_metrics(solid, ts)}}
        s_alone = solid.astype(np.uint8)
        row["front_alone"] = score(s_alone, np.where(solid, tm, 0).astype(np.uint16), ts, tm)
        if model is not None and row["split"] == "val":
            src_inp, src_dims = (inp_f, dims_f) if free is not None else (inp, ts.shape)
            if args.passes > 1:
                ps, pm = decode(model, M.model_input(src_inp, src_dims), src_dims, passes=args.passes, temperature=0.0)
            else:
                ps, pm = predict_grid(model, M.model_input(src_inp, src_dims), src_dims)
            if free is not None:                                       # into the target's frame for scoring
                from mcvoxel.blockstates import rotate_y
                from training.score import _place
                ia = free["input_alignment"]                           # the same free-frame prediction, put on
                rs, rm = rotate_y(ps, ia["turns"], pm) if ia["turns"] % 4 else (ps, pm)   # the target by the input's
                row["end_to_end_input_aligned"] = score(_place(rs, ia["shift"], ts.shape, 0),  # own alignment
                                                        _place(rm, ia["shift"], ts.shape, 0), ts, tm)
                ps, pm, pinfo2 = align(ps, pm, ts, tm)
                free["prediction_alignment"] = pinfo2
            row["end_to_end"] = score(ps, pm, ts, tm)
            np.savez_compressed(out_dir / f"{bid}.pred.npz", shape=ps, material=pm)
        rows.append(row)
        f = row["front"]
        if free is not None:
            print(" " * 58 + f" free placement: up off {free['up_error_deg']:.1f} deg, walls off the grid "
                  f"{free['yaw_off_grid_deg']:.1f} deg, grid {free['dims']}", flush=True)
        print(f"{bid:58s} cams {np.mean(cam_err):.1f} deg / {np.mean(centre_err):.1f} blk | recall {f['recall']:.2f} "
              f"(1 blk {f['recall_1']:.2f}) prec {f.get('precision', 0):.2f} (1 blk {f.get('precision_1', 0):.2f}) "
              f"false-on-air {f['false_solid_on_seen_air']:.3f} | alone IoU {row['front_alone']['iou']:.3f}"
              + (f" | e2e IoU {row['end_to_end']['iou']:.3f}" if "end_to_end" in row else ""), flush=True)
    summary = {"views": args.views, "renders": args.renders, "set": args.set, "ground": args.ground,
               "builds": len(rows), "run": args.run, "checkpoint": args.checkpoint,
               "front_end": args.views_from or "mapanything", "hints": None if args.views_from else args.hints,
               "refine": args.refine, "res": args.res, "coarse_scale": args.coarse_scale, "carve": args.carve,
               "placement": args.placement, "free_scale": args.free_scale if args.placement == "free" else None,
               "front": {k: round(float(np.mean([r["front"][k] for r in rows if k in r["front"]])), 4)
                         for k in ("recall", "recall_1", "precision", "precision_1", "false_solid_on_seen_air",
                                   "false_air", "coverage_1")},
               "camera_rotation_error_deg": round(float(np.mean([np.mean(r["camera_rotation_error_deg"]) for r in rows])), 2),
               "camera_centre_error_blocks": round(float(np.mean([np.mean(r["camera_centre_error_blocks"]) for r in rows])), 2),
               "front_alone": summarize([r["front_alone"] for r in rows])}
    val = [r for r in rows if "end_to_end" in r]
    if val:
        summary["end_to_end_val"] = summarize([r["end_to_end"] for r in val])
        ia = [r["end_to_end_input_aligned"] for r in val if "end_to_end_input_aligned" in r]
        if ia:
            summary["end_to_end_val_input_aligned"] = summarize(ia)
        summary["front_alone_val"] = summarize([r["front_alone"] for r in val])
    fr = [r["free"] for r in rows if r.get("free")]
    if fr:
        summary["free_placement"] = {
            "up_error_deg_median": round(float(np.median([f["up_error_deg"] for f in fr])), 2),
            "up_error_deg_p90": round(float(np.percentile([f["up_error_deg"] for f in fr], 90)), 2),
            "yaw_off_grid_deg_median": round(float(np.median([f["yaw_off_grid_deg"] for f in fr])), 2),
            "yaw_off_grid_deg_p90": round(float(np.percentile([f["yaw_off_grid_deg"] for f in fr], 90)), 2)}
        print("free placement:", summary["free_placement"])
    (pf / f"front_eval_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float),
                                              encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k in ("front", "camera_rotation_error_deg", "camera_centre_error_blocks")}, indent=1))
    for name in ("front_alone_val", "end_to_end_val", "end_to_end_val_input_aligned"):
        if name in summary:
            s = summary[name]
            print(name, {k: (s[k]["mean"] if s.get(k) else None) for k in ("iou", "f1_tol1", "precision", "recall", "material_acc", "block_acc")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
