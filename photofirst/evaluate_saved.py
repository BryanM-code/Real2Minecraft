"""A completion model on front-end inputs already saved (`evaluate_mv`, `evaluate_front`):
block scores, where the details are and the quality stats, without rerunning the front end.

    r2m_train\\Scripts\\python -m photofirst.evaluate_saved --dataset <BDC>/data/real-schems-v1
        --inputs ma_rough_res2 --run pf-b1 [--checkpoint best_front.pt] [--passes 8] [--tag ...]

Reads <dataset>/photofirst/front/<inputs>/<id>.npz (1 or 2 cells per block), writes the
predictions to front/<tag>/<id>.pred.npz and eval_saved_<tag>.json: per build and on average
`training.score.score`, `training.quality.parts` and `training.quality.quality`, the human
builds' quality stats beside them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from training import quality as Q
from training.score import score, summarize

from . import data as P
from . import front_mv as M
from .views import OUT_DIR

PART_KEYS = ("stairs", "slab", "thin", "glass")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--inputs", required=True, help="front/<this>/ (saved inputs)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best_front.pt")
    ap.add_argument("--passes", type=int, default=1, help="> 1: a masked model decoded over that many passes")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shape-bias", type=float, default=0.0,
                    help="decoding calibration: solid shapes reweighted by (the run's class weight / full's) ** this "
                         "(training.predict.predict_grid shape_prior); 0 = plain argmax")
    ap.add_argument("--free", action="store_true",
                    help="inputs saved by evaluate_mv --placement free: the model on the input as placed (its own "
                         "grid, as on real photos), the prediction put on the human build by the input's alignment")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    from training.predict import predict_grid
    from .masked import decode, load_any
    dataset = args.dataset.resolve()
    pf = dataset / OUT_DIR
    tag = args.tag or (f"{args.inputs}__{args.run}" + (f"_p{args.passes}" if args.passes > 1 else "")
                       + (f"_sb{args.shape_bias:g}" if args.shape_bias else "") + ("_free" if args.free else ""))
    out_dir = pf / "front" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    store = P.Store(dataset)
    model, ck = load_any(pf / "runs" / args.run / args.checkpoint)
    prior = None
    if args.shape_bias:
        cw = np.asarray(json.loads((pf / "runs" / args.run / "config.json").read_text(encoding="utf-8"))["class_weights"])
        prior = (cw / cw[1]) ** args.shape_bias
    rows = []
    for f in sorted((pf / "front" / args.inputs).glob("*.npz")):
        if f.name.endswith(".pred.npz"):
            continue
        bid = f.stem
        if store.builds.get(bid, {}).get("split") not in args.splits:
            continue
        ts, tm = (np.asarray(a) for a in store.target(bid))
        z = np.load(f)
        if args.free:
            from mcvoxel.blockstates import rotate_y
            from training.score import _place
            raw = z["free_input"]
            dims = tuple(int(d) // 2 for d in raw.shape[:3])
        else:
            raw, dims = z["input"], ts.shape
        inp = M.model_input(raw, dims)
        t = time.time()
        if args.passes > 1:
            ps, pm = decode(model, inp, dims, passes=args.passes, temperature=args.temperature)
        else:
            ps, pm = predict_grid(model, inp, dims, shape_prior=prior)
        if args.free:                                       # onto the human build by the input's own alignment
            k = int(z["free_turns"]) % 4
            ps, pm = rotate_y(ps, k, pm) if k else (ps, pm)
            ps, pm = (_place(ps, [int(v) for v in z["free_shift"]], ts.shape, 0),
                      _place(pm, [int(v) for v in z["free_shift"]], ts.shape, 0))
        np.savez_compressed(out_dir / f"{bid}.pred.npz", shape=ps, material=pm)
        r = {"id": bid, "score": score(ps, pm, ts, tm), "parts": Q.parts(ps, pm, ts, tm),
             "quality": Q.quality(ps, pm), "human": Q.quality(ts, tm, target=True), "seconds": time.time() - t}
        rows.append(r)
        s = r["score"]
        print(f"{bid:58s} IoU {s['iou']:.3f}  F1~1 {s['f1_tol1']:.3f}  material {s['material_acc']:.3f}  "
              f"stairs F1 {r['parts']['stairs']['f1']:.2f}  glass F1 {r['parts']['glass']['f1']:.2f}  "
              f"({r['seconds']:.0f} s)", flush=True)
    summary = {"inputs": args.inputs, "free": args.free, "shape_bias": args.shape_bias, "run": args.run, "checkpoint": args.checkpoint, "step": ck.get("step"),
               "passes": args.passes, "temperature": args.temperature, "builds": len(rows),
               "score": summarize([r["score"] for r in rows]),
               "parts": {k: {m: float(np.nanmean([r["parts"][k][m] for r in rows])) for m in ("precision", "recall", "f1")}
                         for k in PART_KEYS},
               "stair_facing": float(np.nanmean([r["parts"]["stair_facing"] for r in rows])),
               "quality": {k: float(np.nanmean([r["quality"][k] for r in rows])) for k in Q.KEYS},
               "human_quality": {k: float(np.nanmean([r["human"][k] for r in rows])) for k in Q.KEYS}}
    (pf / f"eval_saved_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float),
                                               encoding="utf-8", newline="\n")
    sc = summary["score"]
    print(f"{tag}: IoU {sc['iou']['mean']:.4f}  F1~1 {sc['f1_tol1']['mean']:.4f}  shape {sc['shape_acc']['mean']:.4f}  "
          f"material {sc['material_acc']['mean']:.4f}  dE p50 {sc['material_de_p50']['mean']:.1f}  "
          f"exact {sc['block_acc']['mean']:.4f}")
    print("  parts F1: " + "  ".join(f"{k} {summary['parts'][k]['f1']:.2f}" for k in PART_KEYS)
          + f"  stair facing {summary['stair_facing']:.2f}")
    print("  quality:  " + "  ".join(f"{k} {summary['quality'][k]:.3f}" for k in Q.KEYS))
    print("  human:    " + "  ".join(f"{k} {summary['human_quality'][k]:.3f}" for k in Q.KEYS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
