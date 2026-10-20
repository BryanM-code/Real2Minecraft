"""Several photos of one build: the front end's blocks from each, fused, then the completion model.

    r2m_train\\Scripts\\python -m photofirst.multiview --dataset <BDC>/data/real-schems-v1 --run pf-a1
        --front persp_known_none persp_known_none_v2 [persp_known_none_v1 persp_known_none_v3]

Each photo's front-end input (`evaluate_front --photos persp --view k`, already in the target's
frame) contributes its seen blocks and its seen air; the fused input is their union (a block
seen solid in any photo is solid, seen air is air no photo saw solid), colours from the first
photo that saw the block. The truth for the same photos (`render --views`: what each camera
really sees) gives the ceiling. Scores over the val builds that have every photo: the fused
input alone (the seen blocks as the prediction) and the completion model's output, for the
front end and for the truth. Writes <dataset>/photofirst/multiview_<tags>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.score import score, summarize

from . import data as P
from .views import OUT_DIR

KEYS = ("iou", "f1_tol1", "precision", "recall", "shape_acc", "material_acc", "material_de_p50", "block_acc")


fuse = P.fuse_views


def truth_input(dataset: Path, key: str, ts, tm, table) -> np.ndarray:
    """What a rendered photo's camera truly sees, encoded like the front end's input."""
    packed = np.load(dataset / OUT_DIR / "persp" / f"{key}.seen.npy")
    solid = np.unpackbits(packed[0])[:ts.size].reshape(ts.shape).astype(bool)
    air = np.unpackbits(packed[1])[:ts.size].reshape(ts.shape).astype(bool)
    return P.encode(tm, solid, air, table)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True, help="a photofirst run (completion model)")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--front", nargs="+", required=True, help="front-end tags, one per photo")
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--carve", action="store_true", help="space carving across the photos (data.fuse_views)")
    ap.add_argument("--save", default=None,
                    help="also save the fused front-end inputs as photofirst/front/<this>/ (e.g. for train --val-front)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    from training.evaluate import load_model
    from training.predict import predict_grid
    from .render import view_key
    model, ck = load_model(dataset / OUT_DIR / "runs" / args.run / args.checkpoint)
    store = P.Store(dataset)
    table = P.material_rgb()
    cams = json.loads((dataset / OUT_DIR / "persp" / "cameras.json").read_text(encoding="utf-8"))
    views = []                                      # the view each tag's photo was taken from (..._v<k>)
    for t in args.front:
        last = t.rsplit("_", 1)[-1]
        views.append(int(last[1:]) if last.startswith("v") and last[1:].isdigit() else 0)
    res = {k: [] for k in ("front_alone", "front_model", "truth_alone", "truth_model")}
    ids = []
    for bid in sorted({c["build"] if "build" in c else k for k, c in cams.items()}):
        if store.builds.get(bid, {}).get("split") not in args.splits:
            continue
        paths = [dataset / OUT_DIR / "front" / t / f"{bid}.npz" for t in args.front]
        keys = [view_key(bid, v) for v in views]
        if not all(p.is_file() for p in paths) or not all(k in cams for k in keys):
            continue
        ts, tm = store.target(bid)
        ts, tm = np.asarray(ts), np.asarray(tm)
        front = fuse([np.load(p)["input"] for p in paths], carve=args.carve)
        if args.save:
            (dataset / OUT_DIR / "front" / args.save).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dataset / OUT_DIR / "front" / args.save / f"{bid}.npz", input=front)
        truth = fuse([truth_input(dataset, k, ts, tm, table) for k in keys])
        for name, x in (("front", front), ("truth", truth)):
            s = x[..., 0].astype(bool)
            res[f"{name}_alone"].append(score(s.astype(np.uint8), np.where(s, tm, 0).astype(np.uint16), ts, tm))
            ps, pm = predict_grid(model, P.upsample(x), ts.shape)
            res[f"{name}_model"].append(score(ps, pm, ts, tm))
            if name == "front" and args.save:
                np.savez_compressed(dataset / OUT_DIR / "front" / args.save / f"{bid}.pred.npz", shape=ps, material=pm)
        ids.append(bid)
        print(f"{bid:58s} front: alone {res['front_alone'][-1]['iou']:.3f} model {res['front_model'][-1]['iou']:.3f}"
              f" | truth: alone {res['truth_alone'][-1]['iou']:.3f} model {res['truth_model'][-1]['iou']:.3f}", flush=True)
    out = {"run": args.run, "step": ck["step"], "front": args.front, "builds": ids,
           **{k: summarize(v, KEYS) for k, v in res.items()}}
    tag = "+".join(args.front) + f"__{args.run}__{'_'.join(args.splits)}" + ("__carve" if args.carve else "")
    (dataset / OUT_DIR / f"multiview_{tag}.json").write_text(json.dumps(out, indent=1, default=float),
                                                            encoding="utf-8", newline="\n")
    print(f"\n{len(ids)} builds, {len(args.front)} photo(s)")
    print(f"{'':16s}" + "".join(f"{k:>14s}" for k in res))
    for k in ("iou", "f1_tol1", "precision", "recall", "material_acc", "block_acc"):
        print(f"{k:16s}" + "".join(f"{(out[r][k] or {}).get('mean', float('nan')):14.3f}" for r in res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
