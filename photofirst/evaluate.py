"""Score a completion model, next to what its input alone gives and to the TRELLIS pipeline.

    r2m_train\\Scripts\\python -m photofirst.evaluate --dataset <BDC>/data/real-schems-v1 --run pf-a1
        [--checkpoint best.pt] [--split val|test]

Per build of the split and per view (view 0 is the evaluation photos' viewpoint, 1-3 the
other corners): the model's scores (`training.score.score`) and those of the input alone
(the seen blocks as the prediction: geometry only). On the real-TRELLIS evaluation builds
(the smooth set) view 0 is the photo TRELLIS was given, so the TRELLIS pipeline's results on
those builds (`training.evaluate`'s eval_<split>.json of runs r1 and r3, and the baseline's
building and sculpture modes) sit beside it, over the same builds. Writes
<dataset>/photofirst/runs/<run>/eval_<split>.json and the predictions,
pred_<split>/view<k>/<id>.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.evaluate import load_model
from training.predict import predict_grid
from training.score import score, summarize

from . import data as P
from .views import OUT_DIR

KEYS = ("iou", "f1_tol1", "shape_acc", "material_acc", "material_de_p50", "block_acc")
TRELLIS_RUNS = ("r1", "r3")


def input_alone(ts, tm, solid):
    """The seen blocks as a prediction (full blocks with the target's material)."""
    ps = np.where(solid, 1, 0).astype(np.uint8)
    pm = np.where(solid, np.asarray(tm), 0).astype(np.uint16)
    return score(ps, pm, np.asarray(ts), np.asarray(tm))


def trellis_rows(dataset: Path, split: str, ids: list) -> dict:
    """The TRELLIS pipeline on the smooth set, over `ids`: per run its model summary, and the
    baseline modes' summaries over the same builds (from the r1 file)."""
    out = {}
    for run in TRELLIS_RUNS:
        p = dataset / "runs" / run / f"eval_{split}.json"
        if not p.is_file():
            continue
        sm = json.loads(p.read_text(encoding="utf-8"))["sets"].get("smooth", {})
        rows = [sm["per_build"][b] for b in ids if b in sm.get("per_build", {})]
        if rows:
            out[run] = summarize(rows)
        for mode in ("building", "sculpture"):
            if f"baseline_{mode}" in sm and mode not in out:
                out[mode] = sm[f"baseline_{mode}"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    run = dataset / OUT_DIR / "runs" / args.run
    model, ck = load_model(run / args.checkpoint)
    store = P.Store(dataset)
    table = P.material_rgb()
    ids = store.ids(args.split)
    smooth = json.loads((dataset / "pairs" / "real_trellis_smooth" / "index.json").read_text(encoding="utf-8"))
    smooth_ids = sorted(b["id"] for b in smooth["builds"] if b.get("status") == "ok" and b["split"] == args.split)
    per = {}                                   # view -> id -> {"model": ..., "input": ...}
    for bid in ids:
        ts, tm = store.target(bid)
        for k in range(store.n_views(bid)):
            solid, _ = store.view(bid, k)
            ps, pm = predict_grid(model, P.full_input(store, bid, k, table), ts.shape)
            d = run / f"pred_{args.split}" / f"view{k}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{bid}.npz", shape=ps, material=pm)
            per.setdefault(k, {})[bid] = {"model": score(ps, pm, np.asarray(ts), np.asarray(tm)),
                                          "input": input_alone(ts, tm, solid)}
    result = {"run": args.run, "checkpoint": args.checkpoint, "step": ck["step"], "split": args.split,
              "views": {}, "smooth_builds": smooth_ids}
    for k, rows in per.items():
        entry = {"model": summarize([r["model"] for r in rows.values()]),
                 "input_alone": summarize([r["input"] for r in rows.values()]),
                 "per_build": {b: {"model": r["model"], "input": {x: r["input"][x] for x in ("iou", "f1_tol1")}}
                               for b, r in rows.items()}}
        if k == 0:
            sub = [b for b in smooth_ids if b in rows]
            entry["smooth_builds"] = {"model": summarize([rows[b]["model"] for b in sub]),
                                      "input_alone": summarize([rows[b]["input"] for b in sub]),
                                      "trellis": trellis_rows(dataset, args.split, sub)}
        result["views"][str(k)] = entry
    out = run / f"eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8", newline="\n")

    def fmt(s, key):
        v = (s or {}).get(key)
        return "-" if not v or v.get("mean") is None else f"{v['mean']:.3f}"
    print(f"{args.run} @ step {ck['step']}, {args.split}: {len(ids)} builds")
    for k, e in result["views"].items():
        print(f"== view {k} ({'the evaluation photo viewpoint' if k == '0' else 'another corner'})")
        print(f"{'':18s}{'model':>10s}{'input alone':>14s}")
        for key in KEYS:
            print(f"{key:18s}{fmt(e['model'], key):>10s}{fmt(e['input_alone'], key):>14s}")
    sb = result["views"]["0"]["smooth_builds"]
    cols = [("photo-first", sb["model"]), ("input alone", sb["input_alone"])] + list(sb["trellis"].items())
    print(f"\n== the {len(smooth_ids)} real-TRELLIS evaluation builds, view 0 (the photo TRELLIS was given)")
    print(f"{'':18s}" + "".join(f"{c:>14s}" for c, _ in cols))
    for key in KEYS:
        print(f"{key:18s}" + "".join(f"{fmt(s, key):>14s}" for _, s in cols))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
