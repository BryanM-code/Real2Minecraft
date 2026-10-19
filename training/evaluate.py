"""Score a trained model next to the rule-based baseline, on the same builds and inputs.

    r2m_train\\Scripts\\python.exe -m training.evaluate --dataset <BDC>/data/real-schems-v1 --run r1
        [--checkpoint best.pt] [--split val|test] [--sets synthetic smooth blocky photos]

Sets: `synthetic` (all four variants of each build of the split; the model's own domain),
`smooth` and `blocky` (the real-TRELLIS evaluation sets: the same builds through TRELLIS,
from smooth renders and from block renders), `photos` (the real photos of builds; only
those whose build is in the split -- photos of train builds would score a model on
targets it trained on). The baseline's scores come from `training.baseline`'s run
(<dataset>/runs/baseline/): building and sculpture mode, on the same builds of the same sets
(the synthetic pairs keep no mesh, so the baseline has no synthetic column).

Test is for the final report: run it once, on the checkpoint chosen on val.

Writes <dataset>/runs/<run>/eval_<split>.json (eval_<split>_<checkpoint>.json for a checkpoint
other than best.pt, e.g. eval_val_best_trellis.json): per set the per-build scores of the model and
the summaries of the model and both baseline modes over the same builds, the shape confusion
and per-shape recall / precision, where the false solid lies against the input's own solid
(`input_excess`); and the predictions, pred_<split>/<set>/<id>.npz (shape,
material; for `synthetic` variant 0 only; the same suffix), for `training.gallery`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .model import UNet3D
from .predict import input_from_pair, predict_grid
from .score import score, summarize

KEYS = ("iou", "f1_tol1", "shape_acc", "material_acc", "material_de_p50", "block_acc")


def load_model(path: Path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]["model"]
    m = UNet3D(tuple(cfg["widths"]), cfg["blocks"], cfg["in_channels"], cfg["n_shapes"], cfg["n_materials"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def inputs(dataset: Path, cache: D.Cache, name: str, split: str):
    """(id, label, input loader) for one set and split."""
    if name == "synthetic":
        for p in cache.split(split):
            yield p["id"], f"v{p['variant']}", (lambda p=p: np.asarray(cache.input(p["id"], p["variant"])))
        return
    if name == "photos":
        sys.path.insert(0, str(dataset.parents[1]))
        from schemset.real_photo import PHOTOS, build_dir
        idx = json.loads((dataset / "pairs" / "real_photo_index.json").read_text(encoding="utf-8"))
        rows = {r["id"]: r for r in idx["builds"] if r.get("status") == "ok"}
        for p in PHOTOS:
            r = rows.get(p["id"])
            if r and r["split"] == split:
                d = build_dir(dataset, p)
                yield p["id"], "photo", (lambda d=d: input_from_pair(np.load(d / "input.npz")))
        return
    folder = dataset / "pairs" / {"smooth": "real_trellis_smooth", "blocky": "real_trellis"}[name]
    for b in json.loads((folder / "index.json").read_text(encoding="utf-8"))["builds"]:
        if b.get("status") == "ok" and b["split"] == split:
            yield b["id"], name, (lambda b=b: input_from_pair(np.load(folder / b["id"] / "input.npz")))


def input_blocks(inp: np.ndarray, dims) -> np.ndarray:
    """The input's occupancy at block level (half the cells of a block or more), as the
    validator counts it: where the input itself says solid."""
    occ = inp[..., 0] > 0
    X, Y, Z = (int(d) for d in dims)
    return occ[:2 * X, :2 * Y, :2 * Z].reshape(X, 2, Y, 2, Z, 2).mean(axis=(1, 3, 5)) >= 0.5


def baseline_rows(dataset: Path, name: str, mode: str, ids) -> dict:
    d = dataset / "runs" / "baseline" / name / mode
    out = {}
    for bid in ids:
        p = d / f"{bid}.json"
        if p.is_file():
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("scores"):
                out[bid] = r["scores"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--sets", nargs="*", default=["synthetic", "smooth", "blocky", "photos"])
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    run = dataset / "runs" / args.run
    model, ck = load_model(run / args.checkpoint)
    cache = D.Cache(dataset)
    tag = "" if args.checkpoint == "best.pt" else "_" + Path(args.checkpoint).stem
    result = {"run": args.run, "checkpoint": args.checkpoint, "step": ck["step"], "split": args.split, "sets": {}}
    from mcvoxel.vocab import SHAPES
    for name in args.sets:
        per = {}
        confusion = np.zeros((len(SHAPES), len(SHAPES)), np.int64)      # target x predicted
        excess = {"false_solid": 0, "false_solid_in_input": 0, "air_filled_by_input": 0, "true_solid": 0,
                  "missed_solid": 0, "missed_solid_in_input": 0}
        for bid, label, load in inputs(dataset, cache, name, args.split):
            ts, tm = cache.target(bid)
            ts, tm = np.asarray(ts), np.asarray(tm)
            inp = load()
            ps, pm = predict_grid(model, inp, ts.shape)
            occ = input_blocks(inp, ts.shape)
            sup, t, p = ts != D.IGNORE_SHAPE, (ts != D.IGNORE_SHAPE) & (ts != 0), (ts != D.IGNORE_SHAPE) & (ps != 0)
            for k, v in (("false_solid", p & ~t), ("false_solid_in_input", p & ~t & occ),
                         ("air_filled_by_input", sup & (ts == 0) & occ), ("true_solid", t),
                         ("missed_solid", t & ~p), ("missed_solid_in_input", t & ~p & occ)):
                excess[k] += int(v.sum())
            per.setdefault(bid, []).append(score(ps, pm, ts, tm))
            if label in ("v0", name, "photo"):
                d = run / f"pred_{args.split}{tag}" / name
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(d / f"{bid}.npz", shape=ps, material=pm)
            sup = ts != D.IGNORE_SHAPE
            confusion += np.bincount(ts[sup].astype(np.int64) * len(SHAPES) + ps[sup],
                                     minlength=len(SHAPES) ** 2).reshape(len(SHAPES), len(SHAPES))
        # one row per build: the mean over its variants (synthetic) or its one input
        model_rows = {bid: {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0] if isinstance(rs[0][k], float)}
                      for bid, rs in per.items()}
        support, predicted, hit = confusion.sum(1), confusion.sum(0), np.diag(confusion)
        entry = {"builds": sorted(model_rows), "model": summarize(list(model_rows.values())),
                 "per_build": model_rows, "shape_confusion": confusion.tolist(),
                 "per_shape": {SHAPES[i]: {"support": int(support[i]), "predicted": int(predicted[i]),
                                           "recall": round(float(hit[i] / support[i]), 4) if support[i] else None,
                                           "precision": round(float(hit[i] / predicted[i]), 4) if predicted[i] else None}
                               for i in range(len(SHAPES)) if support[i] or predicted[i]},
                 "input_excess": {**excess,
                                  "false_solid_inside_input_share": round(excess["false_solid_in_input"] / max(excess["false_solid"], 1), 4),
                                  "input_fills_air_x_true_solid": round(excess["air_filled_by_input"] / max(excess["true_solid"], 1), 3)}}
        if name != "synthetic":
            for mode in ("building", "sculpture"):
                b = baseline_rows(dataset, name, mode, model_rows)
                same = sorted(set(b) & set(model_rows))
                entry[f"baseline_{mode}"] = summarize([b[i] for i in same])
                entry[f"model_on_baseline_{mode}_builds"] = summarize([model_rows[i] for i in same])
                entry[f"baseline_{mode}_builds"] = len(same)
        result["sets"][name] = entry
        print(f"\n== {name} ({args.split}, {len(model_rows)} builds)")
        cols = ["model"] + [c for c in ("baseline_building", "baseline_sculpture") if c in entry]
        print(f"{'metric':16s}" + "".join(f"{c:>22s}" for c in cols))
        for k in KEYS:
            vals = [entry[c][k]["mean"] if entry[c].get(k) else None for c in cols]
            print(f"{k:16s}" + "".join(f"{('-' if v is None else f'{v:.3f}'):>22s}" for v in vals))
        ex = entry["input_excess"]
        print(f"  false solid {ex['false_solid']:,}, {ex['false_solid_inside_input_share']:.0%} of it where the input is solid; "
              f"the input fills supervised air {ex['input_fills_air_x_true_solid']}x the true solid")
        print("  per shape (voxels): support / predicted / recall / precision")
        for sh, v in sorted(entry["per_shape"].items(), key=lambda kv: -kv[1]["support"])[:16]:
            print(f"    {sh:20s} {v['support']:>10,} {v['predicted']:>10,} {v['recall']!s:>8} {v['precision']!s:>8}")
    out = run / f"eval_{args.split}{tag}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
