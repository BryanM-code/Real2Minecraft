"""Test B: how much of what a photo shows the front end recovers, next to TRELLIS.

    r2m_train\\Scripts\\python -m photofirst.evaluate_front --dataset <BDC>/data/real-schems-v1
        [--photos ortho|persp] [--frame known|estimated] [--stretch auto|none] [--run pf-a1]

On the real-TRELLIS evaluation builds (val and test), the front end runs on each photo
(`front.photo_to_blocks`, scaled to the target's height): `ortho`, the smooth set's renders
(the photos TRELLIS was given; orthographic, so the field of view is given as 2 degrees), or
`persp`, the same builds through a 50-degree perspective camera (`photofirst.render`).
`known` gives the front end the camera's orientation (and field of view); `estimated` leaves
both to the photo, as for a real one. Its blocks are put into the target's frame by the shift
(and, for an estimated frame, the quarter turn) that best overlaps them with the target's
solid (`training.score.align`; TRELLIS's input was placed with the target too). They are
scored against what that photo truly shows (`visibility.cast` / `cast_from` from the
photo's camera):

  seen solid  recall of the truly seen blocks, exact and within one block; precision
              (blocks that are target solid), exact and within one block
  air         front-end solid on air the photo shows (false solid), front-end seen air on
              target solid (false air)

TRELLIS's input (its occupancy at block resolution) gets the same recall and false-solid
numbers, against what the orthographic photo it was given shows. With `--run`, the
completion model also runs on the front end's input (end to end, val builds), next to the
same model on the true seen blocks (test A). Writes
<dataset>/photofirst/front/<photos>_<frame>_<stretch>/<id>.npz (the front end's input in
the target frame) and front_eval_<photos>_<frame>_<stretch>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from training.score import align, score, summarize

from . import data as P
from . import front as F
from .views import OUT_DIR
from .visibility import EVAL_VIEW, IGNORE, view_axes

BALL = np.ones((3, 3, 3), bool)


def seen_metrics(solid, air, ts, t_seen, t_air) -> dict:
    """The front end's (or any input's) seen solid / seen air against the truth."""
    sup = ts != IGNORE
    t_solid = sup & (ts != 0)
    f = solid & sup
    m = {"recall": float((f & t_seen).sum() / max(t_seen.sum(), 1)),
         "recall_1": float((binary_dilation(solid, BALL) & t_seen).sum() / max(t_seen.sum(), 1)),
         "false_solid_on_seen_air": float((f & t_air).sum() / max(t_air.sum(), 1))}
    if f.any():
        m["precision"] = float((f & t_solid).sum() / f.sum())
        m["precision_1"] = float((f & binary_dilation(t_solid, BALL)).sum() / f.sum())
    if air is not None and (air & sup).any():
        m["false_air"] = float((air & t_solid).sum() / (air & sup).sum())
    return m


def photo_metrics(solid, ts) -> dict:
    """Without knowing what the photo shows (a real photo): precision of the solid (exact and
    within one block) and how much of the target's solid lies within one block of it."""
    sup = ts != IGNORE
    t_solid = sup & (ts != 0)
    f = solid & sup
    m = {"coverage_1": float((binary_dilation(solid, BALL) & t_solid).sum() / max(t_solid.sum(), 1))}
    if f.any():
        m["precision"] = float((f & t_solid).sum() / f.sum())
        m["precision_1"] = float((f & binary_dilation(t_solid, BALL)).sum() / f.sum())
    return m


def target_height(ts) -> int:
    ys = np.nonzero(((ts != 0) & (ts != IGNORE)).any((0, 2)))[0]
    return int(ys.max() - ys.min() + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--photos", default="ortho", choices=("ortho", "persp", "real"))
    ap.add_argument("--frame", default="known", choices=("known", "estimated"))
    ap.add_argument("--moge", type=int, default=2, choices=(2, 3), help="MoGe-2 or MoGe-3 (front.MODELS)")
    ap.add_argument("--view", type=int, default=0,
                    help="persp only: which corner's photo (yaw 35 + 90 view; `render --views`)")
    ap.add_argument("--stretch", default="none", choices=("auto", "none"),
                    help="auto: estimate a depth stretch (helps on the orthographic renders only)")
    ap.add_argument("--run", default=None, help="a photofirst run: the completion model end to end")
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    store = P.Store(dataset)
    table = P.material_rgb()
    smooth = dataset / "pairs" / "real_trellis_smooth"
    index = (dataset / "pairs" / "real_photo_index.json") if args.photos == "real" else (smooth / "index.json")
    builds = [b for b in json.loads(index.read_text(encoding="utf-8"))["builds"] if b.get("status") == "ok"]
    if args.photos == "real":
        args.frame = "estimated"                                  # a real photo's camera is unknown
    tag = f"{args.photos}_{args.frame}_{args.stretch}" + (f"_v{args.view}" if args.view else "")         + (f"_moge{args.moge}" if args.moge != 2 else "")
    persp = dataset / OUT_DIR / "persp"
    cams = json.loads((persp / "cameras.json").read_text(encoding="utf-8")) if args.photos == "persp" else {}
    out_dir = dataset / OUT_DIR / "front" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    moge = F.load_moge(version=args.moge)
    R_known = F.known_frame(*view_axes((EVAL_VIEW[0] + 90.0 * args.view) % 360.0, EVAL_VIEW[1]))
    from .render import view_key
    model = None
    if args.run:
        from training.evaluate import load_model
        model, _ = load_model(dataset / OUT_DIR / "runs" / args.run / args.checkpoint)
    from training.evaluate import input_blocks
    from training.predict import input_from_pair, predict_grid
    rows = []
    for b in builds:
        bid = b["id"]
        ts, tm = store.target(bid)
        ts, tm = np.asarray(ts), np.asarray(tm)
        folder = dataset / "pairs" / b.get("folder", "") / bid
        if args.photos == "real":
            if (folder / "cutout.png").is_file():
                rgba = np.array(Image.open(folder / "cutout.png").convert("RGBA"))
            else:                                                # cutout rejected: the whole photo, as TRELLIS got it
                photo = next(p for p in sorted(folder.glob("photo.*")))
                rgba = np.array(Image.open(photo).convert("RGBA"))
                rgba[..., 3] = 255
            t_seen = t_air = fov = None
        elif args.photos == "persp":
            key = view_key(bid, args.view)
            if key not in cams:
                continue
            packed = np.load(persp / f"{key}.seen.npy")
            t_seen = np.unpackbits(packed[0])[:ts.size].reshape(ts.shape).astype(bool)
            t_air = np.unpackbits(packed[1])[:ts.size].reshape(ts.shape).astype(bool)
            rgba = np.array(Image.open(persp / f"{key}.png").convert("RGBA"))
            fov = cams[key]["fov_deg"]
        else:
            t_seen, t_air = store.view(bid, 0)
            rgba = np.array(Image.open(smooth / bid / "photo.png").convert("RGBA"))
            fov = 2.0
        known = args.frame == "known"
        fb = F.photo_to_blocks(moge, rgba, target_height(ts), R=R_known if known else None,
                               fov_x=fov if known else None, stretch=args.stretch == "auto")
        _, _, info = align(fb["solid"].astype(np.uint8), np.zeros(fb["solid"].shape, np.uint16), ts, tm,
                           turns=(0,) if known else (0, 1, 2, 3))
        k, shift = info["turns"], info["shift"]
        solid = F.place(fb["solid"], k, shift, ts.shape, False)
        air = F.place(fb["air"], k, shift, ts.shape, False)
        rgb = F.place(fb["rgb"], k, shift, ts.shape, 0)
        inp = np.zeros(ts.shape + (5,), np.uint8)
        inp[..., 0], inp[..., 1], inp[..., 2:] = solid, air & ~solid, rgb
        np.savez_compressed(out_dir / f"{bid}.npz", input=inp)
        row = {"id": bid, "split": b["split"], "stretch": fb["stretch"], "fov_deg": round(fb["fov_deg"], 2),
               "align": info}
        if t_seen is None:
            row["front"] = photo_metrics(solid, ts)
            occ = input_blocks(input_from_pair(np.load(folder / "input.npz")), ts.shape)
            row["trellis"] = photo_metrics(occ, ts)
        else:
            row["front"] = {**seen_metrics(solid, air, ts, t_seen, t_air), **photo_metrics(solid, ts)}
            occ = input_blocks(input_from_pair(np.load(smooth / bid / "input.npz")), ts.shape)
            o_seen, o_air = store.view(bid, 0)
            row["trellis"] = {**seen_metrics(occ, None, ts, o_seen, o_air), **photo_metrics(occ, ts)}
        if model is not None and b["split"] == "val":
            ps, pm = predict_grid(model, P.upsample(inp), ts.shape)
            row["end_to_end"] = score(ps, pm, ts, tm)
            np.savez_compressed(out_dir / f"{bid}.pred.npz", shape=ps, material=pm)
            if t_seen is not None:
                ps, pm = predict_grid(model, P.full_input(store, bid, 0, table), ts.shape)
                row["true_seen"] = score(ps, pm, ts, tm)
        rows.append(row)
        f, t = row["front"], row["trellis"]
        seen = (f"front recall {f['recall']:.2f} (1 blk {f['recall_1']:.2f}) false-on-air {f['false_solid_on_seen_air']:.3f} "
                f"| TRELLIS recall {t['recall']:.2f} (1 blk {t['recall_1']:.2f}) false-on-air {t['false_solid_on_seen_air']:.3f} "
                if "recall" in f else "")
        print(f"{bid:58s} stretch {fb['stretch']:.2f} fov {fb['fov_deg']:.0f} turns {k}  {seen}"
              f"| front prec {f.get('precision', 0):.2f} (1 blk {f.get('precision_1', 0):.2f}) cover~1 {f['coverage_1']:.2f} "
              f"| TRELLIS prec {t.get('precision', 0):.2f} (1 blk {t.get('precision_1', 0):.2f}) cover~1 {t['coverage_1']:.2f}"
              + (f"  | e2e IoU {row['end_to_end']['iou']:.3f}" if "end_to_end" in row else "")
              + (f" (true-seen {row['true_seen']['iou']:.3f})" if "true_seen" in row else ""), flush=True)

    def mean(key, sub, sel=lambda r: True):
        v = [r[sub][key] for r in rows if sel(r) and key in r.get(sub, {})]
        return round(float(np.mean(v)), 4) if v else None
    summary = {"photos": args.photos, "frame": args.frame, "stretch": args.stretch, "builds": len(rows)}
    for sub in ("front", "trellis"):
        summary[sub] = {k: mean(k, sub) for k in ("recall", "recall_1", "precision", "precision_1",
                                                   "false_solid_on_seen_air", "false_air", "coverage_1")}
    if model is not None:
        val = [r for r in rows if "end_to_end" in r]
        summary["end_to_end_val"] = summarize([r["end_to_end"] for r in val])
        if any("true_seen" in r for r in val):
            summary["true_seen_val"] = summarize([r["true_seen"] for r in val if "true_seen" in r])
        summary["val_ids"] = [r["id"] for r in val]
    (dataset / OUT_DIR / f"front_eval_{tag}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k in ("front", "trellis")}, indent=1))
    if model is not None:
        for name in ("end_to_end_val", "true_seen_val"):
            if name not in summary:
                continue
            s = summary[name]
            print(name, {k: (s[k]["mean"] if s.get(k) else None) for k in ("iou", "f1_tol1", "shape_acc", "material_acc", "block_acc")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
