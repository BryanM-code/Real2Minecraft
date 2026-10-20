"""The hybrid: TRELLIS's geometry, corrected by what the photo shows (the front end).

    r2m_train\\Scripts\\python -m photofirst.hybrid --dataset <BDC>/data/real-schems-v1
        --front ortho_known_auto [--front persp_known_none ...] [--trellis-set real_trellis_smooth]
        [--run r1]

TRELLIS fills open space and the front end places surfaces less exactly (BuildDataCollector
`docs/phase5.md`, "Photo-first pipeline"), so each corrects the other where it is reliable:

  carve       TRELLIS's input (2x cells) minus every block the photo shows as empty space (the
              front end's seen air, from one or several photos)
  carve_m1/m2 ... only seen air more than 1 / 2 blocks from any seen surface: the air near a
              surface depends on the depth the front end estimated, the air far from every
              surface (the silhouette's) does not
  carve_add   ... plus the front end's seen blocks that lie more than a block from any of
              TRELLIS's (parts of the building TRELLIS lost), with their photo colours
  carve_colour  carve, and the photo's colours on TRELLIS's surface cells it can see

After carving, the surface channel is recomputed and newly exposed cells take the nearest
surface colour, so the input keeps the TRELLIS pipeline's layout and its model (`--run`,
`training.evaluate.load_model` from <dataset>/runs/) applies as it is.

Several `--front` tags fuse several photos: seen air is the union (each photo shows its own
empty space); seen solid the union. Every input is in the target's frame already
(`evaluate_front` placed it). Scores: the fused input's own blocks (input as the prediction)
and, with `--run`, the model's output, over the evaluation builds that have every input.
Writes <dataset>/photofirst/hybrid_<tags>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from training.score import score, summarize

from . import data as P
from .views import OUT_DIR

MODES = ("trellis", "carve", "carve_m1", "carve_m2", "carve_add", "carve_colour")
KEYS = ("iou", "f1_tol1", "precision", "recall", "shape_acc", "material_acc", "material_de_p50", "block_acc")


def up2(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)


def surface_of(occ: np.ndarray) -> np.ndarray:
    pad = np.pad(occ, 1, constant_values=False)
    inner = np.ones_like(occ)
    for ax in range(3):
        for s in (-1, 1):
            inner &= np.roll(pad, s, axis=ax)[1:-1, 1:-1, 1:-1]
    return occ & ~inner


def fuse(trellis: np.ndarray, fronts: list, mode: str) -> np.ndarray:  # noqa: C901
    """TRELLIS's 2x input (occ, surface, R, G, B) and the front ends' block inputs (seen solid,
    seen air, R, G, B) in the same frame -> the fused 2x input."""
    if mode == "trellis":
        return trellis
    seen_air = np.zeros(fronts[0].shape[:3], bool)
    seen = np.zeros(fronts[0].shape[:3], bool)
    rgb_seen = np.zeros(fronts[0].shape[:3] + (3,), np.uint8)
    for f in fronts:
        seen_air |= f[..., 1].astype(bool)
        s = f[..., 0].astype(bool)
        rgb_seen[s & ~seen] = f[..., 2:][s & ~seen]
        seen |= s
    seen_air &= ~seen
    margin = {"carve_m1": 1, "carve_m2": 2}.get(mode, 0)
    if margin:
        seen_air &= ~binary_dilation(seen, np.ones((3, 3, 3), bool), iterations=margin)
    occ = trellis[..., 0].astype(bool) & ~up2(seen_air)
    rgb = trellis[..., 2:].copy()
    if mode == "carve_add":
        t_blocks = occ.reshape(occ.shape[0] // 2, 2, occ.shape[1] // 2, 2, occ.shape[2] // 2, 2).any((1, 3, 5))
        far = seen & ~binary_dilation(t_blocks, np.ones((3, 3, 3), bool))
        occ |= up2(far)
        rgb = np.where(up2(far)[..., None], up2(rgb_seen), rgb)
    surf = surface_of(occ)
    # colours: every surface cell needs one; newly exposed cells take the nearest coloured one
    has = trellis[..., 1].astype(bool) & surf | (mode == "carve_add") & up2(seen) & surf
    if has.any() and (surf & ~has).any():
        _, idx = distance_transform_edt(~has, return_indices=True)
        fill = surf & ~has
        rgb[fill] = rgb[idx[0][fill], idx[1][fill], idx[2][fill]]
    if mode == "carve_colour":
        near_seen = up2(binary_dilation(seen, np.ones((3, 3, 3), bool)))
        if seen.any():
            _, idx = distance_transform_edt(~up2(seen), return_indices=True)
            use = surf & near_seen
            src = up2(rgb_seen)
            rgb[use] = src[idx[0][use], idx[1][use], idx[2][use]]
    out = np.zeros_like(trellis)
    out[..., 0], out[..., 1] = occ, surf
    out[..., 2:] = np.where(surf[..., None], rgb, 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--front", nargs="+", required=True, help="front-end tags (photofirst/front/<tag>/)")
    ap.add_argument("--trellis-set", default="real_trellis_smooth",
                    help="a pairs/ set, or photofirst/<name> for TRELLIS on photofirst's own photos")
    ap.add_argument("--run", default="r1", help="a TRELLIS-pipeline run (runs/<run>/best.pt); '' for inputs only")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--save", default=None, help="save the model's output for the carve mode as photofirst/hybrid/<this>/")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    from training.evaluate import input_blocks
    from training.predict import input_from_pair, predict_grid
    store = P.Store(dataset)
    tset = (dataset / OUT_DIR / args.trellis_set.split("/", 1)[1]) if args.trellis_set.startswith("photofirst/") \
        else (dataset / "pairs" / args.trellis_set)
    index = json.loads((tset / "index.json").read_text(encoding="utf-8"))
    model = None
    if args.run:
        from training.evaluate import load_model
        model, _ = load_model(dataset / "runs" / args.run / args.checkpoint)
    rows = {m: [] for m in MODES}
    inputs_alone = {m: [] for m in MODES}
    ids = []
    for b in index["builds"]:
        bid = b["id"]
        if b.get("status") != "ok" or b.get("split") not in args.splits or b["validation"].get("failures"):
            continue
        paths = [dataset / OUT_DIR / "front" / tag / f"{bid}.npz" for tag in args.front]
        if not all(p.is_file() for p in paths) or not (tset / bid / "input.npz").is_file():
            continue
        ts, tm = store.target(bid)
        ts, tm = np.asarray(ts), np.asarray(tm)
        trellis = input_from_pair(np.load(tset / bid / "input.npz"))
        fronts = [np.load(p)["input"] for p in paths]
        line = []
        for m in MODES:
            x = fuse(trellis, fronts, m)
            occ = input_blocks(x, ts.shape)
            ps = np.where(occ, 1, 0).astype(np.uint8)
            inputs_alone[m].append(score(ps, np.where(occ, tm, 0).astype(np.uint16), ts, tm))
            if model is not None:
                ps, pm = predict_grid(model, x, ts.shape)
                rows[m].append(score(ps, pm, ts, tm))
                if args.save and m == "carve":
                    d = dataset / OUT_DIR / "hybrid" / args.save
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(d / f"{bid}.npz", shape=ps, material=pm)
                line.append(f"{m} {rows[m][-1]['iou']:.3f}")
            else:
                line.append(f"{m} {inputs_alone[m][-1]['iou']:.3f}")
        ids.append(bid)
        print(f"{bid:58s} " + "  ".join(line), flush=True)
    result = {"front": args.front, "trellis_set": args.trellis_set, "run": args.run, "splits": args.splits,
              "builds": ids, "input_alone": {m: summarize(r, KEYS) for m, r in inputs_alone.items()}}
    if model is not None:
        result["model"] = {m: summarize(r, KEYS) for m, r in rows.items()}
    tag = "+".join(args.front) + f"__{Path(args.trellis_set).name}__{args.run or 'inputs'}__{'_'.join(args.splits)}"
    (dataset / OUT_DIR / f"hybrid_{tag}.json").write_text(json.dumps(result, indent=1, default=float),
                                                         encoding="utf-8", newline="\n")
    print(f"\n{len(ids)} builds ({', '.join(args.splits)})")
    print(f"{'':16s}" + "".join(f"{m:>14s}" for m in MODES))
    for part in (["input_alone"] + (["model"] if model is not None else [])):
        print(f"-- {part}")
        for k in ("iou", "f1_tol1", "precision", "recall", "material_acc", "block_acc"):
            print(f"{k:16s}" + "".join(f"{(result[part][m][k] or {}).get('mean', float('nan')):14.3f}" for m in MODES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
