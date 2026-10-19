"""A page to look at a run: the human build, the TRELLIS input, the model and the baseline.

Runs in BuildDataCollector's environment, for its renderer (`schemset.pairs_gallery`):

    ..\\BuildDataCollector\\.venv\\Scripts\\python -m training.gallery --dataset <BDC>/data/real-schems-v1
        --run r1 [--split val] [--checkpoint best.pt]

One row per build of the split: target · the smooth-photo TRELLIS input (voxelised) · the
model on it · building mode · sculpture mode · the model on the synthetic input (variant
0), each tile with its IoU, exact blocks and material accuracy. Every tile is the same
view of the same frame, so the columns line up. Reads `training.evaluate`'s predictions
(pred_<split>/) and scores (eval_<split>.json) and `training.baseline`'s aligned grids.
Writes <dataset>/runs/<run>/gallery_<split>/index.html and img/; for a checkpoint other than
best.pt, every name carries its stem (eval_val_best_trellis.json -> gallery_val_best_trellis/).
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

COLUMNS = (("target", "Human build (target)"), ("input", "TRELLIS input (smooth photo)"),
           ("model", "Model"), ("building", "Building mode"), ("sculpture", "Sculpture mode"),
           ("model_syn", "Model, synthetic input"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--checkpoint", default="best.pt", help="the checkpoint `training.evaluate` scored")
    args = ap.parse_args(argv)
    tag = "" if args.checkpoint == "best.pt" else "_" + Path(args.checkpoint).stem
    dataset = args.dataset.resolve()
    sys.path.insert(0, str(dataset.parents[1]))
    from schemset.pairs_gallery import _input_image, _target_image
    run = dataset / "runs" / args.run
    ev = json.loads((run / f"eval_{args.split}{tag}.json").read_text(encoding="utf-8"))
    out = run / f"gallery_{args.split}{tag}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    smooth = ev["sets"].get("smooth", {})
    syn = ev["sets"].get("synthetic", {})
    ids = sorted(set(smooth.get("builds", [])) | set(syn.get("builds", [])))
    base = dataset / "runs" / "baseline" / "smooth"
    rows = []
    for bid in ids:
        z = np.load(dataset / "normalized" / f"{bid}.npz")
        tiles = {}
        grids = {"target": (z["target_shape"], z["target_material"])}
        for key, path in (("model", run / f"pred_{args.split}{tag}" / "smooth" / f"{bid}.npz"),
                          ("model_syn", run / f"pred_{args.split}{tag}" / "synthetic" / f"{bid}.npz"),
                          ("building", base / "building" / f"{bid}.npz"),
                          ("sculpture", base / "sculpture" / f"{bid}.npz")):
            if path.is_file():
                g = np.load(path)
                grids[key] = (g["shape"], g["material"])
        for key, (s, m) in grids.items():
            img = _target_image(np.asarray(s), np.asarray(m))
            if img is not None:
                img.save(out / "img" / f"{bid}__{key}.png")
                tiles[key] = f"img/{bid}__{key}.png"
        inp = dataset / "pairs" / "real_trellis_smooth" / bid / "input.npz"
        if inp.is_file():
            zi = np.load(inp)
            img = _input_image(zi["input_occ"], zi["input_rgb"])
            if img is not None:
                img.save(out / "img" / f"{bid}__input.png")
                tiles["input"] = f"img/{bid}__input.png"
        scores = {"model": smooth.get("per_build", {}).get(bid), "model_syn": syn.get("per_build", {}).get(bid)}
        for mode in ("building", "sculpture"):
            p = base / mode / f"{bid}.json"
            if p.is_file():
                r = json.loads(p.read_text(encoding="utf-8"))
                scores[mode] = r.get("scores")
                if r.get("failed"):
                    scores[mode] = {**(r.get("scores") or {}), "failed": True}
        rows.append((bid, tiles, scores))
        print(f"  {bid}: {len(tiles)} tiles", flush=True)

    def caption(sc):
        if not sc:
            return ""
        if sc.get("failed"):
            return "<div class=s>failed: no build</div>"
        f = lambda k: "-" if sc.get(k) is None or sc[k] != sc[k] else f"{sc[k]:.2f}"
        return f"<div class=s>IoU {f('iou')} · F1~1 {f('f1_tol1')} · exact {f('block_acc')} · material {f('material_acc')}</div>"

    head = "".join(f"<th>{html.escape(t)}</th>" for _, t in COLUMNS)
    body = []
    for bid, tiles, scores in rows:
        cells = []
        for key, _ in COLUMNS:
            img = f"<img src='{tiles[key]}' loading=lazy>" if key in tiles else "<div class=e>-</div>"
            cells.append(f"<td>{img}{caption(scores.get(key))}</td>")
        body.append(f"<tr><th class=id>{html.escape(bid)}</th>{''.join(cells)}</tr>")
    summ = {k: v.get("model", {}).get("iou", {}).get("mean") if isinstance(v, dict) and v.get("model") and v["model"].get("iou") else None
            for k, v in ev["sets"].items()}
    page = f"""<!doctype html><meta charset=utf-8><title>{html.escape(args.run)}: {args.split}</title>
<style>body{{font:13px system-ui;margin:16px;background:#fafaf8;color:#222}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px;vertical-align:top;text-align:center}} img{{width:220px;height:220px;object-fit:contain}}
th.id{{max-width:120px;word-break:break-all;font-weight:600}} .s{{font-size:11px;color:#555}} .e{{width:220px;height:220px;line-height:220px;color:#aaa}}</style>
<h1>{html.escape(args.run)} · {args.split} · step {ev['step']}</h1>
<p>Mean IoU by set: {html.escape(json.dumps(summ))}. Every tile is the same view of the target's frame.</p>
<table><tr><th></th>{head}</tr>{''.join(body)}</table>"""
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"-> {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
