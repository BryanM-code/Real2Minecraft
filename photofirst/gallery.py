"""A page to look at the photo-first pipeline next to the TRELLIS pipeline, build by build.

Runs in BuildDataCollector's environment, for its renderer (`schemset.pairs_gallery`):

    ..\\BuildDataCollector\\.venv\\Scripts\\python -m photofirst.gallery --dataset <BDC>/data/real-schems-v1
        --run pf-a1 [--front persp_estimated_none] [--trellis-run r1]

One row per real-TRELLIS evaluation build of val: the human build (target) · the photo ·
the blocks the front end took from it · the photo-first result (front end + completion
model, `evaluate_front --run`) · the completion model given exactly what the photo shows
(test A, `photofirst.evaluate`) · the TRELLIS pipeline's model · sculpture mode, each with
its IoU, exact blocks and material accuracy. Every tile is the same view of the target's
frame. Writes <dataset>/photofirst/gallery_<front>/index.html and img/.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .views import OUT_DIR

COLUMNS = (("target", "Human build (target)"), ("photo", "Photo"), ("front", "Blocks the front end saw"),
           ("pf", "Photo-first (front end + completion)"), ("pf_true", "Completion of exactly what the photo shows"),
           ("trellis", "TRELLIS pipeline (model)"), ("sculpture", "Sculpture mode (TRELLIS)"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--front", default="persp_estimated_none")
    ap.add_argument("--trellis-run", default="r1")
    args = ap.parse_args(argv)
    dataset = args.dataset.resolve()
    sys.path.insert(0, str(dataset.parents[1]))
    from schemset.pairs_gallery import _input_image, _target_image
    pf = dataset / OUT_DIR
    front_dir = pf / "front" / args.front
    ev = json.loads((pf / f"front_eval_{args.front}.json").read_text(encoding="utf-8"))
    rows_ev = {r["id"]: r for r in ev["rows"]}
    tr = json.loads((dataset / "runs" / args.trellis_run / "eval_val.json").read_text(encoding="utf-8"))
    tr_rows = tr["sets"]["smooth"]["per_build"]
    out = pf / f"gallery_{args.front}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    photos = args.front.split("_")[0]
    rows = []
    for bid, r in sorted(rows_ev.items()):
        if "end_to_end" not in r:
            continue                                            # val builds only
        z = np.load(dataset / "normalized" / f"{bid}.npz")
        grids = {"target": (z["target_shape"], z["target_material"])}
        for key, path in (("pf", front_dir / f"{bid}.pred.npz"),
                          ("pf_true", pf / "runs" / args.run / "pred_val" / "view0" / f"{bid}.npz"),
                          ("trellis", dataset / "runs" / args.trellis_run / "pred_val" / "smooth" / f"{bid}.npz"),
                          ("sculpture", dataset / "runs" / "baseline" / "smooth" / "sculpture" / f"{bid}.npz")):
            if path.is_file():
                g = np.load(path)
                grids[key] = (g["shape"], g["material"])
        tiles = {}
        for key, (s, m) in grids.items():
            img = _target_image(np.asarray(s), np.asarray(m))
            if img is not None:
                img.save(out / "img" / f"{bid}__{key}.png")
                tiles[key] = f"img/{bid}__{key}.png"
        inp = np.load(front_dir / f"{bid}.npz")["input"]
        up = lambda a: np.repeat(np.repeat(np.repeat(a, 2, 0), 2, 1), 2, 2)
        img = _input_image(up(inp[..., 0]), up(inp[..., 2:]))
        if img is not None:
            img.save(out / "img" / f"{bid}__front.png")
            tiles["front"] = f"img/{bid}__front.png"
        src = (pf / "persp" / f"{bid}.png") if photos == "persp" else \
            (dataset / "pairs" / "real_trellis_smooth" / bid / "photo.png")
        if src.is_file():
            shutil.copyfile(src, out / "img" / f"{bid}__photo.png")
            tiles["photo"] = f"img/{bid}__photo.png"
        scores = {"pf": r.get("end_to_end"), "pf_true": r.get("true_seen"), "trellis": tr_rows.get(bid)}
        p = dataset / "runs" / "baseline" / "smooth" / "sculpture" / f"{bid}.json"
        if p.is_file():
            scores["sculpture"] = json.loads(p.read_text(encoding="utf-8")).get("scores")
        rows.append((bid, tiles, scores))
        print(f"  {bid}: {len(tiles)} tiles", flush=True)

    def caption(sc):
        if not sc:
            return ""
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
    page = f"""<!doctype html><meta charset=utf-8><title>photo-first: {html.escape(args.front)}</title>
<style>body{{font:13px system-ui;margin:16px;background:#fafaf8;color:#222}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px;vertical-align:top;text-align:center}} img{{width:220px;height:220px;object-fit:contain}}
th.id{{max-width:120px;word-break:break-all;font-weight:600}} .s{{font-size:11px;color:#555}} .e{{width:220px;height:220px;line-height:220px;color:#aaa}}</style>
<h1>Photo-first ({html.escape(args.front)}, completion model {html.escape(args.run)}) next to TRELLIS ({html.escape(args.trellis_run)})</h1>
<p>Val builds of the real-TRELLIS evaluation set. Every block tile is the same view of the target's frame.</p>
<table><tr><th></th>{head}</tr>{''.join(body)}</table>"""
    (out / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"-> {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
