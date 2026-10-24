"""One model through the street-photo evaluation protocols, one table (`evaluate_mv` underneath).

    r2m_train\\Scripts\\python -m photofirst.eval_matrix --dataset <BDC>/data/real-schems-v2 --run pf-c1
        [--checkpoint best_front.pt] [--sets street3 street2 around4] [--placements oracle free]
        [--splits val] [--only-table]

Per photo set (`front_mv.STREET_EVAL`: three or two photos across the street front, four round
the build) and placement (`oracle`: the reconstruction put on the human build, as the training
pairs are; `free`: as on real photos, `place.free_placement` with the ground at the lowest
visible point and the true scale, the prediction put on the human build only for scoring), with compass hints (direction within 12 degrees, tilt from each photo): the street
renders (`render --street`) of every build of `--splits` through MapAnything and the model.
Writes front_eval_<run>_<set>_<placement>.json per cell (`evaluate_mv`) and
eval_matrix_<run>.json / .md (the table). Cells already evaluated are read back, not re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .views import OUT_DIR

KEYS = ("iou", "f1_tol1", "material_acc", "block_acc", "precision", "recall")


def cell_tag(run: str, s: str, placement: str) -> str:
    return f"{run}_{s}_{placement}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best_front.pt")
    ap.add_argument("--sets", nargs="*", default=["street3", "street2", "around4"])
    ap.add_argument("--placements", nargs="*", default=["oracle", "free"])
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--only-table", action="store_true", help="read the cells already evaluated, run nothing")
    args = ap.parse_args(argv)
    from . import evaluate_mv
    pf = args.dataset.resolve() / OUT_DIR
    table = {}
    for s in args.sets:
        for placement in args.placements:
            tag = cell_tag(args.run, s, placement)
            path = pf / f"front_eval_{tag}.json"
            if not path.is_file() and not args.only_table:
                argv_e = ["--dataset", str(args.dataset), "--renders", "persp_street", "--set", s, "--hints", "compass",
                          "--res", "2", "--run", args.run, "--checkpoint", args.checkpoint, "--tag", tag,
                          "--splits", *args.splits, "--passes", str(args.passes)]
                argv_e += ["--refine"] if placement == "oracle" else ["--placement", "free", "--ground", "points",
                                                                       "--free-scale", "cameras"]
                evaluate_mv.main(argv_e)
            if not path.is_file():
                continue
            summ = json.loads(path.read_text(encoding="utf-8"))["summary"]
            e2e = summ.get("end_to_end_val_input_aligned" if placement == "free" else "end_to_end_val") or {}
            table[f"{s}/{placement}"] = {"builds": summ["builds"],
                                         "front_recall_1": summ["front"].get("recall_1"),
                                         "front_precision_1": summ["front"].get("precision_1"),
                                         **{k: (e2e.get(k) or {}).get("mean") for k in KEYS}}
    out = {"run": args.run, "checkpoint": args.checkpoint, "splits": args.splits, "cells": table}
    (pf / f"eval_matrix_{args.run}.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    lines = [f"# {args.run} ({args.checkpoint}), street photos, compass hints, {'/'.join(args.splits)}", "",
             "| set / placement | builds | front recall~1 | IoU | F1~1 | material | exact |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, c in table.items():
        f = lambda v: "-" if v is None else f"{v:.3f}"
        lines.append(f"| {name} | {c['builds']} | {f(c['front_recall_1'])} | {f(c['iou'])} | {f(c['f1_tol1'])} | "
                     f"{f(c['material_acc'])} | {f(c['block_acc'])} |")
    (pf / f"eval_matrix_{args.run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
