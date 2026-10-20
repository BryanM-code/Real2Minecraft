"""Per build: its target (memory-mappable copies) and what several photo viewpoints see of it
(`visibility.cast`): the completion model's training and evaluation inputs.

    r2m_train\\Scripts\\python -m photofirst.views --dataset <BDC>/data/real-schems-v1 [--workers 8]

Train builds get TRAIN_VIEWS viewpoints: yaws spread around the build (one per sector, at a
random place in it), pitch uniform in PITCH (street level to a raised camera). Val and test
builds get the evaluation photos' viewpoint first (`visibility.EVAL_VIEW`), then the other
three corners at the same pitch. Reads the dataset's index.json (splits) and
normalized/<id>.npz (targets); writes <dataset>/photofirst/: targets/<id>.shape.npy and
<id>.material.npy, views/<id>.npy (views x [seen solid, seen air] x packed bits) and
manifest.json (split, category, dims, viewpoints per build). Builds already done are kept;
--force redoes them (after the targets change).
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .visibility import EVAL_VIEW, cast

OUT_DIR = "photofirst"
TRAIN_VIEWS = 16
PITCH = (5.0, 30.0)
SEED = 20261018


def viewpoints(bid: str, split: str) -> list:
    """[(yaw, pitch), ...] for one build: fixed for val and test, seeded by the id for train."""
    if split != "train":
        yaw, pitch = EVAL_VIEW
        return [((yaw + 90.0 * k) % 360.0, pitch) for k in range(4)]
    rng = np.random.default_rng([SEED, zlib.crc32(bid.encode())])
    sector = 360.0 / TRAIN_VIEWS
    return [(round(float((k + rng.random()) * sector), 3), round(float(rng.uniform(*PITCH)), 3))
            for k in range(TRAIN_VIEWS)]


def _one(job):
    """One build (a process-pool job) -> (id, its manifest entry)."""
    dataset, row, out = job
    bid, split = row["id"], row["split"]
    z = np.load(Path(dataset) / "normalized" / f"{bid}.npz")
    ts, tm = z["target_shape"], z["target_material"]
    out = Path(out)
    np.save(out / "targets" / f"{bid}.shape.npy", ts)
    np.save(out / "targets" / f"{bid}.material.npy", tm)
    views = viewpoints(bid, split)
    packed = []
    for yaw, pitch in views:
        solid, air = cast(ts, yaw, pitch)
        packed.append(np.stack([np.packbits(solid.ravel()), np.packbits(air.ravel())]))
    np.save(out / "views" / f"{bid}.npy", np.stack(packed))
    return bid, {"split": split, "category": row.get("category_top"), "dims": [int(d) for d in ts.shape],
                 "views": [list(v) for v in views]}


def build(dataset: Path, workers: int = 8, force: bool = False, log=print) -> dict:
    dataset = Path(dataset)
    out = dataset / OUT_DIR
    (out / "targets").mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() and not force else {"builds": {}}
    rows = [r for r in json.loads((dataset / "index.json").read_text(encoding="utf-8"))["rows"]
            if r.get("split") in ("train", "val", "test")]
    todo = [(str(dataset), r, str(out)) for r in rows
            if force or r["id"] not in manifest["builds"] or not (out / "views" / f"{r['id']}.npy").is_file()]
    if log:
        log(f"{len(rows)} builds, {len(todo)} to do")
    manifest["settings"] = {"train_views": TRAIN_VIEWS, "pitch": list(PITCH), "eval_view": list(EVAL_VIEW),
                            "seed": SEED}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (bid, entry) in enumerate(pool.map(_one, todo), 1):
            manifest["builds"][bid] = entry
            if i % 25 == 0 or i == len(todo):            # saved as it goes: an interrupted run keeps its builds
                mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
                if log:
                    log(f"  {i}/{len(todo)} {bid}")
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    m = build(args.dataset.resolve(), args.workers, args.force)
    from collections import Counter
    print(dict(Counter(e["split"] for e in m["builds"].values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
