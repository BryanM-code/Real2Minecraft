"""Training pairs from the multi-photo front end (`front_mv`): sets of 2-4 of each training
build's photos (`render --train-views`, sets by `front_mv.view_subsets`) through MapAnything
with camera hints, or another front end's saved arrays (`front_da3 --train`), placed on the
target (`front_mv.refine_to_target`: input and target must line up, as the TRELLIS pairs'
placement does) and turned into blocks (`front_mv.to_grid`).

    r2m_train\\Scripts\\python -m photofirst.mv_pairs --dataset <BDC>/data/real-schems-v1
        [--hints rough] [--tag ma_rough] [--views-from da3_train_rough] [--splits train]

Writes <dataset>/photofirst/mv_pairs/<tag>/<id>_s<j>.npz (input (X, Y, Z, 5) uint8, block
resolution) and index.json (per pair: photos, placement, how well its seen blocks match what
those photos truly show). Resumable: pairs already written are kept. The completion model
trains on them with `train --real-pairs <tag>`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from PIL import Image

from . import data as P
from . import front_mv as M
from .evaluate_front import seen_metrics
from .render import TRAIN_DIR
from .views import OUT_DIR


def truth_seen(folder: Path, keys: list, dims):
    """The union of what photos `keys` truly show (`render_build`'s .seen.npy), or None when
    they were rendered without it."""
    if not all((folder / f"{k}.seen.npy").is_file() for k in keys):
        return None
    n = int(np.prod(dims))
    seen = np.zeros(tuple(dims), bool)
    air = np.zeros(tuple(dims), bool)
    for k in keys:
        packed = np.load(folder / f"{k}.seen.npy")
        seen |= np.unpackbits(packed[0])[:n].reshape(tuple(dims)).astype(bool)
        air |= np.unpackbits(packed[1])[:n].reshape(tuple(dims)).astype(bool)
    return seen, air & ~seen


def _worker_init():
    M.FFT_WORKERS = 2


def free_pair(views, masks, sim, ts, tm, eye: float, res: int) -> tuple:
    """A photo set placed as on real photos (`place.free_placement`, the true scale; the
    ground at the lowest visible point, or with `eye` at eye height below the cameras) and the
    human build moved into that frame: the
    inverse of the input's alignment onto it (`front_mv.onto_target`, `front_mv.target_into`).
    Returns (input, target shape, target material, what was done)."""
    from . import place as PL
    sim_f, dims_f, pinfo = PL.free_placement(views, masks, metres_per_block=1.0 / sim["s"],
                                             eye_units=None if eye is None else eye / sim["s"])
    inp = M.to_grid(views, sim_f, dims_f, masks, res=res)
    _, ainfo = M.onto_target(inp, dims_f, ts, tm)
    t_s, t_m = M.target_into(ts, tm, ainfo, dims_f)
    return inp, t_s, t_m, {"free": pinfo, "alignment": ainfo}


def place_pair(job) -> tuple:
    """The CPU part of one photo set (in a worker process when `--workers` > 1): placement on
    the target, blocks, written atomically (a stopped run leaves no half file), metrics.
    With placement "oracle+free" also the pair placed as on real photos (`free_pair`), into
    `<out_dir>_free`. Returns (name, index entry, seconds, the free pair's entry or None)."""
    import os
    dataset, out_dir, bid, name, keys, views, res, placement, renders = job
    t = time.time()
    pf = Path(dataset) / OUT_DIR
    folder = pf / renders / bid
    cams = json.loads((folder / "cameras.json").read_text(encoding="utf-8"))
    ts = np.load(pf / "targets" / f"{bid}.shape.npy")
    h, w = views[0]["pts"].shape[:2]
    masks = [np.array(Image.fromarray(np.array(Image.open(folder / f"{k}.png").convert("RGBA"))[..., 3] > 127)
                      .resize((w, h), Image.NEAREST)) for k in keys]
    sim = M.similarity_from_cameras([v["cam2world"] for v in views], [cams[k]["centre"] for k in keys],
                                    M.true_rotations(cams, keys))
    free_entry = None
    if placement == "oracle+free":
        tm = np.load(pf / "targets" / f"{bid}.material.npy")
        f_inp, f_s, f_m, finfo = free_pair(views, masks, sim, ts, tm, None, res)   # ground: lowest point
        free_dir = Path(str(out_dir) + "_free")
        tmp = free_dir / f"{name}.part.npz"
        np.savez_compressed(tmp, input=f_inp, target_shape=f_s, target_material=f_m)
        os.replace(tmp, free_dir / f"{name}.npz")
        free_entry = {"build": bid, "views": [int(k) for k in keys], "dims": list(f_s.shape), **finfo}
    if placement == "free":
        from . import place as PL
        tm = np.load(pf / "targets" / f"{bid}.material.npy")
        sim_f, dims_f, pinfo = PL.free_placement(views, masks, metres_per_block=1.0 / sim["s"])
        inp, ainfo = M.onto_target(M.to_grid(views, sim_f, dims_f, masks, res=res), dims_f, ts, tm)
        sim = {**sim, "refine": {"free": pinfo, "alignment": ainfo}}
    else:
        sim = M.refine_to_target(views, sim, (ts != 0) & (ts != 255), masks,
                                 coarse_scales=np.exp(np.linspace(np.log(0.75), np.log(1.33), 11)), fast=True)
        inp = M.to_grid(views, sim, ts.shape, masks, res=res)
    tmp = Path(out_dir) / f"{name}.part.npz"
    np.savez_compressed(tmp, input=inp)
    os.replace(tmp, Path(out_dir) / f"{name}.npz")
    truth = truth_seen(folder, keys, ts.shape)
    blocks = M.to_blocks(inp, ts.shape)
    m = seen_metrics(blocks[..., 0].astype(bool), blocks[..., 1].astype(bool), ts, *truth) if truth else {}
    entry = {"build": bid, "views": [int(k) for k in keys], "yaw": [cams[k]["yaw"] for k in keys],
             "pitch": [cams[k]["pitch"] for k in keys], "scale": sim["s"], "refine": sim["refine"],
             "front": {k: round(float(v), 4) for k, v in m.items()}}
    return name, entry, time.time() - t, free_entry


def write_index(path: Path, index: dict) -> None:
    """`index` merged into the one on disk (another process may be writing pairs to the same
    folder), written whole through a temporary file."""
    import os
    if path.is_file():
        try:
            disk = json.loads(path.read_text(encoding="utf-8"))
            index["pairs"] = {**disk.get("pairs", {}), **index["pairs"]}
        except (OSError, ValueError):
            pass
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(index, indent=1, default=float), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--hints", choices=M.HINTS, default="rough")
    ap.add_argument("--views-from", default=None, help="saved arrays: <dataset>/photofirst/mv_views/<tag>/<id>_s<j>.npz")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--splits", nargs="*", default=["train"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0, help="at most this many builds (a trial)")
    ap.add_argument("--res", type=int, default=2, choices=(1, 2), help="front_mv.to_grid cells per block")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the CPU part (placement, blocks) while MapAnything runs the next set")
    ap.add_argument("--placement", choices=("oracle", "free", "oracle+free"), default="oracle",
                    help="free: placed as on real photos (place.free_placement, at the true cameras' scale), then "
                         "turned and shifted onto the target by quarter turns and whole blocks (front_mv.onto_target); "
                         "oracle+free: the oracle pairs and, into <tag>_free, the same reconstruction placed as on "
                         "real photos with the ground from the cameras and the human build moved into that frame "
                         "(free_pair; the pair carries its own target)")
    ap.add_argument("--renders", default=TRAIN_DIR,
                    help="the training photos: persp_train (render --train-views) or persp_street (render --street)")
    ap.add_argument("--reverse", action="store_true",
                    help="builds in reverse order: a second process alongside a first, the same --tag")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dataset = args.dataset.resolve()
    pf = dataset / OUT_DIR
    tag = args.tag or (args.views_from or f"ma_{args.hints}")
    out_dir = pf / "mv_pairs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"pairs": {}}
    free_dir = Path(str(out_dir) + "_free")
    free_path = free_dir / "index.json"
    free_index = {"pairs": {}}
    if args.placement == "oracle+free":
        free_dir.mkdir(parents=True, exist_ok=True)
        if free_path.is_file():
            free_index = json.loads(free_path.read_text(encoding="utf-8"))
        free_index.update({"front_end": "mapanything", "hints": args.hints, "res": args.res,
                           "placement": "free_ground", "renders": args.renders, "own_targets": True})
    index.update({"front_end": args.views_from or "mapanything", "hints": args.hints, "res": args.res,
                  "placement": args.placement, "renders": args.renders})
    store = P.Store(dataset)
    ma = None if args.views_from else M.load_mapanything()
    tmp = pf / "tmp_mv" / "train"
    tmp.mkdir(parents=True, exist_ok=True)
    builds = [b for b in sorted(store.builds) if store.builds[b]["split"] in args.splits
              and (not args.ids or b in args.ids) and (pf / args.renders / b / "cameras.json").is_file()]
    if args.reverse:
        builds = builds[::-1]
    if args.limit:
        builds = builds[:args.limit]
    t_all = time.time()
    if args.workers > 1:
        import os
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # inherited by the workers
            os.environ[k] = "2"
    pool = ProcessPoolExecutor(args.workers, initializer=_worker_init) if args.workers > 1 else None
    pending, done = set(), 0

    def collect(block: bool):
        nonlocal pending, done
        if not pending:
            return
        ready, pending = wait(pending, return_when=FIRST_COMPLETED) if block else             ({f for f in pending if f.done()}, {f for f in pending if not f.done()})
        for f in ready:
            name, entry, seconds, free_entry = f.result()
            index["pairs"][name] = entry
            if free_entry is not None:
                free_index["pairs"][name] = free_entry
            done += 1
            m = entry["front"]
            print(f"[{done}] {name:50s} {len(entry['views'])} photos {seconds:5.1f} s | recall "
                  f"{m.get('recall', float('nan')):.2f} (1 blk {m.get('recall_1', float('nan')):.2f}) "
                  f"prec {m.get('precision', float('nan')):.2f} (1 blk {m.get('precision_1', float('nan')):.2f})",
                  flush=True)
        if ready:
            write_index(index_path, index)
            if args.placement == "oracle+free":
                write_index(free_path, free_index)

    for i, bid in enumerate(builds, 1):
        folder = pf / args.renders / bid
        cams = json.loads((folder / "cameras.json").read_text(encoding="utf-8"))
        for j, keys, K, poses in M.pair_jobs(bid, cams, args.hints):
            name = f"{bid}_s{j}"
            if (out_dir / f"{name}.npz").is_file() and (args.placement != "oracle+free" or (free_dir / f"{name}.npz").is_file()):
                if name not in index["pairs"]:          # written by a process whose index entry was lost
                    index["pairs"][name] = {"build": bid, "views": [int(k) for k in keys], "adopted": True}
                continue
            paths = []
            for k in keys:
                paths.append(tmp / f"{k}.png")
                if ma is not None:
                    im = np.array(Image.open(folder / f"{k}.png").convert("RGBA"))
                    Image.fromarray(np.where(im[..., 3:] > 127, im[..., :3], 255).astype(np.uint8)).save(paths[-1])
            if args.views_from:
                src = pf / "mv_views" / args.views_from / f"{name}.npz"
                if not src.is_file():
                    continue
                views = M.load_views(src)
            else:
                views = M.run_mapanything(ma, paths, K, poses)
            job = (dataset, out_dir, bid, name, keys, views, args.res, args.placement, args.renders)
            if pool is None:
                name, entry, seconds, free_entry = place_pair(job)
                index["pairs"][name] = entry
                if free_entry is not None:
                    free_index["pairs"][name] = free_entry
                print(f"[{i}/{len(builds)}] {name:50s} {len(keys)} photos {seconds:5.1f} s", flush=True)
                continue
            pending.add(pool.submit(place_pair, job))
            while len(pending) >= 2 * args.workers:
                collect(block=True)
            collect(block=False)
        if pool is None:
            write_index(index_path, index)
            if args.placement == "oracle+free":
                write_index(free_path, free_index)
    while pending:
        collect(block=True)
    if pool is not None:
        pool.shutdown()
    write_index(index_path, index)
    if args.placement == "oracle+free":
        write_index(free_path, free_index)
    rows = [r for r in index["pairs"].values() if "front" in r]
    print(f"{len(index['pairs'])} pairs, {time.time() - t_all:.0f} s; mean recall within 1 "
          f"{np.nanmean([r['front'].get('recall_1', np.nan) for r in rows] or [np.nan]):.3f}, precision within 1 "
          f"{np.nanmean([r['front'].get('precision_1', np.nan) for r in rows] or [np.nan]):.3f} (pairs with seen sets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
