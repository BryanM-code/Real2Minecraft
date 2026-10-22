"""Depth Anything 3 for the multi-photo front end (`front_mv`), in its own environment.

    r2m_da3\\Scripts\\python -m photofirst.front_da3 --dataset <BDC>/data/real-schems-v1
        --views 0 1 2 3 [--splits val test] [--hints none|rough|coarse|true] [--tag da3]

DA3 (`depth-anything/DA3-LARGE-1.1`, Apache 2.0) needs numpy < 2, so it runs in the r2m_da3
venv (photofirst/requirements.txt) and only writes arrays. Per build with renders of all
`--views`, the photos go through DA3 together: with --hints, conditioned on camera hints
(`front_mv.render_hints`; DA3 takes poses only with intrinsics). The per-photo arrays
`front_mv` uses (world points, camera-frame rays, cam2world, mask, colours) go to
<dataset>/photofirst/mv_views/<tag>/<id>.npz; `evaluate_mv --views-from <tag>` scores them
as it scores MapAnything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import front_mv as M
from .render import view_key
from .views import OUT_DIR

DA3 = "depth-anything/DA3-LARGE-1.1"


def load_da3(device="cuda"):
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(DA3).to(device).eval()


def run_da3(model, images: list, intrinsics=None, poses=None, process_res: int = 504) -> list:
    """Photos (uint8 RGB arrays or paths) -> per photo the `front_mv` view dict at DA3's
    resolution. Hints as in `front_mv.run_mapanything`; the result keeps DA3's own cameras,
    put in the hints' frame (a similarity), not the hinted cameras themselves."""
    ext = None if poses is None else np.stack([np.linalg.inv(np.asarray(T, float)) for T in poses]).astype(np.float32)
    ixt = None if intrinsics is None else np.stack([np.asarray(K, float) for K in intrinsics]).astype(np.float32)
    pred = model.inference(list(images), extrinsics=ext, intrinsics=ixt, align_to_input_ext_scale=False,
                           process_res=process_res)
    out = []
    for i in range(len(images)):
        depth = np.asarray(pred.depth[i], np.float64)
        K = np.asarray(pred.intrinsics[i], np.float64)
        w2c = np.eye(4)
        w2c[:3] = np.asarray(pred.extrinsics[i], np.float64)[:3]
        c2w = np.linalg.inv(w2c)
        h, w = depth.shape
        x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64), indexing="xy")
        d = np.stack([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(x)], -1)   # z = 1
        pts = (d * depth[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]
        conf = np.asarray(pred.conf[i], np.float64) if pred.conf is not None else np.ones_like(depth)
        mask = np.isfinite(depth) & (depth > 0) & (conf > np.quantile(conf, 0.10)) & ~M.depth_edges(depth)
        out.append({"pts": pts.astype(np.float32), "ray": (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32),
                    "cam2world": c2w.astype(np.float32), "mask": mask,
                    "rgb": np.asarray(pred.processed_images[i], np.uint8)})
    return out


def train_sets(model, pf: Path, out_dir: Path, args) -> int:
    from .render import TRAIN_DIR
    manifest = json.loads((pf / "manifest.json").read_text(encoding="utf-8"))["builds"]
    for bid in sorted(manifest):
        folder = pf / TRAIN_DIR / bid
        if manifest[bid]["split"] not in args.splits or (args.ids and bid not in args.ids) \
                or not (folder / "cameras.json").is_file():
            continue
        cams = json.loads((folder / "cameras.json").read_text(encoding="utf-8"))
        t = time.time()
        jobs = M.pair_jobs(bid, cams, args.hints)
        for j, keys, K, poses in jobs:
            if (out_dir / f"{bid}_s{j}.npz").is_file():
                continue
            images = []
            for k in keys:
                im = np.array(Image.open(folder / f"{k}.png").convert("RGBA"))
                images.append(np.where(im[..., 3:] > 127, im[..., :3], 255).astype(np.uint8))
            M.save_views(out_dir / f"{bid}_s{j}.npz", run_da3(model, images, K, poses))
        print(f"{bid:58s} {len(jobs)} sets {time.time() - t:5.1f} s", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--splits", nargs="*", default=None, help="default: val test (train with --train)")
    ap.add_argument("--hints", choices=[h for h in M.HINTS if h != "intrinsics"], default="none")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--train", action="store_true",
                    help="the training builds' photo sets (render --train-views, front_mv.pair_jobs) for mv_pairs")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    pf = args.dataset.resolve() / OUT_DIR
    out_dir = pf / "mv_views" / (args.tag or f"da3_{'train_' if args.train else ''}{args.hints}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_da3()
    if args.train:
        args.splits = args.splits or ["train"]
        return train_sets(model, pf, out_dir, args)
    args.splits = args.splits or ["val", "test"]
    cams = json.loads((pf / "persp" / "cameras.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    for bid in sorted({c.get("build", k) for k, c in cams.items()}):
        keys = [view_key(bid, v) for v in args.views]
        if (args.ids and bid not in args.ids) or not all(k in cams for k in keys) \
                or cams[keys[0]].get("split") not in args.splits:
            continue
        images = []
        for k in keys:
            im = np.array(Image.open(pf / "persp" / f"{k}.png").convert("RGBA"))
            images.append(np.where(im[..., 3:] > 127, im[..., :3], 255).astype(np.uint8))
        K, poses = M.render_hints(cams, keys, args.hints, rng)
        t = time.time()
        views = run_da3(model, images, K, poses)
        M.save_views(out_dir / f"{bid}.npz", views)
        print(f"{bid:58s} {time.time() - t:5.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
