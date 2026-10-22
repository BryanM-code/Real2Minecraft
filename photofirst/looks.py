"""Does an output look like the human build? Renders compared with image features.

The block scores (`training.score`) count exact voxels, the quality stats
(`training.quality`) count kinds of blocks; neither says whether a build reads like the human
one at a glance -- the frame posts, the roof, the window rhythm -- which is how the outputs
are judged. Here both are rendered the same way from the same corners and compared with
DINOv2 image features (ViT-L/14 with registers, the locally cached `torch.hub` weights):
`global` the cosine similarity of the image embeddings (class token), `patch` the mean over
the human render's foreground patches of their best match in the output's (local structure:
a roof edge, a window bay).

    1. ..\\BuildDataCollector\\.venv\\Scripts\\python -m photofirst.looks render --dataset <BDC>/data/real-schems-v1
           --pred pf-a3=photofirst/front/persp_known_4carve_a3 r1=runs/r1/pred_val/smooth
    2. r2m_train\\Scripts\\python -m photofirst.looks score --dataset <BDC>/data/real-schems-v1 --sets pf-a3 r1

Renders go to <dataset>/photofirst/looks/<set>/<id>_<yaw>.png (the human builds under
`human`); scores to looks/scores.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .views import OUT_DIR

YAWS = (45, 225)
PITCH = 30
SIZE = 448


def _render_grid(shape, material, yaw):
    from mcvoxel import vocab
    from schemset.review import render
    sup = (shape != 0) & (shape != vocab.IGNORE_SHAPE)
    if not sup.any():
        return None
    keys = shape.astype(np.int64) * 65536 + material.astype(np.int64)
    uniq, inv = np.unique(keys[sup], return_inverse=True)
    palette = ["minecraft:air"] + [vocab.decode(int(k // 65536), int(k % 65536)) for k in uniq]
    grid = np.zeros(shape.shape, dtype=np.uint16)
    grid[sup] = inv.ravel() + 1
    return render(grid, palette, yaw=yaw, pitch=PITCH, size=SIZE)


def render_sets(dataset: Path, sets: dict, force: bool = False) -> None:
    """Every set's builds and the matching human builds, from each of `YAWS` (BuildDataCollector's
    environment: its orbit renderer)."""
    from PIL import Image
    from training.quality import load_pred, pred_files
    sys.path.insert(0, str(dataset.parents[1]))
    root = dataset / OUT_DIR / "looks"
    ids = set()
    for name, folder in sets.items():
        for bid, f in pred_files(folder).items():
            ids.add(bid)
            _save(root / name, bid, lambda: load_pred(f), force)
    for bid in sorted(ids):
        z = np.load(dataset / "normalized" / f"{bid}.npz")
        _save(root / "human", bid, lambda: (z["target_shape"], z["target_material"]), force)


def _save(folder: Path, bid: str, load, force: bool):
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=True)
    todo = [y for y in YAWS if force or not (folder / f"{bid}_{y}.png").is_file()]
    if not todo:
        return
    s, m = (np.asarray(a) for a in load())
    for y in todo:
        im = _render_grid(s, m, y)
        if im is None:
            continue
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB").save(folder / f"{bid}_{y}.png")


def _dino(device="cuda"):
    import torch
    hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub), "dinov2_vitl14_reg", source="local", pretrained=True)
    return model.to(device).eval()


def _features(model, path: Path, device="cuda"):
    """(class embedding (C,), foreground patch embeddings (n, C)), L2-normalised."""
    import torch
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB").resize((448, 448), Image.BICUBIC), np.float32) / 255.0
    fg = (np.abs(im - 1.0).max(-1) > 0.02)
    x = (im - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    xt = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().to(device)
    with torch.no_grad():
        out = model.forward_features(xt)
    cls = torch.nn.functional.normalize(out["x_norm_clstoken"][0], dim=-1).cpu().numpy()
    patches = torch.nn.functional.normalize(out["x_norm_patchtokens"][0], dim=-1).cpu().numpy()
    g = 448 // 14
    fgp = fg.reshape(g, 14, g, 14).mean((1, 3)).ravel() > 0.3
    return cls, patches[fgp] if fgp.any() else patches


def score_sets(dataset: Path, names: list) -> dict:
    root = dataset / OUT_DIR / "looks"
    model = _dino()
    cache = {}

    def feats(p):
        if p not in cache:
            cache[p] = _features(model, p)
        return cache[p]
    out = {}
    for name in names:
        rows = {}
        for p in sorted((root / name).glob("*.png")):
            h = root / "human" / p.name
            if not h.is_file():
                continue
            (cp, pp), (ch, ph) = feats(p), feats(h)
            bid = p.stem.rsplit("_", 1)[0]
            r = rows.setdefault(bid, {"global": [], "patch": []})
            r["global"].append(float(cp @ ch))
            r["patch"].append(float((ph @ pp.T).max(1).mean()))
        out[name] = {"builds": len(rows), "global": float(np.mean([np.mean(r["global"]) for r in rows.values()])),
                     "patch": float(np.mean([np.mean(r["patch"]) for r in rows.values()])),
                     "per_build": {b: {k: float(np.mean(v)) for k, v in r.items()} for b, r in rows.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("render", "score"))
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pred", nargs="*", default=[], help="render: name=folder (relative to --dataset or absolute)")
    ap.add_argument("--sets", nargs="*", default=[], help="score: set names under looks/")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    if args.step == "render":
        sets = {}
        for item in args.pred:
            name, folder = item.split("=", 1)
            sets[name] = Path(folder) if Path(folder).is_absolute() else dataset / folder
        render_sets(dataset, sets, args.force)
        return 0
    res = score_sets(dataset, args.sets)
    path = dataset / OUT_DIR / "looks" / "scores.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1), encoding="utf-8", newline="\n")
    for name, r in res.items():
        print(f"{name:28s} builds {r['builds']:3d}  global {r['global']:.3f}  patch {r['patch']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
