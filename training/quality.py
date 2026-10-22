"""Build quality without a target: does a block grid look like a human build or like "block
vomit"? Measured the same way on the human builds (the reference) and on any pipeline's
output, over the blocks seen from outside (solid, with a face on the outside air).

    coherence   share of face-adjacent pairs of visible blocks with the same material: walls,
                floors and roofs are runs of one material
    speckle     share of visible blocks with two or more visible face neighbours, none of the
                same material: single odd blocks scattered over a surface
    palette90   number of materials that make up 90% of the visible blocks
    fragments   share of solid blocks in 26-connected pieces of under `FRAGMENT` blocks
    lonely      share of visible blocks with no solid face neighbour
    partial     share of visible blocks that are not full cubes (stairs, slabs, fences, panes,
                walls, trapdoors, doors: roofs, trim, railings, windows)
    roughness   exposed faces per visible block: 1 on a flat wall, more on edges and bumps
    roof_steps  share of the blocks seen from above (an exposed top face) that are stairs or
                slabs: a roof built as a roof, not a stepped pile of cubes
    glass       share of visible blocks of glass (windows; panes and blocks)
    posts       share of visible blocks alone in their horizontal layer (outside air on all four
                sides): posts, hanging lines, single blocks sticking out. Human builds have some
                (fence posts, lanterns, pillars); a model unsure where they go scatters them

A target (`score.IGNORE_SHAPE` inside and on unsupervised blocks): its air is the outside air
and an ignored voxel is neither solid nor air (it still joins pieces and shades faces). A
prediction: the outside air is the air connected to the grid's border.

    r2m_app\\Scripts\\python -m training.quality --dataset <BDC>/data/real-schems-v1
        --pred pf-a3=photofirst/front/persp_known_4carve_a3 r1=runs/r1/pred_val/smooth

prints the human builds' values (the builds the predictions cover) beside each set's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .score import IGNORE_SHAPE

FRAGMENT = 8
FULL_SHAPES = (1, 25, 26)              # full, full_axis_x, full_axis_z
KEYS = ("coherence", "speckle", "palette90", "fragments", "lonely", "partial", "roughness", "roof_steps", "glass",
        "posts")
STEP_SHAPES = (2, 3, 4, 5, 6, 7, 10, 11, 12, 13)    # slabs and stairs


def _glass_ids():
    from mcvoxel.vocab import MATERIALS
    return np.array([i for i, m in enumerate(MATERIALS) if "glass" in m])
_DIRS = [(a, s) for a in range(3) for s in (1, -1)]


def _shift(a: np.ndarray, axis: int, step: int) -> np.ndarray:
    """a[x + step] along `axis` (a padded grid: the border wraps onto the padding)."""
    return np.roll(a, -step, axis=axis)


def outside_air(shape: np.ndarray) -> np.ndarray:
    """Air connected to the grid's border through face-adjacent air."""
    from scipy import ndimage
    air = shape == 0
    lab, _ = ndimage.label(air)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    return np.isin(lab, border[border > 0])


def quality(shape, material, target: bool = False) -> dict:
    """The metrics above for one grid (shape uint8, material uint16)."""
    from scipy import ndimage
    shape = np.asarray(shape)
    material = np.asarray(material)
    ignored = shape == IGNORE_SHAPE
    solid = (shape != 0) & ~ignored
    out = (shape == 0) if target else outside_air(shape)
    pad = lambda a, v: np.pad(a, 1, constant_values=v)
    solid_p, out_p, mass_p = pad(solid, False), pad(out, True), pad(solid | ignored, False)
    mat_p = pad(np.where(solid, material, 0).astype(np.int64), -1)
    exposed = sum(_shift(out_p, a, s).astype(np.int32) for a, s in _DIRS) * solid_p
    vis = exposed > 0
    n_vis = int(vis.sum())
    if not n_vis:
        return {k: float("nan") for k in KEYS} | {"visible": 0}
    pairs = same = 0
    nb_vis = np.zeros(vis.shape, np.int32)
    nb_same = np.zeros(vis.shape, np.int32)
    nb_mass = np.zeros(vis.shape, np.int32)
    for a, s in _DIRS:
        v2, m2 = _shift(vis, a, s), _shift(mat_p, a, s)
        both = vis & v2
        eq = both & (m2 == mat_p)
        if s == 1:
            pairs += int(both.sum())
            same += int(eq.sum())
        nb_vis += both
        nb_same += eq
        nb_mass += _shift(mass_p, a, s)
    counted = vis & (nb_vis >= 2)
    counts = np.sort(np.bincount(mat_p[vis]))[::-1]
    lab, n = ndimage.label(solid | ignored, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel())
    pieces = np.nonzero(sizes < FRAGMENT)[0]
    small = np.isin(lab, pieces[pieces > 0]) & solid
    shp_p = pad(shape, 0)
    top = vis & _shift(out_p, 1, 1)
    return {"coherence": same / pairs if pairs else float("nan"),
            "speckle": float((counted & (nb_same == 0)).sum() / max(int(counted.sum()), 1)),
            "palette90": int(np.searchsorted(np.cumsum(counts) / counts.sum(), 0.9) + 1),
            "fragments": float(small.sum() / max(int(solid.sum()), 1)),
            "lonely": float((vis & (nb_mass == 0)).sum() / n_vis),
            "partial": float((vis & ~np.isin(shp_p, FULL_SHAPES)).sum() / n_vis),
            "roughness": float(exposed[vis].mean()),
            "roof_steps": float((top & np.isin(shp_p, STEP_SHAPES)).sum() / max(int(top.sum()), 1)),
            "glass": float((vis & np.isin(mat_p, _glass_ids())).sum() / n_vis),
            "posts": float((vis & _shift(out_p, 0, 1) & _shift(out_p, 0, -1) & _shift(out_p, 2, 1)
                            & _shift(out_p, 2, -1)).sum() / n_vis),
            "visible": n_vis}


def load_pred(path: Path) -> tuple:
    z = np.load(path)
    return z["shape"], z["material"]


def pred_files(folder: Path) -> dict:
    """{build id: file} for a folder of predictions (`<id>.pred.npz` or `<id>.npz` with
    shape and material)."""
    out = {}
    for f in sorted(folder.glob("*.npz")):
        bid = f.name[:-len(".pred.npz")] if f.name.endswith(".pred.npz") else f.stem
        if f.name.endswith(".pred.npz") or (bid not in out and "shape" in np.load(f).files):
            out[bid] = f
    return out


def table(rows: dict) -> str:
    lines = [f"{'set':24s} {'builds':>6s} " + " ".join(f"{k:>10s}" for k in KEYS)]
    for name, r in rows.items():
        lines.append(f"{name:24s} {r['builds']:6d} " + " ".join(f"{r[k]:10.3f}" for k in KEYS))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="name=folder (relative to --dataset or absolute)")
    ap.add_argument("--out", type=Path, default=None, help="JSON with every build's values")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = args.dataset.resolve()
    sets = {}
    for item in args.pred:
        name, folder = item.split("=", 1)
        folder = Path(folder) if Path(folder).is_absolute() else dataset / folder
        sets[name] = pred_files(folder)
    ids = sorted(set().union(*sets.values()))
    per = {"human": {}}
    for bid in ids:
        z = np.load(dataset / "normalized" / f"{bid}.npz")
        per["human"][bid] = quality(z["target_shape"], z["target_material"], target=True)
    for name, files in sets.items():
        per[name] = {bid: quality(*load_pred(f)) for bid, f in files.items()}
    rows = {}
    for name, d in per.items():
        rows[name] = {"builds": len(d), **{k: float(np.nanmean([r[k] for r in d.values()])) for k in KEYS}}
    for name, files in sets.items():           # the human builds over the same builds as each set
        sub = [per["human"][b] for b in files]
        rows[f"  human, {name}'s builds"] = {"builds": len(sub), **{k: float(np.nanmean([r[k] for r in sub])) for k in KEYS}}
    print(table(rows))
    if args.out:
        args.out.write_text(json.dumps({"summary": rows, "builds": per}, indent=1, default=float), encoding="utf-8",
                            newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ── where the details are, against the human build ──────────────────────────────────────
PART_SHAPES = {"stairs": (4, 5, 6, 7, 10, 11, 12, 13), "slab": (2, 3),
               "thin": (8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29)}


def _near(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    return binary_dilation(a, np.ones((3, 3, 3), bool))


def parts(pred_shape, pred_material, tgt_shape, tgt_material) -> dict:
    """Per kind of detail (stairs, slabs, thin parts: fences, walls, panes, trapdoors, doors,
    carpets; glass: windows), within one block: precision (a predicted one has one of the
    target's within a block), recall (a target's one has a predicted one within a block) and F1;
    and `stair_facing`: of the predicted stairs on a target stair voxel, the share with the
    target's facing and half. Both grids in the target's frame; only voxels the target
    supervises count."""
    ps, pm = np.asarray(pred_shape), np.asarray(pred_material)
    ts, tm = np.asarray(tgt_shape), np.asarray(tgt_material)
    sup = ts != IGNORE_SHAPE
    glass = _glass_ids()
    kinds = {k: (np.isin(ps, v), np.isin(ts, v)) for k, v in PART_SHAPES.items()}
    kinds["glass"] = ((ps != 0) & np.isin(pm, glass), (ts != 0) & sup & np.isin(tm, glass))
    out = {}
    for k, (p, t) in kinds.items():
        p = p & sup
        t = t & sup
        prec = float((p & _near(t)).sum() / p.sum()) if p.any() else float("nan")
        rec = float((t & _near(p)).sum() / t.sum()) if t.any() else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else float("nan")
        out[k] = {"precision": prec, "recall": rec, "f1": f1, "pred": int(p.sum()), "target": int(t.sum())}
    both = kinds["stairs"][0] & kinds["stairs"][1] & sup
    out["stair_facing"] = float((ps[both] == ts[both]).mean()) if both.any() else float("nan")
    return out
