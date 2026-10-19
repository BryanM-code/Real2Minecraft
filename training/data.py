"""Phase 5 data: the pairs as a memory-mapped cache, training windows and their symmetries.

The contract is BuildDataCollector docs/pairs-v1.md ("Loader contract"): 64^3 target windows
with their 128^3 input windows (input cell 2s <-> target block s), padding with IGNORE /
zero, the 8 symmetries of the square applied to both grids through `mcvoxel.blockstates`,
IGNORE in the loss, and metrics with and without the `low_iou` pairs.

    python -m training.data cache --dataset <BDC>/data/real-schems-v1
    python -m training.data trellis --dataset <BDC>/data/real-schems-v1    # real-TRELLIS train inputs

writes <dataset>/pairs_cache/: per build the target once (`target_shape.npy` uint8,
`target_material.npy` uint16, X x Y x Z) and per variant `v<k>.input.npy` (uint8,
2X x 2Y x 2Z x 5: occupancy, surface, R, G, B), plain .npy so a window reads only its own
pages, and `manifest.json` (the pairs' settings hash, every pair's split, dims and flags).
The pairs' npz are compressed; a large build's input is 400 MB unpacked, too slow to
unpack per window. The cache is about 14 GB; rebuild it when the pairs change (the
manifest's hash no longer matches pairs/index.json).

Everything here but `WindowDataset` is numpy, so it runs (and is tested) without torch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WINDOW = 64                    # target blocks per side; the input window is 2x
SCALE = 2                      # input cells per block (D16)
IN_CHANNELS = 5                # occ, surface, R, G, B in the cache
IGNORE_SHAPE = 255
IGNORE_MATERIAL = 65535
CACHE_DIR = "pairs_cache"
MIN_SOLID = 64                 # a training window needs this many supervised solid voxels
TRIES = 12                     # ... found within this many random starts, else the best one


# --------------------------------------------------------------------------- cache

def build_cache(dataset: Path, force: bool = False, log=print) -> dict:
    """Unpack every usable pair of <dataset>/pairs/index.json into the cache (resumable)."""
    dataset = Path(dataset)
    index = json.loads((dataset / "pairs" / "index.json").read_text(encoding="utf-8"))
    root = dataset / CACHE_DIR
    root.mkdir(exist_ok=True)
    pairs = [p for p in index["pairs"] if p.get("status") == "ok"]
    rows = []
    t0 = time.time()
    for n, p in enumerate(pairs):
        d = root / p["id"]
        d.mkdir(exist_ok=True)
        inp = d / f"v{p['variant']}.input.npy"
        tgt = d / "target_shape.npy"
        if force or not inp.is_file() or not tgt.is_file():
            z = np.load(dataset / "pairs" / p["file"])
            if force or not tgt.is_file():
                np.save(tgt, np.ascontiguousarray(z["target_shape"]))
                np.save(d / "target_material.npy", np.ascontiguousarray(z["target_material"]))
            stack = np.empty(z["input_occ"].shape + (IN_CHANNELS,), np.uint8)
            stack[..., 0] = z["input_occ"]
            stack[..., 1] = z["input_surface"]
            stack[..., 2:] = z["input_rgb"]
            np.save(inp, stack)
        flags = (p.get("validation") or {}).get("flags", [])
        rows.append({"id": p["id"], "variant": p["variant"], "split": p["split"],
                     "target_dims": p["target_shape"], "low_iou": "low_iou" in flags,
                     "category": p.get("category"), "source": "synthetic"})
        if log and (n + 1) % 100 == 0:
            log(f"  {n + 1}/{len(pairs)} pairs cached ({time.time() - t0:.0f} s)")
    manifest = {"pairs_version": index["version"], "settings": index["settings"],
                "settings_hash": _settings_hash(index), "pairs": rows}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest


def _settings_hash(index: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(index["settings"], sort_keys=True).encode()).hexdigest()[:12]


class Cache:
    """Read access to the cache: memory-mapped arrays per (build, variant)."""

    def __init__(self, dataset: Path, check: bool = True):
        self.dataset = Path(dataset)
        self.root = self.dataset / CACHE_DIR
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if check:
            index = json.loads((self.dataset / "pairs" / "index.json").read_text(encoding="utf-8"))
            synthetic = [p for p in self.manifest["pairs"] if p.get("source", "synthetic") == "synthetic"]
            if _settings_hash(index) != self.manifest["settings_hash"] or \
                    len([p for p in index["pairs"] if p.get("status") == "ok"]) != len(synthetic):
                raise SystemExit("pairs_cache is stale: rebuild it (python -m training.data cache)")
        self.pairs = self.manifest["pairs"]

    def split(self, name: str, without_low_iou: bool = False) -> list:
        return [p for p in self.pairs if p["split"] == name and not (without_low_iou and p["low_iou"])]

    def target(self, bid: str):
        d = self.root / bid
        return (np.load(d / "target_shape.npy", mmap_mode="r"),
                np.load(d / "target_material.npy", mmap_mode="r"))

    def input(self, bid: str, variant: int):
        return np.load(self.root / bid / f"v{variant}.input.npy", mmap_mode="r")


TRELLIS_SET = "real_trellis_train"


def add_trellis(dataset: Path, force: bool = False, log=print) -> dict:
    """The real-TRELLIS inputs of the train builds (BuildDataCollector
    `schemset.real_trellis --set train`: each train build's smooth photo through TRELLIS,
    placed in its target frame), added to the cache as variant "trellis". Only placements
    that validated are added (no failures: a mesh that misses most of the build, or a frame
    that does not line up, is no reliable pair). Rebuilding the cache drops them: run this
    again."""
    dataset = Path(dataset)
    root = dataset / CACHE_DIR
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["pairs"] = [p for p in manifest["pairs"] if p.get("source", "synthetic") != "trellis"]
    known = {p["id"]: p for p in manifest["pairs"]}
    idx = json.loads((dataset / "pairs" / TRELLIS_SET / "index.json").read_text(encoding="utf-8"))
    added, skipped = [], {}
    for b in idx["builds"]:
        bid = b["id"]
        if b.get("status") != "ok":
            skipped[bid] = b.get("status")
            continue
        if b["validation"]["failures"]:
            skipped[bid] = b["validation"]["failures"]
            continue
        if b.get("split") != "train" or bid not in known or known[bid]["split"] != "train":
            skipped[bid] = "not a train build of the cache"
            continue
        out = root / bid / "vtrellis.input.npy"
        if force or not out.is_file():
            z = np.load(dataset / "pairs" / TRELLIS_SET / bid / "input.npz")
            stack = np.empty(z["input_occ"].shape + (IN_CHANNELS,), np.uint8)
            stack[..., 0] = z["input_occ"]
            stack[..., 1] = z["input_surface"]
            stack[..., 2:] = z["input_rgb"]
            np.save(out, stack)
        manifest["pairs"].append({"id": bid, "variant": "trellis", "split": "train",
                                  "target_dims": known[bid]["target_dims"],
                                  "low_iou": "low_iou" in b["validation"].get("flags", []),
                                  "category": known[bid].get("category"), "source": "trellis"})
        added.append(bid)
    manifest["trellis"] = {"set": TRELLIS_SET, "added": len(added), "skipped": skipped}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    if log:
        log(f"{len(added)} real-TRELLIS train inputs added, {len(skipped)} skipped")
    return manifest["trellis"]


# --------------------------------------------------------------------------- windows

def crop(inp, ts, tm, start, size: int = WINDOW):
    """The target window [start, start + size) and its input window [2 start, 2 start +
    2 size), padded on the high side: target IGNORE, input zero (contract item 2)."""
    start = np.asarray(start, dtype=np.int64)
    w_s = np.full((size,) * 3, IGNORE_SHAPE, np.uint8)
    w_m = np.full((size,) * 3, IGNORE_MATERIAL, np.uint16)
    w_i = np.zeros((size * SCALE,) * 3 + (inp.shape[-1],), np.uint8)
    hi = np.minimum(start + size, ts.shape)
    t_sl = tuple(slice(a, b) for a, b in zip(start, hi))
    n = tuple(b - a for a, b in zip(start, hi))
    w_s[:n[0], :n[1], :n[2]] = ts[t_sl]
    w_m[:n[0], :n[1], :n[2]] = tm[t_sl]
    i_sl = tuple(slice(a * SCALE, min(b * SCALE, m)) for a, b, m in zip(start, hi, inp.shape[:3]))
    ni = tuple(s.stop - s.start for s in i_sl)
    w_i[:ni[0], :ni[1], :ni[2]] = inp[i_sl]
    return w_i, w_s, w_m


def supervised_solid(w_s) -> int:
    return int(((w_s != IGNORE_SHAPE) & (w_s != 0)).sum())


def sample_start(dims, rng, size: int = WINDOW):
    """A uniform start over [0, max(dim - size, 0)] per axis (contract item 2)."""
    return np.array([int(rng.integers(0, max(int(d) - size, 0) + 1)) for d in dims], dtype=np.int64)


def sample_window(inp, ts, tm, rng, size: int = WINDOW, min_solid: int = MIN_SOLID):
    """(input, shape, material, start): a random window with at least `min_solid` supervised
    solid voxels, or the most solid of `TRIES` random starts."""
    best = None
    for _ in range(TRIES):
        start = sample_start(ts.shape, rng, size)
        w = crop(inp, ts, tm, start, size)
        n = supervised_solid(w[1])
        if n >= min_solid:
            return w + (start,)
        if best is None or n > best[0]:
            best = (n, w + (start,))
    return best[1]


def symmetry(w_i, w_s, w_m, k: int, mirror: bool):
    """The same symmetry on the input and target windows (contract item 3): k quarter turns
    about +Y (`blockstates.rotate_y`, which remaps oriented shapes), then the east-west
    mirror (`blockstates.mirror_x`). Material ids never change."""
    from mcvoxel.blockstates import mirror_x, rotate_y
    if k % 4:
        w_s, w_m = rotate_y(w_s, k, w_m)
        w_i = np.rot90(w_i, k, axes=(0, 2))
    if mirror:
        w_s, w_m = mirror_x(w_s, w_m)
        w_i = w_i[::-1]
    return np.ascontiguousarray(w_i), np.ascontiguousarray(w_s), np.ascontiguousarray(w_m)


# TRELLIS-like variation added to each training input on the fly; the target never changes.
# Four fixed degraded inputs per build let a model memorise them (r1 peaked at 4,000 steps);
# these also cover what the synthetic pairs lack (docs/pairs-v1.md, TRELLIS artefacts):
# spurious geometry, thin parts closed up, colours that drift.
AUGMENT = {"brightness": (0.6, 1.6),        # log-uniform gain on the surface colours
           "channel_sd": 0.08,              # per-channel gain sd (white balance)
           "noise_sd": (0.0, 8.0),          # per-cell RGB noise sd
           "grey_p": 0.1,                   # share of windows with colour washed to grey
           "blobs": (0, 3),                 # spurious solid balls on the surface (floaters, clutter)
           "dents": (0, 3),                 # balls cut out of the surface
           "radius_cells": (2, 6),
           "close_p": 0.15,                 # one-cell closing: thin gaps and slots filled
           "shift_p": 0.5}                  # the input moved by one cell (half a block) on a random axis

_SIX = np.zeros((3, 3, 3), bool)
_SIX[1, 1, :] = _SIX[1, :, 1] = _SIX[:, 1, 1] = True


def _ball(r: int) -> np.ndarray:
    g = np.arange(-r, r + 1)
    return (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) <= r * r


def _clip_slices(centre, r, shape):
    lo = np.maximum(centre - r, 0)
    hi = np.minimum(centre + r + 1, shape)
    blo = lo - (centre - r)
    return (tuple(slice(a, b) for a, b in zip(lo, hi)),
            tuple(slice(a, a + (b - c)) for a, b, c in zip(blo, hi, lo)))


def augment(w_i: np.ndarray, rng, aug: dict = AUGMENT) -> np.ndarray:
    """A perturbed copy of an input window (2w, 2w, 2w, 5) uint8. Geometry first (blobs,
    dents, closing; the surface flag is recomputed only around what changed, so faces the
    window cut keep their flags; new surface cells take the colour where they were made),
    then the shift, then the colours of the surface cells. Off the surface rgb stays 0."""
    from scipy import ndimage
    x = np.array(w_i, copy=True)
    occ = x[..., 0] > 0
    surf = x[..., 1] > 0
    rgb = x[..., 2:].astype(np.float32)
    occ0 = occ.copy()
    cells = np.argwhere(surf)
    if len(cells):
        for kind in ("blobs", "dents"):
            for _ in range(int(rng.integers(aug[kind][0], aug[kind][1] + 1))):
                c = cells[int(rng.integers(len(cells)))]
                r = int(rng.integers(aug["radius_cells"][0], aug["radius_cells"][1] + 1))
                sl, bsl = _clip_slices(c, r, occ.shape)
                ball = _ball(r)[bsl]
                colour = rgb[tuple(c)].copy()
                if kind == "blobs":
                    occ[sl] |= ball
                else:
                    occ[sl] &= ~ball
                view = rgb[sl]
                fill = occ[sl] & (view.sum(axis=-1) == 0)
                view[fill] = colour
        if rng.random() < aug["close_p"]:
            occ = ndimage.binary_closing(occ, structure=_SIX, border_value=0) | occ
    changed = occ ^ occ0
    if changed.any():
        boundary = occ & ~ndimage.binary_erosion(occ, structure=_SIX, border_value=1)
        region = ndimage.binary_dilation(changed, structure=np.ones((3, 3, 3), bool))
        surf = np.where(region, boundary, surf)
        missing = surf & (rgb.sum(axis=-1) == 0)
        if missing.any():                        # e.g. cells the closing filled: a neighbour's colour
            near = np.stack([ndimage.grey_dilation(rgb[..., k], size=3) for k in range(3)], axis=-1)
            rgb[missing] = near[missing]
    if rng.random() < aug["shift_p"]:
        axis, step = int(rng.integers(3)), int(rng.choice((-1, 1)))
        occ, surf = (np.roll(a, step, axis=axis) for a in (occ, surf))
        rgb = np.roll(rgb, step, axis=axis)
        edge = [slice(None)] * 3
        edge[axis] = slice(0, 1) if step > 0 else slice(-1, None)
        occ[tuple(edge)] = surf[tuple(edge)] = False
        rgb[tuple(edge)] = 0
    gain = np.exp(rng.uniform(*np.log(aug["brightness"]))) * (1.0 + rng.normal(0.0, aug["channel_sd"], 3))
    s = rgb[surf] * gain
    s += rng.normal(0.0, rng.uniform(*aug["noise_sd"]), s.shape)
    if rng.random() < aug["grey_p"]:
        s = s * 0.3 + s.mean(axis=-1, keepdims=True) * 0.7
    rgb[:] = 0
    rgb[surf] = s
    x[..., 0] = occ
    x[..., 1] = surf & occ
    x[..., 2:] = np.clip(rgb, 0, 255).astype(np.uint8)
    return x


def build_weights(pairs: list) -> np.ndarray:
    """Sampling weight per pair: builds count by the square root of how many windows they
    span (a 300-block hall is not 100 small houses), shared among the build's variants."""
    from collections import Counter
    per_build = Counter(p["id"] for p in pairs)
    w = np.array([np.sqrt(np.prod([max(1, -(-int(d) // WINDOW)) for d in p["target_dims"]]))
                  / per_build[p["id"]] for p in pairs], dtype=np.float64)
    return w / w.sum()


# --------------------------------------------------------------------------- torch

def to_tensors(w_i, w_s, w_m, start):
    """Window arrays -> torch tensors: input uint8 (5, 2w, 2w, 2w), shape int64, material
    int64 (IGNORE kept as 255 / 65535), the window's start in blocks (for the height input)."""
    import torch
    return (torch.from_numpy(np.ascontiguousarray(np.moveaxis(w_i, -1, 0))),
            torch.from_numpy(w_s.astype(np.int64)), torch.from_numpy(w_m.astype(np.int64)),
            torch.from_numpy(np.asarray(start, dtype=np.int64)))


class WindowDataset:
    """Random training windows: a build drawn by `build_weights`, then its real-TRELLIS input
    with probability `trellis_share` (when it has one, `add_trellis`) or else one of its
    synthetic variants; a random window, a random symmetry. `length` samples make an epoch;
    every sample draws afresh from its own seed. A map-style dataset for
    `torch.utils.data.DataLoader` (no torch import needed here)."""

    def __init__(self, dataset: Path, split: str = "train", length: int = 4000, seed: int = 0,
                 without_low_iou: bool = False, symmetries: bool = True, augment: bool = False,
                 trellis_share: float = 0.0):
        self.dataset, self.length, self.seed = Path(dataset), int(length), int(seed)
        self.symmetries, self.augment, self.trellis_share = symmetries, augment, float(trellis_share)
        self.cache = None
        self.pairs = Cache(self.dataset).split(split, without_low_iou)
        self.synthetic, self.trellis = {}, {}
        for p in self.pairs:
            if p.get("source") == "trellis":
                self.trellis[p["id"]] = p
            else:
                self.synthetic.setdefault(p["id"], []).append(p)
        self.restrict(sorted(self.synthetic))
        self.epoch = 0

    def restrict(self, ids, variants=None):
        """Sample only these builds (and, if given, only these synthetic variants)."""
        self.builds = [b for b in ids if b in self.synthetic]
        if variants is not None:
            self.synthetic = {b: [p for p in self.synthetic[b] if p["variant"] in variants] for b in self.builds}
            self.trellis = {b: p for b, p in self.trellis.items() if b in self.builds}
        self.weights = build_weights([self.synthetic[b][0] for b in self.builds])

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.cache is None:                         # per worker: open lazily
            self.cache = Cache(self.dataset, check=False)
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        bid = self.builds[int(rng.choice(len(self.builds), p=self.weights))]
        if self.trellis_share and bid in self.trellis and rng.random() < self.trellis_share:
            p = self.trellis[bid]
        else:
            syn = self.synthetic[bid]
            p = syn[int(rng.integers(len(syn)))]
        ts, tm = self.cache.target(p["id"])
        w_i, w_s, w_m, start = sample_window(self.cache.input(p["id"], p["variant"]), ts, tm, rng)
        if self.symmetries:
            w_i, w_s, w_m = symmetry(w_i, w_s, w_m, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
        if self.augment:
            w_i = augment(w_i, rng)
        return to_tensors(w_i, w_s, w_m, start)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache", help="unpack the pairs into the memory-mapped cache")
    c.add_argument("--dataset", type=Path, required=True)
    c.add_argument("--force", action="store_true")
    t = sub.add_parser("trellis", help="add the real-TRELLIS train inputs to the cache")
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "trellis":
        print(json.dumps(add_trellis(args.dataset, args.force), indent=1)[:2000])
        return 0
    if args.cmd == "cache":
        t0 = time.time()
        m = build_cache(args.dataset, args.force)
        print(f"{len(m['pairs'])} pairs cached, settings {m['settings_hash']}, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
