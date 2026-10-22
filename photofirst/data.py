"""The completion model's inputs and training windows: what one photo shows of a build -> the
whole build.

Input, five uint8 channels like the TRELLIS pipeline's cache, so `training.model.features`
(which adds the height channel), the U-Net and `training.predict.predict_grid` apply as they
are: seen solid, seen air, and R, G, B of the seen blocks. The colour is the block
material's (`training/material_colors.json`) under a random exposure, white balance and
noise, as a photo would give it. Built at block resolution from a precomputed view
(`views`), then repeated to the model's 2x grid. Targets are the build's own (shape,
material, IGNORE). Windows, symmetries and sampling weights are `training.data`'s.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training import data as D

from .views import OUT_DIR

COLOUR = {"gain": (0.6, 1.5),        # log-uniform exposure
          "channel_sd": 0.08,        # per-channel gain sd (white balance)
          "noise_sd": (0.0, 10.0)}   # per-block RGB noise sd

# The front end's errors (`front`, measured on the perspective renders of the evaluation
# builds with the camera known): of the truly seen blocks it has 39% exactly, 37% one block
# off, 23% further off or missing; 44% of its blocks sit in air the photo shows, 7% in hidden
# air; it marks 71% of the seen air, 1.6% of its seen air is solid. FRONT_NOISE imitates that
# on the true seen blocks, in patches (the errors come from smooth depth errors), so a
# completion model learns not to take the seen blocks literally. PHOTO_COLOUR adds a photo's
# exposure and shade: the photos' colours are much darker than the materials'.
FRONT_NOISE = {"patch": 4,               # blocks per side of a patch
               "shift_p": 0.65,          # patches of the seen surface moved ...
               "shift2_p": 0.35,         # ... by two blocks rather than one
               "drop_p": 0.3,            # patches of seen blocks lost
               "spurious_p": 0.3,        # seen air next to the surface turned solid
               "air_keep": (0.6, 0.9),   # share of the seen air kept (patches)
               "false_air_p": 0.25}      # a moved block's old place marked seen air
PHOTO_COLOUR = {"gain": (0.25, 1.3), "channel_sd": 0.1, "noise_sd": (0.0, 14.0), "shade": (0.35, 1.0)}

_SIX_OFFSETS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])


def _patches(shape, rng, size: int, p: float) -> np.ndarray:
    """A random boolean field of `size`-block patches, each on with probability `p`."""
    low = rng.random(tuple(-(-n // size) for n in shape)) < p
    return np.repeat(np.repeat(np.repeat(low, size, 0), size, 1), size, 2)[:shape[0], :shape[1], :shape[2]]


def front_noise(w_i: np.ndarray, rng, noise=FRONT_NOISE) -> np.ndarray:
    """The front end's kind of errors on a block-resolution input window (in place, returned)."""
    solid, air, rgb = w_i[..., 0].astype(bool), w_i[..., 1].astype(bool), w_i[..., 2:].copy()
    shape, size = solid.shape, noise["patch"]
    # patches of the surface moved one block, each patch along its own axis direction
    choice = rng.integers(len(_SIX_OFFSETS), size=tuple(-(-n // size) for n in shape))
    choice = np.repeat(np.repeat(np.repeat(choice, size, 0), size, 1), size, 2)[:shape[0], :shape[1], :shape[2]]
    moving = solid & _patches(shape, rng, size, noise["shift_p"])
    far = _patches(shape, rng, size, noise.get("shift2_p", 0.0))
    new_solid, new_rgb = solid & ~moving, np.where((solid & ~moving)[..., None], rgb, 0)
    for k, off in [(k, o) for k, o in enumerate(_SIX_OFFSETS)] + [(k + 6, 2 * o) for k, o in enumerate(_SIX_OFFSETS)]:
        src = moving & (choice == k % 6) & (far if k >= 6 else ~far)
        if not src.any():
            continue
        idx = np.argwhere(src) + off
        ok = ((idx >= 0) & (idx < shape)).all(1)
        frm = np.argwhere(src)[ok]
        idx = idx[ok]
        new_solid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        new_rgb[idx[:, 0], idx[:, 1], idx[:, 2]] = rgb[frm[:, 0], frm[:, 1], frm[:, 2]]
    vacated = solid & ~new_solid
    air = (air | (vacated & (rng.random(shape) < noise["false_air_p"]))) & ~new_solid
    # patches of seen blocks lost
    lost = new_solid & _patches(shape, rng, size, noise["drop_p"])
    new_solid &= ~lost
    # spurious blocks in the seen air next to the surface
    from scipy.ndimage import binary_dilation
    near = air & binary_dilation(new_solid) & (rng.random(shape) < noise["spurious_p"])
    if near.any():
        src = np.argwhere(new_solid)
        if len(src):                                         # coloured like a random seen block
            pick = src[rng.integers(len(src), size=int(near.sum()))]
            new_rgb[near] = new_rgb[pick[:, 0], pick[:, 1], pick[:, 2]]
    new_solid |= near
    air &= ~near
    # part of the seen air not marked
    air &= _patches(shape, rng, size, rng.uniform(*noise["air_keep"]))
    out = np.zeros_like(w_i)
    out[..., 0], out[..., 1] = new_solid, air & ~new_solid
    out[..., 2:] = np.where(new_solid[..., None], new_rgb, 0)
    return out


def material_rgb() -> np.ndarray:
    """(materials, 3) uint8: each material id's colour (air black)."""
    path = Path(D.__file__).with_name("material_colors.json")
    mats = json.loads(path.read_text(encoding="utf-8"))["materials"]
    return np.array([rgb or (0, 0, 0) for _, rgb in mats], np.uint8)


class Store:
    """Targets and views written by `views.build`, memory-mapped."""

    def __init__(self, dataset: Path):
        self.root = Path(dataset) / OUT_DIR
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.builds = self.manifest["builds"]

    def ids(self, split: str) -> list:
        return sorted(b for b, e in self.builds.items() if e["split"] == split)

    def target(self, bid: str):
        return (np.load(self.root / "targets" / f"{bid}.shape.npy", mmap_mode="r"),
                np.load(self.root / "targets" / f"{bid}.material.npy", mmap_mode="r"))

    def n_views(self, bid: str) -> int:
        return len(self.builds[bid]["views"])

    def view(self, bid: str, k: int):
        """(seen solid, seen air) boolean grids of view `k`."""
        dims = tuple(self.builds[bid]["dims"])
        packed = np.load(self.root / "views" / f"{bid}.npy", mmap_mode="r")[k]
        n = int(np.prod(dims))
        return (np.unpackbits(packed[0])[:n].reshape(dims).astype(bool),
                np.unpackbits(packed[1])[:n].reshape(dims).astype(bool))


def encode(tm, solid, air, table, rng=None, colour=COLOUR) -> np.ndarray:
    """Block-resolution input (..., 5) uint8 from the target materials and one view; colours
    varied with `rng` (None: the materials' own colours)."""
    out = np.zeros(solid.shape + (5,), np.uint8)
    out[..., 0] = solid
    out[..., 1] = air
    if solid.any():
        rgb = table[np.asarray(tm)[solid].astype(np.int64)].astype(np.float32)
        if rng is not None:
            g = np.exp(rng.uniform(*np.log(colour["gain"])))
            rgb *= g * rng.normal(1.0, colour["channel_sd"], 3)
            if "shade" in colour:                               # lit and shaded sides, in patches
                shade = rng.uniform(*colour["shade"], size=tuple(-(-n // 4) for n in solid.shape))
                shade = np.repeat(np.repeat(np.repeat(shade, 4, 0), 4, 1), 4, 2)[:solid.shape[0], :solid.shape[1], :solid.shape[2]]
                rgb *= shade[solid][:, None]
            rgb += rng.normal(0.0, rng.uniform(*colour["noise_sd"]), rgb.shape)
        out[solid, 2:] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def upsample(w: np.ndarray) -> np.ndarray:
    """Block resolution -> the model's 2x grid (each block's 8 cells alike)."""
    return np.repeat(np.repeat(np.repeat(w, 2, 0), 2, 1), 2, 2)


def full_input(store: Store, bid: str, k: int, table=None) -> np.ndarray:
    """A whole build's input for view `k` at 2x (`predict_grid`), with the materials' own
    colours."""
    _, tm = store.target(bid)
    solid, air = store.view(bid, k)
    return upsample(encode(tm, solid, air, material_rgb() if table is None else table))


def window(store: Store, bid: str, k: int, rng, table, size: int = D.WINDOW,
           min_solid: int = D.MIN_SOLID, colour=COLOUR, start=None):
    """(input at block resolution, shape, material, start) for a random window of build `bid`
    seen from view `k`, with at least `min_solid` supervised solid voxels (or the most solid
    of `training.data.TRIES` starts), or the window at `start`; colours varied by `colour`
    (None: as they are)."""
    ts, tm = store.target(bid)
    if start is None:
        best = None
        for _ in range(D.TRIES):
            start = D.sample_start(ts.shape, rng, size)
            hi = np.minimum(start + size, ts.shape)
            sl = tuple(slice(a, b) for a, b in zip(start, hi))
            w = np.asarray(ts[sl])
            n = int(((w != D.IGNORE_SHAPE) & (w != 0)).sum())
            if n >= min_solid:
                best = (n, start)
                break
            if best is None or n > best[0]:
                best = (n, start)
        start = best[1]
    start = np.asarray(start, dtype=np.int64)
    hi = np.minimum(start + size, ts.shape)
    sl = tuple(slice(a, b) for a, b in zip(start, hi))
    n = tuple(b - a for a, b in zip(start, hi))
    w_s = np.full((size,) * 3, D.IGNORE_SHAPE, np.uint8)
    w_m = np.full((size,) * 3, D.IGNORE_MATERIAL, np.uint16)
    w_s[:n[0], :n[1], :n[2]] = ts[sl]
    w_m[:n[0], :n[1], :n[2]] = tm[sl]
    solid, air = store.view(bid, k)
    w_i = np.zeros((size,) * 3 + (5,), np.uint8)
    w_i[:n[0], :n[1], :n[2]] = encode(w_m[:n[0], :n[1], :n[2]], solid[sl], air[sl], table,
                                      rng if colour is not None else None, colour or COLOUR)
    return w_i, w_s, w_m, start


def fuse_views(inputs: list, carve: bool = False, margin: int = 1) -> np.ndarray:
    """Block-resolution inputs of the same window or build from several photos (seen solid,
    seen air, R, G, B) -> their union. With `carve` (space carving): a block one photo saw
    solid is dropped where another photo saw clear through it, more than `margin` blocks from
    that photo's own seen surface (near a surface the air depends on its estimated depth).
    Colours come from the first photo that saw the block."""
    from scipy.ndimage import binary_dilation
    solids = [x[..., 0].astype(bool) for x in inputs]
    airs = [x[..., 1].astype(bool) for x in inputs]
    if carve and len(inputs) > 1:
        deep = [a & ~binary_dilation(s, np.ones((3, 3, 3), bool), iterations=margin) if margin else a
                for s, a in zip(solids, airs)]
        solids = [s & ~np.logical_or.reduce([d for j, d in enumerate(deep) if j != k])
                  for k, s in enumerate(solids)]
    out = np.zeros_like(inputs[0])
    solid = np.zeros(inputs[0].shape[:3], bool)
    air = np.zeros(inputs[0].shape[:3], bool)
    for x, s, a in zip(inputs, solids, airs):
        new = s & ~solid
        out[..., 2:][new] = x[..., 2:][new]
        solid |= s
        air |= a
    out[..., 0], out[..., 1] = solid, air & ~solid
    out[..., 2:][~solid] = 0
    return out


class CompletionDataset:
    """Random training windows: a build drawn by `training.data.build_weights`, `n_views` of its
    views (a random number in that range; several are fused by `fuse_views`, with space carving
    if `carve`), a random window, a random symmetry, the colours varied (`COLOUR`, or the darker
    `PHOTO_COLOUR`), optionally the front end's kind of errors (`noise`: `FRONT_NOISE`, on each
    view before fusing). A map-style dataset for `torch.utils.data.DataLoader`; every sample
    draws afresh from its own seed."""

    def __init__(self, dataset: Path, split: str = "train", length: int = 4000, seed: int = 0,
                 symmetries: bool = True, colour: bool = True, ids=None, noise=None, photo_colour: bool = False,
                 n_views=(1, 1), carve: bool = True, real_pairs=None, real_share: float = 0.0):
        self.dataset, self.length, self.seed = Path(dataset), int(length), int(seed)
        self.symmetries, self.colour, self.noise = symmetries, colour, noise
        self.colour_params = PHOTO_COLOUR if photo_colour else COLOUR
        self.n_views, self.carve = tuple(n_views), carve
        self.store = None
        store = Store(self.dataset)
        self.builds = list(ids) if ids is not None else store.ids(split)
        self.views = {b: store.n_views(b) for b in self.builds}
        self.weights = D.build_weights([{"id": b, "target_dims": store.builds[b]["dims"]} for b in self.builds])
        self.real_dir, self.real_share, self.real = None, float(real_share), {}
        if real_pairs is not None:               # the front end's own inputs (`mv_pairs`)
            self.real_dir = Path(real_pairs)
            index = json.loads((self.real_dir / "index.json").read_text(encoding="utf-8"))["pairs"]
            for name, e in sorted(index.items()):
                if e["build"] in self.views:
                    self.real.setdefault(e["build"], []).append(name)
            self.real_builds = sorted(self.real)
            self.real_weights = D.build_weights([{"id": b, "target_dims": store.builds[b]["dims"]} for b in self.real_builds])
        self.epoch = 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.store is None:                       # opened per worker process
            self.store = Store(self.dataset)
            self.table = material_rgb()
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        if self.real and rng.random() < self.real_share:
            return self._real_item(rng)
        bid = self.builds[int(rng.choice(len(self.builds), p=self.weights))]
        n = int(rng.integers(self.n_views[0], self.n_views[1] + 1))
        ks = rng.choice(self.views[bid], size=min(n, self.views[bid]), replace=False)
        colour = self.colour_params if self.colour else None
        w_i, w_s, w_m, start = window(self.store, bid, int(ks[0]), rng, self.table, colour=colour)
        views = [w_i]
        for k in ks[1:]:                             # the same window, seen from the other photos
            views.append(window(self.store, bid, int(k), rng, self.table, colour=colour, start=start)[0])
        if self.noise:
            views = [front_noise(v, rng, self.noise) for v in views]
        w_i = fuse_views(views, carve=self.carve) if len(views) > 1 else views[0]
        if self.symmetries:
            w_i, w_s, w_m = D.symmetry(w_i, w_s, w_m, int(rng.integers(4)), bool(rng.integers(2)))
        return D.to_tensors(upsample(w_i), w_s, w_m, start)

    def _real_item(self, rng):
        """A window of one of the front end's real inputs of a build (a random photo set)."""
        bid = self.real_builds[int(rng.choice(len(self.real_builds), p=self.real_weights))]
        names = self.real[bid]
        inp = np.load(self.real_dir / f"{names[int(rng.integers(len(names)))]}.npz")["input"]
        _, w_s, w_m, start = window(self.store, bid, 0, rng, self.table, colour=None)
        f = 2 if tuple(inp.shape[:3]) == tuple(2 * d for d in self.store.builds[bid]["dims"]) else 1
        size = f * D.WINDOW                      # 2: the input is at the model's resolution already
        hi = np.minimum(f * start + size, inp.shape[:3])
        n = tuple(b - a for a, b in zip(f * start, hi))
        w_i = np.zeros((size,) * 3 + (5,), np.uint8)
        w_i[:n[0], :n[1], :n[2]] = inp[tuple(slice(a, b) for a, b in zip(f * start, hi))]
        if self.symmetries:
            w_i, w_s, w_m = D.symmetry(w_i, w_s, w_m, int(rng.integers(4)), bool(rng.integers(2)))
        return D.to_tensors(upsample(w_i) if f == 1 else w_i, w_s, w_m, start)
