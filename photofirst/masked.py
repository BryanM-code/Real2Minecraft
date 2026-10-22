"""Iterative completion: the model also sees blocks already decided, and a build is decided
over several passes, the surest blocks first (MaskGIT, Chang et al. 2022, on block grids).

Why: one pass takes each block's most likely shape and material on its own, so where the
photos allow several good answers (which bay has the window, which way a roof's stairs face,
where the frame posts stand) every block takes the safest one: flat walls, full cubes on the
roof, the dominant material everywhere. Decided blocks are inputs to the next pass, so the
undecided ones follow them: one window rhythm, one roof direction.

    model     `training.model.UNet3D` plus embeddings of each block's decided shape and
              material (an extra index: not decided), added after the stem
    training  per window a random share of the supervised blocks is given, the share masked
              drawn as cos(pi/2 u) with u uniform (every share from all to none); the loss is
              on the blocks not given
    decoding  `decode`: `passes` passes over the whole build (windows as `predict_grid`);
              after each the undecided blocks' shapes are sampled from their probabilities
              (Gumbel, the temperature falling to 0) and the surest are fixed, a growing share
              each pass (cosine), materials the best given what is fixed. Blocks inside the
              build (no face on the outside air) are never fed back: training never shows
              them (the targets leave the inside unsupervised)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from training import data as D
from training.model import N_MATERIALS, N_SHAPES, UNet3D, features, valid_table
from training.predict import OVERLAP, WINDOW

UNDECIDED_SHAPE = N_SHAPES               # the embeddings' extra index
UNDECIDED_MATERIAL = N_MATERIALS


class MaskedUNet3D(UNet3D):
    def __init__(self, widths=(32, 64, 128, 256), blocks: int = 1, **kw):
        super().__init__(widths, blocks, **kw)
        self.config["masked"] = True
        self.shape_emb = nn.Embedding(N_SHAPES + 1, widths[0])
        self.material_emb = nn.Embedding(N_MATERIALS + 1, widths[0])
        nn.init.normal_(self.shape_emb.weight, std=0.02)
        nn.init.normal_(self.material_emb.weight, std=0.02)

    def forward(self, x, tok_s=None, tok_m=None):
        """`UNet3D.forward` with the decided blocks (B, w, w, w) int64 (UNDECIDED_* where not)."""
        if tok_s is None:
            b, _, d, h, w = x.shape
            tok_s = torch.full((b, d // 2, h // 2, w // 2), UNDECIDED_SHAPE, dtype=torch.int64, device=x.device)
            tok_m = torch.full_like(tok_s, UNDECIDED_MATERIAL)
        extra = (self.shape_emb(tok_s) + self.material_emb(tok_m)).permute(0, 4, 1, 2, 3)
        return super().forward(x, extra)


def load_masked(path: Path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]["model"]
    m = MaskedUNet3D(tuple(cfg["widths"]), cfg["blocks"], in_channels=cfg["in_channels"],
                     n_shapes=cfg["n_shapes"], n_materials=cfg["n_materials"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def given_tokens(ts: torch.Tensor, tm: torch.Tensor, share_masked: torch.Tensor, wrong: float = 0.0):
    """(tok_s, tok_m, given) for target windows (B, w, w, w): each supervised block given with
    probability 1 - share_masked (per window). `wrong`: that share of the given blocks is given
    a wrong token instead (solid for air and air for solid, or another material), as decoding
    will give the model its own mistakes: it must not take every decided block on trust."""
    sup = ts != D.IGNORE_SHAPE
    given = (torch.rand(ts.shape, device=ts.device) >= share_masked.view(-1, 1, 1, 1)) & sup
    solid = (ts != 0) & (tm != D.IGNORE_MATERIAL)
    tok_s = torch.where(given, ts, torch.full_like(ts, UNDECIDED_SHAPE))
    tok_m = torch.where(given & solid, tm, torch.where(given & (ts == 0), torch.zeros_like(tm),
                                                       torch.full_like(tm, UNDECIDED_MATERIAL)))
    if wrong > 0:
        flip = given & (torch.rand(ts.shape, device=ts.device) < wrong)
        rand_s = torch.randint(1, N_SHAPES, ts.shape, device=ts.device)
        rand_m = torch.randint(1, N_MATERIALS, ts.shape, device=ts.device)
        was_air = ts == 0
        tok_s = torch.where(flip, torch.where(was_air, rand_s, torch.where(torch.rand(ts.shape, device=ts.device) < 0.5,
                                                                          torch.zeros_like(ts), rand_s)), tok_s)
        tok_m = torch.where(flip, torch.where(tok_s == 0, torch.zeros_like(tm), rand_m), tok_m)
    return tok_s, tok_m, given


def masked_losses(model, batch, weights, device, wrong: float = 0.0):
    """`training.train.losses` on the blocks not given (`given_tokens`' `wrong`: mistakes among
    the given ones)."""
    x8, ts, tm, start = (t.to(device, non_blocking=True) for t in batch)
    share = torch.cos(torch.rand(ts.shape[0], device=device) * (math.pi / 2))
    tok_s, tok_m, given = given_tokens(ts, tm, share, wrong)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, feats = model(features(x8, start[:, 1]), tok_s, tok_m)
    target = torch.where(given, torch.full_like(ts, D.IGNORE_SHAPE), ts)
    shape_loss = F.cross_entropy(logits.float(), target, weight=weights, ignore_index=D.IGNORE_SHAPE)
    solid = (ts != 0) & (ts != D.IGNORE_SHAPE) & (tm != D.IGNORE_MATERIAL) & ~given
    if solid.any():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ml = model.material_logits(feats, solid)
        material_loss = F.cross_entropy(ml.float(), tm[solid])
    else:
        material_loss = logits.sum() * 0.0
    return shape_loss, material_loss


@torch.no_grad()
def _pass(model, inp, tok_s, tok_m, dims, valid, window, overlap, device):
    """One pass over every window: (shape probabilities (30, X, Y, Z) float16, material
    (X, Y, Z) int64: the best valid one for the most likely solid shape, most central window)."""
    from mcvoxel.tiling import iter_windows, ramp
    acc = np.zeros((N_SHAPES,) + dims, np.float16)
    wsum = np.zeros(dims, np.float32)
    best = np.zeros(dims, np.float32)
    material = np.zeros(dims, np.int64)
    for start, gsl, wsl in iter_windows(dims, window, overlap):
        x = np.zeros((2 * window,) * 3 + (inp.shape[-1],), np.uint8)
        sl = tuple(slice(2 * a, min(2 * (a + window), n)) for a, n in zip(start, inp.shape[:3]))
        n = tuple(s.stop - s.start for s in sl)
        x[:n[0], :n[1], :n[2]] = inp[sl]
        ws = np.full((window,) * 3, UNDECIDED_SHAPE, np.int64)
        wm = np.full((window,) * 3, UNDECIDED_MATERIAL, np.int64)
        ws[wsl], wm[wsl] = tok_s[gsl], tok_m[gsl]
        xt = torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, 0)))[None].to(device)
        st, mt = torch.from_numpy(ws)[None].to(device), torch.from_numpy(wm)[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, feats = model(features(xt, torch.tensor([int(start[1])])), st, mt)
            probs = torch.softmax(logits.float(), dim=1)
            maybe = probs[:, 0] < 0.9
            mat = torch.zeros(probs.shape[:1] + probs.shape[2:], dtype=torch.int64, device=device)
            if maybe.any():
                best_solid = probs[:, 1:].argmax(dim=1) + 1
                ml = model.material_logits(feats, maybe).float()
                ml = ml.masked_fill(~valid[best_solid[maybe]], float("-inf"))
                mat[maybe] = ml.argmax(dim=1)
        p, m = probs[0].cpu().numpy(), mat[0].cpu().numpy()
        ramps = [ramp(g.stop - g.start, overlap, g.start == 0, g.stop == nn_) for g, nn_ in zip(gsl, dims)]
        w = ramps[0][:, None, None] * ramps[1][None, :, None] * ramps[2][None, None, :]
        acc[(slice(None),) + gsl] += (p[(slice(None),) + wsl] * w[None]).astype(np.float16)
        wsum[gsl] += w
        better = w > best[gsl]
        view_m = material[gsl]
        view_m[better] = m[wsl][better]
        material[gsl] = view_m
        view_b = best[gsl]
        view_b[better] = w[better]
        best[gsl] = view_b
    probs = acc.astype(np.float32) / np.maximum(wsum, 1e-6)[None]
    return probs, material


def _inside(shape: np.ndarray) -> np.ndarray:
    """Solid blocks with no face on air connected to the grid's border."""
    from training.quality import outside_air
    out = np.pad(outside_air(shape), 1, constant_values=True)
    touch = np.zeros(shape.shape, bool)
    for a in range(3):
        for s in (1, -1):
            touch |= np.roll(out, s, axis=a)[1:-1, 1:-1, 1:-1]
    return (shape != 0) & ~touch


@torch.no_grad()
def decode(model, inp: np.ndarray, target_dims, passes: int = 8, temperature: float = 1.0, seed: int = 0,
           window: int = WINDOW, overlap: int = OVERLAP, device="cuda") -> tuple:
    """(shape uint8, material uint16) over `target_dims` for one input (2X, 2Y, 2Z, 5) uint8,
    decided over `passes` passes (module docstring). passes=1 is plain one-pass prediction."""
    from mcvoxel.vocab import VALID
    model.eval()
    valid = valid_table(device)
    dims = tuple(int(d) for d in target_dims)
    total = int(np.prod(dims))
    rng = np.random.default_rng(seed)
    dec_s = np.full(dims, -1, np.int64)                # decided shape (-1: not yet)
    dec_m = np.zeros(dims, np.int64)
    tok_s = np.full(dims, UNDECIDED_SHAPE, np.int64)
    tok_m = np.full(dims, UNDECIDED_MATERIAL, np.int64)
    for t in range(passes):
        probs, mat = _pass(model, inp, tok_s, tok_m, dims, valid, window, overlap, device)
        open_ = dec_s < 0
        n_open = int(open_.sum())
        goal = total if t == passes - 1 else int(math.ceil(total * (1.0 - math.cos(math.pi / 2 * (t + 1) / passes))))
        n_new = min(max(goal - (total - n_open), 0), n_open)
        if n_new:
            idx = np.flatnonzero(open_)
            logp = np.log(np.maximum(probs.reshape(N_SHAPES, -1)[:, idx].T, 1e-9))     # (n, 30)
            tau = temperature * (1.0 - (t + 1) / passes)
            if tau > 0:
                g = -np.log(-np.log(rng.uniform(1e-9, 1.0, logp.shape)))
                pick = np.argmax(logp + tau * g, axis=1)
                conf = logp[np.arange(len(idx)), pick] + tau * g[np.arange(len(idx)), pick]
            else:
                pick = np.argmax(logp, axis=1)
                conf = logp[np.arange(len(idx)), pick]
            take = idx[np.argsort(-conf, kind="stable")[:n_new]]
            chosen = pick[np.argsort(-conf, kind="stable")[:n_new]]
            dec_s.flat[take] = chosen
            dec_m.flat[take] = np.where(chosen != 0, mat.flat[take], 0)
        # what the next pass sees: decided blocks, except those inside the current build
        now = np.where(dec_s >= 0, dec_s, np.argmax(probs, axis=0))
        feed = (dec_s >= 0) & ~_inside(now)
        tok_s = np.where(feed, dec_s, UNDECIDED_SHAPE)
        tok_m = np.where(feed, np.where(dec_s > 0, dec_m, 0), UNDECIDED_MATERIAL)
    shape = dec_s.astype(np.uint8)
    material = dec_m.astype(np.uint16)
    shape[(shape != 0) & (material == 0)] = 0
    material[shape == 0] = 0
    bad = (shape != 0) & ~VALID[shape, material]
    shape[bad] = 1
    return shape, material


def load_any(path: Path, device="cuda"):
    """A photofirst checkpoint, masked or not: (model, checkpoint)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["config"]["model"].get("masked"):
        return load_masked(path, device)
    from training.evaluate import load_model
    return load_model(path, device)
