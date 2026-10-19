"""Train the Phase 5 model on the pairs, validating on whole builds as the app will run them.

    r2m_train\\Scripts\\python.exe -m training.train --dataset <BDC>/data/real-schems-v1 --run <name>
        [--steps 20000] [--batch 8] [--lr 1e-3] [--widths 32 64 128 256] [--blocks 1]
        [--workers 6] [--val-every 2000] [--augment] [--weight-power 0.5] [--weight-cap N]
        [--weight-decay 1e-4] [--trellis-share 0.5] [--smoke] [--resume]

Training windows come from `training.data.WindowDataset` (train split, the contract's
windows and symmetries). The loss is the contract's: shape cross-entropy over every
supervised voxel (exterior air included), weighted by inverse square-root class frequency
over the train targets, plus material cross-entropy over the supervised solid voxels.
bf16 autocast, AdamW, linear warm-up then cosine decay, gradient clipping. `--augment` adds
`data.AUGMENT`'s TRELLIS-like input variation; `--weight-power` / `--weight-cap` soften the
shape weights; `--trellis-share` mixes in the train builds' real TRELLIS inputs
(`python -m training.data trellis`).

Validation runs whole builds through `predict.predict_grid` (the app's tiling) and scores
them with `score.score`: variant 0 of every val build (synthetic), and the real-TRELLIS
smooth-photo inputs of the val builds. The checkpoint with the best synthetic-val
`select` score (mean of IoU and exact-block accuracy) is kept as best.pt, the one with the
best real-TRELLIS (smooth) val score as best_trellis.pt (12 builds: noisier, but the input
the app will give); last.pt resumes. Test builds are never touched here.

`--smoke` overfits two small train builds (no symmetries) to check that the pipeline can
learn at all: the loss should fall towards zero and their own score towards 1.

Writes <dataset>/runs/<name>/: config.json, class_weights.json, train_log.jsonl,
val_log.jsonl, best.pt, best_trellis.pt, last.pt. Weights trained on tier B / C builds stay
local (D14).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .model import N_MATERIALS, N_SHAPES, UNet3D, features
from .predict import input_from_pair, predict_grid
from .score import score, summarize


def class_weights(cache: D.Cache, pairs: list, power: float = 0.5, cap: float = None) -> torch.Tensor:
    """Shape weights: each label's frequency over the supervised voxels of the train targets
    to the power -`power`, normalised to a frequency-weighted mean of 1, then capped at `cap`
    times the smallest weight; unseen labels get 1. Steep weights trade precision for
    recall: r1 (power 0.5, a 277x spread) predicted twice the blocks the targets have."""
    counts = np.zeros(N_SHAPES, np.int64)
    for bid in sorted({p["id"] for p in pairs}):
        ts, _ = cache.target(bid)
        c = np.bincount(np.asarray(ts).ravel(), minlength=256)
        counts += c[:N_SHAPES]
    f = counts / counts.sum()
    w = np.ones(N_SHAPES)
    seen = counts > 0
    w[seen] = f[seen] ** -power
    w[seen] /= (f[seen] * w[seen]).sum()
    if cap:
        w[seen] = np.minimum(w[seen], cap * w[seen].min())
    return torch.tensor(w, dtype=torch.float32), counts


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t))))


class Sampler:
    """Sample indices offset..total-1, so every window of the run has its own seed and a
    resumed run continues where it stopped."""

    def __init__(self, offset: int, total: int):
        self.offset, self.total = offset, total

    def __iter__(self):
        return iter(range(self.offset, self.total))

    def __len__(self):
        return self.total - self.offset


def losses(model, batch, weights, device):
    x8, ts, tm, start = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, feats = model(features(x8, start[:, 1]))
    shape_loss = F.cross_entropy(logits.float(), ts, weight=weights, ignore_index=D.IGNORE_SHAPE)
    solid = (ts != 0) & (ts != D.IGNORE_SHAPE) & (tm != D.IGNORE_MATERIAL)
    if solid.any():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ml = model.material_logits(feats, solid)
        material_loss = F.cross_entropy(ml.float(), tm[solid])
    else:
        material_loss = logits.sum() * 0.0
    return shape_loss, material_loss


def validation_sets(dataset: Path, cache: D.Cache, smoke_ids=None) -> list:
    """(name, id, input loader) for synthetic val (variant 0) and the real-TRELLIS smooth
    val builds; for --smoke, the smoke builds themselves."""
    out = []
    if smoke_ids:
        for bid in smoke_ids:
            out.append(("smoke", bid, lambda bid=bid: np.asarray(cache.input(bid, 0))))
        return out
    for p in cache.split("val"):
        if p["variant"] == 0:
            out.append(("val_synthetic", p["id"], lambda p=p: np.asarray(cache.input(p["id"], 0))))
    smooth = dataset / "pairs" / "real_trellis_smooth"
    idx = json.loads((smooth / "index.json").read_text(encoding="utf-8"))
    for b in idx["builds"]:
        if b.get("status") == "ok" and b["split"] == "val":
            out.append(("val_trellis_smooth", b["id"],
                        lambda b=b: input_from_pair(np.load(smooth / b["id"] / "input.npz"))))
    return out


def validate(model, dataset: Path, cache: D.Cache, sets: list, device) -> dict:
    rows = {}
    t0 = time.time()
    for name, bid, load in sets:
        ts, tm = cache.target(bid)
        ps, pm = predict_grid(model, load(), ts.shape, device=device)
        r = score(ps, pm, np.asarray(ts), np.asarray(tm))
        r["id"] = bid
        rows.setdefault(name, []).append(r)
    out = {name: summarize(rs) for name, rs in rows.items()}
    for name, rs in rows.items():
        sel = [0.5 * (r["iou"] + r["block_acc"]) for r in rs if np.isfinite(r["iou"]) and np.isfinite(r["block_acc"])]
        out[name]["select"] = round(float(np.mean(sel)), 4) if sel else None
        out[name]["per_build"] = {r["id"]: {k: (round(v, 4) if isinstance(v, float) else v)
                                            for k, v in r.items() if k != "id"} for r in rs}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--widths", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--without-low-iou", action="store_true", help="leave the low_iou pairs out of training")
    ap.add_argument("--augment", action="store_true", help="TRELLIS-like input variation (data.AUGMENT)")
    ap.add_argument("--trellis-share", type=float, default=0.0,
                    help="probability of a build's real-TRELLIS input instead of a synthetic one (data.add_trellis)")
    ap.add_argument("--weight-power", type=float, default=0.5, help="shape weights: frequency ** -power")
    ap.add_argument("--weight-cap", type=float, default=None, help="cap: this many times the smallest weight")
    ap.add_argument("--smoke", action="store_true", help="overfit two small train builds")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    dataset = args.dataset.resolve()
    run = dataset / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    cache = D.Cache(dataset)

    ds = D.WindowDataset(dataset, "train", length=args.steps * args.batch, seed=args.seed,
                         without_low_iou=args.without_low_iou, symmetries=not args.smoke,
                         augment=args.augment and not args.smoke,
                         trellis_share=0.0 if args.smoke else args.trellis_share)
    smoke_ids = None
    if args.smoke:
        small = sorted({(int(np.prod(p["target_dims"])), p["id"]) for p in ds.pairs
                        if max(p["target_dims"]) <= D.WINDOW})[:2]
        smoke_ids = [bid for _, bid in small]
        ds.restrict(smoke_ids, variants={0})
        ds.pairs = [p for p in ds.pairs if p["id"] in smoke_ids and p["variant"] == 0]
    weights, counts = class_weights(cache, ds.pairs, args.weight_power, args.weight_cap)
    (run / "class_weights.json").write_text(json.dumps({"weights": weights.tolist(), "counts": counts.tolist()}),
                                            encoding="utf-8", newline="\n")
    model = UNet3D(tuple(args.widths), args.blocks, checkpointing=args.checkpointing).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, best, best_trellis = 0, None, None
    if args.resume and (run / "last.pt").is_file():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best, best_trellis = ck["step"], ck.get("best"), ck.get("best_trellis")
        print(f"resumed at step {step} (best {best}, best on TRELLIS input {best_trellis})")
    config = {**vars(args), "dataset": str(dataset), "params": sum(p.numel() for p in model.parameters()),
              "train_pairs": len(ds.pairs), "train_builds": len(ds.builds),
              "train_builds_with_trellis": len(set(ds.trellis) & set(ds.builds)),
              "smoke_ids": smoke_ids, "torch": torch.__version__, "model": model.config,
              "pairs_settings_hash": cache.manifest["settings_hash"]}
    (run / "config.json").write_text(json.dumps(config, indent=1, default=str), encoding="utf-8", newline="\n")
    print(f"run {run.name}: {config['params'] / 1e6:.1f} M params, {config['train_pairs']} pairs "
          f"({config['train_builds']} builds), {args.steps} steps of {args.batch}")

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=Sampler(step * args.batch, len(ds)),
                                         num_workers=args.workers, pin_memory=True, drop_last=True,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    val_sets = validation_sets(dataset, cache, smoke_ids)
    w = weights.to(device)
    model.train()
    t_log, acc = time.time(), {"shape": 0.0, "material": 0.0, "n": 0, "wait": 0.0}
    t_wait = time.time()
    for batch in loader:
        acc["wait"] += time.time() - t_wait
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        shape_loss, material_loss = losses(model, batch, w, device)
        loss = shape_loss + material_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        acc["shape"] += shape_loss.item()
        acc["material"] += material_loss.item()
        acc["n"] += 1
        if step % args.log_every == 0 or step == args.steps:
            dt = time.time() - t_log
            rec = {"step": step, "lr": opt.param_groups[0]["lr"], "shape_loss": acc["shape"] / acc["n"],
                   "material_loss": acc["material"] / acc["n"], "s_per_step": dt / acc["n"],
                   "data_wait_share": acc["wait"] / max(dt, 1e-9),
                   "gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            with open(run / "train_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:6d}  shape {rec['shape_loss']:.4f}  material {rec['material_loss']:.4f}  "
                  f"lr {rec['lr']:.2e}  {rec['s_per_step']:.2f} s/step  data wait {rec['data_wait_share']:.0%}")
            t_log, acc = time.time(), {"shape": 0.0, "material": 0.0, "n": 0, "wait": 0.0}
        if step % args.val_every == 0 or step == args.steps:
            v = validate(model, dataset, cache, val_sets, device)
            key = "smoke" if args.smoke else "val_synthetic"
            sel = v[key]["select"]
            improved = best is None or (sel is not None and sel > best)
            sel_t = v.get("val_trellis_smooth", {}).get("select")
            improved_t = sel_t is not None and (best_trellis is None or sel_t > best_trellis)
            with open(run / "val_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **v}) + "\n")
            line = "  ".join(f"{name}: IoU {v[name]['iou']['mean']} block {v[name]['block_acc']['mean']} "
                             f"shape {v[name]['shape_acc']['mean']} material {v[name]['material_acc']['mean']}"
                             for name in v if isinstance(v[name], dict) and v[name].get("iou"))
            print(f"val @ {step}: {line}  ({v['seconds']} s){'  *best*' if improved else ''}"
                  f"{'  *best on TRELLIS input*' if improved_t else ''}")
            if improved:
                best = sel
            if improved_t:
                best_trellis = sel_t
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                     "best": best, "best_trellis": best_trellis, "config": config, "val": v}
            if improved:
                torch.save(state, run / "best.pt")
            if improved_t:
                torch.save(state, run / "best_trellis.pt")
            torch.save(state, run / "last.pt")
            model.train()
        if step >= args.steps:
            break
        t_wait = time.time()
    print(f"done: {step} steps, best {best}, best on TRELLIS input {best_trellis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
