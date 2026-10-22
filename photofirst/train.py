"""Train the completion model: what one photo shows of a build -> the whole build.

    r2m_train\\Scripts\\python -m photofirst.train --dataset <BDC>/data/real-schems-v1 --run pf-a1
        [--steps 12000] [--batch 8] [--val-every 1000] [--no-colour] [--resume]

The model, loss, schedule and class weights are the TRELLIS pipeline's (`training.model`,
`training.train`); only the input differs (`photofirst.data`). Validation runs every val
build seen from the evaluation photos' viewpoint (view 0) through `predict_grid` and scores
it with `training.score.score`; best.pt keeps the best mean of IoU and exact blocks (the
TRELLIS runs' selection score, so the numbers compare). Writes
<dataset>/photofirst/runs/<name>/: config.json, train_log.jsonl, val_log.jsonl, best.pt,
last.pt. Weights trained on tier B / C builds stay local (D14).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from training.model import UNet3D
from training.predict import predict_grid
from training.score import score, summarize
from training.train import Sampler, class_weights, losses, lr_at

from . import data as P
from .views import OUT_DIR


def validate(model, store, ids, table, device="cuda", front_dir=None, passes: int = 1, temperature: float = 1.0) -> dict:
    """Scores on the val builds: their true view-0 inputs, or (`front_dir`) the front end's
    inputs saved by `evaluate_front` (the builds it has). `passes` > 1: a masked model decoded
    over that many passes (`masked.decode`)."""
    rows = []
    t0 = time.time()
    for bid in ids:
        ts, tm = store.target(bid)
        if front_dir is not None:
            p = Path(front_dir) / f"{bid}.npz"
            if not p.is_file():
                continue
            inp = np.load(p)["input"]
            inp = P.upsample(inp) if inp.shape[:3] == tuple(ts.shape) else inp     # 1 or 2 cells per block
        else:
            inp = P.full_input(store, bid, 0, table)
        if passes > 1:
            from .masked import decode
            ps, pm = decode(model, inp, ts.shape, passes=passes, temperature=temperature, device=device)
        else:
            ps, pm = predict_grid(model, inp, ts.shape, device=device)
        r = score(ps, pm, np.asarray(ts), np.asarray(tm))
        r["id"] = bid
        rows.append(r)
    out = summarize(rows)
    sel = [0.5 * (r["iou"] + r["block_acc"]) for r in rows if np.isfinite(r["iou"]) and np.isfinite(r["block_acc"])]
    out["select"] = round(float(np.mean(sel)), 4) if sel else None
    out["per_build"] = {r["id"]: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "id"}
                        for r in rows}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--widths", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--weight-power", type=float, default=0.5, help="shape weights: frequency ** -power")
    ap.add_argument("--weight-cap", type=float, default=None, help="cap: this many times the smallest weight")
    ap.add_argument("--no-colour", action="store_true", help="the materials' own colours, unvaried")
    ap.add_argument("--front-noise", action="store_true", help="the front end's kind of errors (data.FRONT_NOISE)")
    ap.add_argument("--photo-colour", action="store_true", help="photo-like exposure and shade (data.PHOTO_COLOUR)")
    ap.add_argument("--views-per-sample", type=int, nargs=2, default=[1, 1], metavar=("MIN", "MAX"),
                    help="photos fused per training sample (data.fuse_views), e.g. 1 4 for multi-photo input")
    ap.add_argument("--no-carve", action="store_true", help="fuse several photos without space carving")
    ap.add_argument("--val-front", default=None,
                    help="also validate on the front end's inputs of the val builds (photofirst/front/<this>/), "
                         "keeping best_front.pt")
    ap.add_argument("--real-pairs", default=None,
                    help="also train on the front end's own inputs: photofirst/mv_pairs/<this>/ (mv_pairs)")
    ap.add_argument("--real-share", type=float, default=0.7, help="share of samples from --real-pairs")
    ap.add_argument("--init", default=None, help="start from this run's best_front.pt (or a .pt path)")
    ap.add_argument("--masked", action="store_true",
                    help="iterative completion (photofirst.masked): decided blocks as inputs, loss on the rest")
    ap.add_argument("--val-passes", type=int, default=1, help="with --masked: decoding passes on the front-end val inputs")
    ap.add_argument("--val-temperature", type=float, default=1.0, help="with --val-passes: the decoding temperature")
    ap.add_argument("--wrong-tokens", type=float, default=0.0,
                    help="with --masked: share of the given blocks given wrong (masked.given_tokens)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    dataset = args.dataset.resolve()
    run = dataset / OUT_DIR / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    store = P.Store(dataset)
    table = P.material_rgb()
    ds = P.CompletionDataset(dataset, "train", length=args.steps * args.batch, seed=args.seed,
                             colour=not args.no_colour, noise=P.FRONT_NOISE if args.front_noise else None,
                             photo_colour=args.photo_colour, n_views=tuple(args.views_per_sample),
                             carve=not args.no_carve,
                             real_pairs=dataset / OUT_DIR / "mv_pairs" / args.real_pairs if args.real_pairs else None,
                             real_share=args.real_share)
    weights, counts = class_weights(store, [{"id": b} for b in ds.builds], args.weight_power, args.weight_cap)
    if args.masked:
        from .masked import MaskedUNet3D, masked_losses
        model = MaskedUNet3D(tuple(args.widths), args.blocks).to(device)
    else:
        model = UNet3D(tuple(args.widths), args.blocks).to(device)
    loss_fn = (lambda m, b, w, d: masked_losses(m, b, w, d, args.wrong_tokens)) if args.masked else losses
    if args.init and not args.resume:
        p = Path(args.init) if args.init.endswith(".pt") else dataset / OUT_DIR / "runs" / args.init / "best_front.pt"
        missing, unexpected = model.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model"],
                                                    strict=not args.masked)
        print(f"initialised from {p}" + (f" (new: {', '.join(sorted({k.split('.')[0] for k in missing}))})" if missing else ""))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, best, best_front = 0, None, None
    if args.resume and (run / "last.pt").is_file():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best, best_front = ck["step"], ck.get("best"), ck.get("best_front")
        print(f"resumed at step {step} (best {best}, best on front-end input {best_front})")
    front_dir = dataset / OUT_DIR / "front" / args.val_front if args.val_front else None
    val_ids = store.ids("val")
    config = {**vars(args), "dataset": str(dataset), "params": sum(p.numel() for p in model.parameters()),
              "train_builds": len(ds.builds), "train_views": sum(ds.views.values()),
              "real_pairs": sum(len(v) for v in ds.real.values()), "real_builds": len(ds.real), "val_builds": len(val_ids),
              "torch": torch.__version__, "model": model.config, "views": store.manifest.get("settings"),
              "class_weights": weights.tolist()}
    (run / "config.json").write_text(json.dumps(config, indent=1, default=str), encoding="utf-8", newline="\n")
    print(f"run {run.name}: {config['params'] / 1e6:.1f} M params, {len(ds.builds)} builds x views "
          f"({config['train_views']}), {args.steps} steps of {args.batch}")

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, sampler=Sampler(step * args.batch, len(ds)),
                                         num_workers=args.workers, pin_memory=True, drop_last=True,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    w = weights.to(device)
    model.train()
    t_log, acc = time.time(), {"shape": 0.0, "material": 0.0, "n": 0, "wait": 0.0}
    t_wait = time.time()
    for batch in loader:
        acc["wait"] += time.time() - t_wait
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        shape_loss, material_loss = loss_fn(model, batch, w, device)
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
            v = validate(model, store, val_ids, table, device)
            sel = v["select"]
            improved = best is None or (sel is not None and sel > best)
            if improved:
                best = sel
            vf = validate(model, store, val_ids, table, device, front_dir,
                          passes=args.val_passes if args.masked else 1, temperature=args.val_temperature) if front_dir else None
            improved_f = vf is not None and vf["select"] is not None and (best_front is None or vf["select"] > best_front)
            if improved_f:
                best_front = vf["select"]
            with open(run / "val_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val": v, "val_front": vf}) + "\n")
            print(f"val @ {step}: IoU {v['iou']['mean']:.4f}  F1~1 {v['f1_tol1']['mean']:.4f}  "
                  f"shape {v['shape_acc']['mean']:.4f}  material {v['material_acc']['mean']:.4f}  "
                  f"exact {v['block_acc']['mean']:.4f}  ({v['seconds']} s){'  *best*' if improved else ''}"
                  + (f"  | front-end input: IoU {vf['iou']['mean']:.4f}  F1~1 {vf['f1_tol1']['mean']:.4f}  "
                     f"material {vf['material_acc']['mean']:.4f}  exact {vf['block_acc']['mean']:.4f}"
                     f"{'  *best*' if improved_f else ''}" if vf else ""))
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "best": best,
                     "best_front": best_front, "config": config,
                     "val": {k: v[k] for k in v if k != "per_build"}}
            if improved:
                torch.save(state, run / "best.pt")
            if improved_f:
                torch.save(state, run / "best_front.pt")
            torch.save(state, run / "last.pt")
            model.train()
        if step >= args.steps:
            break
        t_wait = time.time()
    print(f"done: {step} steps, best {best}, best on front-end input {best_front}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
