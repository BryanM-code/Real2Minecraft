"""2-4 photos of a building -> a Minecraft `.schem` (the photo-first pipeline end to end, no
human build anywhere: for real photos).

    r2m_train\\Scripts\\python -m photofirst.photos_to_schem front.jpg left.jpg back.jpg right.jpg
        --out outputs/photofirst/house [--yaws 0 90 180 270] [--metres-per-block 1 | --height 24]
        [--run pf-b1] [--checkpoint best_front.pt] [--passes 1] [--no-hints] [--dataset <BDC>/data/real-schems-v1]

1. each photo upright (EXIF orientation); its compass heading and 35 mm focal from EXIF
   (`real.exif_info`); the building cut out by the app's own background removal (a photo that
   already has transparency is taken as its own cutout)
2. camera hints: the direction each photo looks (the compass heading, or `--yaws` in degrees,
   north 0, east 90: any consistent reference works, e.g. 0 90 180 270 walking round), the tilt
   from the photo (`real.tilt`, MoGe-2), the focal; without directions MapAnything runs on the
   photos alone (a four-sided building can then come out with a camera turned 90 degrees)
3. MapAnything on all photos together (`front_mv.run_mapanything`), placed without a target
   (`place.free_placement`: up, walls along the grid, metric scale or `--height`), turned into
   blocks at two cells per block (`front_mv.to_grid`)
4. the completion model (`--run`, a photofirst run: pf-b1 is the best so far) completes the
   build; `--passes` > 1 decodes a masked model iteratively (`masked.decode`)
5. `<out>/build.schem` (Sponge v2, 1.21.11), `blocks.npz` (shape, material, the input),
   `placement.json` (hints, tilt, placement, timings), the cutouts, and preview renders when
   BuildDataCollector's renderer is there

Weights trained on tier B / C builds stay local (D14): the output is for your own use.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import front_mv as M
from . import place as PL
from . import real as RL
from .views import OUT_DIR

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "BuildDataCollector" / "data" / "real-schems-v1"
BDC_PYTHON = ROOT.parent / "BuildDataCollector" / ".venv" / "Scripts" / "python.exe"


def free_gpu():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def prepare_photos(photos: list, out: Path) -> tuple:
    """(rgba arrays, exif dicts, cutout flags): upright photos and their cutouts."""
    infos, uprights, own_alpha = [], [], []
    for k, p in enumerate(photos):
        infos.append(RL.exif_info(p))
        dst = out / f"photo_{k}.png"
        with Image.open(p) as im:
            has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            if has_alpha:
                a = np.array(im.convert("RGBA"))
                if (a[..., 3] > 127).mean() < 0.97:      # already cut out (a render, an app cutout)
                    Image.fromarray(a).save(dst)
                    own_alpha.append(True)
                    uprights.append(dst)
                    continue
        RL.upright(p, dst)
        own_alpha.append(False)
        uprights.append(dst)
    need = [k for k, a in enumerate(own_alpha) if not a]
    rgba = [np.array(Image.open(u).convert("RGBA")) for u in uprights]
    flags = list(own_alpha)
    if need:
        cut, ok = RL.cutouts([uprights[k] for k in need], out)
        for k, c, g in zip(need, cut, ok):
            rgba[k], flags[k] = c, bool(g)
    return rgba, infos, flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--yaws", type=float, nargs="*", default=None,
                    help="each photo's compass direction in degrees (overrides EXIF)")
    ap.add_argument("--no-hints", action="store_true", help="MapAnything on the photos alone")
    ap.add_argument("--metres-per-block", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=None, help="the build's height in blocks (instead of metric scale)")
    ap.add_argument("--dataset", type=Path, default=DATASET, help="where the photofirst runs are")
    ap.add_argument("--run", default="pf-b1")
    ap.add_argument("--checkpoint", default="best_front.pt")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--max-blocks", type=int, default=192)
    ap.add_argument("--solid", action="store_true",
                    help="keep the model's guessed inside (default: a shell, the blocks next to the outside)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    times, report = {}, {"photos": [str(p) for p in args.photos], "run": args.run, "checkpoint": args.checkpoint}
    t = time.time()
    rgba, infos, cut_ok = prepare_photos(args.photos, out)
    times["photos_and_cutouts"] = time.time() - t
    report["exif"] = infos
    report["cutout_used"] = cut_ok
    whites, masks_full = [], []
    for k, im in enumerate(rgba):
        a = im[..., 3] > 127
        masks_full.append(a)
        p = out / f"white_{k}.png"
        Image.fromarray(np.where(a[..., None], im[..., :3], 255).astype(np.uint8)).save(p)
        whites.append(p)
        print(f"photo {k}: {im.shape[1]}x{im.shape[0]}, building {a.mean():.0%} of the frame"
              + ("" if cut_ok[k] else " (no cutout: the whole photo)"), flush=True)
    yaws = None
    if not args.no_hints:
        if args.yaws:
            if len(args.yaws) != len(args.photos):
                ap.error("--yaws needs one direction per photo")
            yaws = list(args.yaws)
        elif all(i["heading_deg"] is not None for i in infos):
            yaws = [float(i["heading_deg"]) for i in infos]
    K = poses = None
    if yaws is not None:
        t = time.time()
        from .front import load_moge
        moge = load_moge()
        tilts = [RL.tilt(moge, im[..., :3], m) for im, m in zip(rgba, masks_full)]
        del moge
        free_gpu()
        times["tilt"] = time.time() - t
        pitches, replaced = RL.steady_pitches([tl["pitch_deg"] for tl in tilts])
        for tl, p, r in zip(tilts, pitches, replaced):
            tl["pitch_used_deg"], tl["replaced_by_median"] = p, r
        report["tilt"] = tilts
        f = [RL.focal_px(i, im.shape[1], im.shape[0]) for i, im in zip(infos, rgba)]
        if all(v is not None for v in f):
            K = [np.array([[fk, 0.0, im.shape[1] / 2.0], [0.0, fk, im.shape[0] / 2.0], [0.0, 0.0, 1.0]])
                 for fk, im in zip(f, rgba)]
        poses = M.rough_poses(yaws, pitches)
        print("hints: directions " + ", ".join(f"{y:.0f}" for y in yaws) + " | tilt (down +) "
              + ", ".join(f"{p:.1f}" for p in pitches) + (" | focal from EXIF" if K is not None else ""), flush=True)
    else:
        print("no camera directions: MapAnything on the photos alone", flush=True)
    report["hints"] = {"yaws": yaws, "intrinsics_from_exif": K is not None}
    t = time.time()
    ma = M.load_mapanything()
    views = M.run_mapanything(ma, whites, K, poses)
    metric_height = None
    if args.height is None and poses is not None:
        # the hinted run takes the hints' unit-free scale: the photos alone give it in metres
        plain = M.run_mapanything(ma, whites, K, None)
        hp, wp = plain[0]["pts"].shape[:2]
        mp = [np.array(Image.fromarray(a).resize((wp, hp), Image.NEAREST)) for a in masks_full]
        metric_height = PL.free_placement(plain, mp, metres_per_block=1.0)[2]["height_units"]
        report["metric_height_m"] = metric_height
    del ma
    free_gpu()
    times["mapanything"] = time.time() - t
    h, w = views[0]["pts"].shape[:2]
    masks = [np.array(Image.fromarray(a).resize((w, h), Image.NEAREST)) for a in masks_full]
    t = time.time()
    height = args.height if args.height is not None else (
        metric_height / args.metres_per_block if metric_height is not None else None)
    sim, dims, info = PL.free_placement(views, masks, metres_per_block=args.metres_per_block,
                                        height_blocks=height, max_blocks=args.max_blocks)
    inp = M.to_grid(views, sim, dims, masks, res=2)
    times["placement_and_blocks"] = time.time() - t
    report["placement"] = info
    if metric_height is not None:
        print(f"metric height (photos alone): {metric_height:.1f} m -> {height:.0f} blocks at "
              f"{args.metres_per_block:g} m a block", flush=True)
    print(f"placement: grid {dims[0]} x {dims[1]} x {dims[2]} blocks, {info['blocks_per_unit']:.2f} blocks a unit"
          f" (height {info['height_units']:.1f} units), levelled {info['levelling_deg']:.1f} deg from the cameras, "
          f"walls {info['wall_share']:.0%} of the surface", flush=True)
    t = time.time()
    from training.predict import predict_grid
    from .masked import decode, load_any
    model, ck = load_any(Path(args.dataset) / OUT_DIR / "runs" / args.run / args.checkpoint)
    if args.passes > 1:
        shape, material = decode(model, inp, dims, passes=args.passes, temperature=0.0)
    else:
        shape, material = predict_grid(model, inp, dims)
    del model
    free_gpu()
    times["completion"] = time.time() - t
    from .schem import hollow, write_schem
    np.savez_compressed(out / "blocks.npz", shape=shape, material=material, input=inp)
    if not args.solid:
        shape, material = hollow(shape, material)
    written = write_schem(shape, material, out / "build.schem")
    report["schem"] = written
    report["seconds"] = {k: round(v, 1) for k, v in times.items()}
    (out / "placement.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8", newline="\n")
    print(f"{out / 'build.schem'}: {written['blocks']} blocks, {written['palette']} block states, "
          f"{written['dims'][0]} x {written['dims'][1]} x {written['dims'][2]}", flush=True)
    if BDC_PYTHON.is_file():                               # preview renders with BuildDataCollector's renderer
        code = ("import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import numpy as np; "
                "from PIL import Image; from photofirst.looks import _render_grid; "
                "z = np.load(sys.argv[3]); "
                "[(lambda im: im is not None and Image.alpha_composite(Image.new('RGBA', im.size, (255, 255, 255, 255)), "
                "im.convert('RGBA')).convert('RGB').save(sys.argv[4] + f'/preview_{y}.png'))(_render_grid(z['shape'], z['material'], y)) "
                "for y in (45, 225)]")
        subprocess.run([str(BDC_PYTHON), "-c", code, str(ROOT), str(ROOT.parent / "BuildDataCollector"),
                        str(out / "blocks.npz"), str(out)], cwd=str(ROOT), capture_output=True)
        if (out / "preview_45.png").is_file():
            print(f"previews: {out / 'preview_45.png'}, {out / 'preview_225.png'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
