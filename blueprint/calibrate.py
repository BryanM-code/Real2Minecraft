"""Calibration of the evaluator: does it prefer a known-good build to the same build with one
deliberate mistake, and does it say which mistake? A reference blueprint made by hand from a
prepared photo (`refine prepare` + `measure` in the same folder) is changed one way at a
time (`MISTAKES`: a wing's roof hipped, the split level gone, a garage door missing, the
parts' widths 60/40, half the depth, 45-degree roofs, the parts swapped, red walls, a wing
missing, flat roofs), and each version is scored by the computer measures (`score`) and,
with --critic, critiqued by the vision model, which sees the version beside the photo.

A measure that does not rank the reference first for a mistake it should see, or a critic
that rates a mistake as high as the reference or names the wrong problem, is not ready to
steer corrections. Runs in r2m_vlm (the critic) or any environment with the renderer:

    r2m_vlm\\Scripts\\python -m blueprint.calibrate outputs\\blueprint\\woodland\\woodland.json
        --photo-dir outputs\\vlm\\m3test --id woodland --out outputs\\vlm\\calib [--critic qwen3.8-27b]

Writes <out>/calibration.json (and .png: the versions side by side).

`synthetic` does the same over many buildings at once, without a hand-made reference: each
photo's reading is composed (the reference) and its render stands in for the photo (its
outline, colours, openings and plan, so nothing is hidden or out of perspective), and each
mistake is one of the loop's own corrections applied the wrong way (`MISTAKE_EDITS`: a part's
roof type or ridge, a part's level, a garage door more or less, the main block wider, half
the depth, a steeper or flatter roof, mirrored, red walls, a part removed, a storey more or
less). It says, per mistake and per kind of building, how often the measures rank the
reference first and by how much: what they can see at best, on every kind the builder makes.

    r2m_vlm\\Scripts\\python -m blueprint.calibrate synthetic outputs\\vlm\\run3 --out outputs\\vlm\\calib_synth
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path


def _numbered(bp: dict) -> dict:
    """The blueprint with its box masses named p0, p1, ... as compose names a form's parts
    (the critique is told about those, `edits.describe_state`)."""
    names = {m["id"]: f"p{k}" for k, m in enumerate(m for m in bp["masses"] if m.get("shape", "box") == "box")}
    for m in bp["masses"]:
        m["id"] = names.get(m["id"], m["id"])
    for item in bp.get("openings", []) + bp.get("features", []):
        if item.get("mass") in names:
            item["mass"] = names[item["mass"]]
    return bp


def _wing(bp: dict) -> dict:
    """The reference's second part (the wing)."""
    return [m for m in bp["masses"] if m.get("shape", "box") == "box"][1]


def _hip_wing(bp):
    _wing(bp)["roof"]["type"] = "hip"
    return bp


def _no_split(bp):
    w = _wing(bp)
    w["y"] = 0
    return bp


def _one_garage(bp):
    g = [o for o in bp["openings"] if o["kind"] == "garage"]
    if len(g) > 1:
        bp["openings"].remove(g[-1])
    return bp


def _shares_60_40(bp):
    main, wing = bp["masses"][0], _wing(bp)
    total = main["w"] + wing["w"]
    new_main = round(0.6 * total)
    dx = new_main - main["w"]
    main["w"], wing["w"] = new_main, total - new_main
    wing["x"] += dx
    for o in bp["openings"]:
        m = main if o["mass"] == main["id"] else wing if o["mass"] == wing["id"] else None
        if m is not None and o["side"] in ("south", "north"):
            o["u"] = max(0, min(o["u"], m["w"] - o.get("w", 1) - 1))
    for f in bp.get("features", []):
        if f.get("mass") == wing["id"] and "u" in f:
            f["u"] = max(0, min(f["u"], wing["w"] - f.get("w", 1) - 1))
    return bp


def _shallow(bp):
    for m in bp["masses"]:
        if m.get("shape", "box") == "box":
            front = m["z"] + m["d"]
            m["d"] = max(5, round(m["d"] * 0.55))
            m["z"] = front - m["d"]
    for o in bp["openings"]:
        m = next(mm for mm in bp["masses"] if mm["id"] == o["mass"])
        if o["side"] in ("east", "west"):
            o["u"] = max(0, min(o["u"], m["d"] - o.get("w", 1) - 1))
    for f in bp.get("features", []):
        if f.get("kind") == "chimney":
            f["z"] = max(f["z"], min(m["z"] for m in bp["masses"]) + 1)
    return bp


def _pitch45(bp):
    for m in bp["masses"]:
        if m["roof"]["type"] in ("gable", "hip"):
            m["roof"]["pitch"] = 1.0
    return bp


def _mirrored(bp):
    from .compose import mirror
    return mirror(bp)


def _red_walls(bp):
    bp["materials"]["wall"] = "red_terracotta"
    return bp


def _no_wing(bp):
    wing = _wing(bp)
    bp["masses"] = [m for m in bp["masses"] if m is not wing]
    bp["openings"] = [o for o in bp["openings"] if o["mass"] != wing["id"]]
    bp["features"] = [f for f in bp.get("features", []) if f.get("mass") != wing["id"]]
    return bp


def _flat_roofs(bp):
    for m in bp["masses"]:
        m["roof"] = {"type": "flat", "overhang": 0}
    bp["features"] = [f for f in bp.get("features", []) if f["kind"] != "skylight"]
    return bp


# name -> (change, the measures that should see it, the critique answers that would name it)
MISTAKES = {
    "hip_wing": (_hip_wing, ("roof_line", "sections"), ("roof 2",)),
    "no_split_level": (_no_split, ("roof_line", "sections", "garages", "windows"), ("level 2", "height 2")),
    "one_garage": (_one_garage, ("garages",), ("garage",)),
    "shares_60_40": (_shares_60_40, ("sections", "windows", "roof_line"), ("width",)),
    "half_depth": (_shallow, ("outline", "proportions", "roof_line"), ("depth",)),
    "roofs_45": (_pitch45, ("roof_line", "sections", "outline"), ("pitch",)),
    "mirrored": (_mirrored, ("outline", "sections", "garages", "windows"), ("mirrored",)),
    "red_walls": (_red_walls, ("colour",), ("wall_colour",)),
    "no_wing": (_no_wing, ("outline", "proportions", "sections"), ("missing_part",)),
    "flat_roofs": (_flat_roofs, ("roof_line", "outline"), ("roof",)),
}


def named(crit: dict, hints: tuple) -> bool:
    """Whether a critique names the mistake: any of its answers the hints point to is off."""
    parts = [p for p in crit.get("parts") or [] if isinstance(p, dict)]
    for h in hints:
        key, _, num = h.partition(" ")
        if key in ("roof", "pitch", "level", "height", "width", "depth"):
            for p in parts:
                if num and str(p.get("part")) != num:
                    continue
                if key == "roof" and p.get("roof_ok") is False:
                    return True
                if key != "roof" and str(p.get(key) or "ok").lower() not in ("ok", "none", ""):
                    return True
            if key in ("width", "depth", "height") and str((crit.get("proportions") or {}).get(key, "ok")).lower() != "ok":
                return True
        elif key == "garage":
            if str(crit.get("garage") or "ok").lower() not in ("ok", "none") or any(
                    isinstance(p.get("garage_doors"), int) and p["garage_doors"] >= 2 for p in parts):
                return True
        elif key == "mirrored" and crit.get("mirrored") is True:
            return True
        elif key == "wall_colour" and str(crit.get("wall_colour") or "ok").lower() not in ("ok", "same", "none"):
            return True
        elif key == "missing_part" and str(crit.get("missing_part") or "none").lower() in ("left", "right"):
            return True
    return False


def run(reference: Path, photo_dir: Path, stem: str, out: Path, critic: str | None = None) -> dict:
    from PIL import Image, ImageDraw

    from .refine import context, evaluate_bp, view
    from .score import summary
    out.mkdir(parents=True, exist_ok=True)
    ctx, photo, _ = context(stem, photo_dir)
    ref = _numbered(dict(json.loads(reference.read_text(encoding="utf-8")), site={}))
    versions = {"reference": copy.deepcopy(ref)}
    for name, (change, _, _) in MISTAKES.items():
        versions[name] = change(copy.deepcopy(ref))
    vlm = None
    if critic:
        from .describe import load_model
        vlm = load_model(critic)
    from .edits import describe_state
    from .vision import critique
    rec = {"reference": str(reference), "photo": stem, "critic": critic, "versions": {}}
    tiles = []
    for name, bp in versions.items():
        res = evaluate_bp(bp, ctx, place=False)
        row = {"measured": summary(res["scores"]) if res["scores"] else {}}
        if vlm is not None:
            t = time.time()
            feats = [f["kind"] for f in res["bp"].get("features", [])]
            form = {"parts": [{"part": "main"}] + [{"part": "wing"}] * (len(res["bp"]["masses"]) - 1),
                    "chimneys": feats.count("chimney"), "skylights": feats.count("skylight")}
            text = describe_state({"form": form, "style": "suburban", "kind": "house", "adjust": {}}, res["bp"])[0]
            crit = critique(vlm, photo, view(res), text)
            row["critique"] = {k: v for k, v in crit.items() if k != "text"}
            row["resemblance"] = crit.get("resemblance")
            row["seconds"] = round(time.time() - t, 1)
            if name in MISTAKES:
                row["named"] = named(crit, MISTAKES[name][2])
        rec["versions"][name] = row
        tiles.append((name, view(res)))
        print(name, row["measured"].get("total"), row.get("resemblance"), row.get("named"), flush=True)
    ref_row = rec["versions"]["reference"]
    checks = {}
    for name, (_, should, _) in MISTAKES.items():
        row = rec["versions"][name]
        m, r = row["measured"], ref_row["measured"]
        checks[name] = {
            "total_prefers_reference": (m.get("total") or 0) < (r.get("total") or 0),
            "measures_that_see_it": [k for k in should if k in m and k in r and m[k] < r[k] - 0.005],
            "measures_that_should": list(should),
        }
        if vlm is not None:
            checks[name].update(critic_prefers_reference=(row.get("resemblance") or 0) < (ref_row.get("resemblance") or 0),
                                critic_names_it=row.get("named"))
    rec["checks"] = checks
    (out / "calibration.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    side = 300
    cols = 4
    sheet = Image.new("RGB", (side * cols, (side + 34) * ((len(tiles) + 1 + cols - 1) // cols)), "white")
    d = ImageDraw.Draw(sheet)
    ph = photo.copy()
    ph.thumbnail((side, side))
    for k, (name, im) in enumerate([("photo", ph)] + tiles):
        im = im.copy()
        im.thumbnail((side, side))
        x, y = (k % cols) * side, (k // cols) * (side + 34)
        sheet.paste(im, (x + (side - im.width) // 2, y + 34))
        row = rec["versions"].get(name, {})
        label = name if name == "photo" else (f"{name}: measured {row['measured'].get('total')}"
                                              + (f", rated {row.get('resemblance')}" if vlm is not None else ""))
        d.text((x + 4, y + 4), label, fill=(0, 0, 0))
    sheet.save(out / "calibration.png")
    return rec


def mistake_edits(state: dict, bp: dict) -> list:
    """(name, edit) of each kind of mistake that applies to a composed reading: the loop's own
    corrections, the wrong way."""
    from .edits import built_parts
    parts = [p for p in state["form"].get("parts") or [] if isinstance(p, dict)]
    main = next((i for i, p in enumerate(parts) if p.get("part") == "main"), 0)
    shown = dict(built_parts(bp))
    out = []
    m = shown.get(main)
    if m is not None:
        t = m["roof"]["type"]
        swap = {"gable": "hip", "hip": "gable", "flat": "gable", "cross_gable": "hip", "mansard": "gable",
                "gambrel": "hip", "pyramid": "gable", "shed": "gable"}.get(t)
        if swap:
            out.append(("roof_type", {"op": "roof", "part": main, "value": swap}))
        if t in ("gable", "hip", "gambrel", "mansard"):
            out.append(("ridge", {"op": "ridge", "part": main}))
            out.append(("pitch_up", {"op": "pitch", "part": main, "delta": 1}))
            out.append(("pitch_down", {"op": "pitch", "part": main, "delta": -1}))
    others = [i for i in shown if i != main]
    if others:
        out.append(("part_level", {"op": "part_level", "part": others[-1], "delta": -4}))
        out.append(("part_removed", {"op": "remove_part", "part": others[-1]}))
        out.append(("shares", {"op": "share", "part": main, "factor": 1.5}))
        if shown[others[-1]]["roof"]["type"] in ("gable", "hip"):
            out.append(("wing_roof_type", {"op": "roof", "part": others[-1],
                                           "value": "hip" if shown[others[-1]]["roof"]["type"] == "gable" else "gable"}))
    garages = {i: sum(1 for o in bp["openings"] if o["mass"] == mm["id"] and o["kind"] == "garage") for i, mm in shown.items()}
    gi = max(garages, key=garages.get) if garages else None
    if gi is not None and garages[gi]:
        out.append(("garage_count", {"op": "garage_doors", "part": gi, "value": garages[gi] - 1}))
    out.append(("half_depth", {"op": "depth", "factor": 0.5}))
    out.append(("storey_more", {"op": "storeys", "part": main, "delta": 1}))
    out.append(("mirrored", {"op": "mirror"}))
    out.append(("red_walls", {"op": "colour", "role": "wall", "value": "red"}))
    return out


def synthetic(readings: Path, out: Path, yaw: float = -35.0, pitch: float = 22.0, off: tuple = (0.0, 0.0),
              log=print) -> dict:
    """The measures against each reading's own build, standing in for its photo (see the
    module docstring); `off` (yaw, pitch degrees): the stand-in photo taken that far from the
    camera the builds are rendered from, as MapAnything's camera from one photo is off."""
    from .build import build
    from .compose import compose
    from .edits import apply, initial
    from .facade import fractions
    from .refine import evaluate_bp
    from .render import render_rgba
    from .score import facts, frame_of, summary
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(readings.glob("*.read.json")):
        stem = f.name[:-len(".read.json")]
        state = initial(json.loads(f.read_text(encoding="utf-8")))
        try:
            ref = compose(state["form"], state["style"], state["kind"])
            b = build(ref)
        except Exception as ex:  # noqa: BLE001  a reading the builder cannot take is not a test
            log(f"{stem}: skipped ({type(ex).__name__})")
            continue
        rgba = render_rgba(b["shape"], b["material"], yaw + off[0], pitch + off[1], 512)
        if rgba is None:
            continue
        mask = rgba[..., 3] > 0
        pf = facts(rgba[..., :3], mask, frame_of(mask))
        pf["openings3d"] = [{k: o[k] for k in ("kind", "wall", "along", "up")} for o in fractions(ref, b, "right")]
        boxes = [m for m in ref["masses"] if m.get("shape", "box") == "box"] or ref["masses"]
        pf["plan"] = (max(m["z"] + m["d"] for m in boxes) - min(m["z"] for m in boxes)) / max(
            1, max(m["x"] + m["w"] for m in boxes) - min(m["x"] for m in boxes))
        ctx = {"measure": None, "photo_facts": pf, "yaw": yaw, "pitch": pitch, "size": 512, "openings3d": None,
               "side": "right"}
        base = evaluate_bp(ref, ctx, place=False)
        for name, edit in mistake_edits(state, ref):
            new, note = apply(state, edit, ref)
            if note or new == state:
                continue
            try:
                bp = compose(new["form"], new["style"], new["kind"], adjust=new["adjust"])
                res = evaluate_bp(bp, ctx, place=False)
            except Exception as ex:  # noqa: BLE001
                log(f"{stem} {name}: {type(ex).__name__}: {ex}")
                continue
            if res["key"] == base["key"]:
                continue                               # the mistake changed no block here
            sm, sb = summary(res["scores"]), summary(base["scores"])
            rows.append({"id": stem, "kind": state["kind"], "mistake": name, "cost": round(sb["total"] - sm["total"], 4),
                         "measures": {k: round(sb[k] - sm.get(k, 0.0), 4) for k in sb if k != "total"}})
        log(f"{stem} ({state['kind']}): {sum(1 for r in rows if r['id'] == stem)} mistakes", flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["mistake"], []).append(r)
    table = {}
    for name, rs in sorted(by.items()):
        costs = [r["cost"] for r in rs]
        seen = {k: sum(1 for r in rs if r["measures"].get(k, 0) > 0.005) for k in rs[0]["measures"]}
        table[name] = {"buildings": len(rs), "reference_first": sum(1 for c in costs if c > 0),
                       "share_first": round(sum(1 for c in costs if c > 0) / len(rs), 3),
                       "median_cost": round(float(sorted(costs)[len(costs) // 2]), 4),
                       "seen_by": dict(sorted(seen.items(), key=lambda kv: -kv[1]))}
    kinds = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], [0, 0])
        k[0] += r["cost"] > 0
        k[1] += 1
    rec = {"camera": [yaw, pitch], "photo_camera_off": list(off), "rows": rows, "table": table,
           "by_kind": {k: {"reference_first": v[0], "mistakes": v[1]} for k, v in sorted(kinds.items())}}
    (out / "synthetic.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "synthetic":
        ap = argparse.ArgumentParser(description="the measures against many readings' own builds")
        ap.add_argument("readings", type=Path, help="a folder of <id>.read.json")
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--off", type=float, nargs=2, default=(0.0, 0.0), metavar=("YAW", "PITCH"),
                        help="the stand-in photo's camera this far off the builds' (degrees)")
        args = ap.parse_args(argv[1:])
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        rec = synthetic(args.readings, args.out, off=tuple(args.off), log=lambda *a, **k: print(*a, flush=True))
        print(json.dumps({"table": rec["table"], "by_kind": rec["by_kind"]}, indent=1))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference", type=Path, help="a blueprint made by hand from the photo")
    ap.add_argument("--photo-dir", type=Path, required=True, help="the folder `refine prepare` and `measure` wrote")
    ap.add_argument("--id", required=True, help="the photo's id in that folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--critic", default=None, help="a vision model to critique each version (describe.load_model)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    rec = run(args.reference, args.photo_dir, args.id, args.out, args.critic)
    print(json.dumps(rec["checks"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
