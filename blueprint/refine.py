"""The evaluator: a build is checked against its photo and corrected, one change at a time,
until it is close enough or nothing more helps.

Two evaluations, combined (`combined`, 0-1):
- the computer measures (`score`): outline, proportions, roof line, windows and doors,
  colours. Repeatable, but only as good as the segmentation and the camera angle;
- the vision model (`vision`): it looks at a render of the build beside the photo, says what
  is wrong in the builder's own terms (style, kind, proportions, roofs, missing or extra
  features, mirrored, windows, colours) and how alike they are (0-10), and judges which of
  two builds is closer.

Each round every correction the critique asks for and every one the measures point to
(`edits`) is built and measured (a build and two renders take about half a second). Three are
tried in turn, the model's own first. A change of proportions the measures found is kept on
the measures; any other is checked by critiquing the candidate build on its own (`_verify`):
kept when the model's complaint is gone, nothing in the new critique asks to undo it and its
rating does not drop. (Asked instead which of two builds is closer, the 9B model picked the
image shown last 59 times in 70, so `compare` decides nothing here.) A correction the next
critique contradicts is undone. The loop stops at the target score, after `rounds` rounds,
or when no correction helps, and returns the best build it rated.

Corrections only exist in the builder's terms, so the loop cannot ask for what the builder
cannot draw. What it could not do is logged per photo: what the model asked for outside the
list ("gaps"), corrections that hit a limit (no roof steeper than 72 degrees: "limits") and
corrections that changed no block ("no_effect"); `summary` counts them over a run, which
says what to build next.

Stages, from this repository's root, MapAnything (r2m_train) measuring in between:

    r2m_vlm\\Scripts\\python -m blueprint.refine prepare photos... --out DIR
    r2m_train\\Scripts\\python -m blueprint.measure photos... --out DIR
    r2m_vlm\\Scripts\\python -m blueprint.refine run photos... --out DIR [--rounds 4]
    r2m_vlm\\Scripts\\python -m blueprint.refine judge photos... --out DIR
    r2m_vlm\\Scripts\\python -m blueprint.refine summary --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

TARGET = 0.8            # a combined score this good ends the loop, with the measures alone at least
MEASURED_TARGET = 0.7   # this (the model's resemblance is lenient: 9 of 10 for a first build measured 0.49)
SHORTLIST = 3           # corrections the vision model judges per round, at most
# the measured gain that keeps a correction the measures found (and the model does not
# object to): low for a change of proportions (what the measures judge best), high for a
# change of what the building is (a feature, the roof's type: what they cannot see), between
# for the rest; a storey, the style, the kind or a mirror image of theirs only when the model
# rates the result higher (a flip the measures wanted took a chalet's rating from 6 to 2)
SHAPE_GAIN, CLEAR_GAIN, MEANING_GAIN = 0.02, 0.05, 0.08
SHAPE_OPS = ("width", "height", "depth", "share")
MEANING_OPS = ("roof", "feature", "add_part", "remove_part", "units", "garage_wall", "skylights_on", "window_style",
               "wall_material")
MODEL_ONLY = ("storeys", "style", "kind", "mirror")
WORSE = -0.03           # a correction the measures find this much worse is not shown
# kinds whose windows and doors the vision model can box one by one (elsewhere they are too
# many, or not windows: on a 9-storey pagoda it drew a grid that is not there, in 80 s)
BOXED_KINDS = ("house", "mansion", "cottage", "townhouse", "shop", "restaurant", "barn", "gas_station", "warehouse",
               "factory", "greenhouse", "other")


def stem_of(p: Path) -> str:
    return p.parent.name if p.name == "photo.jpg" else p.stem


def photos_of(paths) -> list:
    """Photos, a folder holding photo.jpg, or a folder of <id>/photo.jpg folders."""
    out = []
    for p in paths:
        if p.is_dir():
            out += [p / "photo.jpg"] if (p / "photo.jpg").is_file() else sorted(p.glob("*/photo.jpg"))
        else:
            out.append(p)
    return out


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --- prepare: the photo's reading, its building's pixels, windows and doors --------------------

def prepare(path: Path, out: Path, vlm, seg) -> dict:
    """The local model's reading (`describe.run_read`), the main building's box and visible
    side, its windows and doors (BOXED_KINDS only), and its pixels;
    saved as <stem>.read.json, <stem>.photo.json and _work/<stem>/ (photo, masks, white.png
    for MapAnything)."""
    from PIL import Image

    from . import segment, vision
    from .compose import kind_of
    from .describe import photo, run_read
    stem = stem_of(path)
    work = out / "_work" / stem
    work.mkdir(parents=True, exist_ok=True)
    image = photo(path)
    read = _json(out / f"{stem}.read.json")
    if read is None:
        read = {"photo": str(path), "mode": "read", **run_read(*vlm, image)}
        (out / f"{stem}.read.json").write_text(json.dumps(read, indent=1, ensure_ascii=False), encoding="utf-8")
    t = time.time()
    loc = vision.locate(vlm, image)
    ops = {"boxes": [], "complete": False, "text": ""}
    if kind_of(read.get("kind")) in BOXED_KINDS:
        ops = vision.openings(vlm, image, 1000)
    lab = segment.labels(seg, image)
    sb = segment.main_building(lab, loc["box"])
    rgb = np.array(image)
    image.save(work / "photo.png")
    Image.fromarray(lab.astype(np.uint8)).save(work / "labels.png")
    Image.fromarray((sb["mask"] * 255).astype(np.uint8)).save(work / "mask.png")
    Image.fromarray((sb["occluders"] * 255).astype(np.uint8)).save(work / "occluders.png")
    Image.fromarray(np.where(sb["mask"][..., None], rgb, 255).astype(np.uint8)).save(work / "white.png")
    info = {k: sb.get(k) for k in ("box", "frame", "share", "ok", "occluded")}
    (work / "seg.json").write_text(json.dumps(info), encoding="utf-8")
    facts = {"id": stem, "photo": str(path), "size": [image.width, image.height], "locate": loc, **info,
             "openings": ops["boxes"], "openings_complete": ops["complete"], "openings_text": ops["text"][-300:],
             "seconds": round(time.time() - t, 1)}
    (out / f"{stem}.photo.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    return facts


# --- run: the loop ------------------------------------------------------------------------------

def trusted_measure(out: Path, stem: str, facts: dict | None) -> dict | None:
    """MapAnything's measurement when the building was segmented cleanly, with the frame's
    height / width for when its size fails; else None (typical sizes)."""
    m = _json(out / f"{stem}.measure.json")
    if not m or not m.get("cutout") or not 2.0 <= m.get("height_m", 0) <= 600:
        return None
    if facts and facts.get("frame"):
        x0, y0, x1, y1 = facts["frame"]
        m["mask_aspect"] = (y1 - y0) / max(1, x1 - x0)
    return m


def camera(facts: dict, measure: dict | None) -> tuple:
    """(yaw, pitch) to render the build as the photo sees it: which side from the vision
    model (MapAnything's left and right agreed with the better outline on only 26 of 46
    photos), how far round and how high from MapAnything (else 30 and 12 degrees). The
    renderer's positive yaw looks from the front left."""
    side = ((facts or {}).get("locate") or {}).get("side", "none")
    yaw_m = abs(float(measure.get("camera_yaw_deg", 30.0))) if measure else 30.0
    pitch = float(measure.get("camera_pitch_deg", 12.0)) if measure else 12.0
    yaw = 0.0 if side == "none" else max(12.0, min(60.0, yaw_m)) * (1 if side == "left" else -1)
    return yaw, max(-25.0, min(50.0, pitch))


def evaluate(state: dict, ctx: dict) -> dict:
    """Compose, build, render and measure a state."""
    from . import score
    from .build import build
    from .compose import compose
    from .render import render_rgba
    bp = compose(state["form"], state["style"], state["kind"], ctx["measure"], adjust=state["adjust"])
    b = build(bp)
    key = hashlib.md5(b["shape"].tobytes() + b["material"].tobytes() + str(b["dims"]).encode()).hexdigest()
    rgba = render_rgba(b["shape"], b["material"], ctx["yaw"], ctx["pitch"], ctx["size"])
    lab = render_rgba(b["shape"], b["material"], ctx["yaw"], ctx["pitch"], ctx["size"], labels=True)
    bf = score.build_facts(rgba, lab[..., :3] if lab is not None else None) if rgba is not None else None
    pf = ctx["photo_facts"]
    scores = score.compare(pf, bf) if bf is not None and pf is not None else {}
    return {"state": state, "bp": bp, "build": b, "key": key, "rgba": rgba, "scores": scores,
            "measured": scores["total"][0] if scores else None,
            "diag": score.diagnose(pf, bf) if scores else {}}


def view(res: dict):
    """The render the vision model sees: the build on a plain background, cropped."""
    from PIL import Image

    from .render import finish
    from .vision import SIDE
    return finish(Image.fromarray(res["rgba"]), SIDE)


def combined(res: dict):
    """The measures' total and the vision model's resemblance (out of 10), averaged; either
    alone when the other is missing."""
    parts = [v for v in (res.get("measured"), None if res.get("resemblance") is None else res["resemblance"] / 10)
             if v is not None]
    return round(float(np.mean(parts)), 3) if parts else None


def _kept(hist: list) -> list:
    """The corrections the final build has: those kept, less those undone."""
    out = []
    for h in hist:
        if h.get("reverted") and h["reverted"] in out:
            out.remove(h["reverted"])
        if h.get("accepted"):
            out.append(h["accepted"])
    return out


def _verify(edit: dict, before: dict, after: dict, gain: float) -> str:
    """Whether a correction is kept, judged from the critique of the build before it and of
    the candidate on its own: "kept", or why not. Not when the new critique asks to undo it or
    rates the build lower; a correction the model asked for is kept when its complaint is
    gone; one the measures found when they gain enough (a storey, style, kind or mirror image
    of theirs only when the model rates the result higher)."""
    from .edits import _feature_name
    if _contradicts(after, edit):
        return "contradicted"
    ra, rb = after.get("resemblance"), before.get("resemblance")
    if ra is not None and rb is not None and ra < rb:
        return f"rated lower ({rb:g} -> {ra:g})"
    op = edit.get("op")
    if edit.get("why") != "vision":
        if op in MODEL_ONLY:
            return "kept" if ra is not None and rb is not None and ra > rb else "not rated higher"
        need = MEANING_GAIN if op in MEANING_OPS else CLEAR_GAIN
        return "kept" if gain >= need else "too small a gain"
    still = False                                      # the complaint that asked for it, still there?
    if op == "feature":
        words = after.get("missing" if edit.get("on", True) else "extra") or []
        still = edit.get("name") in {_feature_name(w) for w in (words if isinstance(words, list) else [words])}
    elif op in ("roof", "pitch"):
        want = {"pitch": "steeper" if edit.get("delta", 0) > 0 else "flatter"}.get(op)
        still = any(isinstance(r, dict) and (str(r.get("pitch")).lower() == want if want else
                                             (r.get("type_ok") is False and str(r.get("type")).lower() == edit.get("value")))
                    for r in after.get("roofs") or [])
    elif op in ("style", "kind"):
        still = after.get(f"{op}_ok") is False
    elif op == "mirror":
        still = after.get("mirrored") is True
    elif op in ("width", "height", "depth", "storeys"):
        axis = "height" if op == "storeys" else op
        still = str((after.get("proportions") or {}).get(axis, "ok")).lower() != "ok"
    elif op == "windows":
        still = str(after.get("windows") or "ok").lower() != "ok"
    elif op == "colour":
        still = str(after.get(f"{edit.get('role')}_colour") or "ok").lower() not in ("ok", "same", "correct", "none")
    elif op == "add_part":
        still = str(after.get("missing_part") or "none").lower() in ("left", "right")
    elif op == "garage_wall":
        still = str(after.get("garage") or "ok").lower() == "other_side"
    return "complaint still there" if still else "kept"


def _contradicts(crit: dict, edit: dict) -> bool:
    """Whether a critique asks to undo `edit`: a feature just added now called extra (or
    removed, now missing), a pitch just raised now to be flatter, a style or kind just set now
    called wrong, a mirror just made now called mirrored."""
    from .edits import _feature_name
    op = edit.get("op")
    if op == "feature":
        words = crit.get("extra" if edit.get("on", True) else "missing") or []
        return edit.get("name") in {_feature_name(w) for w in (words if isinstance(words, list) else [words])}
    if op == "pitch":
        want = "flatter" if edit.get("delta", 0) > 0 else "steeper"
        return any(isinstance(r, dict) and str(r.get("pitch")).lower() == want for r in crit.get("roofs") or [])
    if op in ("style", "kind"):
        return crit.get(f"{op}_ok") is False
    if op == "mirror":
        return crit.get("mirrored") is True
    return False


def _scores(res: dict) -> dict:
    from .score import summary
    return summary(res["scores"]) if res.get("scores") else {}


def refine_one(stem: str, out: Path, vlm, rounds: int = 4, log=print) -> dict:
    from PIL import Image

    from . import edits as E
    from . import score, vision
    from .spec import BlueprintError
    t0 = time.time()
    read = _json(out / f"{stem}.read.json")
    facts = _json(out / f"{stem}.photo.json") or {}
    work = out / "_work" / stem
    photo = Image.open(work / "photo.png").convert("RGB")
    measure = trusted_measure(out, stem, facts)
    pf = None
    if facts.get("ok"):
        seg = {"mask": np.array(Image.open(work / "mask.png")) > 127,
               "occluders": np.array(Image.open(work / "occluders.png")) > 127, "frame": facts["frame"]}
        boxes = facts.get("openings") or []
        pf = score.photo_facts(np.array(photo), seg, boxes if (facts.get("openings_complete") or len(boxes) >= 3)
                               else None)
    yaw, pitch = camera(facts, measure)
    ctx = {"measure": measure, "photo_facts": pf, "yaw": yaw, "pitch": pitch, "size": 512}
    cur = evaluate(E.initial(read), ctx)
    first = cur
    hist, gaps, no_effect, limits, invalid = [], [], [], [], []
    tried = set()
    prev = None                                        # (the build before the last kept correction, that correction)
    scored = []                                        # every build the model rated: (round, build, its corrections)
    for rnd in range(rounds + 1):
        if cur.get("critique") is not None:            # critiqued as a candidate last round
            text, numbers, crit = cur["critique"]
        else:
            text, numbers = E.describe_state(cur["state"], cur["bp"])
            crit = vision.critique(vlm, photo, view(cur), text)
            cur["critique"] = (text, numbers, crit)
        cur["resemblance"] = crit.get("resemblance")
        entry = {"round": rnd, "measured": _scores(cur), "resemblance": cur["resemblance"], "combined": combined(cur),
                 "critique": {k: v for k, v in crit.items() if k != "text"}}
        scored.append((rnd, cur, _kept(hist)))
        hist.append(entry)
        log(f"  round {rnd}: combined {entry['combined']}, measured {entry['measured'].get('total')}, "
            f"resemblance {cur['resemblance']}")
        if prev is not None and _contradicts(crit, prev[1]):
            # the model now asks to undo what it just asked for (a porch added, then called extra)
            entry["reverted"] = E.label(prev[1])
            log(f"  undid {entry['reverted']}: the next critique contradicts it")
            cur, prev = prev[0], None
            if rnd == rounds:
                break
            continue
        good = entry["combined"] is not None and entry["combined"] >= TARGET and (
            cur["measured"] is None or cur["measured"] >= MEASURED_TARGET)
        if rnd == rounds or good:
            entry["stop"] = "rounds" if rnd == rounds else "target"
            break
        cands, g = E.from_critique(crit, cur["state"], numbers)
        gaps += [{"round": rnd, "gap": x} for x in g]
        cands += E.from_measures(cur["diag"], cur["state"], cur["bp"]) if cur["diag"] else []
        evaluated = []
        for e in cands:
            name = E.label(e)
            if (cur["key"], name) in tried:
                continue
            tried.add((cur["key"], name))
            new, note = E.apply(cur["state"], e, cur["bp"])
            if note:
                (invalid if note.startswith("invalid") else limits).append({"round": rnd, "edit": name, "note": note})
                continue
            if new == cur["state"]:                    # already so (a chimney it has): nothing to try
                continue
            try:
                res = evaluate(new, ctx)
            except (BlueprintError, ValueError, KeyError, TypeError, IndexError) as ex:
                invalid.append({"round": rnd, "edit": name, "note": f"{type(ex).__name__}: {ex}"[:200]})
                continue
            if res["key"] == cur["key"]:
                no_effect.append({"round": rnd, "edit": name, "why": e.get("why")})
                continue
            res["edit"] = e
            res["gain"] = (res["measured"] - cur["measured"]) if res["measured"] is not None and cur["measured"] is not None else 0.0
            evaluated.append(res)
        evaluated.sort(key=lambda r: -r["gain"])
        entry["candidates"] = [{"edit": E.label(r["edit"]), "why": r["edit"].get("why"), "gain": round(r["gain"], 3)}
                               for r in evaluated]
        # the model's own corrections and the measures' in turn, the model's first: led by the
        # measured gain alone, small changes of proportion always came first and its asks
        # (dormers, skylights) were never shown to it
        fine = [r for r in evaluated if r["gain"] > WORSE]
        short = []
        for pair in itertools.zip_longest([r for r in fine if r["edit"].get("why") == "vision"],
                                          [r for r in fine if r["edit"].get("why") != "vision"]):
            short += [r for r in pair if r is not None]
        short = short[:SHORTLIST]
        accepted = None
        for r in short:
            op, why = r["edit"]["op"], r["edit"].get("why")
            if op in SHAPE_OPS and why != "vision":
                # proportions the measures found are kept on the measures: the next critique and
                # the best-build rule catch one the model finds worse
                verdict = "kept" if r["gain"] >= SHAPE_GAIN else "too small a gain"
            else:
                # the rest is checked by critiquing the candidate on its own: asked which of two
                # builds is closer, the model picked the image shown last 59 times in 70
                text_c, numbers_c = E.describe_state(r["state"], r["bp"])
                crit_c = vision.critique(vlm, photo, view(r), text_c)
                r["critique"] = (text_c, numbers_c, crit_c)
                verdict = _verify(r["edit"], crit, crit_c, r["gain"])
            entry.setdefault("judged", []).append({"edit": E.label(r["edit"]), "verdict": verdict,
                                                   "gain": round(r["gain"], 3),
                                                   "resemblance": (r.get("critique") or (0, 0, {}))[2].get("resemblance")})
            if verdict == "kept":
                accepted = r
                break
        if accepted is None:
            entry["stop"] = "no correction helped"
            break
        entry["accepted"] = E.label(accepted["edit"])
        log(f"  kept {entry['accepted']} (measured gain {accepted['gain']:+.3f})")
        prev = (cur, accepted["edit"])
        cur = accepted
    # the best build the loop saw, not just its last: each correction looked better at the
    # time, yet a run of them can end worse (a chalet widened three times and mirrored)
    best_round, cur, kept = max(scored, key=lambda t: (combined(t[1]) or -1.0, t[1]["measured"] or -1.0))
    d = out / stem
    d.mkdir(parents=True, exist_ok=True)
    from photofirst.schem import write_schem
    for tag, res in (("first", first), ("final", cur)):
        view(res).save(d / f"{tag}.png")
        (d / f"{tag}.blueprint.json").write_text(json.dumps(res["bp"], indent=1), encoding="utf-8")
        try:
            write_schem(res["build"]["shape"], res["build"]["material"], d / f"{tag}.schem")
        except ImportError:                            # no nbtlib here (r2m_vlm): `python -m blueprint` writes it
            pass
    rec = {"id": stem, "camera": [yaw, pitch], "measured": measure is not None, "photo_measures": pf is not None,
           "first": {"measured": _scores(first), "resemblance": first.get("resemblance"), "combined": combined(first)},
           "final": {"measured": _scores(cur), "resemblance": cur.get("resemblance"), "combined": combined(cur),
                     "state": cur["state"], "round": best_round},
           "kept": kept, "rounds": hist, "gaps": gaps,
           "limits": limits, "no_effect": no_effect, "invalid": invalid, "seconds": round(time.time() - t0, 1)}
    (out / f"{stem}.refine.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return rec


# --- judge: the first build against the final one, both orders -----------------------------

def judge(stem: str, out: Path, vlm) -> dict:
    """The vision model's verdict, first build against final, asked twice with the order
    swapped: "final" or "first" only when both answers agree (with Qwen3.5-9B mostly "tie": it
    picks the image shown last; kept to test other models)."""
    from PIL import Image

    from .describe import ask, parse_json
    from .vision import COMPARE, _small
    d = out / stem
    rec = _json(out / f"{stem}.refine.json") or {}
    if not rec.get("kept"):                            # the loop changed nothing: nothing to judge
        rec = {"id": stem, "votes": [], "verdict": "unchanged"}
        (out / f"{stem}.judge.json").write_text(json.dumps(rec), encoding="utf-8")
        return rec
    photo = Image.open(out / "_work" / stem / "photo.png").convert("RGB")
    a, b = Image.open(d / "first.png"), Image.open(d / "final.png")
    votes = []
    for first_shown in ("first", "final"):
        ims = [a, b] if first_shown == "first" else [b, a]
        js = parse_json(ask(*vlm, [_small(photo), _small(ims[0]), _small(ims[1])], COMPARE, 60)) or {}
        pick = js.get("closer")
        votes.append(None if pick not in (2, 3) else (first_shown if pick == 2 else
                                                      ("final" if first_shown == "first" else "first")))
    verdict = votes[0] if votes[0] == votes[1] else "tie"
    rec = {"id": stem, "votes": votes, "verdict": verdict}
    (out / f"{stem}.judge.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def summary(out: Path) -> dict:
    """Over a run: first and final scores, the corrections kept, and the capability gaps."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.refine.json"))]
    judged = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.judge.json"))]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    s = {"photos": len(recs)}
    for tag in ("first", "final"):
        s[tag] = {"combined": mean(r[tag]["combined"] for r in recs),
                  "resemblance": mean(r[tag]["resemblance"] for r in recs),
                  "measured": mean(r[tag]["measured"].get("total") for r in recs),
                  "outline": mean(r[tag]["measured"].get("outline") for r in recs),
                  "proportions": mean(r[tag]["measured"].get("proportions") for r in recs)}
    s["changed"] = sum(1 for r in recs if r["kept"])
    s["kept"] = Counter(k.split("(")[0] for r in recs for k in r["kept"]).most_common()
    s["no_effect"] = Counter(n["edit"] for r in recs for n in r["no_effect"]).most_common(25)
    s["limits"] = Counter(n["note"] for r in recs for n in r["limits"]).most_common(10)
    s["gaps"] = Counter(g["gap"].lower() for r in recs for g in r["gaps"]).most_common(40)
    s["judge"] = dict(Counter(j["verdict"] for j in judged))
    s["seconds"] = mean(r["seconds"] for r in recs)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("prepare", "run", "judge", "summary"))
    ap.add_argument("photos", nargs="*", type=Path, help="photos, or folders of <id>/photo.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="run photos that already have a result again")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "summary":
        print(json.dumps(summary(out), indent=1))
        return 0
    photos = photos_of(args.photos)
    suffix = {"prepare": "photo", "run": "refine", "judge": "judge"}[args.stage]
    todo = [p for p in photos if args.redo or not (out / f"{stem_of(p)}.{suffix}.json").is_file()]
    if not todo:
        return 0
    from .describe import load_model
    t = time.time()
    vlm = load_model()
    seg = None
    if args.stage == "prepare":
        from . import segment
        seg = segment.load()
    print(f"models loaded in {time.time() - t:.0f} s", flush=True)
    for p in todo:
        stem = stem_of(p)
        t = time.time()
        try:
            if args.stage == "prepare":
                f = prepare(p, out, vlm, seg)
                print(f"{stem}: {time.time() - t:.0f} s, building {f['share']:.0%} of the photo, side {f['locate']['side']}, "
                      f"{len(f['openings'])} openings", flush=True)
            elif args.stage == "run":
                print(f"{stem}:", flush=True)
                r = refine_one(stem, out, vlm, args.rounds)
                print(f"{stem}: {r['first']['combined']} -> {r['final']['combined']} in {r['seconds']} s, "
                      f"kept {r['kept']}", flush=True)
            else:
                r = judge(stem, out, vlm)
                print(f"{stem}: {r['verdict']} {r['votes']}", flush=True)
        except Exception as ex:  # noqa: BLE001  one photo's failure must not stop the run
            import traceback
            print(f"{stem}: FAILED {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
