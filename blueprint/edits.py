"""The evaluator's corrections, and only these. Each one changes the form or compose's
`adjust` (`compose` turns both into blocks), so a correction can never ask the builder for
something it cannot build. The vision model's critique is asked in this list's terms
(`roof_menu`, `FEATURE_ON`, the styles and kinds) and its answers are mapped onto it
(`from_critique`); the computer measures propose their own (`from_measures`). What the model
asks for beyond the list, a correction that hits a limit (no roof steeper than 72 degrees),
and one that changes no block are logged as capability gaps: the parts to build next.

A state is {"form", "style", "kind", "adjust"}; an edit is {"op": ..., ...}.
"""
from __future__ import annotations

import copy

from .compose import ROOF_KINDS, WALLS2, kind_of, style_of
from .styles import KINDS, STYLES

PITCH_STEPS = (0.5, 1.0, 2.0, 3.0)
PITCHED = ("gable", "cross_gable", "hip", "shed", "mansard", "gambrel", "pyramid", "pagoda")   # roofs a pitch changes
PITCH_NAMES = {0.5: "low (27 degrees)", 1.0: "medium (45 degrees)", 2.0: "steep (63 degrees)",
               3.0: "very steep (72 degrees)"}
SCALE_RANGE = (0.4, 2.5)
# a feature switched on (the values a form gives it), and off
FEATURE_ON = {"porch": "front", "portico": {"columns": 4, "pediment": True}, "garage": {"doors": 1, "wall": "front"},
              "dormers": 2, "bay_windows": 1, "chimneys": 1, "shutters": True, "skylights": 2, "balconies": "some",
              "towers": {"count": 1, "shape": "square", "roof": "pyramid", "where": "corners"}, "spire": True,
              "buttresses": True, "dome": {"size": "small", "drum": False}, "minarets": 2, "pilasters": True,
              "arcade": True, "curtain_wall": True, "tiers": "setbacks", "crown": "stepped", "half_timbering": True,
              "battlements": True, "smokestacks": 1, "sawtooth": True, "big_doors": 1, "silo": True,
              "flared_eaves": True, "awning": True, "canopy": True, "sails": True, "gallery": True, "lantern": True,
              "podium": {"storeys": 3}, "units": 4, "gables": "front"}
FEATURE_OFF = {"porch": "none", "portico": {"columns": 0}, "garage": {"doors": 0, "wall": "none"},
               "balconies": "none", "towers": {"count": 0}, "dome": {"size": "none"}, "tiers": [], "crown": "flat",
               "podium": {"storeys": 0}, "units": 1, "gables": "side"}
# what the model may call a feature, as the list's name
FEATURE_SYNONYMS = {"turret": "towers", "turrets": "towers", "tower": "towers", "steeple": "spire", "columns": "portico",
                    "colonnade": "portico", "veranda": "porch", "verandah": "porch", "balcony": "balconies",
                    "dormer": "dormers", "chimney": "chimneys", "bay_window": "bay_windows", "bay": "bay_windows",
                    "skylight": "skylights", "minaret": "minarets", "setbacks": "tiers", "arches": "arcade",
                    "garage_door": "garage", "garage_doors": "garage", "cupola": "dome", "smokestack": "smokestacks",
                    "glass_facade": "curtain_wall", "crenellations": "battlements", "pinnacles": "crown",
                    "row_of_houses": "units", "row": "units", "terrace": "units", "terraced_houses": "units"}


def initial(read: dict) -> dict:
    """The state the local model's reading gives."""
    return {"form": copy.deepcopy(read.get("form") or {}), "style": style_of(read.get("style")),
            "kind": kind_of(read.get("kind")), "adjust": {}}


def _parts(form: dict) -> list:
    raw = form.get("parts")
    if not isinstance(raw, list) or not any(isinstance(p, dict) for p in raw):
        form["parts"] = [{"part": "main"}]
    form["parts"] = [p for p in form["parts"] if isinstance(p, dict)]
    return form["parts"]


def built_parts(bp: dict) -> list:
    """[(part number, its mass)] in the order a viewer sees them, left to right."""
    out = [(int(m["id"][1:]), m) for m in bp["masses"] if m["id"][:1] == "p" and m["id"][1:].isdigit()]
    out.sort(key=lambda im: im[1]["x"])
    return out


def describe_state(state: dict, bp: dict) -> tuple:
    """(text, numbers): the build as the critique sees it, its parts numbered 1, 2, ... left to
    right as drawn, and {shown number: part number}."""
    numbers, lines = {}, []
    form = state["form"]
    fparts = [p for p in (form.get("parts") or []) if isinstance(p, dict)]
    main = next((i for i, p in enumerate(fparts) if p.get("part") == "main"), 0)
    row = (bp.get("meta") or {}).get("row")
    for k, (i, m) in enumerate(built_parts(bp), 1):
        # a row built from one described house: every house is that part
        numbers[k] = main if row and len(fparts) != row["n"] else i
        p = fparts[numbers[k]] if numbers[k] < len(fparts) else {}
        r = m["roof"]
        pitch = PITCH_NAMES.get(float(r.get("pitch", 1.0)), "") if r["type"] not in ("flat", "dome") else ""
        storeys = max(1, int(m["walls"]) // max(1, int(m.get("storey_h", 5))))
        lines.append(f"- part {k}: {'house in the row' if row else p.get('part', 'main')}, {storeys} storeys, "
                     f"{r['type']} roof" + (f", {pitch}" if pitch else ""))
    feats = [f for f in FEATURE_ON if _is_on(f, form.get(f))]
    text = (f"Style {state['style']}, kind {state['kind']}.\n" + "\n".join(lines) +
            f"\nFeatures: {', '.join(feats) if feats else 'none'}.")
    return text, numbers


def _is_on(name: str, v) -> bool:
    """Whether a form's feature field says the building has it."""
    if isinstance(v, dict):
        return any(_is_on(name, x) for k, x in v.items() if k in ("columns", "doors", "count", "size", "storeys"))
    if isinstance(v, str):
        return v.lower() not in ("", "none", "no", "false", "flat", "side", "0")
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    if isinstance(v, bool) or v is None:
        return bool(v)
    try:
        return float(v) > (1 if name == "units" else 0)
    except (TypeError, ValueError):
        return False


def roof_menu() -> str:
    return "|".join(ROOF_KINDS)


def feature_menu() -> list:
    """The features the critique may call missing or extra: the form's, a row of houses by
    a plain name, and not "gables" (how a row's roofs meet the street, not a thing to add)."""
    return ["row_of_houses" if f == "units" else f for f in FEATURE_ON if f != "gables"]


def apply(state: dict, edit: dict, bp: dict | None = None) -> tuple:
    """(new state, note): note is None, or why the edit could not do all it asked (a limit),
    or "invalid: ..." when it cannot apply at all."""
    s = copy.deepcopy(state)
    form, adj = s["form"], s["adjust"]
    op = edit.get("op")
    parts = _parts(form)
    i = edit.get("part", 0)
    if op in ("storeys", "roof", "pitch", "ridge", "share", "remove_part", "window_style", "skylights_on"):
        if not isinstance(i, int) or not 0 <= i < len(parts):
            return state, f"invalid: no part {i}"
    if op == "style":
        if edit.get("value") not in STYLES:
            return state, f"invalid: no style {edit.get('value')!r}"
        s["style"] = edit["value"]
    elif op == "kind":
        if edit.get("value") not in KINDS:
            return state, f"invalid: no kind {edit.get('value')!r}"
        s["kind"] = edit["value"]
    elif op == "storeys":
        n = int(parts[i].get("storeys") or 1) + int(edit.get("delta", 0))
        if n < 1:
            return state, "limit: a part needs at least 1 storey"
        parts[i]["storeys"] = min(120, n)
    elif op == "roof":
        if edit.get("value") not in ROOF_KINDS:
            return state, f"invalid: no roof {edit.get('value')!r}"
        parts[i]["roof"] = edit["value"]
        parts[i].pop("roof_ends", None)
    elif op == "pitch":
        cur = 1.0
        if bp is not None:
            cur = next((float(m["roof"].get("pitch", 1.0)) for k, m in built_parts(bp) if k == i), 1.0)
        k = min(range(len(PITCH_STEPS)), key=lambda j: abs(PITCH_STEPS[j] - cur))
        k2 = max(0, min(len(PITCH_STEPS) - 1, k + int(edit.get("delta", 0))))
        if k2 == k:
            return state, ("limit: no pitched roof steeper than 72 degrees" if edit.get("delta", 0) > 0
                           else "limit: no pitched roof gentler than 27 degrees")
        adj.setdefault("pitch", {})[str(i)] = PITCH_STEPS[k2]
    elif op == "ridge":
        parts[i]["ridge"] = "parallel" if parts[i].get("ridge") == "perpendicular" else "perpendicular"
    elif op in ("width", "depth", "height"):
        sc = adj.setdefault("scale", {})
        v = float(sc.get(op, 1.0)) * float(edit.get("factor", 1.0))
        if not SCALE_RANGE[0] <= v <= SCALE_RANGE[1]:
            return state, f"limit: {op} already scaled {sc.get(op, 1.0):.2f}x"
        sc[op] = round(v, 3)
    elif op == "share":
        parts[i]["share"] = max(5.0, float(parts[i].get("share") or 100.0 / len(parts)) * float(edit.get("factor", 1.0)))
    elif op == "mirror":
        adj["mirror"] = not adj.get("mirror", False)
    elif op == "add_part":
        role = edit.get("role", "wing")
        new = {"part": role if role in ("wing", "tower", "annex") else "wing", "share": 30,
               "storeys": int(edit.get("storeys") or 1), "roof": edit.get("roof") or "gable"}
        if edit.get("side") == "left":
            parts.insert(0, new)
        else:
            parts.append(new)
    elif op == "remove_part":
        if len(parts) < 2:
            return state, "limit: the building's only part"
        parts.pop(i)
        if not any(p.get("part") == "main" for p in parts):
            parts[0]["part"] = "main"
    elif op == "feature":
        name, on = edit.get("name"), edit.get("on", True)
        if name not in FEATURE_ON:
            return state, f"invalid: no feature {name!r}"
        if name == "tiers":
            n = max(int(p.get("storeys") or 1) for p in parts)
            form["tiers"] = [max(1, round(n * 0.7)), max(1, n - round(n * 0.7))] if on else []
        elif on:
            form[name] = copy.deepcopy(FEATURE_ON[name])
        else:
            form[name] = copy.deepcopy(FEATURE_OFF.get(name, 0 if isinstance(FEATURE_ON[name], int) and
                                                      not isinstance(FEATURE_ON[name], bool) else False))
            if name == "towers":                        # the parts called towers no longer rise
                for p in parts:
                    if p.get("part") == "tower":
                        p["part"] = "wing"
    elif op == "windows":                              # by half again (at least one): 3 -> 5 -> 7 -> 11
        for p in parts:
            n = int(p.get("windows_across") or 2)
            p["windows_across"] = max(n + 1, round(n * 1.5)) if edit.get("delta", 0) > 0 else max(0, min(
                n - 1, round(n / 1.5)))
    elif op == "window_style":
        parts[i]["window_style"] = edit.get("value", "rect")
    elif op == "colour":
        role, word = edit.get("role"), str(edit.get("value") or "")
        if role == "wall":
            wall = form["wall"] if isinstance(form.get("wall"), dict) else {}
            form["wall"] = dict(wall, colour=word)
        elif role in ("roof", "trim", "door"):
            form[f"{role}_colour"] = word
        else:
            return state, f"invalid: no colour role {role!r}"
    elif op == "wall_material":
        if edit.get("value") not in WALLS2:
            return state, f"invalid: no wall material {edit.get('value')!r}"
        wall = form["wall"] if isinstance(form.get("wall"), dict) else {}
        form["wall"] = dict(wall, material=edit["value"])
    elif op == "skylights_on":
        adj["skylights_on"] = i
        form.setdefault("skylights", 2)
    elif op == "units":
        form["units"] = max(1, int(form.get("units") or 1) + int(edit.get("delta", 0)))
    elif op == "garage_wall":
        g = form["garage"] if isinstance(form.get("garage"), dict) else {"doors": 1}
        form["garage"] = dict(g, wall=edit.get("value", "front"))
    else:
        return state, f"invalid: no edit {op!r}"
    return s, None


def label(edit: dict) -> str:
    """A short name for an edit (logs, the report)."""
    keys = [k for k in ("part", "value", "delta", "factor", "name", "role", "on", "side") if k in edit]
    return edit["op"] + "(" + ", ".join(f"{k}={edit[k]}" for k in keys) + ")"


def _feature_name(word) -> str | None:
    w = str(word or "").lower().strip().replace(" ", "_").replace("-", "_")
    w = FEATURE_SYNONYMS.get(w, w)
    if w in FEATURE_ON:
        return w
    if w.endswith("s") and w[:-1] in FEATURE_ON:
        return w[:-1]
    return FEATURE_SYNONYMS.get(w.rstrip("s"))


def from_critique(crit: dict, state: dict, numbers: dict) -> tuple:
    """(edits, gaps) from the vision model's critique (`vision.critique`); `numbers` maps the
    critique's part numbers to the form's (`describe_state`)."""
    edits, gaps = [], []

    def part(k):
        try:
            return numbers.get(int(k))
        except (TypeError, ValueError):
            return None
    if crit.get("style_ok") is False:
        s = style_of(crit.get("style"))
        if s != state["style"] and s != "other":
            edits.append({"op": "style", "value": s})
    if crit.get("kind_ok") is False:
        k = kind_of(crit.get("kind"))
        if k != state["kind"] and k != "other":
            edits.append({"op": "kind", "value": k})
    prop = crit.get("proportions") if isinstance(crit.get("proportions"), dict) else {}
    for axis, small, big in (("width", "too_narrow", "too_wide"), ("height", "too_low", "too_tall"),
                             ("depth", "too_shallow", "too_deep")):
        v = str(prop.get(axis, "ok")).lower()
        # (not a storey more or less for "too low": the loop's three such corrections on the
        # 50-photo run were all wrong against the photos' storey counts)
        if v == small:
            edits.append({"op": axis, "factor": 1.25})
        elif v == big:
            edits.append({"op": axis, "factor": 0.8})
    roofs = crit.get("roofs") or []
    for r in [roofs] if isinstance(roofs, dict) else roofs:
        if not isinstance(r, dict):
            continue
        i = part(r.get("part"))
        if i is None:
            continue
        t = str(r.get("type") or "").lower()
        if r.get("type_ok") is False:
            if t in ROOF_KINDS:
                edits.append({"op": "roof", "part": i, "value": t})
            elif t:
                gaps.append(f"roof type {t!r} (part {r.get('part')})")
        p = str(r.get("pitch") or "ok").lower()
        if p in ("steeper", "flatter"):
            edits.append({"op": "pitch", "part": i, "delta": 1 if p == "steeper" else -1})
    for key, on in (("missing", True), ("extra", False)):
        words = crit.get(key) or []
        if isinstance(words, str):                     # "porch, dormers" rather than a list
            words = [w for w in words.split(",") if w.strip().lower() not in ("", "none")]
        for word in words if isinstance(words, list) else []:
            name = _feature_name(word)
            if name:
                edits.append({"op": "feature", "name": name, "on": on})
            elif str(word).strip():
                gaps.append(f"{'missing' if on else 'extra'} {str(word).strip()!r}")
    if crit.get("mirrored") is True:
        edits.append({"op": "mirror"})
    extra_parts = crit.get("extra_parts") or []
    for k in extra_parts if isinstance(extra_parts, list) else [extra_parts]:
        i = part(k)
        if i is not None and len({numbers[n] for n in numbers}) > 1:     # not the only part
            edits.append({"op": "remove_part", "part": i})
    side = str(crit.get("missing_part") or "none").lower()
    if side in ("left", "right"):
        edits.append({"op": "add_part", "role": "wing", "side": side, "storeys": 1})
    garage = str(crit.get("garage") or "ok").lower()
    if garage == "other_side":
        wall = (state["form"].get("garage") or {}).get("wall") if isinstance(state["form"].get("garage"), dict) else None
        edits.append({"op": "garage_wall", "value": {"left": "right", "right": "left"}[wall]} if wall in ("left", "right")
                     else {"op": "mirror"})
    elif garage == "missing":
        edits.append({"op": "feature", "name": "garage", "on": True})
    elif garage == "not_in_photo":
        edits.append({"op": "feature", "name": "garage", "on": False})
    w = str(crit.get("windows") or "ok").lower()
    if w in ("too_few", "too_many"):
        edits.append({"op": "windows", "delta": 1 if w == "too_few" else -1})
    for role in ("wall", "roof"):
        c = str(crit.get(f"{role}_colour") or "ok").strip().lower()
        if c and c not in ("ok", "same", "correct", "none"):
            edits.append({"op": "colour", "role": role, "value": c})
    other = str(crit.get("other") or "").strip()
    if other and other.lower() not in ("none", "nothing", "n/a", "-", "no"):
        gaps.append(other[:200])
    for e in edits:
        e["why"] = "vision"
    return edits, gaps


def from_measures(diag: dict, state: dict, bp: dict) -> list:
    """Edits the computer measures point to (`score.diagnose`)."""
    edits = []
    # proportions only: the measures cannot count storeys (the reading gets them right on 41
    # of 44 photos), so a storey more or less is the vision model's call. A small step, and
    # one sized to the error (to at most twice) when the shape is far off: in 15% steps a
    # 65-storey needle against a square photo never got there
    ratio = float(diag.get("aspect_ratio") or 1.0)
    if diag.get("shape") in ("taller", "wider"):
        f = min(2.0, max(1.15, (ratio if ratio > 1 else 1 / ratio) ** 0.6))
        tall = diag["shape"] == "taller"
        edits += [{"op": "height", "factor": 0.85 if tall else 1.18}, {"op": "width", "factor": 1.15 if tall else 0.87}]
        if f > 1.25:
            edits += [{"op": "height", "factor": round(1 / f if tall else f, 2)},
                      {"op": "width", "factor": round(f if tall else 1 / f, 2)}]
    shown = built_parts(bp)
    if shown:
        x0 = min(m["x"] for _, m in shown)
        x1 = max(m["x"] + m["w"] for _, m in shown)
        for k, v in enumerate(diag.get("roof_thirds") or []):
            if v not in ("higher", "lower"):
                continue
            mid = x0 + (k + 0.5) * (x1 - x0) / 3
            i, m = next(((i, m) for i, m in shown if m["x"] <= mid < m["x"] + m["w"]), (None, None))
            if i is not None and m["roof"]["type"] in PITCHED:       # a flat roof has no pitch to change
                edits.append({"op": "pitch", "part": i, "delta": -1 if v == "higher" else 1})
    if diag.get("windows") in ("more", "fewer"):
        edits.append({"op": "windows", "delta": -1 if diag["windows"] == "more" else 1})
    if diag.get("mirror_gain", 0) > 0.04:
        edits.append({"op": "mirror"})
    for role in ("wall", "roof"):
        if diag.get(f"{role}_colour"):
            edits.append({"op": "colour", "role": role, "value": diag[f"{role}_colour"]})
    for e in edits:
        e["why"] = "measures"
    out, seen = [], set()
    for e in edits:
        key = label(e)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out
